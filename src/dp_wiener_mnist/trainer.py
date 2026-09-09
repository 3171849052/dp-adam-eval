"""DP-SGD/Fisher-Wiener training with Poisson accounting semantics."""

from __future__ import annotations

from collections.abc import Callable
import os
import time

import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

from .config import Config
from .data import build_data
from .model import SimpleCNN
from .privacy import clip_and_noise_gradients, _compute_per_sample_norms_squared
from .fisher_wiener import (
    synthetic_samples,
    build_covariances,
    build_fisher_state,
    apply_fisher_wiener,
    state_bytes,
)
from .utils import RNGStream, set_seed, digest, configure_runtime, require_finite
from .run_logging import RunLog, write_json, write_yaml


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gradient_norm(model: torch.nn.Module) -> float:
    return float(
        torch.cat([p.grad.detach().flatten() for p in model.parameters()]).norm()
    )


def private_update(
    model: torch.nn.Module,
    active: dict | None,
    optimizer: torch.optim.Optimizer,
    rng: RNGStream,
    sigma: float,
    max_grad_norm: float,
    sample_count: int,
    algorithm: str,
    expected_batch_size: int | None = None,
    on_private_mechanism: Callable[[], None] | None = None,
) -> dict:
    """Run one DP mechanism, optional Wiener post-processing, and SGD.

    The callback is invoked immediately after the Gaussian mechanism returns,
    before any Fisher-Wiener operation or optimizer step.  It is deliberately
    optional so the fixed-batch trajectory regression can call this low-level
    routine without an accountant.
    """
    if algorithm not in ("dp_sgd", "dp_fisher_wiener"):
        raise ValueError(algorithm)
    if expected_batch_size is None:
        expected_batch_size = sample_count
    if sample_count < 0 or expected_batch_size <= 0:
        raise ValueError("Invalid sample counts")

    device = next(model.parameters()).device
    if sample_count:
        sq = _compute_per_sample_norms_squared(
            [p for p in model.parameters() if p.requires_grad], sample_count, device
        )
        require_finite(sq)
        clipped_count = int((sq.sqrt() > max_grad_norm).sum())
    else:
        clipped_count = 0

    with rng.use():
        clip_and_noise_gradients(
            model,
            sigma,
            max_grad_norm,
            sample_count=sample_count,
            expected_batch_size=expected_batch_size,
            store_summed_grad=True,
        )
    # Once this point is reached, the Gaussian mechanism has happened.  Any
    # subsequent Fisher or optimizer failure must not erase its privacy cost.
    if on_private_mechanism is not None:
        on_private_mechanism()

    require_finite(*(p.grad for p in model.parameters()))
    norm = gradient_norm(model)
    sync(device)
    start = time.perf_counter()
    if algorithm == "dp_fisher_wiener":
        apply_fisher_wiener(model, active)
    sync(device)
    elapsed = time.perf_counter() - start if algorithm == "dp_fisher_wiener" else 0.0
    require_finite(*(p.grad for p in model.parameters()))
    filtered = gradient_norm(model)
    optimizer.step()
    return dict(
        sample_count=sample_count,
        clipped_count=clipped_count,
        clip_rate=(clipped_count / sample_count if sample_count else None),
        gradient_norm_after_dp=norm,
        filtered_gradient_norm=filtered,
        filter_time=elapsed,
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    loss = correct = count = 0
    try:
        for x, y in loader:
            y = y.to(device)
            output = model(x.to(device))
            require_finite(output)
            batch_loss = F.cross_entropy(output, y, reduction="sum")
            require_finite(batch_loss)
            loss += float(batch_loss)
            correct += int((output.argmax(1) == y).sum())
            count += len(y)
    finally:
        model.train()
    if count == 0:
        raise ValueError("Evaluation dataset is empty")
    return dict(test_loss=loss / count, test_accuracy=correct / count)


def _resolved_config(
    c: Config,
    *,
    device: torch.device | None,
    train_size: int | None,
    test_size: int | None,
    steps_per_epoch: int | None,
    total_steps: int | None,
    sigma: float | None,
    sample_rate: float | None,
    privacy_steps: int,
    epsilon_spent: float | None,
    run_directory,
) -> dict:
    resolved = c.to_dict()
    resolved["data"].update(train_size=train_size, test_size=test_size)
    resolved["privacy"].update(
        sample_rate=sample_rate,
        noise_multiplier=sigma,
        expected_batch_size=c.data.batch_size,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        privacy_steps=privacy_steps,
        epsilon_spent=epsilon_spent,
    )
    actual_device = str(device) if device is not None else None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_name = None
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        visible_index = 0 if visible is not None else c.runtime.gpu
        try:
            gpu_name = torch.cuda.get_device_name(visible_index)
        except (AssertionError, IndexError, RuntimeError):
            gpu_name = None
    resolved["runtime"].update(
        actual_device=actual_device,
        physical_gpu_index=c.runtime.gpu,
        cuda_visible_devices=visible,
        gpu_name=gpu_name,
    )
    resolved["rng"] = dict(
        init_seed=c.seed,
        train_sampler_seed=c.seed + 1,
        test_seed=c.seed + 2,
        synthetic_seed=c.seed + 3,
        noise_seed=c.seed + 4,
    )
    resolved["run"] = {"directory": str(run_directory.resolve())}
    resolved["wiener"]["enabled"] = c.algorithm == "dp_fisher_wiener"
    return resolved


def train(c: Config, log: RunLog, data_override: tuple | None = None) -> dict:
    """Run one experiment and preserve partial outputs on every failure."""
    c.validate()
    started = time.perf_counter()
    model = None
    device = None
    train_size = test_size = None
    steps_per_epoch = total_steps = None
    sigma = sample_rate = None
    active = None
    completed_steps = 0
    privacy_steps = 0
    epochs_completed = 0
    current_step = None
    accountant = RDPAccountant()
    evaluations = []
    all_refresh_times = []
    all_filter_times = []
    summary = dict(
        status="failed",
        algorithm=c.algorithm,
        seed=c.seed,
        planned_steps=None,
        privacy_steps=0,
        completed_steps=0,
        epochs_completed=0,
        epsilon_target=c.privacy.epsilon,
        epsilon_spent=0.0,
        delta=c.privacy.delta,
        noise_multiplier=None,
        sample_rate=None,
        expected_batch_size=c.data.batch_size,
        final_accuracy=None,
        best_accuracy=None,
        final_test_loss=None,
        final_model_hash=None,
        number_of_refreshes=0,
        total_refresh_time=0.0,
        mean_refresh_time=0.0,
        total_filter_time=0.0,
        mean_filter_time=0.0,
        active_state_bytes=0,
        wall_time=0.0,
        diverged_step=None,
        error=None,
    )

    def account_private_mechanism() -> None:
        nonlocal privacy_steps
        accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
        privacy_steps += 1

    try:
        device = configure_runtime(c)
        loader, test_loader = build_data(c, data_override)
        train_size = len(loader.dataset)
        test_size = len(test_loader.dataset)
        steps_per_epoch = train_size // c.data.batch_size
        total_steps = steps_per_epoch * c.training.epochs
        sample_rate = c.data.batch_size / train_size
        sigma = get_noise_multiplier(
            target_epsilon=c.privacy.epsilon,
            target_delta=c.privacy.delta,
            sample_rate=sample_rate,
            steps=total_steps,
            accountant="rdp",
        )
        set_seed(c.seed)
        model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=c.training.learning_rate, momentum=0, weight_decay=0
        )
        synthetic_rng = RNGStream(c.seed + 3, device)
        noise_rng = RNGStream(c.seed + 4, device)
        write_yaml(
            log.root / "resolved_config.yaml",
            _resolved_config(
                c,
                device=device,
                train_size=train_size,
                test_size=test_size,
                steps_per_epoch=steps_per_epoch,
                total_steps=total_steps,
                sigma=sigma,
                sample_rate=sample_rate,
                privacy_steps=privacy_steps,
                epsilon_spent=0.0,
                run_directory=log.root,
            ),
        )
        summary["initial_model_hash"] = digest(model.parameters())

        for epoch in range(1, c.training.epochs + 1):
            epoch_loss_sum = 0.0
            epoch_sampled = 0
            epoch_clipped = 0
            epoch_batch_sizes = []
            epoch_dp_norms = []
            epoch_filtered_norms = []
            epoch_filter_times = []
            epoch_refresh_times = []
            iterator = iter(loader)
            for _ in range(steps_per_epoch):
                current_step = completed_steps
                refreshed = (
                    c.algorithm == "dp_fisher_wiener"
                    and completed_steps % c.wiener.refresh_interval == 0
                )
                if refreshed:
                    sync(device)
                    refresh_start = time.perf_counter()
                    samples, _ = synthetic_samples(
                        c.fisher_config(), device, synthetic_rng
                    )
                    cov = build_covariances(
                        model._module.state_dict(), samples, c.fisher_config(), device
                    )
                    active = build_fisher_state(
                        cov, sigma, c.privacy.max_grad_norm, c.data.batch_size
                    )
                    del samples, cov
                    sync(device)
                    refresh_elapsed = time.perf_counter() - refresh_start
                    epoch_refresh_times.append(refresh_elapsed)
                    all_refresh_times.append(refresh_elapsed)

                batch = next(iterator)
                model.zero_grad(set_to_none=True)
                if batch is None:
                    sample_count = 0
                    loss = None
                else:
                    x, y = batch
                    sample_count = len(x)
                    loss = F.cross_entropy(
                        model(x.to(device)), y.to(device), reduction="sum"
                    )
                    require_finite(loss)
                    loss.backward()

                row = private_update(
                    model,
                    active,
                    optimizer,
                    noise_rng,
                    sigma,
                    c.privacy.max_grad_norm,
                    sample_count,
                    c.algorithm,
                    expected_batch_size=c.data.batch_size,
                    on_private_mechanism=account_private_mechanism,
                )
                # optimizer.step() returned successfully; count this step even
                # if the post-step finite check detects a divergence.
                completed_steps += 1
                require_finite(*model.parameters())

                epoch_batch_sizes.append(sample_count)
                epoch_sampled += row["sample_count"]
                epoch_clipped += row["clipped_count"]
                epoch_dp_norms.append(row["gradient_norm_after_dp"])
                epoch_filtered_norms.append(row["filtered_gradient_norm"])
                epoch_filter_times.append(row["filter_time"])
                all_filter_times.append(row["filter_time"])
                if loss is not None:
                    epoch_loss_sum += float(loss.detach())

            ev = evaluate(model, test_loader, device)
            evaluations.append(ev)
            epsilon_spent = float(accountant.get_epsilon(c.privacy.delta))
            log.append(
                dict(
                    epoch=epoch,
                    global_step=completed_steps,
                    privacy_steps=privacy_steps,
                    train_loss=(
                        epoch_loss_sum / epoch_sampled if epoch_sampled else None
                    ),
                    test_loss=ev["test_loss"],
                    test_accuracy=ev["test_accuracy"],
                    epsilon_spent=epsilon_spent,
                    noise_multiplier=sigma,
                    sample_rate=sample_rate,
                    expected_batch_size=c.data.batch_size,
                    mean_sampled_batch_size=(
                        sum(epoch_batch_sizes) / len(epoch_batch_sizes)
                    ),
                    total_sampled_examples=epoch_sampled,
                    clip_rate=(
                        epoch_clipped / epoch_sampled if epoch_sampled else None
                    ),
                    gradient_norm_after_dp=(
                        sum(epoch_dp_norms) / len(epoch_dp_norms)
                    ),
                    filtered_gradient_norm=(
                        sum(epoch_filtered_norms) / len(epoch_filtered_norms)
                    ),
                    refresh_count=len(epoch_refresh_times),
                    mean_refresh_time=(
                        sum(epoch_refresh_times) / len(epoch_refresh_times)
                        if epoch_refresh_times
                        else 0.0
                    ),
                    mean_filter_time=(
                        sum(epoch_filter_times) / len(epoch_filter_times)
                        if epoch_filter_times
                        else 0.0
                    ),
                    active_state_bytes=state_bytes(active),
                )
            )
            epochs_completed = epoch

        summary["status"] = "completed"
    except FloatingPointError as error:
        summary.update(status="diverged", diverged_step=current_step, error=str(error))
        log.logger.exception("Training diverged")
    except Exception as error:
        summary.update(status="failed", error=str(error))
        log.logger.exception("Training failed")
        raise
    finally:
        epsilon_spent = (
            float(accountant.get_epsilon(c.privacy.delta)) if privacy_steps else 0.0
        )
        summary.update(
            planned_steps=total_steps,
            privacy_steps=privacy_steps,
            completed_steps=completed_steps,
            epochs_completed=epochs_completed,
            noise_multiplier=sigma,
            sample_rate=sample_rate,
            epsilon_spent=epsilon_spent,
            final_accuracy=evaluations[-1]["test_accuracy"] if evaluations else None,
            best_accuracy=(
                max(e["test_accuracy"] for e in evaluations) if evaluations else None
            ),
            final_test_loss=evaluations[-1]["test_loss"] if evaluations else None,
            number_of_refreshes=len(all_refresh_times),
            total_refresh_time=sum(all_refresh_times),
            mean_refresh_time=sum(all_refresh_times) / max(len(all_refresh_times), 1),
            total_filter_time=sum(all_filter_times),
            mean_filter_time=sum(all_filter_times) / max(len(all_filter_times), 1),
            active_state_bytes=state_bytes(active),
            wall_time=time.perf_counter() - started,
        )
        if model is not None:
            summary["final_model_hash"] = digest(model.parameters())
            model.remove_hooks()
        if device is not None:
            write_yaml(
                log.root / "resolved_config.yaml",
                _resolved_config(
                    c,
                    device=device,
                    train_size=train_size,
                    test_size=test_size,
                    steps_per_epoch=steps_per_epoch,
                    total_steps=total_steps,
                    sigma=sigma,
                    sample_rate=sample_rate,
                    privacy_steps=privacy_steps,
                    epsilon_spent=epsilon_spent,
                    run_directory=log.root,
                ),
            )
        write_json(log.root / "summary.json", summary)
        log.logger.info("Summary: %s", summary)
    return summary
