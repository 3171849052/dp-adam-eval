"""ExpV1b order: refresh, private batch, summed CE, DP noise, Wiener, SGD."""

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
    batch_size: int,
    algorithm: str,
) -> dict:
    """Shared DP mechanism; the optional filter consumes only DP noisy gradients."""
    if algorithm not in ("dp_sgd", "dp_fisher_wiener"):
        raise ValueError(algorithm)
    device = next(model.parameters()).device
    sq = _compute_per_sample_norms_squared(list(model.parameters()), batch_size, device)
    require_finite(sq)
    clip_rate = float((sq.sqrt() > max_grad_norm).float().mean())
    with rng.use():
        clip_and_noise_gradients(
            model, sigma, max_grad_norm, batch_size, store_summed_grad=True
        )
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
        clip_rate=clip_rate,
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
    return dict(test_loss=loss / count, test_accuracy=correct / count)


def train(c: Config, log: RunLog, data_override: tuple | None = None) -> dict:
    """Run one YAML-defined experiment; preserve partial outputs on every failure."""
    c.validate()
    started = time.perf_counter()
    model = None
    step = 0
    active = None
    current_step = 0
    accountant = RDPAccountant()
    sigma = q = None
    evaluations = []
    refresh_times = []
    filter_times = []
    summary = dict(
        status="failed",
        algorithm=c.algorithm,
        seed=c.seed,
        completed_steps=0,
        epsilon_target=c.privacy.epsilon,
        delta=c.privacy.delta,
        learning_rate=c.training.learning_rate,
        max_grad_norm=c.privacy.max_grad_norm,
    )
    try:
        device = configure_runtime(c)
        loader, test_loader = build_data(c, data_override)
        total = len(loader) * c.training.epochs
        q = c.data.batch_size / len(loader.dataset)
        sigma = get_noise_multiplier(
            target_epsilon=c.privacy.epsilon,
            target_delta=c.privacy.delta,
            sample_rate=q,
            steps=total,
            accountant="rdp",
        )
        set_seed(c.seed)
        model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=c.training.learning_rate, momentum=0, weight_decay=0
        )
        synthetic_rng = RNGStream(c.seed + 3, device)
        noise_rng = RNGStream(c.seed + 4, device)
        resolved = c.to_dict()
        resolved.update(
            device=str(device),
            physical_gpu_index=c.runtime.gpu,
            gpu_name=(
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            dataset="mnist",
            train_size=len(loader.dataset),
            test_size=len(test_loader.dataset),
            epochs=c.training.epochs,
            batch_size=c.data.batch_size,
            total_steps=total,
            learning_rate=c.training.learning_rate,
            epsilon_target=c.privacy.epsilon,
            delta=c.privacy.delta,
            noise_multiplier=sigma,
            sample_rate=q,
            max_grad_norm=c.privacy.max_grad_norm,
            accountant="rdp",
            sampling=c.privacy.sampling,
            accounting_convention=c.privacy.accounting_convention,
            poisson_sampling=False,
            rng_seeds=dict(
                zip(
                    ["init", "loader", "test", "synthetic", "noise"],
                    range(c.seed, c.seed + 5),
                )
            ),
        )
        resolved["wiener"]["enabled"] = c.algorithm == "dp_fisher_wiener"
        write_yaml(log.root / "resolved_config.yaml", resolved)
        summary["initial_model_hash"] = digest(model.parameters())
        for epoch in range(1, c.training.epochs + 1):
            iterator = iter(loader)
            for _ in range(len(loader)):
                current_step = step
                refreshed = (
                    c.algorithm == "dp_fisher_wiener"
                    and step % c.wiener.refresh_interval == 0
                )
                elapsed = 0.0
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
                    elapsed = time.perf_counter() - refresh_start
                    refresh_times.append(elapsed)
                x, y = next(iterator)
                model.zero_grad(set_to_none=True)
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
                    len(x),
                    c.algorithm,
                )
                accountant.step(noise_multiplier=sigma, sample_rate=q)
                completed_step = step
                step += 1
                require_finite(*model.parameters())
                row.update(
                    step=completed_step,
                    epoch=epoch,
                    algorithm=c.algorithm,
                    learning_rate=c.training.learning_rate,
                    train_loss=float(loss.detach()) / len(x),
                    test_loss=None,
                    test_accuracy=None,
                    epsilon_spent=float(accountant.get_epsilon(c.privacy.delta)),
                    filter_refresh=refreshed,
                    refresh_time=elapsed,
                    active_state_bytes=state_bytes(active),
                )
                if step % c.logging.eval_interval == 0 or step == total:
                    ev = evaluate(model, test_loader, device)
                    row.update(ev)
                    evaluations.append(ev)
                log.append(row)
                filter_times.append(row["filter_time"])
        summary["status"] = "completed"
    except FloatingPointError as error:
        summary.update(status="diverged", diverged_step=current_step, error=str(error))
        log.logger.exception("Training diverged")
    except Exception as error:
        summary["error"] = str(error)
        log.logger.exception("Training failed")
        raise
    finally:
        summary.update(
            completed_steps=step,
            noise_multiplier=sigma,
            sample_rate=q,
            epsilon_spent=(
                float(accountant.get_epsilon(c.privacy.delta)) if step else 0.0
            ),
            final_accuracy=evaluations[-1]["test_accuracy"] if evaluations else None,
            best_accuracy=(
                max(e["test_accuracy"] for e in evaluations) if evaluations else None
            ),
            final_test_loss=evaluations[-1]["test_loss"] if evaluations else None,
            wall_time=time.perf_counter() - started,
        )
        if model is not None:
            summary["final_model_hash"] = digest(model.parameters())
            model.remove_hooks()
        if c.algorithm == "dp_fisher_wiener":
            summary.update(
                number_of_refreshes=len(refresh_times),
                total_refresh_time=sum(refresh_times),
                mean_refresh_time=sum(refresh_times) / max(len(refresh_times), 1),
                total_filter_time=sum(filter_times),
                mean_filter_time=sum(filter_times) / max(len(filter_times), 1),
                active_state_bytes=state_bytes(active),
            )
        write_json(log.root / "summary.json", summary)
        log.logger.info("Summary: %s", summary)
    return summary
