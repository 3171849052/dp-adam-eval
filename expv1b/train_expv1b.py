"""ExpV1b training harness: ExpV1 Fisher-Wiener with a fixed LR sweep."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from torch.utils.data import DataLoader, Dataset, Subset

from expv1b.common import (
    FISHER_LRS,
    LAYERS,
    ROOT,
    add_effective_signal_metrics,
    check_config,
    datasets,
    digest,
    expected_total_steps,
    fingerprint,
    output_path,
    provenance,
    require_pinned,
    run_spec,
    run_specs,
    save_json,
    set_seed,
    synthetic_batches,
    threshold_steps,
    write_csv,
    RNGStream,
    SimpleCNN,
    read_config,
)
from expv1.fisher_wiener import (
    apply_fisher_wiener,
    build_covariances,
    build_diagnostic_state,
    build_fisher_state,
    pack_layer_gradient,
    state_bytes,
    synthetic_samples,
)
from expv1.metrics import diagnose
from dp_kfac.privacy import (
    _compute_per_sample_norms_squared,
    clip_and_noise_gradients,
)


TRAIN_FIELDS = [
    "seed", "method", "config_fingerprint", "step", "epoch", "train_loss",
    "test_loss", "test_accuracy", "clip_rate", "clean_clipped_norm",
    "actual_noise_norm", "noisy_gradient_norm", "filtered_gradient_norm",
    "relmse_noisy", "relmse_filtered", "cosine_noisy", "cosine_filtered",
    "signal_retention", "noise_retention", "snr_in", "snr_out", "snr_gain_db",
    "mse_reduction", "learning_rate", "signal_amplitude_retention",
    "effective_signal_lr", "optimizer_update_norm", "clean_reference_update_norm",
    "update_to_clean_reference_ratio", "refresh_time", "diagnostic_spectrum_time",
    "wiener_filter_time", "active_state_bytes",
]
LAYER_FIELDS = [
    "seed", "method", "config_fingerprint", "step", "layer", "clean_clipped_norm",
    "actual_noise_norm", "noisy_gradient_norm", "filtered_gradient_norm",
    "relmse_noisy", "relmse_filtered", "cosine_noisy", "cosine_filtered",
    "signal_retention", "noise_retention", "snr_in", "snr_out", "snr_gain_db",
    "mse_reduction", "kappa", "H_mean", "H_std", "H_min", "H_max", "H_q10",
    "H_q25", "H_q50", "H_q75", "H_q90", "trace_A", "trace_G", "trace_F",
    "lambdaF_mean", "lambdaF_median", "lambdaF_q10", "lambdaF_q90",
    "learning_rate", "signal_amplitude_retention", "effective_signal_lr",
    "optimizer_update_norm", "clean_reference_update_norm",
    "update_to_clean_reference_ratio",
]
REFRESH_FIELDS = [
    "seed", "method", "config_fingerprint", "step", "refresh_time",
    "diagnostic_spectrum_time", "active_state_bytes",
]
EIGEN_FIELDS = [
    "seed", "method", "config_fingerprint", "step", "layer", "bin", "count",
    "lambda_min", "lambda_max", "lambda_mean", "log10_lambda_mean", "signal_power",
    "noise_power", "empirical_snr", "H_mean", "mse_before", "mse_after",
    "mse_reduction", "signal_retention", "noise_retention",
]


def sync(model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _all_finite(parameters):
    return all(bool(torch.isfinite(parameter).all()) for parameter in parameters)


def _numeric_values_finite(row):
    return all(
        math.isfinite(float(value))
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


class DivergenceError(FloatingPointError):
    """A run stopped at the first non-finite loss, gradient, or update."""

    def __init__(self, step, stage):
        super().__init__(f"non-finite {stage} at step {step}")
        self.step = step
        self.stage = stage


def make_optimizer(model, lr, c=None):
    """Construct the only permitted optimizer; LR is the run-specific variable."""
    if c is not None:
        if c["optimizer"] != "SGD" or c["momentum"] != 0:
            raise ValueError("ExpV1b requires plain SGD with momentum=0")
    if not math.isfinite(float(lr)) or float(lr) <= 0:
        raise ValueError("learning rate must be finite and positive")
    return torch.optim.SGD(model.parameters(), lr=float(lr), momentum=0)


def private_update(
    model, active, optimizer, rng, sigma, c, batch_size, method, learning_rate,
    diagnostics=True, refresh=False, eigen_budget=0, diagnostic_state=None,
):
    """Run the unchanged ExpV1 private update and add post-hoc diagnostics."""
    if method not in ("dp_sgd", "dp_fisher_wiener"):
        raise ValueError(method)
    diagnostic_seconds = 0.0
    clip_rate = None
    if diagnostics:
        sync(model)
        started = time.perf_counter()
        sq = _compute_per_sample_norms_squared(
            list(model.parameters()), batch_size, next(model.parameters()).device
        )
        clip_rate = float((sq.sqrt() > c["max_grad_norm"]).float().mean())
        sync(model)
        diagnostic_seconds += time.perf_counter() - started

    audit = {"noise_rng_before": rng.audit()}
    # Exact ExpV1 order: raw per-example gradients -> clipping/noise.
    with rng.use():
        clip_and_noise_gradients(
            model,
            noise_multiplier=sigma,
            max_grad_norm=c["max_grad_norm"],
            batch_size=batch_size,
            store_summed_grad=True,
        )
    audit["noise_rng_after"] = rng.audit()
    if not _all_finite(p.grad for p in model.parameters()):
        raise DivergenceError(-1, "noisy gradient")

    if diagnostics:
        sync(model)
        started = time.perf_counter()
        noisy = {name: pack_layer_gradient(getattr(model._module, name)) for name in LAYERS}
        sync(model)
        diagnostic_seconds += time.perf_counter() - started

    sync(model)
    started = time.perf_counter()
    if method == "dp_fisher_wiener":
        apply_fisher_wiener(model, active)
    elif method != "dp_sgd":
        raise ValueError(method)
    sync(model)
    filter_time = time.perf_counter() - started if method != "dp_sgd" else 0.0
    if not _all_finite(p.grad for p in model.parameters()):
        raise DivergenceError(-1, "filtered gradient")

    optimizer.step()
    rows, layers, bins = {}, [], []
    if diagnostics:
        sync(model)
        started = time.perf_counter()
        rows, layers, bins = diagnose(
            model, noisy, active, diagnostic_state, method, refresh, eigen_budget
        )
        del noisy
        sync(model)
        diagnostic_seconds += time.perf_counter() - started
        add_effective_signal_metrics(rows, learning_rate)
        for layer_row in layers:
            add_effective_signal_metrics(layer_row, learning_rate)
    rows.update(clip_rate=clip_rate, wiener_filter_time=filter_time)
    return rows, layers, bins, audit, diagnostic_seconds


class Indexed(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        return x, y, index


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss = correct = count = 0
    for x, y in loader:
        y = y.to(device)
        output = model(x.to(device))
        loss += float(F.cross_entropy(output, y, reduction="sum"))
        correct += int((output.argmax(1) == y).sum())
        count += len(y)
    model.train()
    return {"test_loss": loss / count, "test_accuracy": correct / count}


def _summary_from_rows(
    c, spec, model, rows, refreshes, accountant, sigma, step, wall, diagnostic_seconds,
    active, status, diverged_step, last_finite_accuracy, last_finite_loss,
):
    evaluations = [row for row in rows if row.get("test_accuracy") is not None]
    finite_evaluations = [row for row in evaluations if math.isfinite(float(row["test_accuracy"]))]
    accuracies = [float(row["test_accuracy"]) for row in finite_evaluations]
    if accuracies:
        progress = np.array([(row["step"] + 1) / max(step, 1) for row in finite_evaluations])
        if len(accuracies) > 1:
            trapezoid = np.trapezoid if hasattr(np, "trapezoid") else lambda y, x: ((y[1:] + y[:-1]) * np.diff(x) / 2).sum()
            auc = float(trapezoid(accuracies, progress) / (progress[-1] - progress[0]))
        else:
            auc = accuracies[0]
        late = [row["test_accuracy"] for row in finite_evaluations if row["step"] + 1 > step / 2]
    else:
        auc, late = None, []
    total_refresh = sum(float(row["refresh_time"]) for row in refreshes)
    total_spectrum = sum(float(row["diagnostic_spectrum_time"]) for row in refreshes)
    total_filter = sum(float(row.get("wiener_filter_time", 0.0)) for row in rows)
    epsilon_spent = float(accountant.get_epsilon(c["delta"]))
    complete = status == "completed"
    summary = dict(
        seed=c["seed"], run_id=spec["run_id"], method=spec["method"],
        learning_rate=spec["learning_rate"], fingerprint=fingerprint(c),
        status=status, planned_steps=expected_total_steps(c), completed_steps=step,
        epsilon_spent=epsilon_spent, epsilon_spent_at_divergence=epsilon_spent if not complete else None,
        noise_multiplier=sigma, sample_rate=c["batch_size"] / (c["train_subset"] or 60000),
        final_accuracy=accuracies[-1] if complete and accuracies else None,
        best_accuracy=max(accuracies) if complete and accuracies else None,
        late_mean_accuracy=float(np.mean(late)) if complete and late else None,
        accuracy_auc=auc if complete else None,
        final_test_loss=(finite_evaluations[-1].get("test_loss") if complete and finite_evaluations else None),
        final_model_hash=digest(model.parameters()), parameters_finite=_all_finite(model.parameters()),
        steps_to_acc_50=threshold_steps(finite_evaluations, 0.50) if complete else None,
        steps_to_acc_70=threshold_steps(finite_evaluations, 0.70) if complete else None,
        steps_to_acc_80=threshold_steps(finite_evaluations, 0.80) if complete else None,
        steps_to_acc_85=threshold_steps(finite_evaluations, 0.85) if complete else None,
        wall_time=wall, diagnostic_seconds=diagnostic_seconds,
        core_training_runtime=wall - diagnostic_seconds,
        total_refresh_time=total_refresh, mean_refresh_time=total_refresh / len(refreshes) if refreshes else 0.0,
        number_of_refreshes=len(refreshes), total_diagnostic_spectrum_time=total_spectrum,
        mean_diagnostic_spectrum_time=total_spectrum / len(refreshes) if refreshes else 0.0,
        total_wiener_filter_time=total_filter,
        mean_wiener_filter_time=total_filter / max(step, 1),
        active_state_bytes=state_bytes(active),
        peak_cuda_allocated_memory=(torch.cuda.max_memory_allocated(next(model.parameters()).device)
                                    if next(model.parameters()).device.type == "cuda" else 0),
        diverged_step=diverged_step,
        last_finite_accuracy=last_finite_accuracy,
        last_finite_loss=last_finite_loss,
        single_seed_descriptive_only=True,
    )
    return summary


def train(c, run_id, output, data_override=None, *, diagnostics=True, eigen_budget=0):
    """Train one canonical run.  ``output`` is always confined to expv1b/runs."""
    check_config(c)
    spec = run_spec(run_id, c)
    output = output_path(output)
    root = output / f"seed{c['seed']}" / run_id
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "config.json", c)

    current_provenance = provenance()
    require_pinned(current_provenance, c["smoke"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if c["device"] == "auto" else torch.device(c["device"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    source, test = datasets(c) if data_override is None else data_override
    n = c["train_subset"] or len(source)
    if not c["batch_size"] <= n <= len(source):
        raise ValueError("Invalid training subset")
    data = Subset(source, range(n))
    if c["test_subset"] is not None:
        if not 1 <= c["test_subset"] <= len(test):
            raise ValueError("Invalid test subset")
        test = Subset(test, range(c["test_subset"]))
    loader = DataLoader(
        Indexed(data), batch_size=c["batch_size"], shuffle=True, drop_last=True,
        num_workers=0, generator=torch.Generator().manual_seed(c["seed"] + 1),
    )
    test_loader = DataLoader(
        test, batch_size=c["batch_size"], num_workers=0,
        generator=torch.Generator().manual_seed(c["seed"] + 2),
    )
    total, q = len(loader) * c["epochs"], c["batch_size"] / n
    sigma = get_noise_multiplier(
        target_epsilon=c["epsilon"], target_delta=c["delta"], sample_rate=q,
        steps=total, accountant="rdp",
    )

    # Keep this initialization point identical to ExpV1 for trajectory regression.
    set_seed(c["seed"])
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    optimizer = make_optimizer(model, spec["learning_rate"], c)
    accountant = RDPAccountant()
    synthetic_rng = RNGStream(c["seed"] + 3, device)
    noise_rng = RNGStream(c["seed"] + 4, device)
    meta = dict(
        seed=c["seed"], run_id=run_id, method=spec["method"], learning_rate=spec["learning_rate"],
        fingerprint=fingerprint(c), provenance=current_provenance, device=str(device),
        noise_multiplier=sigma, sample_rate=q, total_steps=total,
        target_epsilon=c["epsilon"], target_delta=c["delta"],
        accountant="rdp", max_grad_norm=c["max_grad_norm"],
        initial_model_hash=digest(model.parameters()), beta=1, covariance_ridge=1e-5,
        rng_seeds=dict(init=c["seed"], loader=c["seed"] + 1, test=c["seed"] + 2,
                       synthetic=c["seed"] + 3, noise=c["seed"] + 4),
        sampling="fixed shuffle/drop_last; inherited RDP convention, not Poisson",
        diagnostics_enabled=diagnostics, eigen_budget=0, eigenmode_diagnostics=False,
        diagnostics="Non-DP research statistics, never consumed by training",
        output=str(root), optimizer="SGD", momentum=0,
        filter_position="after global clipping and DP Gaussian noise",
    )
    save_json(root / "metadata.json", meta)

    rows, layers, bins, refreshes, audits, synthetic_audits = [], [], [], [], [], []
    active = diagnostic_state = None
    step = 0
    diagnostic_seconds = 0.0
    status = "completed"
    diverged_step = None
    last_finite_accuracy = None
    last_finite_loss = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(model)
    started = time.perf_counter()
    try:
        for epoch in range(1, c["epochs"] + 1):
            iterator = iter(loader)
            for _ in range(len(loader)):
                refreshed = step % c["K"] == 0 and spec["method"] == "dp_fisher_wiener"
                refresh_time = 0.0
                diagnostic_spectrum_time = 0.0
                # Refresh happens before consuming the next private batch.
                if refreshed:
                    sync(model)
                    refresh_started = time.perf_counter()
                    samples, synthetic_audit = synthetic_samples(c, device, synthetic_rng)
                    covariances = build_covariances(model._module.state_dict(), samples, c, device)
                    active = build_fisher_state(covariances, sigma, c["max_grad_norm"], c["batch_size"])
                    sync(model)
                    refresh_time = time.perf_counter() - refresh_started
                    if diagnostics:
                        sync(model)
                        spectrum_started = time.perf_counter()
                        diagnostic_state = build_diagnostic_state(covariances, active, spec["method"])
                        sync(model)
                        diagnostic_spectrum_time = time.perf_counter() - spectrum_started
                        diagnostic_seconds += diagnostic_spectrum_time
                    else:
                        diagnostic_state = None
                    del samples, covariances
                    refreshes.append(dict(
                        step=step, refresh_time=refresh_time,
                        diagnostic_spectrum_time=diagnostic_spectrum_time,
                        active_state_bytes=state_bytes(active),
                    ))
                    synthetic_audits.append(dict(step=step, **synthetic_audit))

                x, y, indices = next(iterator)
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
                if not bool(torch.isfinite(loss)):
                    status, diverged_step = "diverged", step
                    break
                loss.backward()
                try:
                    row, layer_rows, eigen_rows, audit, elapsed = private_update(
                        model, active, optimizer, noise_rng, sigma, c, len(x), spec["method"],
                        spec["learning_rate"], diagnostics, refreshed, 0, diagnostic_state,
                    )
                except DivergenceError as exc:
                    status, diverged_step = "diverged", step
                    break
                diagnostic_seconds += elapsed
                accountant.step(noise_multiplier=sigma, sample_rate=q)
                row.update(
                    step=step, epoch=epoch, train_loss=float(loss.detach()) / len(x),
                    refresh_time=refresh_time, diagnostic_spectrum_time=diagnostic_spectrum_time,
                    active_state_bytes=state_bytes(active), test_loss=None, test_accuracy=None,
                )
                parameters_finite = _all_finite(model.parameters())
                diagnostics_finite = _numeric_values_finite(row) and all(
                    _numeric_values_finite(layer_row) for layer_row in layer_rows
                )
                if parameters_finite and diagnostics_finite and (
                    (step + 1) % c["eval_interval"] == 0 or step + 1 == total
                ):
                    row.update(evaluate(model, test_loader, device))
                rows.append(row)
                layers.extend(
                    dict(step=step, **layer_row) for layer_row in layer_rows
                )
                bins.extend(dict(step=step, **eigen_row) for eigen_row in eigen_rows)
                audits.append(dict(
                    step=step, batch_indices=indices.tolist(),
                    model_hash=digest(model.parameters()), parameters_finite=parameters_finite,
                    **audit,
                ))
                step += 1
                if not parameters_finite or not diagnostics_finite:
                    status, diverged_step = "diverged", step - 1
                    break
                evaluations = [r for r in rows if r.get("test_accuracy") is not None]
                if evaluations:
                    last_finite_accuracy = evaluations[-1]["test_accuracy"]
                last_finite_loss = row["train_loss"]
            if status == "diverged":
                break
    finally:
        sync(model)

    wall = time.perf_counter() - started
    summary = _summary_from_rows(
        c, spec, model, rows, refreshes, accountant, sigma, step, wall,
        diagnostic_seconds, active, status, diverged_step,
        last_finite_accuracy, last_finite_loss,
    )
    save_json(root / "summary.json", summary)
    meta.update(summary, complete=status == "completed")
    save_json(root / "metadata.json", meta)
    try:
        save_json(root / "pairing.json", {"private": audits, "synthetic": synthetic_audits})
        write_csv(
            root / "train_metrics.csv",
            [dict(seed=c["seed"], method=spec["method"], config_fingerprint=fingerprint(c), **row) for row in rows],
            fields=TRAIN_FIELDS,
        )
        write_csv(
            root / "layer_metrics.csv",
            [dict(seed=c["seed"], method=spec["method"], config_fingerprint=fingerprint(c), **row) for row in layers],
            fields=LAYER_FIELDS,
        )
        write_csv(
            root / "refresh_metrics.csv",
            [dict(seed=c["seed"], method=spec["method"], config_fingerprint=fingerprint(c), **row) for row in refreshes],
            fields=REFRESH_FIELDS,
        )
        write_csv(
            root / "eigenbin_metrics.csv",
            [dict(seed=c["seed"], method=spec["method"], config_fingerprint=fingerprint(c), **row) for row in bins],
            fields=EIGEN_FIELDS,
        )
    finally:
        model.remove_hooks()
    print(
        f"{status} seed={c['seed']} {run_id}: {step}/{total} steps"
        + (f", accuracy={summary['final_accuracy']:.4f}" if summary["final_accuracy"] is not None else ""),
        flush=True,
    )
    return meta


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "full.json"))
    parser.add_argument("--output", default=str(ROOT / "runs"))
    parser.add_argument("--run", choices=("all", *[spec["run_id"] for spec in run_specs()]), default="all")
    args = parser.parse_args()
    config = read_config(args.config)
    selected = run_specs(config) if args.run == "all" else [run_spec(args.run, config)]
    for spec in selected:
        train(config, spec["run_id"], args.output)


if __name__ == "__main__":
    main()
