"""ExpV2 training harness with beta diagnostics kept outside training control."""

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

from expv2 import beta_estimation
from expv2.beta_estimation import build_interval_rows, build_hold_predictor, build_lagged_rows, window_sensitivity
from expv2.common import (
    EXP1B_REFERENCE_ROOT,
    LAYERS,
    METHODS,
    ROOT,
    RNGStream,
    SimpleCNN,
    check_config,
    datasets,
    digest,
    expected_total_steps,
    fingerprint,
    output_path,
    provenance,
    read_config,
    require_pinned,
    run_specs,
    run_specs_for_seed,
    save_json,
    set_seed,
    write_csv,
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
from dp_kfac.privacy import _compute_per_sample_norms_squared, clip_and_noise_gradients


TRAIN_FIELDS = [
    "seed", "run_id", "method", "config_fingerprint", "step", "epoch", "train_loss",
    "learning_rate",
    "test_loss", "test_accuracy", "clip_rate", "clean_clipped_norm", "actual_noise_norm",
    "noisy_gradient_norm", "filtered_gradient_norm", "relmse_noisy", "relmse_filtered",
    "cosine_noisy", "cosine_filtered", "signal_retention", "noise_retention", "snr_in",
    "snr_out", "snr_gain_db", "mse_reduction", "refresh_time", "diagnostic_spectrum_time",
    "wiener_filter_time", "active_state_bytes", "diagnostics_finite", "beta_diagnostics_finite",
]
LAYER_FIELDS = [
    "seed", "run_id", "method", "config_fingerprint", "step", "layer", "clean_clipped_norm",
    "learning_rate",
    "actual_noise_norm", "noisy_gradient_norm", "filtered_gradient_norm", "relmse_noisy",
    "relmse_filtered", "cosine_noisy", "cosine_filtered", "signal_retention", "noise_retention",
    "snr_in", "snr_out", "snr_gain_db", "mse_reduction", "kappa", "H_mean", "H_std",
    "H_min", "H_max", "H_q10", "H_q25", "H_q50", "H_q75", "H_q90", "trace_A", "trace_G",
    "trace_F", "lambdaF_mean", "lambdaF_median", "lambdaF_q10", "lambdaF_q90",
]
REFRESH_FIELDS = [
    "seed", "run_id", "method", "config_fingerprint", "step", "refresh_index",
    "refresh_time", "beta_trace_time", "diagnostic_spectrum_time", "active_state_bytes", "measurement_only",
]
EIGEN_FIELDS = [
    "seed", "run_id", "method", "config_fingerprint", "step", "layer", "bin", "count",
    "lambda_min", "lambda_max", "lambda_mean", "log10_lambda_mean", "signal_power",
    "noise_power", "empirical_snr", "H_mean", "mse_before", "mse_after", "mse_reduction",
    "signal_retention", "noise_retention",
]
BETA_STEP_FIELDS = [
    "seed", "run_id", "method", "learning_rate", "config_fingerprint", "step", "layer", "r",
    "dimension", "trace_A", "trace_G", "trace_F", "clean_signal_energy", "noisy_gradient_energy",
    "expected_noise_energy", "noise_debiased_energy_raw", "beta_oracle_step", "beta_dp_step_raw",
    "beta_dp_step_positive", "beta_dp_step_negative", "refresh_index", "diagnostic_valid",
]
BETA_INTERVAL_FIELDS = [
    "seed", "run_id", "method", "learning_rate", "interval_index", "start_step", "end_step",
    "n_steps", "layer", "trace_F", "oracle_signal_energy_sum", "noisy_energy_sum",
    "expected_noise_energy_sum", "noise_debiased_energy_sum", "beta_oracle", "beta_dp_raw",
    "beta_dp_positive", "beta_dp_negative", "estimated_signal_energy_positive",
    "estimated_energy_to_noise_ratio", "oracle_estimator_sd", "oracle_observability_snr",
    "conditional_variance_sum", "sample_count",
]
BETA_LAGGED_FIELDS = [
    "seed", "run_id", "method", "learning_rate", "layer", "source_interval", "target_interval",
    "beta_dp_previous_raw", "beta_oracle_current", "lag_prediction_positive", "lag_ratio",
    "lag_log10_abs_ratio_error",
]
BETA_HOLD_FIELDS = [
    "seed", "run_id", "method", "learning_rate", "layer", "source_interval", "target_interval",
    "beta_hold_predictor", "beta_oracle_current", "hold_prediction_positive", "hold_ratio",
    "hold_log10_abs_ratio_error",
]
WINDOW_FIELDS = [
    "window_size", "seed", "run_id", "method", "learning_rate", "layer", "window_index", "n_steps",
    "beta_oracle", "beta_dp_raw", "negative", "log_error", "trace_F", "oracle_observability_snr",
]


class DivergenceError(FloatingPointError):
    """An algorithmic non-finite state; research diagnostics never raise this."""

    def __init__(self, step, stage):
        super().__init__(f"non-finite {stage} at step {step}")
        self.step = step
        self.stage = stage


def sync(model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _all_finite(parameters):
    return all(bool(torch.isfinite(parameter).all()) for parameter in parameters)


def _value_finite(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _diagnostic_row_finite(row):
    return all(_value_finite(value) for value in row.values())


def make_optimizer(model, learning_rate, config=None):
    if config is not None and (config["optimizer"] != "SGD" or config["momentum"] != 0):
        raise ValueError("ExpV2 requires SGD with momentum=0")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("learning rate must be finite and positive")
    return torch.optim.SGD(model.parameters(), lr=float(learning_rate), momentum=0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    for x, y in loader:
        y = y.to(device)
        output = model(x.to(device))
        total_loss += float(F.cross_entropy(output, y, reduction="sum"))
        correct += int((output.argmax(1) == y).sum())
        count += len(y)
    model.train()
    return {"test_loss": total_loss / count, "test_accuracy": correct / count}


@torch.no_grad()
def build_trace_state(covariances):
    """Build only scalar KFAC traces; no eigendecomposition or active state."""
    result = {}
    for name in LAYERS:
        if name not in covariances.A or name not in covariances.G:
            raise ValueError(f"Missing synthetic Fisher factor for {name}")
        A = covariances.A[name].float()
        G = covariances.G[name].float()
        trace_A = float(A.trace())
        trace_G = float(G.trace())
        result[name] = dict(trace_A=trace_A, trace_G=trace_G, trace_F=trace_A * trace_G)
    return result


def _beta_capture(model, trace_state, sigma, config, batch_size, seed, run_id, method,
                  learning_rate, step, refresh_index):
    """Capture only scalar energy aggregates from clean and pre-filter DP gradients."""
    rows = []
    r = beta_estimation.noise_variance(sigma, config["max_grad_norm"], batch_size)
    for name in LAYERS:
        layer = getattr(model._module, name)
        # The actual packed matrix determines dimension, including the bias column.
        clean = pack_layer_gradient(layer, "summed_grad")
        noisy = pack_layer_gradient(layer, "grad")
        dimension = int(noisy.numel())
        trace = trace_state[name]
        clean_energy = float(clean.double().square().sum())
        noisy_energy = float(noisy.double().square().sum())
        estimate = beta_estimation.single_step_beta(
            clean_energy, noisy_energy, dimension, trace["trace_F"], r
        )
        estimate["diagnostic_valid"] = beta_estimation.beta_step_diagnostic_is_finite(estimate)
        rows.append(dict(
            seed=seed, run_id=run_id, method=method, learning_rate=float(learning_rate),
            step=step, layer=name, r=r, dimension=dimension, **trace,
            **estimate, refresh_index=refresh_index,
        ))
    return rows


def private_update(
    model, active, optimizer, rng, sigma, config, batch_size, method, learning_rate,
    trace_state=None, diagnostics=True, beta_measurement=True, refresh_index=None,
    step=0, run_id=None, seed=None, diagnostic_state=None,
):
    """Execute the fixed private update and return scalar-only research outputs.

    The beta capture occurs after clipping/noise and before the optional filter.
    Captured tensors are transient local values and only scalar aggregates leave
    this function; no raw gradient or noise realization is serialized.
    """
    if method not in METHODS:
        raise ValueError(method)
    diagnostic_seconds = 0.0
    clip_rate = None
    if diagnostics:
        sync(model)
        started = time.perf_counter()
        sq = _compute_per_sample_norms_squared(
            list(model.parameters()), batch_size, next(model.parameters()).device
        )
        clip_rate = float((sq.sqrt() > config["max_grad_norm"]).float().mean())
        sync(model)
        diagnostic_seconds += time.perf_counter() - started

    audit = {"noise_rng_before": rng.audit()}
    with rng.use():
        clip_and_noise_gradients(
            model, noise_multiplier=sigma, max_grad_norm=config["max_grad_norm"],
            batch_size=batch_size, store_summed_grad=True,
        )
    audit["noise_rng_after"] = rng.audit()
    if not _all_finite(p.grad for p in model.parameters()):
        raise DivergenceError(step, "noisy gradient")

    noisy_for_diagnostics = None
    beta_rows = []
    if diagnostics or beta_measurement:
        sync(model)
        started = time.perf_counter()
        noisy_for_diagnostics = {
            name: pack_layer_gradient(getattr(model._module, name), "grad")
            for name in LAYERS
        }
        if beta_measurement:
            beta_rows = _beta_capture(
                model, trace_state, sigma, config, batch_size, seed,
                run_id, method, learning_rate, step, refresh_index,
            )
        sync(model)
        diagnostic_seconds += time.perf_counter() - started

    sync(model)
    started = time.perf_counter()
    if method == "dp_fisher_wiener":
        apply_fisher_wiener(model, active)
    elif method != "dp_sgd":
        raise ValueError(method)
    sync(model)
    filter_time = time.perf_counter() - started if method == "dp_fisher_wiener" else 0.0
    if not _all_finite(p.grad for p in model.parameters()):
        raise DivergenceError(step, "filtered gradient")

    optimizer.step()
    rows, layer_rows, bins = {}, [], []
    if diagnostics:
        sync(model)
        started = time.perf_counter()
        rows, layer_rows, bins = diagnose(
            model, noisy_for_diagnostics, active, diagnostic_state,
            method, False, 0,
        )
        sync(model)
        diagnostic_seconds += time.perf_counter() - started
    rows.update(clip_rate=clip_rate, wiener_filter_time=filter_time)
    rows["beta_diagnostics_finite"] = all(
        beta_estimation.beta_step_diagnostic_is_finite(row) for row in beta_rows
    ) if beta_measurement else True
    return rows, layer_rows, bins, beta_rows, audit, diagnostic_seconds


class Indexed(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        return x, y, index


def _summary(c, spec, model, rows, refreshes, accountant, sigma, step, wall,
             diagnostic_seconds, active, beta_rows, status, diverged_step):
    evaluations = [row for row in rows if row.get("test_accuracy") is not None]
    finite_evaluations = [row for row in evaluations if _value_finite(row.get("test_accuracy"))]
    accuracy = [float(row["test_accuracy"]) for row in finite_evaluations]
    if accuracy:
        progress = np.array([(row["step"] + 1) / max(step, 1) for row in finite_evaluations])
        if len(accuracy) > 1:
            trapezoid = np.trapezoid if hasattr(np, "trapezoid") else lambda y, x: ((y[1:] + y[:-1]) * np.diff(x) / 2).sum()
            auc = float(trapezoid(accuracy, progress) / (progress[-1] - progress[0]))
        else:
            auc = accuracy[0]
        late = [row["test_accuracy"] for row in finite_evaluations if row["step"] + 1 > step / 2]
    else:
        auc, late = None, []
    nonfinite_diagnostics = [int(row["step"]) for row in rows if not row.get("diagnostics_finite", True)]
    nonfinite_beta = sorted({
        int(row["step"]) for row in rows if not row.get("beta_diagnostics_finite", True)
    })
    refresh_total = sum(float(row["refresh_time"]) for row in refreshes)
    measurement_only_refresh_total = sum(
        float(row["refresh_time"]) for row in refreshes if row["measurement_only"]
    )
    beta_trace_total = sum(float(row["beta_trace_time"]) for row in refreshes)
    fisher_beta_trace_total = sum(
        float(row["beta_trace_time"]) for row in refreshes if not row["measurement_only"]
    )
    spectrum_total = sum(float(row["diagnostic_spectrum_time"]) for row in refreshes)
    filter_total = sum(float(row.get("wiener_filter_time", 0.0)) for row in rows)
    research_overhead = (
        diagnostic_seconds + measurement_only_refresh_total + fisher_beta_trace_total
    )
    return dict(
        seed=spec["seed"], run_id=spec["run_id"], method=spec["method"],
        learning_rate=spec["learning_rate"], fingerprint=fingerprint(c), status=status,
        planned_steps=expected_total_steps(c), completed_steps=step,
        epsilon_spent=float(accountant.get_epsilon(c["delta"])),
        noise_multiplier=float(sigma), sample_rate=c["batch_size"] / (c["train_subset"] or 60000),
        final_accuracy=accuracy[-1] if status == "completed" and accuracy else None,
        best_accuracy=max(accuracy) if status == "completed" and accuracy else None,
        late_mean_accuracy=float(np.mean(late)) if status == "completed" and late else None,
        accuracy_auc=auc if status == "completed" else None,
        final_test_loss=finite_evaluations[-1].get("test_loss") if status == "completed" and finite_evaluations else None,
        final_model_hash=digest(model.parameters()), parameters_finite=_all_finite(model.parameters()),
        wall_time=wall, diagnostic_seconds=diagnostic_seconds,
        research_overhead_seconds=research_overhead,
        core_training_runtime=wall - research_overhead,
        total_refresh_time=refresh_total,
        total_measurement_only_refresh_time=measurement_only_refresh_total,
        total_beta_trace_time=beta_trace_total,
        fisher_beta_trace_time=fisher_beta_trace_total,
        mean_refresh_time=refresh_total / len(refreshes) if refreshes else 0.0,
        number_of_refreshes=len(refreshes),
        total_diagnostic_spectrum_time=spectrum_total,
        mean_diagnostic_spectrum_time=spectrum_total / len(refreshes) if refreshes else 0.0,
        total_wiener_filter_time=filter_total,
        mean_wiener_filter_time=filter_total / max(step, 1),
        diagnostics_all_finite=not nonfinite_diagnostics,
        diagnostic_nonfinite_count=len(nonfinite_diagnostics),
        diagnostic_nonfinite_steps=nonfinite_diagnostics,
        beta_diagnostics_all_finite=not nonfinite_beta,
        beta_diagnostic_nonfinite_count=len(nonfinite_beta),
        beta_diagnostic_nonfinite_steps=nonfinite_beta,
        active_state_bytes=state_bytes(active),
        peak_cuda_allocated_memory=(torch.cuda.max_memory_allocated(next(model.parameters()).device)
                                    if next(model.parameters()).device.type == "cuda" else 0),
        diverged_step=diverged_step,
        beta_step_rows=len(beta_rows), beta_train=1.0,
        filter_position="after global clipping and DP Gaussian noise",
        synthetic_fisher_role="measurement_only_for_dp_sgd" if spec["method"] == "dp_sgd" else "training_filter_and_measurement",
        contains_non_dp_oracle=True,
        release_safe_under_dp=False,
        deployable_beta_estimator_is_postprocessing=True,
        oracle_is_research_only=True,
    )


def _write_run_artifacts(root, c, spec, beta_rows, interval_rows, lagged_rows, hold_rows, window_rows,
                         rows, layer_rows, bins, refreshes, audits, synthetic_audits):
    fp = fingerprint(c)
    prefix = dict(seed=spec["seed"], run_id=spec["run_id"], method=spec["method"],
                  config_fingerprint=fp, learning_rate=spec["learning_rate"])
    save_json(root / "pairing.json", {"private": audits, "synthetic": synthetic_audits})
    write_csv(root / "train_metrics.csv", [dict(prefix, **row) for row in rows], TRAIN_FIELDS)
    write_csv(root / "layer_metrics.csv", [dict(prefix, **row) for row in layer_rows], LAYER_FIELDS)
    write_csv(root / "refresh_metrics.csv", [dict(prefix, **row) for row in refreshes], REFRESH_FIELDS)
    write_csv(root / "eigenbin_metrics.csv", [dict(prefix, **row) for row in bins], EIGEN_FIELDS)
    write_csv(root / "beta_step_metrics.csv", [dict(prefix, **row) for row in beta_rows], BETA_STEP_FIELDS)
    write_csv(root / "beta_interval_metrics.csv", interval_rows, BETA_INTERVAL_FIELDS)
    write_csv(root / "beta_lagged_metrics.csv", lagged_rows, BETA_LAGGED_FIELDS)
    write_csv(root / "beta_hold_predictor.csv", hold_rows, BETA_HOLD_FIELDS)
    write_csv(root / "beta_window_sensitivity.csv", window_rows, WINDOW_FIELDS)


def train(config, seed, run_name, output, data_override=None, *, diagnostics=True, beta_measurement=True):
    """Run one canonical ExpV2 trajectory; formal runs are explicit CLI work."""
    check_config(config)
    if seed not in config["seeds"]:
        raise ValueError("Unknown seed")
    spec = next((item for item in run_specs(config) if item["run_id"] == run_name), None)
    if spec is None or spec not in run_specs_for_seed(config, seed):
        raise ValueError("Unknown canonical run/seed")
    spec = dict(spec, seed=seed)
    root = output_path(output) / f"seed{seed}" / run_name
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "config.json", config)
    current_provenance = provenance()
    require_pinned(current_provenance, config["smoke"])
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) if config["device"] == "auto" else torch.device(config["device"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    source, test = datasets(config) if data_override is None else data_override
    n = config["train_subset"] or len(source)
    if not config["batch_size"] <= n <= len(source):
        raise ValueError("Invalid training subset")
    data = Subset(source, range(n))
    if config["test_subset"] is not None:
        if not 1 <= config["test_subset"] <= len(test):
            raise ValueError("Invalid test subset")
        test = Subset(test, range(config["test_subset"]))
    loader = DataLoader(
        Indexed(data), batch_size=config["batch_size"], shuffle=True, drop_last=True,
        num_workers=0, generator=torch.Generator().manual_seed(seed + 1),
    )
    test_loader = DataLoader(
        test, batch_size=config["batch_size"], num_workers=0,
        generator=torch.Generator().manual_seed(seed + 2),
    )
    total = len(loader) * config["epochs"]
    q = config["batch_size"] / n
    sigma = get_noise_multiplier(
        target_epsilon=config["epsilon"], target_delta=config["delta"],
        sample_rate=q, steps=total, accountant="rdp",
    )

    set_seed(seed)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    # private_update uses this immutable scalar only for research row labels.
    optimizer = make_optimizer(model, spec["learning_rate"], config)
    accountant = RDPAccountant()
    synthetic_rng, noise_rng = RNGStream(seed + 3, device), RNGStream(seed + 4, device)
    metadata = dict(
        experiment="expv2", seed=seed, run_id=run_name, method=spec["method"],
        learning_rate=spec["learning_rate"], fingerprint=fingerprint(config), provenance=current_provenance,
        device=str(device), dataset="MNIST", model="SimpleCNN", noise_multiplier=sigma,
        sample_rate=q, total_steps=total, target_epsilon=config["epsilon"], target_delta=config["delta"],
        accountant="rdp", max_grad_norm=config["max_grad_norm"], optimizer="SGD", momentum=0,
        initial_model_hash=digest(model.parameters()), beta_train=1.0,
        covariance_ridge=1e-5, eigen_budget=0, eigenmode_diagnostics=False,
        rng_seeds=dict(init=seed, loader=seed + 1, test=seed + 2, synthetic=seed + 3, noise=seed + 4),
        sampling="fixed shuffle/drop_last; inherited RDP convention, not Poisson",
        diagnostics_enabled=diagnostics, beta_measurement_enabled=beta_measurement,
        diagnostics="Non-DP research statistics; beta oracle uses p.summed_grad and is not deployable",
        output=str(root), filter_position="after global clipping and DP Gaussian noise",
        synthetic_fisher_for_dp_sgd=beta_measurement and spec["method"] == "dp_sgd",
        no_private_fisher=True, no_beta_feedback=True,
        contains_non_dp_oracle=True,
        release_safe_under_dp=False,
        deployable_beta_estimator_is_postprocessing=True,
        oracle_is_research_only=True,
    )
    save_json(root / "metadata.json", metadata)

    rows, layer_rows, bins, beta_rows, refreshes, audits, synthetic_audits = [], [], [], [], [], [], []
    active = None
    diagnostic_state = None
    trace_state = None
    step = 0
    diagnostic_seconds = 0.0
    current_refresh_index = None
    status, diverged_step = "completed", None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(model)
    started = time.perf_counter()
    try:
        for epoch in range(1, config["epochs"] + 1):
            iterator = iter(loader)
            for _ in range(len(loader)):
                refreshed = step % config["K"] == 0 and (beta_measurement or spec["method"] == "dp_fisher_wiener")
                refresh_time = 0.0
                beta_trace_time = 0.0
                diagnostic_spectrum_time = 0.0
                refresh_index = current_refresh_index
                if refreshed:
                    sync(model)
                    refresh_started = time.perf_counter()
                    samples, synthetic_audit = synthetic_samples(config, device, synthetic_rng)
                    covariances = build_covariances(model._module.state_dict(), samples, config, device)
                    if beta_measurement:
                        sync(model)
                        trace_started = time.perf_counter()
                        trace_state = build_trace_state(covariances)
                        sync(model)
                        beta_trace_time = time.perf_counter() - trace_started
                    else:
                        trace_state = None
                    if spec["method"] == "dp_fisher_wiener":
                        active = build_fisher_state(covariances, sigma, config["max_grad_norm"], config["batch_size"])
                    else:
                        active = None
                    sync(model)
                    refresh_time = time.perf_counter() - refresh_started
                    diagnostic_state = None
                    if diagnostics and spec["method"] == "dp_fisher_wiener":
                        sync(model)
                        spectrum_started = time.perf_counter()
                        diagnostic_state = build_diagnostic_state(covariances, active, spec["method"])
                        sync(model)
                        diagnostic_spectrum_time = time.perf_counter() - spectrum_started
                        diagnostic_seconds += diagnostic_spectrum_time
                    refresh_index = len(refreshes)
                    current_refresh_index = refresh_index
                    refreshes.append(dict(
                        step=step, refresh_index=refresh_index, refresh_time=refresh_time,
                        beta_trace_time=beta_trace_time,
                        diagnostic_spectrum_time=diagnostic_spectrum_time,
                        active_state_bytes=state_bytes(active), measurement_only=spec["method"] == "dp_sgd",
                    ))
                    synthetic_audits.append(dict(step=step, **synthetic_audit))
                    del samples, covariances

                x, y, indices = next(iterator)
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
                if not bool(torch.isfinite(loss)):
                    status, diverged_step = "diverged", step
                    break
                loss.backward()
                try:
                    row, current_layers, current_bins, current_beta, audit, elapsed = private_update(
                        model, active, optimizer, noise_rng, sigma, config, len(x), spec["method"],
                        spec["learning_rate"], trace_state=trace_state, diagnostics=diagnostics,
                        beta_measurement=beta_measurement, refresh_index=refresh_index, step=step,
                        run_id=run_name, seed=seed, diagnostic_state=diagnostic_state,
                    )
                except DivergenceError as exc:
                    status, diverged_step = "diverged", exc.step
                    break
                diagnostic_seconds += elapsed
                accountant.step(noise_multiplier=sigma, sample_rate=q)
                row.update(
                    step=step, epoch=epoch, train_loss=float(loss.detach()) / len(x),
                    learning_rate=spec["learning_rate"],
                    refresh_time=refresh_time, diagnostic_spectrum_time=diagnostic_spectrum_time,
                    active_state_bytes=state_bytes(active), test_loss=None, test_accuracy=None,
                    diagnostics_finite=(True if not diagnostics else _diagnostic_row_finite(row) and
                                        all(_diagnostic_row_finite(value) for value in current_layers)),
                )
                if (step + 1) % config["eval_interval"] == 0 or step + 1 == total:
                    if _all_finite(model.parameters()):
                        row.update(evaluate(model, test_loader, device))
                rows.append(row)
                layer_rows.extend(dict(step=step, **value) for value in current_layers)
                bins.extend(dict(step=step, **value) for value in current_bins)
                for value in current_beta:
                    value["diagnostic_valid"] = bool(value.get("diagnostic_valid", False))
                beta_rows.extend(current_beta)
                parameters_finite = _all_finite(model.parameters())
                audits.append(dict(
                    step=step, batch_indices=indices.tolist(), model_hash=digest(model.parameters()),
                    parameters_finite=parameters_finite, **audit,
                ))
                step += 1
                if not parameters_finite:
                    status, diverged_step = "diverged", step - 1
                    break
            if status == "diverged":
                break
    finally:
        sync(model)

    wall = time.perf_counter() - started
    summary_spec = dict(spec, seed=seed)
    summary = _summary(
        config, summary_spec, model, rows, refreshes, accountant, sigma, step, wall,
        diagnostic_seconds, active, beta_rows, status, diverged_step,
    )
    metadata.update(summary, complete=status == "completed")
    save_json(root / "summary.json", summary)
    save_json(root / "metadata.json", metadata)

    interval_rows = build_interval_rows(beta_rows, config["beta_primary_window"]) if beta_rows else []
    lagged_rows = build_lagged_rows(interval_rows)
    hold_rows = build_hold_predictor(interval_rows)
    window_rows = window_sensitivity(beta_rows, config["beta_window_sensitivity"]) if beta_rows else []
    try:
        _write_run_artifacts(
            root, config, summary_spec, beta_rows, interval_rows, lagged_rows, hold_rows,
            window_rows, rows, layer_rows, bins, refreshes, audits, synthetic_audits,
        )
    finally:
        model.remove_hooks()
    final = summary.get("final_accuracy")
    suffix = f", accuracy={final:.4f}" if final is not None else ""
    print(f"{status} seed={seed} {run_name}: {step}/{total} steps{suffix}", flush=True)
    return metadata


def selected_run_specs(config, seeds, requested_run="all"):
    """Resolve CLI seed/run selections and reject invalid combinations."""
    selected = []
    for seed in seeds:
        allowed = run_specs_for_seed(config, seed)
        if requested_run == "all":
            chosen = allowed
        else:
            chosen = [spec for spec in allowed if spec["run_id"] == requested_run]
            if not chosen:
                allowed_names = ", ".join(spec["run_id"] for spec in allowed)
                raise ValueError(
                    f"Requested seed {seed} with run {requested_run}; "
                    f"allowed runs for that seed: {allowed_names}"
                )
        selected.extend(dict(spec, seed=seed) for spec in chosen)
    return selected


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "full.json"))
    parser.add_argument("--output", default=str(ROOT / "runs"))
    parser.add_argument("--run", choices=("all", *[spec["run_id"] for spec in run_specs()]), default="all")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    seeds = config["seeds"] if args.seed is None else [args.seed]
    for spec in selected_run_specs(config, seeds, args.run):
        train(config, spec["seed"], spec["run_id"], args.output)
    if config["smoke"] and args.run == "all":
        (ROOT / "runs" / "latest_smoke_path.txt").write_text(str(Path(args.output).resolve()) + "\n")


if __name__ == "__main__":
    main()
