"""Paired ExpV3 training harness; formal runs require explicit CLI invocation."""

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
from expv2.beta_estimation import build_interval_rows
from expv1.fisher_wiener import (
    apply_fisher_wiener,
    build_covariances,
    build_diagnostic_state,
    build_fisher_state,
    pack_layer_gradient,
    synthetic_samples,
    state_bytes,
)
from expv1.metrics import diagnose
from dp_kfac.privacy import _compute_per_sample_norms_squared, clip_and_noise_gradients

from expv3.adaptive_fisher_wiener import (
    build_adaptive_fisher_state,
    h_hash,
    h_stats,
    selected_stats_digest,
    stats_digest,
)
from expv3.beta_controller import AdaptiveBetaController
from expv3.common import (
    DP_METHOD,
    FISHER_METHOD,
    LAYERS,
    ROOT,
    RNGStream,
    SimpleCNN,
    add_effective_step_metrics,
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
    threshold_step,
    write_csv,
)


TRAIN_FIELDS = [
    "seed", "run_id", "method", "config_fingerprint", "step", "epoch", "train_loss",
    "learning_rate", "test_loss", "test_accuracy", "clip_rate", "clean_clipped_norm",
    "actual_noise_norm", "noisy_gradient_norm", "filtered_gradient_norm", "relmse_noisy",
    "relmse_filtered", "cosine_noisy", "cosine_filtered", "signal_retention",
    "noise_retention", "snr_in", "snr_out", "snr_gain_db", "mse_reduction",
    "signal_amplitude_retention", "effective_signal_lr", "optimizer_update_norm",
    "clean_reference_update_norm", "update_to_clean_reference_ratio",
    "lr_compensation_to_dp_sgd_0p5", "refresh_time", "diagnostic_spectrum_time",
    "wiener_filter_time", "beta_controller_time", "active_state_bytes",
    "diagnostics_finite", "beta_diagnostics_finite",
]
LAYER_FIELDS = [
    "seed", "run_id", "method", "config_fingerprint", "step", "layer", "learning_rate",
    "clean_clipped_norm", "actual_noise_norm", "noisy_gradient_norm", "filtered_gradient_norm",
    "relmse_noisy", "relmse_filtered", "cosine_noisy", "cosine_filtered", "signal_retention",
    "noise_retention", "snr_in", "snr_out", "snr_gain_db", "mse_reduction",
    "signal_amplitude_retention", "effective_signal_lr", "optimizer_update_norm",
    "clean_reference_update_norm", "update_to_clean_reference_ratio",
    "lr_compensation_to_dp_sgd_0p5", "kappa", "H_mean", "H_std", "H_min", "H_max",
    "H_q10", "H_q25", "H_q50", "H_q75", "H_q90", "trace_A", "trace_G", "trace_F",
    "lambdaF_mean", "lambdaF_median", "lambdaF_q10", "lambdaF_q90",
]
REFRESH_FIELDS = [
    "seed", "run_id", "method", "learning_rate", "config_fingerprint", "step",
    "refresh_index", "refresh_time", "diagnostic_spectrum_time", "beta_controller_time",
    "active_state_bytes", "measurement_only",
]
BETA_STEP_FIELDS = [
    "seed", "run_id", "method", "learning_rate", "config_fingerprint", "step", "layer",
    "r", "dimension", "trace_A", "trace_G", "trace_F", "clean_signal_energy",
    "noisy_gradient_energy", "expected_noise_energy", "noise_debiased_energy_raw",
    "beta_oracle_step", "beta_dp_step_raw", "beta_dp_step_positive", "beta_dp_step_negative",
    "refresh_index", "active_beta_train", "active_beta_source_interval", "diagnostic_valid",
]
BETA_INTERVAL_FIELDS = [
    "seed", "run_id", "method", "learning_rate", "interval_index", "start_step", "end_step",
    "n_steps", "layer", "trace_F", "oracle_signal_energy_sum", "noisy_energy_sum",
    "expected_noise_energy_sum", "noise_debiased_energy_sum", "beta_oracle", "beta_dp_raw",
    "beta_dp_positive", "beta_dp_negative", "estimated_signal_energy_positive",
    "estimated_energy_to_noise_ratio", "oracle_estimator_sd", "oracle_observability_snr",
    "conditional_variance_sum", "sample_count", "beta_oracle_interval", "beta_dp_raw_interval",
    "raw_positive", "raw_finite", "was_used_by_next_interval", "next_beta_train",
    "applied_to_training",
]
BETA_CONTROLLER_FIELDS = [
    "seed", "run_id", "method", "learning_rate", "config_fingerprint", "refresh_step",
    "interval_index", "layer", "beta_train", "beta_source_interval", "beta_raw_previous",
    "beta_update_accepted", "beta_fallback_used", "beta_fallback_reason",
    "previous_beta_train", "numerator_previous", "denominator_previous",
    "previous_interval_n_steps", "accepted_update_count", "fallback_count", "trace_A",
    "trace_G", "trace_F", "beta_scaled_trace_F", "H_mean", "H_std", "H_min", "H_max",
    "H_q10", "H_q25", "H_q50", "H_q75", "H_q90", "H_beta1_mean", "H_beta1_q10",
    "H_beta1_q50", "H_beta1_q90", "H_fro_ratio_vs_beta1", "H_hash", "H_hash_copy",
    "H_beta1_hash", "H_beta1_hash_copy", "H_stats_digest", "H_beta1_stats_digest",
]


class DivergenceError(FloatingPointError):
    """An algorithmic non-finite state; research diagnostics do not raise it."""

    def __init__(self, step, stage):
        super().__init__(f"non-finite {stage} at step {step}")
        self.step = step
        self.stage = stage


def sync(model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def all_finite(values):
    return all(bool(torch.isfinite(value).all()) for value in values)


def value_finite(value):
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float, np.number)):
        return math.isfinite(float(value))
    return True


def diagnostic_row_finite(row):
    return all(value_finite(value) for value in row.values())


def make_optimizer(model, learning_rate, config=None):
    if config is not None and (config["optimizer"] != "SGD" or config["momentum"] != 0):
        raise ValueError("ExpV3 requires plain SGD with momentum=0")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("learning rate must be finite and positive")
    return torch.optim.SGD(model.parameters(), lr=float(learning_rate), momentum=0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = count = 0
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
    """Build scalar synthetic Fisher traces for beta estimation."""
    result = {}
    for name in LAYERS:
        if name not in covariances.A or name not in covariances.G:
            raise ValueError(f"Missing synthetic Fisher factor for {name}")
        A, G = covariances.A[name].float(), covariances.G[name].float()
        trace_A, trace_G = float(A.trace()), float(G.trace())
        result[name] = {"trace_A": trace_A, "trace_G": trace_G,
                        "trace_F": trace_A * trace_G}
    return result


def beta_capture(model, trace_state, sigma, config, batch_size, seed, run_id, method,
                 learning_rate, step, refresh_index, active_betas, active_source):
    """Create research rows from clean reference and pre-Wiener DP tensors.

    Only scalar values leave this function.  The controller receives its own
    noisy scalar arguments below, never the clean tensor or actual noise.
    """
    rows = []
    r = beta_estimation.noise_variance(sigma, config["max_grad_norm"], batch_size)
    for name in LAYERS:
        layer = getattr(model._module, name)
        clean = pack_layer_gradient(layer, "summed_grad")
        noisy = pack_layer_gradient(layer, "grad")
        dimension = int(noisy.numel())
        trace = trace_state[name]
        estimate = beta_estimation.single_step_beta(
            float(clean.double().square().sum()), float(noisy.double().square().sum()),
            dimension, trace["trace_F"], r,
        )
        rows.append({
            "seed": seed, "run_id": run_id, "method": method,
            "learning_rate": float(learning_rate), "step": step, "layer": name,
            "r": r, "dimension": dimension, **trace, **estimate,
            "refresh_index": refresh_index,
            "active_beta_train": (active_betas[name] if isinstance(active_betas, dict) else active_betas),
            "active_beta_source_interval": active_source,
        })
    return rows


def private_update(model, active, optimizer, rng, sigma, config, batch_size, method,
                   learning_rate, trace_state=None, controller=None, diagnostics=True,
                   beta_measurement=True, refresh_index=None, step=0, run_id=None,
                   seed=None, diagnostic_state=None, active_beta=None, active_source=None):
    """Execute one DP mechanism, optional adaptive filter, and SGD update.

    The fixed order is: per-sample gradients -> global clipping/noise -> capture
    pre-Wiener ``y`` -> scalar controller accumulation -> filter -> SGD.
    """
    if method not in (DP_METHOD, FISHER_METHOD):
        raise ValueError(method)
    diagnostic_seconds = 0.0
    controller_seconds = 0.0
    diagnostics_ok = True
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
    if not all_finite(p.grad for p in model.parameters()):
        raise DivergenceError(step, "noisy gradient")

    noisy_for_diagnostics = None
    beta_rows = []
    if diagnostics or beta_measurement:
        sync(model)
        started = time.perf_counter()
        noisy_for_diagnostics = {
            name: pack_layer_gradient(getattr(model._module, name), "grad") for name in LAYERS
        }
        if beta_measurement:
            beta_rows = beta_capture(
                model, trace_state, sigma, config, batch_size, seed, run_id, method,
                learning_rate, step, refresh_index, active_beta, active_source,
            )
        sync(model)
        diagnostic_seconds += time.perf_counter() - started

    if controller is not None:
        started = time.perf_counter()
        for row in beta_rows:
            # Deliberately pass only DP-safe scalar observations.
            controller.observe(
                row["layer"], row["noisy_gradient_energy"],
                row["expected_noise_energy"], row["trace_F"],
            )
        controller_seconds += time.perf_counter() - started

    sync(model)
    started = time.perf_counter()
    if method == FISHER_METHOD:
        apply_fisher_wiener(model, active)
    elif method != DP_METHOD:
        raise ValueError(method)
    sync(model)
    filter_time = time.perf_counter() - started if method == FISHER_METHOD else 0.0
    if not all_finite(p.grad for p in model.parameters()):
        raise DivergenceError(step, "filtered gradient")

    optimizer.step()
    rows, layer_rows = {}, []
    if diagnostics:
        sync(model)
        started = time.perf_counter()
        try:
            rows, layer_rows, _ = diagnose(
                model, noisy_for_diagnostics, active,
                diagnostic_state, "dp_fisher_wiener" if method == FISHER_METHOD else "dp_sgd",
                False, 0,
            )
            rows = add_effective_step_metrics(rows, learning_rate)
            layer_rows = [add_effective_step_metrics(value, learning_rate) for value in layer_rows]
        except Exception:
            # Research observability cannot turn into algorithmic divergence.
            diagnostics_ok = False
            rows = {"diagnostic_error": True}
            layer_rows = [{"layer": name, "diagnostic_error": True} for name in LAYERS]
        sync(model)
        diagnostic_seconds += time.perf_counter() - started
    rows.update(clip_rate=clip_rate, wiener_filter_time=filter_time,
                beta_controller_time=controller_seconds)
    rows["beta_diagnostics_finite"] = (
        all(beta_estimation.beta_step_diagnostic_is_finite(row) for row in beta_rows)
        if beta_measurement else True
    )
    return rows, layer_rows, beta_rows, audit, diagnostic_seconds, controller_seconds


class Indexed(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        return x, y, index


def _summary(config, spec, model, rows, refreshes, accountant, sigma, step, wall,
             diagnostic_seconds, controller_seconds, active, status, diverged_step):
    evaluations = [row for row in rows if row.get("test_accuracy") is not None]
    finite_evaluations = [row for row in evaluations if value_finite(row.get("test_accuracy"))]
    accuracies = [float(row["test_accuracy"]) for row in finite_evaluations]
    if accuracies:
        progress = np.asarray([(row["step"] + 1) / max(step, 1) for row in finite_evaluations])
        if len(accuracies) > 1:
            trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            auc = float(trapezoid(accuracies, progress) / (progress[-1] - progress[0]))
        else:
            auc = accuracies[0]
        late = [row["test_accuracy"] for row in finite_evaluations if row["step"] + 1 > step / 2]
        prefix_auc = auc
    else:
        auc, late, prefix_auc = None, [], None
    complete = status == "completed"
    total_refresh = sum(float(row["refresh_time"]) for row in refreshes)
    measurement_refresh = sum(float(row["refresh_time"]) for row in refreshes if row["measurement_only"])
    spectrum = sum(float(row["diagnostic_spectrum_time"]) for row in refreshes)
    filter_time = sum(float(row.get("wiener_filter_time", 0.0)) for row in rows)
    prefix_late = float(np.mean(late)) if late else None
    return {
        "seed": spec["seed"], "run_id": spec["run_id"], "method": spec["method"],
        "learning_rate": spec["learning_rate"], "fingerprint": fingerprint(config),
        "status": status, "planned_steps": expected_total_steps(config),
        "completed_steps": step, "epsilon_spent": float(accountant.get_epsilon(config["delta"])),
        "noise_multiplier": float(sigma), "sample_rate": config["batch_size"] / (config["train_subset"] or 60000),
        "final_accuracy": accuracies[-1] if complete and accuracies else None,
        "best_accuracy": max(accuracies) if complete and accuracies else None,
        "late_mean_accuracy": prefix_late if complete else None,
        "accuracy_auc": auc if complete else None,
        "final_test_loss": finite_evaluations[-1].get("test_loss") if complete and finite_evaluations else None,
        "T50": threshold_step(finite_evaluations, .50) if complete else None,
        "T70": threshold_step(finite_evaluations, .70) if complete else None,
        "T80": threshold_step(finite_evaluations, .80) if complete else None,
        "T85": threshold_step(finite_evaluations, .85) if complete else None,
        "T90": threshold_step(finite_evaluations, .90) if complete else None,
        "prefix_accuracy_auc": prefix_auc,
        "prefix_late_mean_accuracy": prefix_late,
        "last_finite_accuracy": accuracies[-1] if accuracies else None,
        "best_accuracy_before_divergence": max(accuracies) if accuracies else None,
        "final_model_hash": digest(model.parameters()),
        "parameters_finite": all_finite(model.parameters()), "wall_time": wall,
        "diagnostic_seconds": diagnostic_seconds,
        "research_overhead_seconds": diagnostic_seconds + measurement_refresh,
        "core_training_runtime": wall - diagnostic_seconds - measurement_refresh,
        "total_refresh_time": total_refresh,
        "total_measurement_only_refresh_time": measurement_refresh,
        "total_beta_controller_time": controller_seconds,
        "total_wiener_filter_time": filter_time,
        "total_diagnostic_spectrum_time": spectrum,
        "number_of_refreshes": len(refreshes),
        "mean_refresh_time": total_refresh / len(refreshes) if refreshes else 0.0,
        "mean_wiener_filter_time": filter_time / max(step, 1),
        "diagnostics_all_finite": all(row.get("diagnostics_finite", True) for row in rows),
        "beta_diagnostics_all_finite": all(row.get("beta_diagnostics_finite", True) for row in rows),
        "beta_diagnostic_nonfinite_steps": sorted({int(row["step"]) for row in rows if not row.get("beta_diagnostics_finite", True)}),
        "active_state_bytes": state_bytes(active),
        "diverged_step": diverged_step,
        "beta_algorithm_active": spec["method"] == FISHER_METHOD,
        "filter_position": "after global clipping and DP Gaussian noise",
        "synthetic_fisher_role": "measurement_only_for_dp_sgd" if spec["method"] == DP_METHOD else "training_filter_and_measurement",
        "contains_non_dp_oracle": True, "release_safe_under_dp": False,
        "deployable_beta_controller_is_postprocessing": True,
        "deployable_beta_estimator_is_postprocessing": True,
        "oracle_is_research_only": True, "actual_noise_saved": False,
        "private_fisher_used": False,
    }


def _controller_rows(config, spec, step, interval, trace_state, active, beta1_state,
                     controller, previous, source_interval, accepted, fallback, reason):
    rows = []
    for name in LAYERS:
        beta = controller.active_beta(name) if controller is not None else None
        trace = trace_state[name]
        if active is None:
            current_stats = {key: None for key in (
                "H_mean", "H_std", "H_min", "H_max", "H_q10", "H_q25", "H_q50", "H_q75", "H_q90")}
            beta1_stats = {"H_beta1_mean": None, "H_beta1_q10": None, "H_beta1_q50": None, "H_beta1_q90": None,
                           "H_fro_ratio_vs_beta1": None}
            hashes = {"H_hash": None, "H_hash_copy": None, "H_beta1_hash": None, "H_beta1_hash_copy": None,
                      "H_stats_digest": None, "H_beta1_stats_digest": None}
            counts = {"accepted_update_count": 0, "fallback_count": 0}
        else:
            current_stats = h_stats(active[name]["H"])
            beta1_stats_full = h_stats(beta1_state[name]["H"])
            beta1_stats = {key.replace("H_", "H_beta1_", 1): beta1_stats_full[key]
                           for key in ("H_mean", "H_q10", "H_q50", "H_q90")}
            denom = float(beta1_state[name]["H"].double().norm())
            ratio = float(active[name]["H"].double().norm()) / (denom + 1e-12)
            beta1_stats["H_fro_ratio_vs_beta1"] = ratio
            hashes = {
                "H_hash": h_hash(active[name]["H"]), "H_hash_copy": h_hash(active[name]["H"]),
                "H_beta1_hash": h_hash(beta1_state[name]["H"]),
                "H_beta1_hash_copy": h_hash(beta1_state[name]["H"]),
            }
            hashes["H_stats_digest"] = stats_digest(current_stats)
            hashes["H_beta1_stats_digest"] = selected_stats_digest({
                "H_mean": beta1_stats["H_beta1_mean"], "H_q10": beta1_stats["H_beta1_q10"],
                "H_q50": beta1_stats["H_beta1_q50"], "H_q90": beta1_stats["H_beta1_q90"],
            })
            counts = controller.snapshot(name)
        prev = previous.get(name, {})
        layer_source = source_interval if prev else None
        layer_accepted = prev.get("accepted", False) if prev else False
        layer_fallback = prev.get("fallback", False) if prev else False
        layer_reason = prev.get("fallback_reason", reason) if prev else reason
        rows.append({
            "seed": spec["seed"], "run_id": spec["run_id"], "method": spec["method"],
            "learning_rate": spec["learning_rate"], "config_fingerprint": fingerprint(config),
            "refresh_step": step, "interval_index": interval, "layer": name,
            "beta_train": beta, "beta_source_interval": layer_source,
            "beta_raw_previous": prev.get("beta_raw"), "beta_update_accepted": layer_accepted,
            "beta_fallback_used": layer_fallback, "beta_fallback_reason": layer_reason,
            "previous_beta_train": prev.get("beta_train_previous", beta),
            "numerator_previous": prev.get("numerator_sum"),
            "denominator_previous": prev.get("denominator_sum"),
            "previous_interval_n_steps": prev.get("observation_count", 0),
            "accepted_update_count": counts.get("accepted_update_count", 0),
            "fallback_count": counts.get("fallback_count", 0), **trace,
            "beta_scaled_trace_F": (beta * trace["trace_F"] if beta is not None else None),
            **current_stats, **beta1_stats, **hashes,
        })
    return rows


def _enrich_intervals(intervals, method, refresh_rows, controller_rows):
    if not intervals:
        return intervals
    used = {int(row["refresh_index"]) - 1 for row in refresh_rows
            if not row["measurement_only"] and int(row["refresh_index"]) > 0}
    next_beta = {}
    for row in controller_rows:
        next_beta[(int(row["interval_index"]), row["layer"])] = row.get("beta_train")
    for row in intervals:
        raw = float(row["beta_dp_raw"])
        finite = math.isfinite(raw)
        row.update(
            beta_oracle_interval=row["beta_oracle"], beta_dp_raw_interval=row["beta_dp_raw"],
            raw_positive=bool(finite and raw > 0), raw_finite=finite,
            was_used_by_next_interval=bool(row["interval_index"] in used),
            next_beta_train=next_beta.get((int(row["interval_index"])+1, row["layer"]))
            if row["interval_index"] in used else None,
            applied_to_training=bool(row["interval_index"] in used),
        )
    return intervals


def _write_artifacts(root, config, spec, rows, layers, refreshes, beta_steps,
                     beta_intervals, controller_rows, pairing, h_certificates):
    fp = fingerprint(config)
    prefix = {"seed": spec["seed"], "run_id": spec["run_id"], "method": spec["method"],
              "learning_rate": spec["learning_rate"], "config_fingerprint": fp}
    write_csv(root / "train_metrics.csv", [dict(prefix, **row) for row in rows], TRAIN_FIELDS)
    write_csv(root / "layer_metrics.csv", [dict(prefix, **row) for row in layers], LAYER_FIELDS)
    write_csv(root / "refresh_metrics.csv", [dict(prefix, **row) for row in refreshes], REFRESH_FIELDS)
    write_csv(root / "beta_step_metrics.csv", [dict(prefix, **row) for row in beta_steps], BETA_STEP_FIELDS)
    write_csv(root / "beta_interval_metrics.csv", beta_intervals, BETA_INTERVAL_FIELDS)
    write_csv(root / "beta_controller_metrics.csv", controller_rows, BETA_CONTROLLER_FIELDS)
    write_csv(root / "eigenbin_metrics.csv", [], ["seed", "step", "layer", "bin"])
    save_json(root / "pairing.json", pairing)
    save_json(root / "h_certificates.json", h_certificates)


def train(config, seed, run_name, output, data_override=None, *, diagnostics=True,
          beta_measurement=True):
    """Run one of the four canonical paired trajectories."""
    check_config(config)
    if seed not in config["seeds"]:
        raise ValueError("Unknown seed")
    spec = next((item for item in run_specs(config) if item["run_id"] == run_name), None)
    if spec is None:
        raise ValueError("Unknown canonical run")
    spec = dict(spec, seed=seed)
    root = output_path(output) / f"seed{seed}" / run_name
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "config.json", config)
    current_provenance = provenance()
    require_pinned(current_provenance, config["smoke"])
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) \
        if config["device"] == "auto" else torch.device(config["device"])
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
    loader = DataLoader(Indexed(data), batch_size=config["batch_size"], shuffle=True,
                        drop_last=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed + 1))
    test_loader = DataLoader(test, batch_size=config["batch_size"], num_workers=0,
                             generator=torch.Generator().manual_seed(seed + 2))
    total = len(loader) * config["epochs"]
    q = config["batch_size"] / n
    sigma = get_noise_multiplier(target_epsilon=config["epsilon"], target_delta=config["delta"],
                                 sample_rate=q, steps=total, accountant="rdp")
    set_seed(seed)
    model = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    optimizer = make_optimizer(model, spec["learning_rate"], config)
    accountant = RDPAccountant()
    synthetic_rng, noise_rng = RNGStream(seed + 3, device), RNGStream(seed + 4, device)
    metadata = {
        "experiment": "expv3", "seed": seed, "run_id": run_name, "method": spec["method"],
        "learning_rate": spec["learning_rate"], "fingerprint": fingerprint(config),
        "provenance": current_provenance, "device": str(device), "dataset": "MNIST",
        "model": "SimpleCNN", "noise_multiplier": sigma, "sample_rate": q,
        "total_steps": total, "target_epsilon": config["epsilon"],
        "target_delta": config["delta"], "accountant": "rdp",
        "max_grad_norm": config["max_grad_norm"], "optimizer": "SGD", "momentum": 0,
        "initial_model_hash": digest(model.parameters()), "covariance_ridge": 1e-5,
        "eigen_budget": 0, "rng_seeds": {"init": seed, "loader": seed + 1,
        "test": seed + 2, "synthetic": seed + 3, "noise": seed + 4},
        "sampling": "fixed shuffle/drop_last; inherited RDP convention, not Poisson",
        "diagnostics_enabled": diagnostics, "beta_measurement_enabled": beta_measurement,
        "beta_algorithm_active": spec["method"] == FISHER_METHOD,
        "beta_update_rule": config["beta_update_rule"], "beta_window": config["beta_window"],
        "filter_position": "after global clipping and DP Gaussian noise",
        "synthetic_fisher_role": "measurement_only_for_dp_sgd" if spec["method"] == DP_METHOD else "training_filter_and_measurement",
        "contains_non_dp_oracle": True, "release_safe_under_dp": False,
        "deployable_beta_controller_is_postprocessing": True,
        "oracle_is_research_only": True, "actual_noise_saved": False,
        "private_fisher_used": False, "output": str(root),
    }
    save_json(root / "metadata.json", metadata)

    rows, layer_rows, beta_steps, refreshes = [], [], [], []
    controller_rows, h_certificates = [], []
    audits, synthetic_audits = [], []
    active = diagnostic_state = trace_state = beta1_state = None
    controller = AdaptiveBetaController(LAYERS, config["beta_initial"]) if spec["method"] == FISHER_METHOD else None
    current_refresh_index = None
    step = 0
    diagnostic_seconds = controller_seconds_total = 0.0
    status, diverged_step = "completed", None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(model)
    started = time.perf_counter()
    try:
        for epoch in range(1, config["epochs"] + 1):
            iterator = iter(loader)
            for _ in range(len(loader)):
                refreshed = step % config["K"] == 0
                refresh_time = 0.0
                diagnostic_spectrum_time = 0.0
                refresh_controller_time = 0.0
                refresh_index = step // config["K"]
                previous = {}
                accepted = fallback = False
                reason = "not_applicable"
                source_interval = None
                if refreshed:
                    if controller is not None and refresh_index > 0:
                        started_controller = time.perf_counter()
                        previous = controller.finalize_all(interval_index=refresh_index - 1, apply=True)
                        refresh_controller_time += time.perf_counter() - started_controller
                        source_interval = refresh_index - 1
                        accepted = all(value["accepted"] for value in previous.values())
                        fallback = any(value["fallback"] for value in previous.values())
                        reasons = {value["fallback_reason"] for value in previous.values()}
                        reason = next(iter(reasons)) if len(reasons) == 1 else "none"
                    elif controller is not None:
                        reason = "initial"

                    sync(model)
                    refresh_started = time.perf_counter()
                    samples, synthetic_audit = synthetic_samples(config, device, synthetic_rng)
                    covariances = build_covariances(model._module.state_dict(), samples, config, device)
                    trace_state = build_trace_state(covariances)
                    if spec["method"] == FISHER_METHOD:
                        beta_by_layer = {name: controller.active_beta(name) for name in LAYERS}
                        # One ExpV1 eigendecomposition supplies both the beta-scaled
                        # active state and the beta=1 counterfactual.
                        beta1_state = build_fisher_state(
                            covariances, sigma, config["max_grad_norm"], config["batch_size"]
                        )
                        from expv3.adaptive_fisher_wiener import rebuild_H_with_beta
                        active = rebuild_H_with_beta(
                            beta1_state, beta_by_layer,
                            (float(sigma) * float(config["max_grad_norm"]) / float(config["batch_size"])) ** 2,
                        )
                    else:
                        active = beta1_state = None
                    sync(model)
                    refresh_time = time.perf_counter() - refresh_started
                    diagnostic_state = None
                    if diagnostics and spec["method"] == FISHER_METHOD:
                        sync(model)
                        spectrum_started = time.perf_counter()
                        diagnostic_state = build_diagnostic_state(covariances, active, "dp_fisher_wiener")
                        sync(model)
                        diagnostic_spectrum_time = time.perf_counter() - spectrum_started
                        diagnostic_seconds += diagnostic_spectrum_time
                    current_refresh_index = refresh_index
                    refreshes.append({
                        "step": step, "refresh_index": refresh_index, "refresh_time": refresh_time,
                        "diagnostic_spectrum_time": diagnostic_spectrum_time,
                        "beta_controller_time": refresh_controller_time,
                        "active_state_bytes": state_bytes(active),
                        "measurement_only": spec["method"] == DP_METHOD,
                    })
                    if spec["method"] == FISHER_METHOD:
                        current_rows = _controller_rows(
                            config, spec, step, refresh_index, trace_state, active, beta1_state,
                            controller, previous, source_interval, accepted, fallback, reason,
                        )
                    else:
                        current_rows = _controller_rows(
                            config, spec, step, refresh_index, trace_state, None, None,
                            None, {}, None, False, False, "not_applicable",
                        )
                    controller_rows.extend(current_rows)
                    h_certificates.extend([{
                        "refresh_step": row["refresh_step"], "interval_index": row["interval_index"],
                        "layer": row["layer"], "H_hash": row["H_hash"],
                        "H_beta1_hash": row["H_beta1_hash"],
                        "beta_train": row["beta_train"],
                        "r": ((float(sigma) * float(config["max_grad_norm"]) /
                               float(config["batch_size"])) ** 2),
                        "lambda_A": beta1_state[row["layer"]]["lambda_A"].tolist()
                        if beta1_state is not None else None,
                        "lambda_G": beta1_state[row["layer"]]["lambda_G"].tolist()
                        if beta1_state is not None else None,
                    } for row in current_rows])
                    synthetic_audits.append({"step": step, **synthetic_audit})
                    del samples, covariances

                x, y, indices = next(iterator)
                model.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x.to(device)), y.to(device), reduction="sum")
                if not bool(torch.isfinite(loss)):
                    status, diverged_step = "diverged", step
                    break
                loss.backward()
                active_beta = ({name: controller.active_beta(name) for name in LAYERS}
                               if controller is not None else None)
                active_source = source_interval if controller is not None else None
                try:
                    current_rows, current_layers, current_beta, audit, elapsed, controller_elapsed = private_update(
                        model, active, optimizer, noise_rng, sigma, config, len(x), spec["method"],
                        spec["learning_rate"], trace_state=trace_state, controller=controller,
                        diagnostics=diagnostics, beta_measurement=beta_measurement,
                        refresh_index=current_refresh_index, step=step, run_id=run_name, seed=seed,
                        diagnostic_state=diagnostic_state, active_beta=active_beta,
                        active_source=active_source,
                    )
                except DivergenceError as exc:
                    status, diverged_step = "diverged", exc.step
                    break
                diagnostic_seconds += elapsed
                controller_seconds_total += controller_elapsed + refresh_controller_time
                accountant.step(noise_multiplier=sigma, sample_rate=q)
                current_rows.update(
                    step=step, epoch=epoch, train_loss=float(loss.detach()) / len(x),
                    learning_rate=spec["learning_rate"], refresh_time=refresh_time,
                    diagnostic_spectrum_time=diagnostic_spectrum_time,
                    active_state_bytes=state_bytes(active), test_loss=None, test_accuracy=None,
                    diagnostics_finite=(True if not diagnostics else not current_rows.get("diagnostic_error", False)
                                        and diagnostic_row_finite(current_rows)
                                        and all(diagnostic_row_finite(value) for value in current_layers)),
                )
                if (step + 1) % config["eval_interval"] == 0 or step + 1 == total:
                    if all_finite(model.parameters()):
                        current_rows.update(evaluate(model, test_loader, device))
                rows.append(current_rows)
                layer_rows.extend(dict(step=step, **value) for value in current_layers)
                beta_steps.extend(current_beta)
                parameters_finite = all_finite(model.parameters())
                audits.append({"step": step, "batch_indices": indices.tolist(),
                               "model_hash": digest(model.parameters()),
                               "parameters_finite": parameters_finite, **audit})
                step += 1
                if not parameters_finite:
                    status, diverged_step = "diverged", step - 1
                    break
            if status == "diverged":
                break
    finally:
        sync(model)

    wall = time.perf_counter() - started
    summary = _summary(
        config, spec, model, rows, refreshes, accountant, sigma, step, wall,
        diagnostic_seconds, controller_seconds_total, active, status, diverged_step,
    )
    metadata.update(summary, complete=status == "completed")
    save_json(root / "summary.json", summary)
    save_json(root / "metadata.json", metadata)
    interval_rows = build_interval_rows(beta_steps, config["beta_window"]) if beta_steps else []
    interval_rows = _enrich_intervals(interval_rows, spec["method"], refreshes, controller_rows)
    try:
        _write_artifacts(
            root, config, spec, rows, layer_rows, refreshes, beta_steps,
            interval_rows, controller_rows,
            {"private": audits, "synthetic": synthetic_audits}, h_certificates,
        )
    finally:
        model.remove_hooks()
    final = summary["final_accuracy"]
    suffix = f", accuracy={final:.4f}" if final is not None else ""
    print(f"{status} seed={seed} {run_name}: {step}/{total} steps{suffix}", flush=True)
    return metadata


def selected_run_specs(config, seeds, requested_run="all"):
    selected = []
    for seed in seeds:
        allowed = run_specs_for_seed(config, seed)
        if requested_run == "all":
            chosen = allowed
        else:
            chosen = [spec for spec in allowed if spec["run_id"] == requested_run]
            if not chosen:
                raise ValueError(f"Unknown run {requested_run} for seed {seed}")
        selected.extend(chosen)
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


if __name__ == "__main__":
    main()
