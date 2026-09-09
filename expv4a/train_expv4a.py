"""Run ExpV3 adaptive Fisher-Wiener and add the ExpV4a diagnostics."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import torch

from expv3 import common as v3_common
from expv3.common import FISHER_METHOD, save_json, write_csv
from expv3.train_expv3 import train as train_v3

from expv4a.gamma_estimators import (
    certificate_holds,
    model_gamma_from_state,
    multiplicative_error,
    ratio_sqrt,
)


FULL_CONFIG = {
    "experiment": "expv4a", "seeds": [42, 7, 91],
    "dp_sgd_learning_rate": 0.5, "fisher_learning_rates": [0.5, 1.0, 5.0],
    "beta_initial": 1.0, "beta_update_rule": "one_interval_lag_hold_last_positive",
    "beta_window": 50, "epochs": 5, "batch_size": 256, "epsilon": 1.0,
    "delta": 1e-5, "max_grad_norm": 1.0, "optimizer": "SGD", "momentum": 0,
    "K": 50, "M_syn": 2560, "eval_interval": 100, "device": "auto", "threads": 4,
    "train_subset": None, "test_subset": None, "smoke": False,
}

SMOKE_CONFIG = {
    **FULL_CONFIG, "seeds": [42], "fisher_learning_rates": [0.5],
    "beta_window": 2, "epochs": 1, "batch_size": 4, "K": 2, "M_syn": 8,
    "eval_interval": 1, "train_subset": 16, "test_subset": 32, "smoke": True,
}

REFRESH_FIELDS = [
    "seed", "run_id", "learning_rate", "step", "refresh_index", "interval_index",
    "layer", "beta_train", "beta_source_interval", "trace_F", "sum_a", "sum_a_H2",
    "H_mean", "H_q50", "H_q90", "mean_H2", "sum_H2", "model_signal_retention",
    "gamma_model_raw", "model_noise_retention_before", "model_noise_retention_after_raw",
]
INTERVAL_FIELDS = [
    "seed", "run_id", "learning_rate", "interval_index", "layer", "n_steps",
    "gamma_model", "gamma_oracle", "gamma_dp_raw", "gamma_dp_valid",
    "model_to_oracle_ratio", "model_multiplicative_error", "clean_signal_energy_in",
    "clean_signal_energy_filtered", "signal_retention_oracle",
    "signal_retention_after_model_gamma", "noise_energy_in", "noise_energy_filtered",
    "noise_retention_oracle", "noise_retention_after_model_gamma", "dp_signal_energy_in",
    "dp_signal_energy_out",
]

EPS = 1e-12


def check_config(config):
    expected = SMOKE_CONFIG if config.get("smoke") else FULL_CONFIG
    if set(config) != set(expected):
        raise ValueError("ExpV4a config keys mismatch")
    for key, value in expected.items():
        if key != "device" and config[key] != value:
            raise ValueError(f"ExpV4a protocol mismatch: {key}")
    if not isinstance(config["device"], str):
        raise ValueError("device must be a string")
    return config


def v3_config(config):
    """Use the exact ExpV3 protocol internally; gamma is not a new trainer."""
    result = dict(config)
    result["experiment"] = "expv3"
    result["fisher_learning_rates"] = [0.5, 1.0, 5.0]
    return result


def run_specs(config):
    return [
        {
            "run_id": v3_common.run_id(FISHER_METHOD, lr),
            "method": FISHER_METHOD,
            "learning_rate": lr,
        }
        for lr in config["fisher_learning_rates"]
    ]


def selected_run_specs(config, seeds, requested_run="all"):
    selected = []
    allowed = {row["run_id"]: row for row in run_specs(config)}
    for seed in seeds:
        if seed not in config["seeds"]:
            raise ValueError(f"Unknown seed {seed}")
        if requested_run == "all":
            chosen = list(allowed.values())
        elif requested_run in allowed:
            chosen = [allowed[requested_run]]
        else:
            raise ValueError(f"Unknown run {requested_run}")
        selected.extend([dict(row, seed=seed) for row in chosen])
    return selected


def _refresh_diagnostics(root, config, spec):
    certificates = json.loads((root / "h_certificates.json").read_text())
    controller = pd.read_csv(root / "beta_controller_metrics.csv")
    controller_by_key = {
        (int(row.interval_index), row.layer): row
        for _, row in controller.iterrows()
    }
    rows = []
    for cert in certificates:
        interval = int(cert["interval_index"])
        layer = cert["layer"]
        control = controller_by_key[(interval, layer)]
        h = torch_h(cert["lambda_A"], cert["lambda_G"], cert["beta_train"], cert["r"])
        stats = model_gamma_from_state({
            "lambda_A": torch.as_tensor(cert["lambda_A"], dtype=torch.float64),
            "lambda_G": torch.as_tensor(cert["lambda_G"], dtype=torch.float64),
            "H": h,
        }, cert["beta_train"])
        rows.append({
            "seed": spec["seed"], "run_id": spec["run_id"],
            "learning_rate": spec["learning_rate"], "step": int(cert["refresh_step"]),
            "refresh_index": interval, "interval_index": interval, "layer": layer,
            "beta_train": float(cert["beta_train"]),
            "beta_source_interval": (None if pd.isna(control.beta_source_interval)
                                      else int(control.beta_source_interval)),
            "trace_F": float(control.trace_F), **stats,
            "H_mean": float(h.mean()), "H_q50": float(h.quantile(.5)),
            "H_q90": float(h.quantile(.9)),
            "sum_H2": float(h.square().sum()),
            "model_noise_retention_before": stats["mean_H2"],
        })
    write_csv(root / "gamma_refresh_metrics.csv", rows, REFRESH_FIELDS)
    return pd.DataFrame(rows, columns=REFRESH_FIELDS)


def torch_h(lambda_a, lambda_g, beta, r):
    lambda_a = torch.as_tensor(lambda_a, dtype=torch.float64)
    lambda_g = torch.as_tensor(lambda_g, dtype=torch.float64)
    lambda_f = lambda_g[:, None] * lambda_a[None, :]
    scaled = float(beta) * lambda_f
    return torch.where(scaled + float(r) > 0, scaled / (scaled + float(r)),
                       torch.zeros_like(scaled))


def _interval_diagnostics(root, config, spec, refresh):
    intervals = pd.read_csv(root / "beta_interval_metrics.csv")
    layers = pd.read_csv(root / "layer_metrics.csv")
    beta_steps = pd.read_csv(root / "beta_step_metrics.csv")
    refresh_by_key = {
        (int(row.interval_index), row.layer): row
        for _, row in refresh.iterrows()
    }
    rows = []
    for _, interval in intervals.iterrows():
        key = (int(interval.interval_index), interval.layer)
        model = refresh_by_key[key]
        step_values = layers[
            (layers.layer == interval.layer)
            & (layers.step >= int(interval.start_step))
            & (layers.step <= int(interval.end_step))
        ]
        dp_values = beta_steps[
            (beta_steps.layer == interval.layer)
            & (beta_steps.step >= int(interval.start_step))
            & (beta_steps.step <= int(interval.end_step))
        ]
        clean_in = float((step_values.clean_clipped_norm ** 2).sum())
        clean_filtered = float((
            step_values.signal_retention
            * (step_values.clean_clipped_norm ** 2 + EPS)
        ).sum())
        noise_in = float((step_values.actual_noise_norm ** 2).sum())
        noise_filtered = float((
            step_values.noise_retention
            * (step_values.actual_noise_norm ** 2 + EPS)
        ).sum())
        dp_in = float((dp_values.noisy_gradient_energy
                       - dp_values.expected_noise_energy).sum())
        h2_sum = float(model.sum_H2)
        dp_out = float((step_values.filtered_gradient_norm ** 2).sum())
        r_values = pd.to_numeric(dp_values.r, errors="coerce")
        dp_out -= float(r_values.sum()) * h2_sum
        gamma_oracle = ratio_sqrt(clean_in, clean_filtered)
        gamma_dp = ratio_sqrt(dp_in, dp_out)
        signal_retention = clean_filtered / clean_in
        noise_retention = noise_filtered / noise_in
        gamma_model = float(model.gamma_model_raw)
        rows.append({
            "seed": spec["seed"], "run_id": spec["run_id"],
            "learning_rate": spec["learning_rate"],
            "interval_index": int(interval.interval_index), "layer": interval.layer,
            "n_steps": int(interval.n_steps), "gamma_model": gamma_model,
            "gamma_oracle": gamma_oracle, "gamma_dp_raw": gamma_dp,
            "gamma_dp_valid": gamma_dp is not None,
            "model_to_oracle_ratio": (gamma_model / gamma_oracle
                                       if gamma_oracle and gamma_oracle > 0 else None),
            "model_multiplicative_error": multiplicative_error(gamma_model, gamma_oracle),
            "clean_signal_energy_in": clean_in,
            "clean_signal_energy_filtered": clean_filtered,
            "signal_retention_oracle": signal_retention,
            "signal_retention_after_model_gamma": gamma_model * gamma_model * signal_retention,
            "noise_energy_in": noise_in, "noise_energy_filtered": noise_filtered,
            "noise_retention_oracle": noise_retention,
            "noise_retention_after_model_gamma": gamma_model * gamma_model * noise_retention,
            "dp_signal_energy_in": dp_in, "dp_signal_energy_out": dp_out,
        })
    write_csv(root / "gamma_interval_metrics.csv", rows, INTERVAL_FIELDS)
    return pd.DataFrame(rows, columns=INTERVAL_FIELDS)


def _postprocess(root, config, spec):
    refresh = _refresh_diagnostics(root, config, spec)
    intervals = _interval_diagnostics(root, config, spec, refresh)
    if refresh.empty or intervals.empty:
        raise ValueError("ExpV4a needs at least one refresh and one interval")
    summary = json.loads((root / "summary.json").read_text())
    summary.update({
        "experiment": "expv4a", "gamma_diagnostic_only": True,
        "gamma_influences_training": False, "gamma_training_trajectory": "expv3_identical",
        "gamma_refresh_rows": len(refresh), "gamma_interval_rows": len(intervals),
        "gamma_model_certificate_all_pass": all(
            certificate_holds(row.gamma_model_raw, row.mean_H2)
            for _, row in refresh.iterrows()
        ),
    })
    save_json(root / "summary.json", summary)
    metadata = json.loads((root / "metadata.json").read_text())
    metadata.update({
        "experiment": "expv4a", "gamma_diagnostic_only": True,
        "gamma_influences_training": False, "gamma_training_trajectory": "expv3_identical",
    })
    save_json(root / "metadata.json", metadata)
    save_json(root / "v4a_config.json", config)


def train(config, seed, run_name, output, data_override=None):
    check_config(config)
    spec = next((row for row in selected_run_specs(config, [seed])
                 if row["run_id"] == run_name), None)
    if spec is None:
        raise ValueError(f"Unknown run {run_name}")
    output = Path(output).resolve()
    destination = output / f"seed{seed}" / run_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temp_root = Path(tempfile.mkdtemp(prefix="_expv4a_", dir=str(v3_common.RUNS_ROOT)))
    try:
        train_v3(v3_config(config), seed, run_name, temp_root, data_override=data_override)
        shutil.copytree(temp_root / f"seed{seed}" / run_name, destination)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    _postprocess(destination, config, spec)
    return json.loads((destination / "metadata.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", default="all")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = check_config(json.loads(Path(args.config).read_text()))
    seeds = config["seeds"] if args.seed is None else [args.seed]
    for spec in selected_run_specs(config, seeds, args.run):
        train(config, spec["seed"], spec["run_id"], args.output)


if __name__ == "__main__":
    main()
