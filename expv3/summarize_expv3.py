"""Descriptive ExpV3 utility, controller, mechanism, and LR summaries."""

import argparse
import math

import numpy as np
import pandas as pd

from expv3.common import FISHER_METHOD, LAYERS, ROOT, run_specs, save_json, write_csv
from expv3.validate_expv3 import cli, load, validate


def _mean(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _sd(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else None


def _median(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(values.median()) if len(values) else None


def _late(frame, total):
    return frame[frame.step + 1 > total / 2]


def summarize(config, runs, output, require_tests=True):
    validate(config, runs, output, require_tests=require_tests)
    summaries, controller_parts, layer_parts, train_parts, interval_parts = [], [], [], [], []
    for seed in config["seeds"]:
        for spec in run_specs(config):
            root = runs / f"seed{seed}" / spec["run_id"]
            summary = load(root / "summary.json")
            summaries.append(summary)
            controller_parts.append(pd.read_csv(root / "beta_controller_metrics.csv"))
            layer_parts.append(pd.read_csv(root / "layer_metrics.csv"))
            train_parts.append(pd.read_csv(root / "train_metrics.csv"))
            interval_parts.append(pd.read_csv(root / "beta_interval_metrics.csv"))
    write_csv(output / "summary_runs.csv", summaries)

    controller_rows = []
    for frame, intervals in zip(controller_parts, interval_parts):
        if frame.empty:
            continue
        for keys, group in frame.groupby(["method", "learning_rate", "seed", "layer"]):
            method, lr, seed, layer = keys
            raw = group.beta_raw_previous
            train = group.beta_train
            train_oracle_ratios, lag_errors, observability = [], [], []
            for _, row in group.iterrows():
                current = intervals[(intervals.layer == row.layer) &
                                    (intervals.interval_index == row.interval_index)]
                if len(current):
                    oracle = float(current.iloc[0].beta_oracle)
                    if pd.notna(row.beta_train) and oracle > 0:
                        train_oracle_ratios.append(float(row.beta_train) / oracle)
                    if pd.notna(row.beta_raw_previous) and float(row.beta_raw_previous) > 0 and oracle > 0:
                        lag_errors.append(abs(math.log10(float(row.beta_raw_previous) / oracle)))
                    if pd.notna(current.iloc[0].oracle_observability_snr):
                        observability.append(float(current.iloc[0].oracle_observability_snr))
            controller_rows.append({
                "method": method, "learning_rate": lr, "seed": seed, "layer": layer,
                "number_of_intervals": len(group), "median_beta_train": _median(train),
                "mean_beta_train": _mean(train), "median_beta_raw": _median(raw),
                "raw_negative_rate": float((pd.to_numeric(raw, errors="coerce") < 0).mean()),
                "fallback_rate": float(group.beta_fallback_used.map(lambda x: str(x).lower() == "true").mean()),
                "accepted_update_rate": float(group.beta_update_accepted.map(lambda x: str(x).lower() == "true").mean()),
                "median_beta_train_to_oracle_ratio": _median(train_oracle_ratios),
                "median_lag_log_error": _median(lag_errors),
                "median_observability_snr": _median(observability),
            })
    write_csv(output / "summary_beta_controller.csv", controller_rows)

    mechanism_rows = []
    for frame in layer_parts:
        if frame.empty:
            continue
        total = int(frame.step.max()) + 1
        for (method, lr, layer), group in frame.groupby(["method", "learning_rate", "layer"]):
            late = _late(group, total)
            mechanism_rows.append({
                "method": method, "learning_rate": lr, "layer": layer,
                "signal_retention": _median(late.signal_retention),
                "noise_retention": _median(late.noise_retention),
                "signal_amplitude_retention": _median(late.signal_amplitude_retention),
                "effective_signal_lr": _median(late.effective_signal_lr),
                "snr_gain_db": _median(late.snr_gain_db),
                "relmse_filtered": _median(late.relmse_filtered),
                "cosine_filtered": _median(late.cosine_filtered),
                "optimizer_update_norm": _median(late.optimizer_update_norm),
                "update_to_clean_reference_ratio": _median(late.update_to_clean_reference_ratio),
            })
    write_csv(output / "summary_mechanism.csv", mechanism_rows)

    layer_rows = []
    for frame in layer_parts:
        if frame.empty:
            continue
        total = int(frame.step.max()) + 1
        for (method, lr, layer), group in frame.groupby(["method", "learning_rate", "layer"]):
            late = _late(group, total)
            layer_rows.append({
                "method": method, "learning_rate": lr, "layer": layer,
                "H_q50": _median(late.H_q50), "H_q10": _median(late.H_q10),
                "H_q90": _median(late.H_q90), "signal_retention": _median(late.signal_retention),
                "noise_retention": _median(late.noise_retention),
                "effective_signal_lr": _median(late.effective_signal_lr),
                "relmse_filtered": _median(late.relmse_filtered),
            })
    write_csv(output / "summary_beta_layers.csv", layer_rows)

    fisher_summaries = [row for row in summaries if row["method"] == FISHER_METHOD]
    lr_rows = []
    for lr in (0.5, 1.0, 5.0):
        group = [row for row in fisher_summaries if row["learning_rate"] == lr]
        fallback = [row for row in controller_rows if row["method"] == FISHER_METHOD and row["learning_rate"] == lr]
        mech = [row for row in mechanism_rows if row["method"] == FISHER_METHOD and row["learning_rate"] == lr and row["layer"] == "fc2"]
        lr_rows.append({
            "method": FISHER_METHOD, "learning_rate": lr, "n_runs": len(group),
            "final_accuracy_mean": _mean([row["final_accuracy"] for row in group]),
            "final_accuracy_sd": _sd([row["final_accuracy"] for row in group]),
            "accuracy_auc_mean": _mean([row["accuracy_auc"] for row in group]),
            "accuracy_auc_sd": _sd([row["accuracy_auc"] for row in group]),
            "late_mean_accuracy_mean": _mean([row["late_mean_accuracy"] for row in group]),
            "late_mean_accuracy_sd": _sd([row["late_mean_accuracy"] for row in group]),
            "final_test_loss_mean": _mean([row["final_test_loss"] for row in group]),
            "final_test_loss_sd": _sd([row["final_test_loss"] for row in group]),
            "number_diverged": sum(row["status"] == "diverged" for row in group),
            "median_fallback_rate": _median([row["fallback_rate"] for row in fallback]),
            "median_beta_train": _median([row["median_beta_train"] for row in fallback]),
            "median_effective_signal_lr": _median([row["effective_signal_lr"] for row in mech]),
        })
    write_csv(output / "summary_learning_rate.csv", lr_rows)

    primary_rows = []
    for seed in config["seeds"]:
        by_name = {row["run_id"]: row for row in summaries if row["seed"] == seed}
        dp = by_name["dp_sgd_lr0p50"]
        adaptive = by_name["dp_fisher_wiener_adaptive_beta_lr0p50"]
        for metric in ("final_accuracy", "accuracy_auc", "late_mean_accuracy"):
            left, right = adaptive.get(metric), dp.get(metric)
            primary_rows.append({"comparison": "adaptive_lr0p50_minus_dp_sgd_lr0p50", "seed": seed,
                                 "metric": metric, "adaptive": left, "dp_sgd": right,
                                 "delta": (left - right if left is not None and right is not None else None),
                                 "mean": None, "sd": None})
    for metric in ("final_accuracy", "accuracy_auc", "late_mean_accuracy"):
        values = [row["delta"] for row in primary_rows if row["metric"] == metric and row["delta"] is not None]
        primary_rows.append({"comparison": "adaptive_lr0p50_minus_dp_sgd_lr0p50", "seed": "aggregate",
                             "metric": metric, "adaptive": None, "dp_sgd": None,
                             "delta": None, "mean": _mean(values), "sd": _sd(values)})
    write_csv(output / "summary_paired_utility.csv", primary_rows)

    analysis = {
        "experiment": "expv3", "primary_comparison": "adaptive beta Fisher-Wiener lr=0.5 vs DP-SGD lr=0.5",
        "fisher_learning_rates": [0.5, 1.0, 5.0], "one_interval_lag": True,
        "hold_last_positive": True, "no_success_thresholds": True,
        "contains_non_dp_oracle": True, "release_safe_under_dp": False,
        "deployable_beta_controller_is_postprocessing": True, "oracle_is_research_only": True,
        "interpretation_cases": {
            "A": "adaptive 0.5 exceeds DP-SGD 0.5 in utility",
            "B": "adaptive 0.5 trails but 1 or 5 recovers utility",
            "C": "utility improves through 5 without divergence; the upper boundary is not an optimum",
            "D": "5 diverges and 1 is best; the stability/utility region is partially bracketed",
            "E": "denoising improves while utility falls, separating Wiener MSE from optimization utility",
            "F": "high fallback, especially in early layers, indicates online observability limits",
        },
        "runs": [f"seed{seed}/{spec['run_id']}" for seed in config["seeds"] for spec in run_specs(config)],
    }
    save_json(output / "summary.json", analysis)
    print(f"ExpV3 summaries generated: {output}")
    return analysis


if __name__ == "__main__":
    summarize(*cli())
