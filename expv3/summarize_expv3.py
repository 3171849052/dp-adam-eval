"""Descriptive ExpV3 utility, controller, mechanism, and LR summaries."""

import argparse
import math

import numpy as np
import pandas as pd

from expv3.common import FISHER_METHOD, run_specs, save_json, write_csv
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


def seed_level_median(rows, field):
    """Median each seed's observed statistic, then median across seeds."""
    by_seed = {}
    for row in rows:
        value = row.get(field)
        try:
            if value is not None and math.isfinite(float(value)):
                by_seed.setdefault(row["seed"], []).append(float(value))
        except (TypeError, ValueError):
            continue
    return _median([_median(values) for values in by_seed.values()])


def _late(frame, total):
    return frame[frame.step + 1 > total / 2]


def _valid_full_metric(row, metric):
    value = row.get(metric)
    try:
        return row.get("status") == "completed" and value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def controller_decision_rates(frame):
    """Return adaptive decision rates, excluding interval zero."""
    if frame.empty or frame.method.iloc[0] != FISHER_METHOD:
        return None, None
    decisions = frame[frame.interval_index > 0]
    if decisions.empty:
        return None, None
    fallback = decisions.beta_fallback_used.map(lambda x: str(x).lower() == "true")
    accepted = decisions.beta_update_accepted.map(lambda x: str(x).lower() == "true")
    return float(fallback.mean()), float(accepted.mean())


def raw_beta_statistics(intervals):
    raw = pd.to_numeric(intervals.beta_dp_raw, errors="coerce")
    raw = raw[np.isfinite(raw)]
    return (float(raw.median()), float((raw < 0).mean())) if len(raw) else (None, None)


def build_paired_utility_rows(summaries, seeds):
    """Build paired rows with explicit status and valid-pair denominators."""
    by_seed = {}
    for row in summaries:
        by_seed.setdefault(row["seed"], {})[row["run_id"]] = row
    metrics = ("final_accuracy", "accuracy_auc", "late_mean_accuracy")
    result = []
    for seed in seeds:
        adaptive = by_seed[seed]["dp_fisher_wiener_adaptive_beta_lr0p50"]
        dp = by_seed[seed]["dp_sgd_lr0p50"]
        for metric in metrics:
            pair_valid = (_valid_full_metric(adaptive, metric)
                          and _valid_full_metric(dp, metric))
            left, right = adaptive.get(metric), dp.get(metric)
            result.append({
                "comparison": "adaptive_lr0p50_minus_dp_sgd_lr0p50", "seed": seed,
                "metric": metric, "adaptive_status": adaptive.get("status"),
                "dp_sgd_status": dp.get("status"), "pair_valid": pair_valid,
                "adaptive": left, "dp_sgd": right,
                "delta": (float(left) - float(right) if pair_valid else None),
                "mean": None, "sd": None, "n_total_pairs": None, "n_valid_pairs": None,
            })
    for metric in metrics:
        values = [row["delta"] for row in result
                  if row["metric"] == metric and row["pair_valid"]]
        result.append({
            "comparison": "adaptive_lr0p50_minus_dp_sgd_lr0p50", "seed": "aggregate",
            "metric": metric, "adaptive_status": "aggregate", "dp_sgd_status": "aggregate",
            "pair_valid": None, "adaptive": None, "dp_sgd": None, "delta": None,
            "mean": _mean(values), "sd": _sd(values), "n_total_pairs": len(seeds),
            "n_valid_pairs": len(values),
        })
    return result


def summarize(config, runs, output, require_tests=True):
    validate(config, runs, output, require_tests=require_tests)
    records = []
    for seed in config["seeds"]:
        for spec in run_specs(config):
            root = runs / f"seed{seed}" / spec["run_id"]
            records.append({
                "seed": seed, "spec": spec, "root": root,
                "summary": load(root / "summary.json"),
                "controller": pd.read_csv(root / "beta_controller_metrics.csv"),
                "layer": pd.read_csv(root / "layer_metrics.csv"),
                "train": pd.read_csv(root / "train_metrics.csv"),
                "interval": pd.read_csv(root / "beta_interval_metrics.csv"),
            })
    summaries = [record["summary"] for record in records]
    write_csv(output / "summary_runs.csv", summaries)

    controller_rows = []
    for record in records:
        frame, intervals, summary = record["controller"], record["interval"], record["summary"]
        if frame.empty:
            continue
        for keys, group in frame.groupby(["method", "learning_rate", "seed", "layer"]):
            method, lr, seed, layer = keys
            observed = intervals[intervals.layer == layer]
            median_raw, negative_rate = raw_beta_statistics(observed)
            fallback_rate, accepted_update_rate = controller_decision_rates(group)
            train_oracle_ratios, lag_errors = [], []
            for _, row in group.iterrows():
                current = intervals[(intervals.layer == row.layer)
                                    & (intervals.interval_index == row.interval_index)]
                if len(current):
                    oracle = float(current.iloc[0].beta_oracle)
                    if pd.notna(row.beta_train) and oracle > 0:
                        train_oracle_ratios.append(float(row.beta_train) / oracle)
                    if pd.notna(row.beta_raw_previous) and float(row.beta_raw_previous) > 0 and oracle > 0:
                        lag_errors.append(abs(math.log10(float(row.beta_raw_previous) / oracle)))
            controller_rows.append({
                "seed": seed, "run_id": summary["run_id"], "status": summary["status"],
                "completed_steps": summary["completed_steps"],
                "metric_scope": "full" if summary["status"] == "completed" else "prefix",
                "method": method, "learning_rate": lr, "layer": layer,
                "number_of_intervals": len(group), "median_beta_train": _median(group.beta_train),
                "mean_beta_train": _mean(group.beta_train), "median_beta_raw": median_raw,
                "raw_negative_rate": negative_rate, "fallback_rate": fallback_rate,
                "accepted_update_rate": accepted_update_rate,
                "median_beta_train_to_oracle_ratio": _median(train_oracle_ratios),
                "median_lag_log_error": _median(lag_errors),
                "median_observability_snr": _median(observed.oracle_observability_snr),
            })
    write_csv(output / "summary_beta_controller.csv", controller_rows)

    mechanism_rows, layer_rows = [], []
    for record in records:
        frame, summary = record["layer"], record["summary"]
        if frame.empty:
            continue
        status = summary["status"]
        metric_scope = "full" if status == "completed" else "prefix"
        total = int(summary["planned_steps"] if status == "completed"
                    else max(summary["completed_steps"], 1))
        for (method, lr, layer), group in frame.groupby(["method", "learning_rate", "layer"]):
            late = _late(group, total)
            identity = {
                "seed": record["seed"], "run_id": summary["run_id"], "status": status,
                "completed_steps": summary["completed_steps"], "metric_scope": metric_scope,
                "method": method, "learning_rate": lr, "layer": layer,
            }
            mechanism_rows.append({
                **identity,
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
            layer_rows.append({
                **identity,
                "H_q50": _median(late.H_q50), "H_q10": _median(late.H_q10),
                "H_q90": _median(late.H_q90), "signal_retention": _median(late.signal_retention),
                "noise_retention": _median(late.noise_retention),
                "effective_signal_lr": _median(late.effective_signal_lr),
                "relmse_filtered": _median(late.relmse_filtered),
            })
    write_csv(output / "summary_mechanism.csv", mechanism_rows)
    write_csv(output / "summary_beta_layers.csv", layer_rows)

    fisher_summaries = [row for row in summaries if row["method"] == FISHER_METHOD]
    lr_rows = []
    for lr in (0.5, 1.0, 5.0):
        group = [row for row in fisher_summaries if row["learning_rate"] == lr]
        completed = [row for row in group if row["status"] == "completed"]
        diverged = [row for row in group if row["status"] == "diverged"]
        fallback = [row for row in controller_rows
                    if row["method"] == FISHER_METHOD and row["learning_rate"] == lr]
        mech = [row for row in mechanism_rows
                if row["method"] == FISHER_METHOD and row["learning_rate"] == lr
                and row["layer"] == "fc2"]
        utility = {
            "final_accuracy_mean": _mean([row["final_accuracy"] for row in completed]),
            "final_accuracy_sd": _sd([row["final_accuracy"] for row in completed]),
            "accuracy_auc_mean": _mean([row["accuracy_auc"] for row in completed]),
            "accuracy_auc_sd": _sd([row["accuracy_auc"] for row in completed]),
            "late_mean_accuracy_mean": _mean([row["late_mean_accuracy"] for row in completed]),
            "late_mean_accuracy_sd": _sd([row["late_mean_accuracy"] for row in completed]),
            "final_test_loss_mean": _mean([row["final_test_loss"] for row in completed]),
            "final_test_loss_sd": _sd([row["final_test_loss"] for row in completed]),
        }
        completed_fallback = [row for row in fallback if row["status"] == "completed"]
        completed_mech = [row for row in mechanism_rows
                          if row["method"] == FISHER_METHOD
                          and row["learning_rate"] == lr
                          and row["layer"] == "fc2"
                          and row["status"] == "completed"]
        lr_rows.append({
            "method": FISHER_METHOD, "learning_rate": lr, "n_runs": len(group),
            "n_completed": len(completed), "n_diverged": len(diverged),
            "utility_aggregate_scope": "completed_only", **utility,
            "final_accuracy_mean_completed": utility["final_accuracy_mean"],
            "final_accuracy_sd_completed": utility["final_accuracy_sd"],
            "accuracy_auc_mean_completed": utility["accuracy_auc_mean"],
            "accuracy_auc_sd_completed": utility["accuracy_auc_sd"],
            "late_mean_accuracy_mean_completed": utility["late_mean_accuracy_mean"],
            "late_mean_accuracy_sd_completed": utility["late_mean_accuracy_sd"],
            "final_test_loss_mean_completed": utility["final_test_loss_mean"],
            "final_test_loss_sd_completed": utility["final_test_loss_sd"],
            "number_diverged": len(diverged),
            "mechanism_aggregate_scope": "all_observed_seed_level",
            "median_fallback_rate": seed_level_median(fallback, "fallback_rate"),
            "median_beta_train": seed_level_median(fallback, "median_beta_train"),
            "median_effective_signal_lr": seed_level_median(mech, "effective_signal_lr"),
            "median_fallback_rate_completed": seed_level_median(
                completed_fallback, "fallback_rate"
            ),
            "median_beta_train_completed": seed_level_median(
                completed_fallback, "median_beta_train"
            ),
            "median_effective_signal_lr_completed": seed_level_median(
                completed_mech, "effective_signal_lr"
            ),
        })
    write_csv(output / "summary_learning_rate.csv", lr_rows)

    primary_rows = build_paired_utility_rows(summaries, config["seeds"])
    write_csv(output / "summary_paired_utility.csv", primary_rows)

    analysis = {
        "experiment": "expv3",
        "primary_comparison": "adaptive beta Fisher-Wiener lr=0.5 vs DP-SGD lr=0.5",
        "fisher_learning_rates": [0.5, 1.0, 5.0], "one_interval_lag": True,
        "hold_last_positive": True, "no_success_thresholds": True,
        "contains_non_dp_oracle": True, "release_safe_under_dp": False,
        "deployable_beta_controller_is_postprocessing": True, "oracle_is_research_only": True,
        "completed_runs": sum(row["status"] == "completed" for row in summaries),
        "diverged_runs": sum(row["status"] == "diverged" for row in summaries),
        "completed_by_fisher_lr": {
            str(lr): sum(row["method"] == FISHER_METHOD and row["learning_rate"] == lr
                         and row["status"] == "completed" for row in summaries)
            for lr in (0.5, 1.0, 5.0)
        },
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
