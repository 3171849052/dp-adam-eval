"""Descriptive beta-estimation summaries; no success threshold is applied."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from expv2.common import LAYERS, ROOT, run_specs_for_seed, save_json, write_csv
from expv2.validate_expv2 import cli, load, validate


def _finite_or_none(value):
    try:
        return float(value) if math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def _corr(x, y, method):
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2 or frame.x.nunique() < 2 or frame.y.nunique() < 2:
        return None
    value = frame.x.corr(frame.y, method=method)
    return _finite_or_none(value)


def _median(values):
    value = pd.Series(values).dropna()
    return _finite_or_none(value.median()) if len(value) else None


def _mean(values):
    value = pd.Series(values).dropna()
    return _finite_or_none(value.mean()) if len(value) else None


def _layer_summary(interval, lagged, hold, seed, n_runs=1, aggregation="run"):
    rows = []
    for (method, learning_rate, layer), group in interval.groupby(["method", "learning_rate", "layer"]):
        positive = group[(group.beta_dp_raw > 0) & (group.beta_oracle > 0)]
        lag = lagged[(lagged.method == method) & (lagged.learning_rate == learning_rate) & (lagged.layer == layer)]
        hold_part = hold[(hold.method == method) & (hold.learning_rate == learning_rate) & (hold.layer == layer)]
        ratio = positive.beta_dp_raw / positive.beta_oracle
        log_error = ratio.map(lambda value: abs(math.log10(value)) if value > 0 else np.nan)
        row = dict(
            aggregation=aggregation, seed=seed, n_runs=n_runs, sample_count=len(group),
            method=method, learning_rate=learning_rate, layer=layer,
            interval_count=len(group), oracle_beta_median=_median(group.beta_oracle),
            oracle_beta_mean=_mean(group.beta_oracle), dp_beta_raw_median=_median(group.beta_dp_raw),
            dp_beta_raw_mean=_mean(group.beta_dp_raw),
            negative_interval_rate=float((group.beta_dp_raw < 0).mean()) if len(group) else None,
            positive_interval_coverage=float((group.beta_dp_raw > 0).mean()) if len(group) else None,
            median_positive_ratio=_median(ratio), median_log10_abs_ratio_error=_median(log_error),
            mean_log10_abs_ratio_error=_mean(log_error), pearson=_corr(group.beta_dp_raw, group.beta_oracle, "pearson"),
            spearman=_corr(group.beta_dp_raw, group.beta_oracle, "spearman"),
            lag_positive_coverage=float(lag.lag_prediction_positive.mean()) if len(lag) else None,
            lag_median_log10_error=_median(lag.lag_log10_abs_ratio_error) if len(lag) else None,
            hold_median_log10_error=_median(hold_part.hold_log10_abs_ratio_error) if len(hold_part) else None,
            oracle_observability_snr_median=_median(group.oracle_observability_snr),
        )
        rows.append(row)
    return rows


def _trajectory_rows(frames, left_name, right_name, relation):
    left = frames[left_name].rename(columns={"beta_oracle": "beta_left"})
    right = frames[right_name].rename(columns={"beta_oracle": "beta_right"})
    joined = left.merge(right, on=["seed", "layer", "interval_index"], suffixes=("_left", "_right"))
    result = []
    for _, row in joined.iterrows():
        a, b = float(row.beta_left), float(row.beta_right)
        difference = abs(math.log10(a) - math.log10(b)) if a > 0 and b > 0 else None
        result.append(dict(
            seed=int(row.seed), layer=row.layer, interval_index=int(row.interval_index),
            comparison=relation, left_run=left_name, right_run=right_name,
            beta_oracle_left=a, beta_oracle_right=b, trajectory_log_difference=difference,
            ratio=(a / b if a > 0 and b > 0 else None),
        ))
    return result


def summarize(config, runs, output, require_tests=True):
    validate(config, runs, output, require_tests=require_tests)
    summaries = []
    interval_frames = {}
    lagged_frames = {}
    hold_frames = {}
    window_frames = {}
    for seed in config["seeds"]:
        for spec in run_specs_for_seed(config, seed):
            key = (seed, spec["run_id"])
            root = runs / f"seed{seed}" / spec["run_id"]
            summary = load(root / "summary.json")
            summaries.append(summary)
            interval_frames[key] = pd.read_csv(root / "beta_interval_metrics.csv")
            lagged_frames[key] = pd.read_csv(root / "beta_lagged_metrics.csv")
            hold_frames[key] = pd.read_csv(root / "beta_hold_predictor.csv")
            window_frames[key] = pd.read_csv(root / "beta_window_sensitivity.csv")

    write_csv(output / "summary_runs.csv", summaries)

    layer_rows = []
    for (seed, run_name), frame in interval_frames.items():
        layer_rows.extend(_layer_summary(frame, lagged_frames[(seed, run_name)], hold_frames[(seed, run_name)], seed))
    # A separate pooled row is used instead of pretending pooled samples came
    # from one seed.  The n_runs column makes the distinction explicit.
    for spec in (run_specs_for_seed(config, 42) if 42 in config["seeds"] else []):
        parts = [interval_frames[(seed, spec["run_id"])] for seed in config["seeds"] if (seed, spec["run_id"]) in interval_frames]
        lag_parts = [lagged_frames[(seed, spec["run_id"])] for seed in config["seeds"] if (seed, spec["run_id"]) in lagged_frames]
        hold_parts = [hold_frames[(seed, spec["run_id"])] for seed in config["seeds"] if (seed, spec["run_id"]) in hold_frames]
        if parts:
            pooled = pd.concat(parts, ignore_index=True)
            pooled_lag = pd.concat(lag_parts, ignore_index=True) if lag_parts else pd.DataFrame(columns=lag_parts[0].columns if lag_parts else [])
            pooled_hold = pd.concat(hold_parts, ignore_index=True) if hold_parts else pd.DataFrame(columns=hold_parts[0].columns if hold_parts else [])
            layer_rows.extend(_layer_summary(pooled, pooled_lag, pooled_hold, "pooled", len(parts), "pooled_across_seeds"))
    write_csv(output / "summary_beta_layers.csv", layer_rows)

    lag_rows = []
    for (seed, run_name), frame in lagged_frames.items():
        for (method, learning_rate, layer), group in frame.groupby(["method", "learning_rate", "layer"]):
            errors = group.lag_log10_abs_ratio_error.dropna()
            lag_rows.append(dict(
                aggregation="run", seed=seed, n_runs=1, method=method, learning_rate=learning_rate,
                layer=layer, sample_count=len(group), median_lag_log10_error=_median(errors),
                mean_lag_log10_error=_mean(errors), positive_lag_coverage=float(group.lag_prediction_positive.mean()),
                spearman_lag=_corr(group.beta_dp_previous_raw, group.beta_oracle_current, "spearman"),
            ))
    write_csv(output / "summary_beta_lagged.csv", lag_rows)

    window_rows = []
    for (seed, run_name), frame in window_frames.items():
        for (window, method, learning_rate, layer), group in frame.groupby(["window_size", "method", "learning_rate", "layer"]):
            errors = group.log_error.dropna()
            window_rows.append(dict(
                seed=seed, n_runs=1, window_size=int(window), method=method,
                learning_rate=learning_rate, layer=layer, sample_count=len(group),
                median_beta_oracle=_median(group.beta_oracle), median_beta_dp_raw=_median(group.beta_dp_raw),
                negative_rate=float(group.negative.mean()), positive_coverage=float((group.beta_dp_raw > 0).mean()),
                median_log_error=_median(errors), mean_log_error=_mean(errors),
            ))
    write_csv(output / "summary_window_sensitivity.csv", window_rows)

    by_name = {(seed, name): frame for (seed, name), frame in interval_frames.items()}
    trajectory_rows = []
    for seed in config["seeds"]:
        if (seed, "dp_sgd_lr0p10") in by_name and (seed, "dp_fisher_wiener_lr0p10") in by_name:
            trajectory_rows.extend(_trajectory_rows(
                {"dp_sgd_lr0p10": by_name[(seed, "dp_sgd_lr0p10")],
                 "dp_fisher_wiener_lr0p10": by_name[(seed, "dp_fisher_wiener_lr0p10")]},
                "dp_sgd_lr0p10", "dp_fisher_wiener_lr0p10", "dp_sgd_vs_fisher_lr0p10",
            ))
    if (42, "dp_fisher_wiener_lr0p10") in by_name and (42, "dp_fisher_wiener_lr0p80") in by_name:
        trajectory_rows.extend(_trajectory_rows(
            {"dp_fisher_wiener_lr0p10": by_name[(42, "dp_fisher_wiener_lr0p10")],
             "dp_fisher_wiener_lr0p80": by_name[(42, "dp_fisher_wiener_lr0p80")]},
            "dp_fisher_wiener_lr0p10", "dp_fisher_wiener_lr0p80", "fisher_lr0p10_vs_lr0p80_seed42",
        ))
    write_csv(output / "beta_trajectory_dependence.csv", trajectory_rows)
    trajectory_summary = []
    if trajectory_rows:
        trajectory = pd.DataFrame(trajectory_rows)
        for comparison, group in trajectory.groupby("comparison"):
            valid = group.dropna(subset=["trajectory_log_difference"])
            trajectory_summary.append(dict(
                comparison=comparison, n_samples=len(group), valid_samples=len(valid),
                median_absolute_log10_beta_difference=_median(valid.trajectory_log_difference),
                mean_absolute_log10_beta_difference=_mean(valid.trajectory_log_difference),
                ratio_median=_median(valid.ratio),
                spearman_oracle=_corr(valid.beta_oracle_left, valid.beta_oracle_right, "spearman"),
            ))
    write_csv(output / "summary_beta_trajectories.csv", trajectory_summary)

    analysis = dict(
        experiment="expv2", beta_train=1.0, primary_window=config["beta_primary_window"],
        window_sensitivity=list(config["beta_window_sensitivity"]), no_success_thresholds=True,
        beta_is_second_moment_scale=True, beta_not_paper_alpha=True,
        interpretation_cases={
            "A": "contemporaneous and lagged DP beta both reliable supports ExpV2b",
            "B": "contemporaneous good but lagged poor indicates dynamics faster than the lag",
            "C": "larger windows improving error indicates estimator sample-size noise",
            "D": "negative deep-layer estimates indicate DP noise-energy dominance",
            "E": "rapid oracle beta changes indicate an unstable online scalar model",
            "F": "beta far below one means calibrated Wiener would shrink more aggressively",
        },
        utility_is_regression_only=True,
        runs=[f"seed{seed}/{name}" for seed, name in interval_frames],
    )
    save_json(output / "summary.json", analysis)
    print(f"ExpV2 summaries generated: {output}")
    return analysis


if __name__ == "__main__":
    summarize(*cli())

