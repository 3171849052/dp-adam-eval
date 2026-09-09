"""Generate the registered ExpV3 utility, beta, and mechanism figures."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from expv3.common import FISHER_METHOD, LAYERS, run_specs
from expv3.validate_expv3 import cli, validate


def aggregate_step_metric(frames, metric):
    """Aggregate finite values by step without joining seed trajectories."""
    parts = []
    for frame in frames:
        if "step" not in frame or metric not in frame:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce")
        numeric = values.to_numpy(dtype=float)
        mask = values.notna() & np.isfinite(numeric)
        if mask.any():
            parts.append(pd.DataFrame({
                "step": pd.to_numeric(frame.loc[mask, "step"], errors="raise").astype(int),
                metric: values.loc[mask].astype(float),
            }))
    if not parts:
        return pd.DataFrame(columns=["step", "mean", "std", "count"])
    combined = pd.concat(parts, ignore_index=True)
    result = combined.groupby("step", as_index=False)[metric].agg(
        mean="mean", std="std", count="count"
    )
    return result.sort_values("step").reset_index(drop=True)


def aggregate_beta_train(frames):
    """Return median beta_train at each interval across the supplied seeds."""
    parts = []
    for frame in frames:
        if "interval_index" not in frame or "beta_train" not in frame:
            continue
        values = pd.to_numeric(frame["beta_train"], errors="coerce")
        numeric = values.to_numpy(dtype=float)
        mask = values.notna() & np.isfinite(numeric)
        if mask.any():
            parts.append(pd.DataFrame({
                "interval_index": pd.to_numeric(
                    frame.loc[mask, "interval_index"], errors="raise"
                ).astype(int),
                "beta_train": values.loc[mask].astype(float),
            }))
    if not parts:
        return pd.DataFrame(columns=["interval_index", "median", "count"])
    combined = pd.concat(parts, ignore_index=True)
    result = combined.groupby("interval_index", as_index=False).beta_train.agg(
        median="median", count="count"
    )
    return result.sort_values("interval_index").reset_index(drop=True)


def beta_train_oracle_ratio(controller, intervals):
    """Join actual beta_train to the oracle from the same interval."""
    keys = ["seed", "method", "learning_rate", "layer", "interval_index"]
    columns = keys + ["beta_train"]
    oracle_columns = keys + ["beta_oracle"]
    if not set(columns) <= set(controller.columns) or not set(oracle_columns) <= set(intervals.columns):
        return pd.DataFrame(columns=keys + ["ratio"])
    merged = controller[columns].merge(
        intervals[oracle_columns], on=keys, how="inner", validate="one_to_one"
    )
    beta_train = pd.to_numeric(merged.beta_train, errors="coerce")
    beta_oracle = pd.to_numeric(merged.beta_oracle, errors="coerce")
    train_numeric = beta_train.to_numpy(dtype=float)
    oracle_numeric = beta_oracle.to_numpy(dtype=float)
    mask = (
        beta_train.notna() & beta_oracle.notna()
        & np.isfinite(train_numeric) & np.isfinite(oracle_numeric)
        & beta_train.gt(0) & beta_oracle.gt(0)
    )
    result = merged.loc[mask, keys].copy()
    result["ratio"] = (beta_train.loc[mask] / beta_oracle.loc[mask]).astype(float)
    return result.reset_index(drop=True)


def aggregate_h_q50(controller):
    """Aggregate actual and beta=1 counterfactual H statistics by layer."""
    required = {"layer", "H_q50", "H_beta1_q50"}
    if not required <= set(controller.columns):
        return pd.DataFrame(columns=["layer", "adaptive", "counterfactual"])
    frame = controller[["layer", "H_q50", "H_beta1_q50"]].copy()
    for column in ("H_q50", "H_beta1_q50"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        numeric = frame[column].to_numpy(dtype=float)
        frame.loc[~np.isfinite(numeric), column] = np.nan
    result = frame.groupby("layer", as_index=False).agg(
        adaptive=("H_q50", "median"), counterfactual=("H_beta1_q50", "median")
    )
    return result.sort_values("layer").reset_index(drop=True)


def _aggregate_interval_ratio(frames, numerator, denominator):
    parts = []
    for frame in frames:
        if numerator not in frame or denominator not in frame:
            continue
        left = pd.to_numeric(frame[numerator], errors="coerce")
        right = pd.to_numeric(frame[denominator], errors="coerce")
        left_numeric, right_numeric = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
        mask = (
            left.notna() & right.notna() & np.isfinite(left_numeric) & np.isfinite(right_numeric)
            & left.gt(0) & right.gt(0)
        )
        if mask.any():
            parts.append(pd.DataFrame({
                "interval_index": frame.loc[mask, "interval_index"].astype(int),
                "ratio": (left.loc[mask] / right.loc[mask]).astype(float),
            }))
    if not parts:
        return pd.DataFrame(columns=["interval_index", "median", "count"])
    combined = pd.concat(parts, ignore_index=True)
    result = combined.groupby("interval_index", as_index=False).ratio.agg(
        median="median", count="count"
    )
    return result.sort_values("interval_index").reset_index(drop=True)


def _plot_aggregate(result, x, y, label, fill=True):
    if result.empty:
        return
    plt.plot(result[x], result[y], marker=".", label=label)
    if fill and {"std", "count"} <= set(result.columns):
        numeric = result["std"].to_numpy(dtype=float)
        mask = (result["count"] >= 2) & np.isfinite(numeric)
        if mask.any():
            mean, std = result.loc[mask, y], result.loc[mask, "std"]
            plt.fill_between(result.loc[mask, x], mean - std, mean + std, alpha=.15)


def plot(config, runs, output, require_tests=True):
    validate(config, runs, output, require_tests=require_tests)
    figure_dir = Path(output) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    specs = run_specs(config)

    def save(name):
        plt.tight_layout()
        plt.savefig(figure_dir / f"{name}.png", dpi=140)
        plt.savefig(figure_dir / f"{name}.pdf")
        plt.close()

    def label(spec):
        return f"{spec['method']} lr={spec['learning_rate']:.2f}"

    train_frames, layer_frames, controller_frames, interval_frames = {}, {}, {}, {}
    summaries = []
    for seed in config["seeds"]:
        for spec in specs:
            root = runs / f"seed{seed}" / spec["run_id"]
            key = (seed, spec["run_id"])
            train_frames[key] = pd.read_csv(root / "train_metrics.csv")
            layer_frames[key] = pd.read_csv(root / "layer_metrics.csv")
            controller_frames[key] = pd.read_csv(root / "beta_controller_metrics.csv")
            interval_frames[key] = pd.read_csv(root / "beta_interval_metrics.csv")
            summaries.append(pd.read_json(root / "summary.json", typ="series"))

    for metric, name, ylabel in (("test_accuracy", "accuracy_vs_step", "Test accuracy"),
                                 ("test_loss", "test_loss_vs_step", "Test CE loss")):
        plt.figure(figsize=(8, 5))
        for spec in specs:
            parts = [train_frames[(seed, spec["run_id"])] for seed in config["seeds"]]
            _plot_aggregate(aggregate_step_metric(parts, metric), "step", "mean", label(spec))
        plt.xlabel("Private step"); plt.ylabel(ylabel); plt.legend(fontsize=7); save(name)

    summary_frame = pd.DataFrame(summaries)
    for metric, name, ylabel in (("final_accuracy", "final_accuracy_vs_lr", "Final test accuracy"),
                                 ("accuracy_auc", "accuracy_auc_vs_lr", "Accuracy AUC")):
        plt.figure(figsize=(7, 5))
        for spec in specs:
            if spec["method"] == FISHER_METHOD:
                part = summary_frame[summary_frame.run_id == spec["run_id"]]
                plt.scatter(part.learning_rate, part[metric], label=f"lr={spec['learning_rate']:.2f}")
        plt.xlabel("Nominal learning rate"); plt.ylabel(ylabel); plt.legend(fontsize=7); save(name)

    plt.figure(figsize=(8, 5))
    for spec in specs:
        if spec["method"] != FISHER_METHOD:
            continue
        frames = [controller_frames[(seed, spec["run_id"])] for seed in config["seeds"]]
        for layer in LAYERS:
            result = aggregate_beta_train([frame[frame.layer == layer] for frame in frames])
            _plot_aggregate(result.rename(columns={"median": "mean"}), "interval_index", "mean",
                            f"{spec['learning_rate']:.2f}/{layer}", fill=False)
    plt.xlabel("Refresh interval"); plt.ylabel("median beta_train"); plt.legend(fontsize=6, ncol=2)
    save("beta_train_vs_interval")

    plt.figure(figsize=(7, 5))
    for spec in specs:
        if spec["method"] != FISHER_METHOD:
            continue
        frames = [interval_frames[(seed, spec["run_id"])] for seed in config["seeds"]]
        for layer in LAYERS:
            part = pd.concat([frame[frame.layer == layer] for frame in frames], ignore_index=True)
            positive = part[(part.beta_oracle > 0) & (part.beta_dp_raw > 0)]
            if len(positive):
                plt.scatter(positive.beta_oracle, positive.beta_dp_raw, s=10,
                            label=f"lr={spec['learning_rate']:.2f}/{layer}")
    plt.xlabel("Oracle beta"); plt.ylabel("DP raw beta"); plt.legend(fontsize=6); save("beta_raw_vs_oracle")

    plt.figure(figsize=(8, 5))
    for spec in specs:
        if spec["method"] != FISHER_METHOD:
            continue
        values = pd.concat([controller_frames[(seed, spec["run_id"])] for seed in config["seeds"]],
                           ignore_index=True)
        values = values[values.interval_index > 0]
        rates = values.groupby("layer").beta_fallback_used.apply(
            lambda x: (x.astype(str).str.lower() == "true").mean()
        )
        plt.plot(list(rates.index), rates.values, marker="o", label=f"lr={spec['learning_rate']:.2f}")
    plt.ylabel("Fallback rate"); plt.legend(fontsize=7); save("fallback_rate_by_layer_lr")

    # Contemporaneous estimator ratio; distinct from active beta/oracle below.
    plt.figure(figsize=(8, 5))
    for spec in specs:
        if spec["method"] != FISHER_METHOD:
            continue
        frames = [interval_frames[(seed, spec["run_id"])] for seed in config["seeds"]]
        result = _aggregate_interval_ratio(frames, "beta_dp_raw", "beta_oracle")
        _plot_aggregate(result, "interval_index", "median", f"lr={spec['learning_rate']:.2f}", fill=False)
    plt.axhline(1, color="black", lw=.7); plt.xlabel("Interval")
    plt.ylabel("DP raw beta / oracle beta"); plt.legend(fontsize=7)
    save("beta_raw_oracle_ratio_over_time")

    # beta_train_m is joined to beta_oracle_m, never to oracle_(m-1).
    plt.figure(figsize=(8, 5))
    for spec in specs:
        if spec["method"] != FISHER_METHOD:
            continue
        controllers = pd.concat([controller_frames[(seed, spec["run_id"])] for seed in config["seeds"]],
                                ignore_index=True)
        intervals = pd.concat([interval_frames[(seed, spec["run_id"])] for seed in config["seeds"]],
                              ignore_index=True)
        ratios = beta_train_oracle_ratio(controllers, intervals)
        for layer in LAYERS:
            part = ratios[ratios.layer == layer]
            if part.empty:
                continue
            result = part.groupby("interval_index", as_index=False).ratio.agg(
                median="median", count="count"
            )
            _plot_aggregate(result, "interval_index", "median",
                            f"lr={spec['learning_rate']:.2f}/{layer}", fill=False)
    plt.axhline(1, color="black", lw=.7); plt.xlabel("Interval")
    plt.ylabel("beta_train / oracle beta"); plt.legend(fontsize=6)
    save("beta_train_oracle_ratio_over_time")

    # Read both actual and beta=1 counterfactual fields from the controller artifact.
    plt.figure(figsize=(8, 5))
    for spec in specs:
        if spec["method"] != FISHER_METHOD:
            continue
        frames = [controller_frames[(seed, spec["run_id"])] for seed in config["seeds"]]
        result = aggregate_h_q50(pd.concat(frames, ignore_index=True))
        if not result.empty:
            plt.plot(result.layer, result.adaptive, marker="o",
                     label=f"lr={spec['learning_rate']:.2f} adaptive")
            plt.plot(result.layer, result.counterfactual, marker="o", linestyle="--",
                     label=f"lr={spec['learning_rate']:.2f} beta1")
    plt.xlabel("Layer"); plt.ylabel("H q50"); plt.legend(fontsize=6, ncol=2)
    save("H_q50_adaptive_vs_beta1")

    for metric, name, ylabel in (("signal_retention", "signal_retention_by_lr_layer", "Signal retention"),
                                 ("noise_retention", "noise_retention_by_lr_layer", "Noise retention"),
                                 ("effective_signal_lr", "effective_signal_lr_by_layer", "Effective signal LR"),
                                 ("update_to_clean_reference_ratio", "update_to_clean_reference_ratio", "Update / clean reference")):
        plt.figure(figsize=(8, 5))
        for spec in specs:
            frame = pd.concat([layer_frames[(seed, spec["run_id"])] for seed in config["seeds"]],
                              ignore_index=True)
            if metric not in frame:
                continue
            values = pd.to_numeric(frame[metric], errors="coerce")
            numeric = values.to_numpy(dtype=float)
            frame = frame.loc[values.notna() & np.isfinite(numeric)].copy()
            frame[metric] = values.loc[frame.index]
            group = frame.groupby("layer")[metric].median()
            plt.plot(list(group.index), group.values, marker="o", label=label(spec))
        plt.xlabel("Layer"); plt.ylabel(ylabel); plt.legend(fontsize=6); save(name)

    divergent = summary_frame[summary_frame.status == "diverged"]
    if len(divergent):
        plt.figure(figsize=(7, 5))
        for _, row in divergent.iterrows():
            plt.scatter(row.learning_rate, row.diverged_step, label=f"seed{row.seed} lr={row.learning_rate}")
        plt.xlabel("Learning rate"); plt.ylabel("Diverged step"); plt.legend(fontsize=7)
        save("divergence_step_by_lr_seed")
    return sorted(path.name for path in figure_dir.glob("*.png"))


if __name__ == "__main__":
    plot(*cli())
