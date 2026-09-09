"""ExpV2 beta figures, each emitted as both PNG and PDF."""

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from expv2.common import LAYERS, run_specs_for_seed
from expv2.validate_expv2 import cli, load, validate


def plot(config, runs, output, require_tests=True):
    validate(config, runs, output, require_tests=require_tests)
    destination = output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for seed in config["seeds"]:
        for spec in run_specs_for_seed(config, seed):
            root = runs / f"seed{seed}" / spec["run_id"]
            entries.append((seed, spec, pd.read_csv(root / "beta_step_metrics.csv"),
                            pd.read_csv(root / "beta_interval_metrics.csv"),
                            pd.read_csv(root / "beta_lagged_metrics.csv")))

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(destination / f"{name}.png", dpi=160)
        fig.savefig(destination / f"{name}.pdf")
        plt.close(fig)

    def label(seed, spec):
        method = "DP-SGD" if spec["method"] == "dp_sgd" else "Fisher-Wiener"
        return f"{method} lr={spec['learning_rate']:.2f} seed={seed}"

    # 1. Oracle beta over primary refresh intervals.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, layer in zip(axes.flat, LAYERS):
        for seed, spec, _, frame, _ in entries:
            part = frame[frame.layer == layer]
            axis.plot(part.interval_index, part.beta_oracle, marker="o", linewidth=1, label=label(seed, spec))
        axis.set_title(layer)
        axis.set_ylabel("oracle beta")
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Primary interval")
    axes[1, 1].set_xlabel("Primary interval")
    axes[0, 0].legend(fontsize=6)
    save(fig, "01_oracle_beta_by_interval")

    # 2. Positive-only log-log comparison. Negative raw estimates are not
    # transformed; their prevalence is shown in figure 4.
    fig, axis = plt.subplots(figsize=(6, 5))
    values = []
    for seed, spec, _, frame, _ in entries:
        valid = frame[(frame.beta_dp_raw > 0) & (frame.beta_oracle > 0)]
        values.extend(valid.beta_oracle.tolist() + valid.beta_dp_raw.tolist())
        axis.scatter(valid.beta_oracle, valid.beta_dp_raw, alpha=0.65, label=label(seed, spec))
    if values:
        lo, hi = min(values), max(values)
        axis.plot([lo, hi], [lo, hi], "k--", label="y=x")
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set(xlabel="Oracle beta", ylabel="DP beta raw (positive only)", title="DP beta vs oracle beta")
    axis.legend(fontsize=7)
    save(fig, "02_dp_vs_oracle_loglog")

    # 3. Positive-valid contemporaneous log error.
    fig, axis = plt.subplots(figsize=(9, 5))
    for seed, spec, _, frame, _ in entries:
        valid = frame[(frame.beta_dp_raw > 0) & (frame.beta_oracle > 0)].copy()
        valid["error"] = (valid.beta_dp_raw / valid.beta_oracle).map(lambda x: abs(math.log10(x)))
        axis.plot(valid.interval_index, valid.error, "o-", label=label(seed, spec))
    axis.set(xlabel="Primary interval", ylabel="|log10(DP beta / oracle beta)|", title="Contemporaneous beta error")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)
    save(fig, "03_contemporaneous_log_error")

    # 4. Negative interval rate by layer.
    rows = []
    for seed, spec, _, frame, _ in entries:
        for layer, group in frame.groupby("layer"):
            rows.append(dict(label=label(seed, spec), layer=layer, rate=float((group.beta_dp_raw < 0).mean())))
    negative = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(10, 5))
    if not negative.empty:
        negative.pivot(index="layer", columns="label", values="rate").plot.bar(ax=axis)
    axis.set(xlabel="Layer", ylabel="Negative interval rate", title="Raw DP beta negative rate")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.2)
    save(fig, "04_negative_interval_rate")

    # 5. Contemporaneous versus lagged errors.
    rows = []
    for seed, spec, _, frame, lag in entries:
        contemporary = frame[(frame.beta_dp_raw > 0) & (frame.beta_oracle > 0)].copy()
        contemporary["error"] = (contemporary.beta_dp_raw / contemporary.beta_oracle).map(lambda x: abs(math.log10(x)))
        lag_error = lag.lag_log10_abs_ratio_error.dropna()
        rows.append(dict(label=label(seed, spec), contemporaneous=contemporary.error.median(), lagged=lag_error.median()))
    error_frame = pd.DataFrame(rows).dropna(how="all", subset=["contemporaneous", "lagged"])
    fig, axis = plt.subplots(figsize=(10, 5))
    if not error_frame.empty:
        error_frame.set_index("label")[["contemporaneous", "lagged"]].plot.bar(ax=axis)
    axis.set(ylabel="Median absolute log10 ratio error", title="Contemporaneous vs lagged beta error")
    axis.grid(axis="y", alpha=0.2)
    save(fig, "05_contemporaneous_vs_lagged")

    # 6. Window-size sensitivity computed offline from per-step files.
    rows = []
    for seed, spec, _, _, _ in entries:
        frame = pd.read_csv(runs / f"seed{seed}" / spec["run_id"] / "beta_window_sensitivity.csv")
        for window, group in frame.groupby("window_size"):
            valid = group.log_error.dropna()
            rows.append(dict(label=label(seed, spec), window_size=window,
                             median_log_error=valid.median() if len(valid) else None))
    sensitivity = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(9, 5))
    for name, group in sensitivity.groupby("label"):
        axis.plot(group.window_size, group.median_log_error, "o-", label=name)
    axis.set(xlabel="Non-overlapping window size", ylabel="Median positive-valid log error", title="Window-size sensitivity")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)
    save(fig, "06_window_size_sensitivity")

    # 7. Oracle trajectory dependence, restricted to the requested comparisons.
    fig, axis = plt.subplots(figsize=(10, 5))
    for seed, spec, _, frame, _ in entries:
        if spec["method"] == "dp_sgd" or spec["learning_rate"] in (0.1, 0.8):
            for layer in LAYERS:
                part = frame[frame.layer == layer]
                axis.plot(part.interval_index, part.beta_oracle, marker=".", linewidth=1,
                           label=f"{label(seed, spec)} / {layer}")
    axis.set(xlabel="Primary interval", ylabel="Oracle beta", title="Trajectory dependence")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=6, ncol=2)
    save(fig, "07_trajectory_dependence")
    print(f"ExpV2 figures generated: {destination}")


if __name__ == "__main__":
    plot(*cli())
