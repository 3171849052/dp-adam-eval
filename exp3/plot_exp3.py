"""Matplotlib seed traces, means, mechanism diagnostics, and resource costs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from exp3.common import ROOT, METHODS, read_config
from exp3.summarize_exp3 import load_runs, summarize, geometric_mean

COLORS = dict(zip(METHODS, ["#4477AA", "#228833", "#CC6677"]))
LABELS = dict(dp_sgd="DP-SGD", syn_diag="SynDiag", dp_kfc="DP-KFC Pink matched")


def plot(root, c, output):
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root, c)
    summary = summarize(root, c, output)
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=160)
        plt.close(fig)
    for metric, filename in [("test_accuracy", "01_test_accuracy"), ("G_diag_new", "02_G_diag_new"), ("G_full_new", "03_G_full_new"),
                             ("G_diag_old", "04_G_diag_old"), ("G_full_old", "05_G_full_old"),
                             ("delta_stale_diag", "11_staleness_diag"), ("delta_stale_full", "12_staleness_full")]:
        is_stale = metric.startswith("delta")
        fig, axes = plt.subplots(2, 2, figsize=(10, 7)) if is_stale else plt.subplots(figsize=(7, 4))
        axes = np.atleast_1d(axes).ravel()
        for ax, layer in zip(axes, ["conv1", "conv2", "fc1", "fc2"] if is_stale else [None]):
            for method in METHODS:
                lines = []
                for _, meta, _, frames in runs:
                    if meta["method"] != method:
                        continue
                    if metric == "test_accuracy":
                        series = frames["train"].set_index("step")[metric].dropna()
                    elif metric.startswith("G_"):
                        frame = frames["oracle"]
                        column = metric.replace("G_", "R_")
                        series = frame.dropna(subset=[column]).groupby("step")[column].agg(lambda v: geometric_mean(v))
                    else:
                        frame = frames["refresh"]
                        if frame.empty:
                            continue
                        series = frame[frame.layer == layer].set_index("step")[metric].dropna()
                    if metric != "test_accuracy":
                        series.index = series.index/meta["total_steps"]
                    ax.plot(series.index, series.values, color=COLORS[method], alpha=.22, linewidth=1)
                    lines.append(series)
                if lines:
                    mean = pd.concat(lines, axis=1).mean(axis=1)
                    ax.plot(mean.index, mean.values, color=COLORS[method], label=LABELS[method], linewidth=2, marker=".")
            if is_stale:
                ax.axhline(0, color="gray", linewidth=.7)
                ax.set_title(layer)
            ax.set(xlabel="Private step" if metric == "test_accuracy" else "Training progress", ylabel=metric)
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
        save(fig, filename)
    for metric, filename in [("late_norm_cv", "06_norm_cv"), ("late_coefficient_cv", "07_coefficient_cv"),
                             ("late_shape_error", "08_shape_error"), ("late_aggregate_cosine", "09_aggregate_cosine"),
                             ("late_snr", "10_snr"), ("mean_refresh_time", "13_refresh_time"),
                             ("core_wall_time", "14_core_wall_time"), ("preconditioner_state_bytes", "15_preconditioner_memory"),
                             ("peak_cuda_memory_core", "16_core_peak_memory")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, method in enumerate(METHODS):
            values = summary.loc[summary.method == method, metric].astype(float).dropna().to_numpy()
            ax.scatter(i+np.linspace(-.08, .08, len(values)), values, color=COLORS[method], alpha=.6)
            if len(values):
                ax.errorbar(i, values.mean(), yerr=values.std(ddof=1) if len(values)>1 else 0,
                            fmt="_", markersize=16, capsize=5, color=COLORS[method])
        ax.set_xticks(range(3), [LABELS[m] for m in METHODS])
        ax.set_ylabel(metric)
        ax.set_title("Seed values and mean ± sample SD")
        ax.grid(axis="y", alpha=.15)
        save(fig, filename)
    print(f"Wrote 16 figures to {figures}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/full.json"))
    p.add_argument("--runs", default=str(ROOT / "runs"))
    p.add_argument("--output", default=str(ROOT))
    a = p.parse_args()
    plot(a.runs, read_config(a.config), a.output)
