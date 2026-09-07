"""Create matplotlib-only Exp3b mechanism figures from offline CSV outputs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp3b.common import DEFAULT_RESULTS, METHODS, LAYERS


COLORS = {"dp_sgd": "#4477AA", "syn_diag": "#228833", "dp_kfc": "#CC6677"}
LABELS = {"dp_sgd": "DP-SGD", "syn_diag": "SynDiag", "dp_kfc": "DP-KFC"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _trajectory_plot(frame, metric, path, title, ylabel=None, highlight=False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method in METHODS:
        group = frame[frame.method == method]
        for seed in sorted(group.seed.unique()):
            trace = group[group.seed == seed]
            ax.plot(trace.step, trace[metric], color=COLORS[method], alpha=.18, linewidth=.8)
        mean = group.groupby("step", as_index=False)[metric].mean()
        ax.plot(mean.step, mean[metric], color=COLORS[method], label=LABELS[method], linewidth=2)
    if highlight:
        ax.axvspan(0, 200, color="gray", alpha=.12, label="early window")
    ax.set(title=title, xlabel="Private step", ylabel=ylabel or metric)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    _save(fig, path)


def _contrib_plot(frame, method, path):
    group = frame[frame.method == method]
    mean = group.groupby("step")[ [f"contrib_{layer}" for layer in LAYERS] ].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stackplot(mean.index, *(mean[f"contrib_{layer}"] for layer in LAYERS),
                 labels=LAYERS, colors=["#88CCEE", "#44AA99", "#DDCC77", "#CC6677"], alpha=.85)
    for seed in sorted(group.seed.unique()):
        trace = group[group.seed == seed]
        ax.plot(trace.step, trace.contrib_fc1 + trace.contrib_fc2, color="black", alpha=.2, linewidth=.5)
    ax.set(title=f"{LABELS[method]} pre-clipping energy allocation", xlabel="Private step", ylabel="Energy fraction")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=.15)
    _save(fig, path)


def _coefficient_alpha_plot(frame, path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for method in METHODS:
        group = frame[frame.method == method]
        for seed in sorted(group.seed.unique()):
            trace = group[group.seed == seed]
            axes[0].plot(trace.step, trace.coefficient_mean, color=COLORS[method], alpha=.15, linewidth=.7)
            axes[0].plot(trace.step, trace.clipping_alpha_star, color=COLORS[method], alpha=.15, linewidth=.7, linestyle="--")
        mean = group.groupby("step")[["coefficient_mean", "clipping_alpha_star"]].mean()
        axes[0].plot(mean.index, mean.coefficient_mean, color=COLORS[method], label=f"{LABELS[method]} coefficient mean")
        axes[0].plot(mean.index, mean.clipping_alpha_star, color=COLORS[method], linestyle="--", label=f"{LABELS[method]} alpha*", alpha=.8)
    axes[0].set_ylabel("value")
    axes[0].set_title("Scalar coefficient mean versus clipping alpha*")
    axes[0].legend(fontsize=7, ncol=2)
    for method in METHODS:
        group = frame[frame.method == method]
        mean = group.groupby("step").alpha_over_mean_coeff.mean()
        axes[1].plot(mean.index, mean, color=COLORS[method], label=LABELS[method], linewidth=2)
    axes[1].axhline(1, color="gray", linewidth=.8)
    axes[1].set(xlabel="Private step", ylabel="alpha*/coefficient mean", title="Scalar attenuation diagnostic")
    axes[1].grid(alpha=.2)
    _save(fig, path)


def _heatmap(results, path):
    main = pd.read_csv(Path(results) / "main_mechanism_table.csv")
    metrics = ["early_G_full", "late_G_full", "early_G_diag_fc", "late_G_diag_fc",
               "late_fc_share", "late_layer_entropy", "late_norm_cv", "late_coefficient_cv",
               "late_alpha_over_mean_coeff", "late_aggregate_cosine", "late_shape_error", "late_snr"]
    values = np.array([[row[f"{metric}_mean"] for metric in metrics] for _, row in main.iterrows()])
    scale = values.std(axis=0, ddof=1)
    scale[scale == 0] = 1
    standardized = (values - values.mean(axis=0)) / scale
    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(standardized, cmap="coolwarm", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(metrics)), metrics, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(main)), [LABELS[x] for x in main.method])
    ax.set_title("Standardized descriptive mechanism metrics (no composite ranking)")
    for i in range(len(main)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{values[i, j]:.3g}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="standardized across methods")
    _save(fig, path)


def _final_accuracy(frame, path):
    final = frame.sort_values("step").groupby(["method", "seed"], as_index=False).tail(1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(METHODS))
    for i, method in enumerate(METHODS):
        values = final[final.method == method].test_accuracy.to_numpy(dtype=float)
        ax.scatter(np.full(len(values), i), values, color=COLORS[method], alpha=.7)
        ax.plot([i - .12, i + .12], [values.mean(), values.mean()], color=COLORS[method], linewidth=3)
    ax.set_xticks(x, [LABELS[m] for m in METHODS])
    ax.set_ylabel("final accuracy")
    ax.set_title("Final accuracy, secondary endpoint")
    ax.grid(axis="y", alpha=.2)
    _save(fig, path)


def _deep_geometry_vs_energy(trajectory, geometry, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, method in zip(axes, ("syn_diag", "dp_kfc")):
        g = geometry[geometry.method == method]
        t = trajectory[trajectory.method == method]
        ratio = g.groupby("step").diag_fc_over_conv.mean()
        fc = t.groupby("step").fc_share.mean()
        left = ax
        right = ax.twinx()
        left.plot(ratio.index, ratio, color="#5555AA", linewidth=2, label="diag FC/conv")
        right.plot(fc.index, fc, color=COLORS[method], linewidth=2, label="fc_share")
        left.axhline(1, color="gray", linewidth=.8)
        left.set_title(LABELS[method])
        left.set_xlabel("Private step")
        left.set_ylabel("G_diag_fc / G_diag_conv")
        right.set_ylabel("fc_share", color=COLORS[method])
        left.grid(alpha=.2)
        left.axvspan(0, 200, color="gray", alpha=.1)
    fig.suptitle("Deep geometry versus pre-clipping energy")
    _save(fig, path)


def plot(results=DEFAULT_RESULTS):
    results = Path(results).resolve()
    figures = results / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    trajectory = pd.read_csv(results / "trajectory_metrics.csv")
    geometry = pd.read_csv(results / "geometry_summary.csv")
    for method in METHODS:
        _contrib_plot(trajectory, method, figures / f"01_contrib_{method}.png")
    for i, (metric, title) in enumerate([
        ("fc_share", "FC pre-clipping energy share"),
        ("fc_to_conv", "FC-to-convolution energy ratio"),
        ("layer_entropy", "Normalized layer energy entropy"),
    ], start=2):
        _trajectory_plot(trajectory, metric, figures / f"{i:02d}_{metric}.png", title)
    for i, metric in enumerate(("G_diag_conv", "G_diag_fc", "diag_fc_over_conv"), start=5):
        _trajectory_plot(geometry, metric, figures / f"{i:02d}_{metric}.png", metric, highlight=False)
    _trajectory_plot(geometry, "G_full_new", figures / "08_G_full_new.png", "Full geometry ratio", highlight=True)
    _trajectory_plot(geometry, "G_diag_new", figures / "09_G_diag_new.png", "Diagonal geometry ratio")
    _coefficient_alpha_plot(trajectory, figures / "10_coefficient_mean_vs_alpha_star.png")
    for i, metric in enumerate(("alpha_over_mean_coeff", "aggregate_cosine", "clipping_shape_error",
                                 "clipped_aggregate_norm", "diagnostic_snr"), start=11):
        _trajectory_plot(trajectory, metric, figures / f"{i:02d}_{metric}.png", metric)
    _heatmap(results, figures / "16_mechanism_summary_heatmap.png")
    _final_accuracy(trajectory, figures / "17_final_accuracy.png")
    _deep_geometry_vs_energy(trajectory, geometry, figures / "deep_geometry_vs_energy.png")
    print(f"Exp3b figures written to {figures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()
    plot(args.results)


if __name__ == "__main__":
    main()
