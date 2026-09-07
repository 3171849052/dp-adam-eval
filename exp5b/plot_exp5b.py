"""Plot Exp5b utility, momentum, beta2, factorial, clipping, and SNR diagnostics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exp5b.analyze_exp5b import load_runs
from exp5b.common import METHODS, ROOT, read_config

COLORS = dict(zip(METHODS, ["#4477AA", "#228833", "#CC6677", "#AA3377", "#EE7733", "#117733"]))


def save(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def curve(ax, frames, column, ylabel):
    for method in METHODS:
        values = []
        for _, meta, _, train, _, _ in frames:
            if meta["method"] != method or column not in train:
                continue
            series = train.set_index("step")[column]
            ax.plot(series.index, series, color=COLORS[method], alpha=.16, linewidth=.7)
            values.append(series.rename(method))
        if values:
            mean = pd.concat(values, axis=1).mean(axis=1)
            ax.plot(mean.index, mean, color=COLORS[method], linewidth=1.5, label=method)
    ax.set(xlabel="Private step", ylabel=ylabel)
    ax.grid(alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7)


def plot(c, runs, output):
    frames, _ = load_runs(c, runs)
    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    curve(axes[0], frames, "test_accuracy", "Test accuracy")
    curve(axes[1], frames, "train_loss", "Train loss")
    save(fig, directory, "fig1_accuracy_vs_step")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    curve(axes[0], frames, "method_update_norm", "Actual method update norm")
    curve(axes[1], frames, "cos_method_adam_noisy", "Method / Adam diagnostic cosine")
    save(fig, directory, "fig2_momentum_direction_diagnostics")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for _, meta, _, _, _, _ in frames:
        refresh_path = Path(runs) / f"seed{meta['seed']}" / meta["method"] / "refresh_metrics.csv"
        refresh = pd.read_csv(refresh_path)
        for column, axis, ylabel in (
            ("beta2_D_innovation", axes[0], "β2 innovation in log geometry"),
            ("beta2_D_ema", axes[1], "β2 EMA move in log geometry"),
        ):
            if column in refresh:
                values = refresh.groupby("step")[column].mean()
                axis.plot(values.index, values, color=COLORS[meta["method"]], alpha=.7, label=meta["method"])
    for axis, ylabel in zip(axes, ("β2 innovation in log geometry", "β2 EMA move in log geometry")):
        axis.set(xlabel="Refresh step", ylabel=ylabel)
        axis.grid(alpha=.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7)
    save(fig, directory, "fig3_beta2_dynamics")

    factorial = pd.read_csv(Path(runs) / "factorial_effects.csv")
    final = factorial[factorial.seed == "all"].set_index("metric")
    fig, ax = plt.subplots(figsize=(9, 4))
    metrics = [metric for metric in ("final_accuracy", "final_test_loss", "late_mean_accuracy") if metric in final.index]
    positions = range(len(metrics))
    width = .24
    for offset, (field, label, color) in enumerate((
        ("beta1_effect_mean", "β1", "#228833"),
        ("beta2_effect_mean", "β2", "#CC6677"),
        ("interaction_mean", "interaction", "#AA3377"),
    )):
        if field in final:
            ax.bar([p + (offset - 1) * width for p in positions], final.loc[metrics, field].astype(float),
                   width=width, label=label, color=color)
    ax.set_xticks(list(positions), metrics, rotation=20, ha="right")
    ax.set_ylabel("Paired effect (Full factorial scale)")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    save(fig, directory, "fig4_factorial_effects")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    curve(axes[0], frames, "clipping_shape_error", "Clipping shape error")
    curve(axes[1], frames, "diagnostic_snr", "Diagnostic SNR")
    save(fig, directory, "fig5_clipping_snr")
    print(f"Saved figures to {directory}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    c = read_config(args.config)
    plot(c, args.runs, args.output or args.runs)


if __name__ == "__main__":
    main()
