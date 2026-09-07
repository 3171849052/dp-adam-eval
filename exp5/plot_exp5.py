"""Plot Exp5 performance, alignment, clipping, and refresh diagnostics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exp5.analyze_exp5 import load_runs
from exp5.common import METHODS, ROOT, read_config

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
    ax.legend(fontsize=7)


def plot(c, runs, output):
    frames, _ = load_runs(c, runs)
    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    curve(axes[0], frames, "test_accuracy", "Test accuracy")
    curve(axes[1], frames, "train_loss", "Train loss")
    save(fig, directory, "figure1_performance")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    curve(axes[0], frames, "cos_method_adam_current_noise_off", "Method / Adam cosine, current noise off")
    curve(axes[1], frames, "cos_method_adam_noisy", "Method / Adam cosine, noisy")
    save(fig, directory, "figure2_adam_alignment")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    curve(axes[0], frames, "aggregate_cosine", "Aggregate clipping cosine")
    curve(axes[1], frames, "clipping_shape_error", "Clipping shape error")
    save(fig, directory, "figure3_clipping")
    fig, ax = plt.subplots(figsize=(9, 4))
    refresh = []
    for _, meta, _, train, _, _ in frames:
        refresh.append(dict(method=meta["method"], value=meta.get("total_refresh_time", 0)))
    # Read the stable summary field from each run's metadata instead of a
    # transient plot-time measurement.
    summaries = pd.DataFrame([dict(method=m["method"], value=float(s["total_refresh_time"]))
                              for _, m, s, _, _, _ in frames])
    for i, method in enumerate(METHODS):
        values = summaries[summaries.method == method].value.to_numpy()
        ax.scatter([i] * len(values), values, color=COLORS[method])
    ax.set_xticks(range(len(METHODS)), METHODS, rotation=25, ha="right")
    ax.set_ylabel("Total refresh time (s)")
    ax.grid(axis="y", alpha=.2)
    save(fig, directory, "figure4_refresh_cost")
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
