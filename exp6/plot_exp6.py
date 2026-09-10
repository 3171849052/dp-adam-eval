"""Plot the small set of Exp6 performance and state diagnostics."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from exp6.analyze_exp6 import load_runs
from exp6.common import METHODS, ROOT, read_config


def plot(config, runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = load_runs(config, runs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in METHODS:
        frames = [train for _, meta, _, train in records if meta["method"] == method]
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        for frame in frames:
            axes[0].plot(frame.step, frame.test_accuracy, alpha=.25, linewidth=.8)
        mean = combined.groupby("step").test_accuracy.mean()
        axes[0].plot(mean.index, mean, label=method, linewidth=1.5)
        for frame in frames:
            axes[1].plot(frame.step, frame.denominator_q50, alpha=.25, linewidth=.8)
        mean_d = combined.groupby("step").denominator_q50.mean()
        axes[1].plot(mean_d.index, mean_d, label=method, linewidth=1.5)
    axes[0].set(xlabel="Private step", ylabel="Test accuracy")
    axes[1].set(xlabel="Private step", ylabel="Median denominator")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "exp6.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output / 'exp6.png'}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    plot(read_config(args.config), args.runs, args.output or args.runs)


if __name__ == "__main__":
    main()
