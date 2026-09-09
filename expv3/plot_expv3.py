"""Generate the registered ExpV3 utility, beta, and mechanism figures."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from expv3.common import FISHER_METHOD, LAYERS, run_specs
from expv3.validate_expv3 import cli, validate


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

    train_frames = {}
    layer_frames = {}
    controller_frames = {}
    summaries = []
    for seed in config["seeds"]:
        for spec in specs:
            root = runs / f"seed{seed}" / spec["run_id"]
            train_frames[(seed, spec["run_id"])] = pd.read_csv(root / "train_metrics.csv")
            layer_frames[(seed, spec["run_id"])] = pd.read_csv(root / "layer_metrics.csv")
            controller_frames[(seed, spec["run_id"])] = pd.read_csv(root / "beta_controller_metrics.csv")
            summaries.append(pd.read_json(root / "summary.json", typ="series"))

    for metric, name, ylabel in (("test_accuracy", "accuracy_vs_step", "Test accuracy"),
                                 ("test_loss", "test_loss_vs_step", "Test CE loss")):
        plt.figure(figsize=(8, 5))
        for spec in specs:
            parts = [train_frames[(seed, spec["run_id"])] for seed in config["seeds"]]
            values = pd.concat(parts, ignore_index=True)
            values = values[values[metric].notna()]
            if len(values):
                plt.plot(values.step, values[metric], label=label(spec), alpha=.8)
        plt.xlabel("Private step"); plt.ylabel(ylabel); plt.legend(fontsize=7); save(name)

    summary_frame = pd.DataFrame(summaries)
    for metric, name, ylabel in (("final_accuracy", "final_accuracy_vs_lr", "Final test accuracy"),
                                 ("accuracy_auc", "accuracy_auc_vs_lr", "Accuracy AUC")):
        plt.figure(figsize=(7, 5))
        for spec in specs:
            if spec["method"] == FISHER_METHOD:
                part = summary_frame[(summary_frame.run_id == spec["run_id"])]
                plt.scatter(part.learning_rate, part[metric], label=f"lr={spec['learning_rate']:.2f}")
        plt.xlabel("Nominal learning rate"); plt.ylabel(ylabel); plt.legend(fontsize=7); save(name)

    # Beta controller figures: DP rows are intentionally absent from these plots.
    plt.figure(figsize=(8, 5))
    for spec in specs:
        if spec["method"] != FISHER_METHOD:
            continue
        frame = pd.concat([controller_frames[(seed, spec["run_id"])] for seed in config["seeds"]])
        for layer in LAYERS:
            part = frame[frame.layer == layer]
            plt.plot(part.interval_index, part.beta_train, marker=".", label=f"{spec['learning_rate']:.2f}/{layer}")
    plt.xlabel("Refresh interval"); plt.ylabel("beta_train"); plt.legend(fontsize=6, ncol=2); save("beta_train_vs_interval")

    plt.figure(figsize=(7, 5))
    for spec in specs:
        if spec["method"] != FISHER_METHOD:
            continue
        values = []
        for seed in config["seeds"]:
            frame = pd.read_csv(runs / f"seed{seed}" / spec["run_id"] / "beta_interval_metrics.csv")
            values.append(frame)
        frame = pd.concat(values)
        plt.scatter(frame.beta_oracle, frame.beta_dp_raw, s=10, label=f"lr={spec['learning_rate']:.2f}")
    plt.xlabel("Oracle beta"); plt.ylabel("DP raw beta"); plt.legend(fontsize=7); save("beta_raw_vs_oracle")

    plt.figure(figsize=(8, 5))
    for spec in specs:
        if spec["method"] != FISHER_METHOD:
            continue
        frame = pd.concat([controller_frames[(seed, spec["run_id"])] for seed in config["seeds"]])
        rates = frame.groupby("layer").beta_fallback_used.apply(lambda x: (x.astype(str).str.lower() == "true").mean())
        plt.plot(list(rates.index), rates.values, marker="o", label=f"lr={spec['learning_rate']:.2f}")
    plt.ylabel("Fallback rate"); plt.legend(fontsize=7); save("fallback_rate_by_layer_lr")

    plt.figure(figsize=(8, 5))
    for spec in specs:
        if spec["method"] != FISHER_METHOD:
            continue
        frame = pd.concat([pd.read_csv(runs / f"seed{seed}" / spec["run_id"] / "beta_interval_metrics.csv") for seed in config["seeds"]])
        positive = frame[(frame.beta_oracle > 0) & (frame.beta_dp_raw > 0)]
        if len(positive):
            plt.plot(positive.interval_index, positive.beta_dp_raw / positive.beta_oracle, ".", label=f"lr={spec['learning_rate']:.2f}")
    plt.axhline(1, color="black", lw=.7); plt.xlabel("Interval"); plt.ylabel("DP/oracle beta"); plt.legend(fontsize=7); save("beta_train_oracle_ratio_over_time")

    for metric, name, ylabel in (("H_q50", "H_q50_adaptive_vs_beta1", "Median H"),
                                 ("signal_retention", "signal_retention_by_lr_layer", "Signal retention"),
                                 ("noise_retention", "noise_retention_by_lr_layer", "Noise retention"),
                                 ("effective_signal_lr", "effective_signal_lr_by_layer", "Effective signal LR"),
                                 ("update_to_clean_reference_ratio", "update_to_clean_reference_ratio", "Update / clean reference")):
        plt.figure(figsize=(8, 5))
        for spec in specs:
            frame = pd.concat([layer_frames[(seed, spec["run_id"])] for seed in config["seeds"]])
            if metric not in frame:
                continue
            group = frame.groupby("layer")[metric].median()
            plt.plot(list(group.index), group.values, marker="o", label=label(spec))
        plt.xlabel("Layer"); plt.ylabel(ylabel); plt.legend(fontsize=6); save(name)

    divergent = summary_frame[summary_frame.status == "diverged"]
    if len(divergent):
        plt.figure(figsize=(7, 5))
        for _, row in divergent.iterrows():
            plt.scatter(row.learning_rate, row.diverged_step, label=f"seed{row.seed} lr={row.learning_rate}")
        plt.xlabel("Learning rate"); plt.ylabel("Diverged step"); plt.legend(fontsize=7); save("divergence_step_by_lr_seed")
    return sorted(path.name for path in figure_dir.glob("*.png"))


if __name__ == "__main__":
    plot(*cli())

