"""ExpV1b LR-sweep figures, each emitted as PNG and PDF."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from expv1b.common import FISHER_LRS, LAYERS, run_specs
from expv1b.validate_expv1b import cli, load, validate


COLORS = {
    "dp_sgd_lr0p10": "black",
    "dp_fisher_wiener_lr0p10": "C0",
    "dp_fisher_wiener_lr0p15": "C1",
    "dp_fisher_wiener_lr0p20": "C2",
    "dp_fisher_wiener_lr0p30": "C3",
    "dp_fisher_wiener_lr0p50": "C4",
    "dp_fisher_wiener_lr0p80": "C5",
}


def plot(c, runs, output, require_tests=True):
    validate(c, runs, output, require_tests=require_tests)
    destination = output / "figures"
    destination.mkdir(parents=True, exist_ok=True)

    specs = run_specs(c)
    summaries = {spec["run_id"]: load(runs / "seed42" / spec["run_id"] / "summary.json") for spec in specs}
    train_frames = {
        spec["run_id"]: pd.read_csv(runs / "seed42" / spec["run_id"] / "train_metrics.csv")
        for spec in specs
    }
    layer_frames = {
        spec["run_id"]: pd.read_csv(runs / "seed42" / spec["run_id"] / "layer_metrics.csv")
        for spec in specs
    }

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(destination / f"{name}.png", dpi=160)
        fig.savefig(destination / f"{name}.pdf")
        plt.close(fig)

    def label(spec):
        return "DP-SGD lr=.10" if spec["method"] == "dp_sgd" else f"Fisher-Wiener lr={spec['learning_rate']:.2f}"

    # Figure 1: the primary LR curve with the fixed DP-SGD reference.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fisher_specs = [spec for spec in specs if spec["method"] == "dp_fisher_wiener"]
    base = summaries["dp_sgd_lr0p10"]
    for axis, metric, title in zip(axes, ("final_accuracy", "accuracy_auc"), ("Final test accuracy", "Accuracy AUC")):
        xs = [spec["learning_rate"] for spec in fisher_specs if summaries[spec["run_id"]].get(metric) is not None]
        ys = [summaries[spec["run_id"]][metric] for spec in fisher_specs if summaries[spec["run_id"]].get(metric) is not None]
        axis.plot(xs, ys, "o-", label="Fisher-Wiener", color="C0")
        axis.axhline(base[metric], color="black", linestyle="--", label="DP-SGD lr=.10")
        axis.set(xlabel="Global learning rate", ylabel=metric, title=title)
        axis.legend()
    save(fig, "lr_sweep_utility")

    # Figures 2 and 3: all run trajectories; a divergent run naturally stops
    # at its last finite row.
    for metric, title, name in (("test_accuracy", "Test accuracy vs private step", "accuracy_vs_step"),
                                ("train_loss", "Train loss vs private step", "train_loss_vs_step")):
        fig, axis = plt.subplots(figsize=(10, 5))
        for spec in specs:
            frame = train_frames[spec["run_id"]].dropna(subset=[metric])
            axis.plot(frame.step, frame[metric], color=COLORS[spec["run_id"]], label=label(spec), linewidth=1.5)
        axis.set(xlabel="Completed private step (CSV step is zero-based)", ylabel=metric, title=title)
        axis.legend(fontsize=8, ncol=2)
        save(fig, name)

    # Figure 4: mechanism diagnostic, one panel per layer.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    selected = [0.10, 0.20, 0.30, 0.50, 0.80]
    for axis, layer_name in zip(axes.flat, LAYERS):
        for spec in fisher_specs:
            if spec["learning_rate"] not in selected:
                continue
            frame = layer_frames[spec["run_id"]]
            frame = frame[frame.layer == layer_name]
            axis.plot(frame.step, frame.effective_signal_lr, color=COLORS[spec["run_id"]], label=f"lr={spec['learning_rate']:.2f}")
        axis.set_title(layer_name)
        axis.set_ylabel("effective signal LR")
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Private step")
    axes[1, 1].set_xlabel("Private step")
    axes[0, 0].legend(fontsize=8)
    save(fig, "effective_signal_lr_vs_step")

    # Figure 5: threshold speed. Unreached thresholds remain absent/NA.
    fig, axis = plt.subplots(figsize=(8, 5))
    for key, marker in zip(("steps_to_acc_50", "steps_to_acc_70", "steps_to_acc_80", "steps_to_acc_85"), ("o", "s", "^", "D")):
        xs = [spec["learning_rate"] for spec in fisher_specs if summaries[spec["run_id"]].get(key) is not None]
        ys = [summaries[spec["run_id"]][key] for spec in fisher_specs if summaries[spec["run_id"]].get(key) is not None]
        axis.plot(xs, ys, marker=marker, linestyle="-", label=key)
    axis.set(xlabel="Global learning rate", ylabel="Completed private steps", title="Accuracy threshold speed")
    axis.legend()
    save(fig, "threshold_steps_vs_lr")

    # Figure 6: late-training mechanism quantities.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = (("late_cosine_filtered", "Late filtered cosine"),
               ("late_snr_gain_db", "Late SNR gain (dB)"),
               ("late_signal_retention", "Late signal retention"),
               ("late_noise_retention", "Late noise retention"))
    summary_lr = pd.read_csv(output / "summary_lr.csv")
    for axis, (metric, title) in zip(axes.flat, metrics):
        frame = summary_lr.dropna(subset=[metric])
        axis.plot(frame.learning_rate, frame[metric], "o-", color="C0")
        axis.set(xlabel="Global learning rate", ylabel=metric, title=title)
        axis.grid(alpha=0.2)
    save(fig, "late_mechanism_vs_lr")
    print(f"Figures generated: {destination}")


if __name__ == "__main__":
    plot(*cli())

