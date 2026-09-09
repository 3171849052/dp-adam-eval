"""Seven scientific figure families, PNG/PDF; no statistical error bars."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from expv1c.common import LAYERS, run_specs
from expv1c.validate_expv1c import cli, load, validate


def plot(c, runs, output, require_tests=True):
    validate(c, runs, output, require_tests=require_tests)
    destination = output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    specs = run_specs(c)
    summaries = {s["run_id"]: load(runs / "seed42" / s["run_id"] / "summary.json") for s in specs}
    frames = {s["run_id"]: pd.read_csv(runs / "seed42" / s["run_id"] / "train_metrics.csv") for s in specs}
    lr = pd.read_csv(output / "summary_lr.csv")
    layers = pd.read_csv(output / "summary_layers.csv")
    base = summaries["dp_sgd_lr0p10"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(destination / f"{name}.png", dpi=160)
        fig.savefig(destination / f"{name}.pdf")
        plt.close(fig)

    def curve(axis, frame, metric, **kwargs):
        values = pd.to_numeric(frame[metric], errors="coerce")
        mask = np.isfinite(values)
        axis.plot(frame.loc[mask, "learning_rate"], values[mask], "o-", **kwargs)
        axis.set(xlabel="Global SGD learning rate", ylabel=metric)
        axis.grid(alpha=.2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric in zip(axes, ("final_accuracy", "accuracy_auc")):
        curve(axis, lr[lr.status == "completed"], metric, label="ExpV1c Fisher-Wiener")
        if base[metric] is not None:
            axis.axhline(base[metric], color="black", linestyle="--", label="DP-SGD .10")
        axis.legend()
    save(fig, "upper_lr_sweep")

    history = pd.read_csv(output / "combined_lr_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric in zip(axes, ("final_accuracy", "accuracy_auc")):
        if history.empty:
            axis.text(.5, .5, "Historical comparison unavailable for smoke", ha="center", transform=axis.transAxes)
        else:
            for source in ("expv1b", "expv1c"):
                segment = history[(history.source == source) & (history.status == "completed")]
                curve(axis, segment, metric, label=source)
            axis.axhline(base[metric], color="black", linestyle="--", label="DP-SGD .10")
            axis.legend()
    save(fig, "combined_lr_history")

    fig, axis = plt.subplots(figsize=(10, 5))
    for spec in specs:
        frame = frames[spec["run_id"]].dropna(subset=["test_accuracy"])
        axis.plot(frame.step + 1, frame.test_accuracy, label=spec["run_id"])
    axis.set(xlabel="Completed private steps", ylabel="Test accuracy")
    axis.legend(fontsize=8)
    save(fig, "accuracy_vs_step")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric in zip(axes, ("train_loss", "test_loss")):
        for spec in specs:
            frame = frames[spec["run_id"]].dropna(subset=[metric])
            axis.plot(frame.step + 1, frame[metric], label=spec["run_id"], linewidth=1)
        axis.set(xlabel="Completed private steps", ylabel=metric)
        axis.legend(fontsize=7)
    save(fig, "loss_vs_step")

    fig, axis = plt.subplots(figsize=(8, 5))
    for threshold in (70, 80, 85, 90):
        curve(axis, lr[lr.status == "completed"], f"steps_to_acc_{threshold}", label=f"T{threshold}")
    axis.set_ylabel("Completed private steps to threshold")
    axis.legend()
    save(fig, "threshold_steps_vs_lr")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, name in zip(axes.flat, LAYERS):
        curve(axis, layers[layers.layer == name], "effective_signal_lr")
        axis.set_title(name)
        axis.set_ylabel("Late median eta * sqrt(R_signal)")
    save(fig, "late_effective_signal_lr_by_layer")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, metric in zip(axes.flat, ("late_signal_retention", "late_noise_retention", "late_snr_gain_db", "late_cosine_filtered")):
        curve(axis, lr, metric)
    save(fig, "late_mechanism_vs_lr")
    print(f"Figures generated: {destination}")


if __name__ == "__main__":
    plot(*cli())
