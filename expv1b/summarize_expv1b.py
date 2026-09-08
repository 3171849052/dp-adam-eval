"""Single-seed descriptive summaries for the ExpV1b LR sweep."""

import pandas as pd

from expv1b.common import FISHER_LRS, LAYERS, save_json, write_csv
from expv1b.validate_expv1b import cli, load, validate


DIAGNOSTIC_METRICS = (
    "relmse_noisy", "relmse_filtered", "cosine_noisy", "cosine_filtered",
    "signal_retention", "noise_retention", "snr_gain_db", "mse_reduction",
    "signal_amplitude_retention", "effective_signal_lr",
)


def _late_medians(frame, total):
    late = frame[frame.step + 1 > total / 2]
    return {
        f"late_{metric}": (float(late[metric].median()) if not late.empty and late[metric].notna().any() else None)
        for metric in DIAGNOSTIC_METRICS
    }


def _ranking(rows, metric, output):
    ranked = [
        {"learning_rate": row["learning_rate"], "value": row[metric]}
        for row in rows
        if row.get("status") == "completed" and row.get(metric) is not None
    ]
    ranked.sort(key=lambda row: row["value"], reverse=True)
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    ranked = [{"rank": row["rank"], "learning_rate": row["learning_rate"], "value": row["value"]} for row in ranked]
    write_csv(output, ranked, fields=["rank", "learning_rate", "value"])


def summarize(c, runs, output, require_tests=True):
    validate(c, runs, output, require_tests=require_tests)
    run_rows, lr_rows, layer_rows = [], [], []
    summaries = {}
    for spec in __import__("expv1b.common", fromlist=["run_specs"]).run_specs(c):
        root = runs / "seed42" / spec["run_id"]
        summary = load(root / "summary.json")
        summaries[spec["run_id"]] = summary
        train = pd.read_csv(root / "train_metrics.csv")
        row = {key: value for key, value in summary.items() if key not in ("final_model_hash",)}
        row.update(_late_medians(train, summary["completed_steps"]))
        run_rows.append(row)
        if spec["method"] != "dp_fisher_wiener":
            continue
        baseline = summaries.get("dp_sgd_lr0p10")
        lr_row = {
            "learning_rate": spec["learning_rate"],
            "status": summary["status"],
            "final_accuracy": summary["final_accuracy"],
            "accuracy_auc": summary["accuracy_auc"],
            "late_mean_accuracy": summary["late_mean_accuracy"],
            "final_test_loss": summary["final_test_loss"],
            "steps_to_acc_50": summary["steps_to_acc_50"],
            "steps_to_acc_70": summary["steps_to_acc_70"],
            "steps_to_acc_80": summary["steps_to_acc_80"],
            "steps_to_acc_85": summary["steps_to_acc_85"],
            "runtime": summary["wall_time"],
            "wall_time": summary["wall_time"],
            "active_state_bytes": summary["active_state_bytes"],
        }
        lr_row.update(_late_medians(train, summary["completed_steps"]))
        lr_row["late_effective_signal_lr_global"] = lr_row["late_effective_signal_lr"]
        if baseline is not None:
            lr_row["delta_final_accuracy_vs_dp_sgd"] = (
                summary["final_accuracy"] - baseline["final_accuracy"]
                if summary["final_accuracy"] is not None else None
            )
            lr_row["delta_auc_vs_dp_sgd"] = (
                summary["accuracy_auc"] - baseline["accuracy_auc"]
                if summary["accuracy_auc"] is not None else None
            )
            lr_row["delta_final_loss_vs_dp_sgd"] = (
                summary["final_test_loss"] - baseline["final_test_loss"]
                if summary["final_test_loss"] is not None else None
            )
        lr_rows.append(lr_row)

        layer = pd.read_csv(root / "layer_metrics.csv")
        late = layer[layer.step + 1 > summary["completed_steps"] / 2]
        for layer_name in LAYERS:
            part = late[late.layer == layer_name]
            layer_row = {
                "learning_rate": spec["learning_rate"],
                "layer": layer_name,
                "status": summary["status"],
            }
            for metric in (
                "signal_retention", "signal_amplitude_retention", "effective_signal_lr",
                "noise_retention", "snr_gain_db", "relmse_filtered", "cosine_filtered",
                "kappa", "H_q50", "H_q10", "H_q90",
            ):
                layer_row[metric] = float(part[metric].median()) if not part.empty and part[metric].notna().any() else None
            layer_rows.append(layer_row)

    write_csv(output / "summary_runs.csv", run_rows)
    write_csv(output / "summary_lr.csv", lr_rows)
    write_csv(output / "summary_layers.csv", layer_rows)
    _ranking(lr_rows, "final_accuracy", output / "ranking_final_accuracy.csv")
    _ranking(lr_rows, "accuracy_auc", output / "ranking_accuracy_auc.csv")

    completed_lr = [row for row in lr_rows if row["status"] == "completed"]
    highest_accuracy = max(completed_lr, key=lambda row: row["final_accuracy"]) if completed_lr else None
    highest_auc = max(completed_lr, key=lambda row: row["accuracy_auc"]) if completed_lr else None
    analysis = dict(
        experiment="expv1b", seed=42, single_seed_descriptive_only=True,
        no_significance_tests=True, learning_rates=list(FISHER_LRS),
        highest_final_accuracy_lr=(highest_accuracy["learning_rate"] if highest_accuracy else None),
        highest_accuracy_auc_lr=(highest_auc["learning_rate"] if highest_auc else None),
        dp_sgd_anchor_learning_rate=0.1,
        runs=[row["run_id"] for row in run_rows],
    )
    save_json(output / "summary.json", analysis)
    print(f"Summary tables generated: {output}")
    return analysis


if __name__ == "__main__":
    summarize(*cli())
