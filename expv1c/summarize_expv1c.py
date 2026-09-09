"""Finite-safe descriptive utility, stability, and historical summaries."""
import math
import numpy as np
import pandas as pd
from expv1c.common import FISHER_LRS, LAYERS, REFERENCE_ROOT, run_specs, save_json, write_csv
from expv1c.validate_expv1c import cli, load, validate, _bool_value, ANCHORS
from expv1b.summarize_expv1b import DIAGNOSTIC_METRICS


def finite_median(series):
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.median()) if len(values) else None


def _late_medians(frame, total):
    late = frame[frame.step + 1 > total / 2]
    return {f"late_{metric}": finite_median(late[metric]) for metric in DIAGNOSTIC_METRICS}


def rank_rows(rows, metric, reverse=True):
    ranked = [row for row in rows if row["status"] == "completed"
              and row.get(metric) is not None and math.isfinite(row[metric])]
    return sorted(ranked, key=lambda row: row[metric], reverse=reverse)


def boundary_analysis(rows):
    accuracy = rank_rows(rows, "final_accuracy")
    auc = rank_rows(rows, "accuracy_auc")
    loss = rank_rows(rows, "final_test_loss", reverse=False)
    candidate = auc[0] if auc else None
    completed = [row for row in rows if row["status"] == "completed"]
    # AUC is the prespecified descriptive utility used for these two flags.
    boundary = bool(candidate and candidate["learning_rate"] == max(FISHER_LRS))
    bracketed = bool(candidate and not boundary and any(
        row["learning_rate"] > candidate["learning_rate"] and row["accuracy_auc"] < candidate["accuracy_auc"]
        for row in auc))
    return dict(
        highest_final_accuracy_lr=accuracy[0]["learning_rate"] if accuracy else None,
        highest_accuracy_auc_lr=candidate["learning_rate"] if candidate else None,
        lowest_final_test_loss_lr=loss[0]["learning_rate"] if loss else None,
        highest_lr_completed=max((row["learning_rate"] for row in completed), default=None),
        number_diverged=sum(row["status"] == "diverged" for row in rows),
        peak_bracketed=bracketed, boundary_best=boundary,
        stability_boundary_observed=any(row["status"] == "diverged" for row in rows),
        boundary_metric="accuracy_auc",
        best_tested_lr_by_auc=candidate["learning_rate"] if candidate else None,
        descriptive_candidate_lr=candidate["learning_rate"] if candidate else None,
        candidate_final_accuracy=candidate["final_accuracy"] if candidate else None,
        candidate_final_loss=candidate["final_test_loss"] if candidate else None,
    )


def combined_history(c, rows, summaries):
    # Never combine tiny smoke observations with full-horizon historical utility.
    if c["smoke"] or not (REFERENCE_ROOT / "summary_lr.csv").exists():
        return []
    anchor = "dp_fisher_wiener_lr0p80"
    reference = load(REFERENCE_ROOT / "seed42" / anchor / "summary.json")
    assert summaries[anchor]["final_model_hash"] == reference["final_model_hash"] == ANCHORS[anchor]
    for key in ("final_accuracy", "accuracy_auc"):
        assert summaries[anchor][key] == reference[key]
    history = pd.read_csv(REFERENCE_ROOT / "summary_lr.csv")
    historical = history[history.learning_rate < .8].to_dict("records")
    return [dict(row, source="expv1b") for row in historical] + [dict(row, source="expv1c") for row in rows]


def summarize(c, runs, output, require_tests=True):
    validate(c, runs, output, require_tests=require_tests)
    run_rows, lr_rows, layer_rows, summaries = [], [], [], {}
    for spec in run_specs(c):
        root = runs / "seed42" / spec["run_id"]
        summary = load(root / "summary.json")
        summaries[spec["run_id"]] = summary
        train = pd.read_csv(root / "train_metrics.csv")
        finite_steps = train.loc[train.diagnostics_finite.map(_bool_value), "step"]
        diagnostics = train[train.step.isin(finite_steps)]
        row = dict(summary, **_late_medians(diagnostics, summary["completed_steps"]))
        run_rows.append(row)
        if spec["method"] != "dp_fisher_wiener":
            continue
        baseline = summaries["dp_sgd_lr0p10"]
        row = dict(row, runtime=summary["wall_time"], late_effective_signal_lr_global=row["late_effective_signal_lr"])
        for metric, suffix in [("final_accuracy", "final_accuracy"), ("accuracy_auc", "auc"), ("final_test_loss", "final_loss")]:
            row[f"delta_{suffix}_vs_dp_sgd"] = (
                summary[metric] - baseline[metric]
                if summary[metric] is not None and baseline[metric] is not None else None)
        lr_rows.append(row)
        layer = pd.read_csv(root / "layer_metrics.csv")
        late = layer[(layer.step + 1 > summary["completed_steps"] / 2) & layer.step.isin(finite_steps)]
        for name in LAYERS:
            part = late[late.layer == name]
            layer_rows.append(dict(learning_rate=spec["learning_rate"], layer=name, status=summary["status"], **{
                metric: finite_median(part[metric]) for metric in (
                    "signal_retention", "signal_amplitude_retention", "effective_signal_lr", "noise_retention",
                    "snr_gain_db", "relmse_filtered", "cosine_filtered", "kappa", "H_q50", "H_q10", "H_q90")
            }))
    for name, rows in [("runs", run_rows), ("lr", lr_rows), ("layers", layer_rows)]:
        write_csv(output / f"summary_{name}.csv", rows)
    for metric, name, reverse in [("final_accuracy", "final_accuracy", True), ("accuracy_auc", "accuracy_auc", True), ("final_test_loss", "final_loss", False)]:
        ranked = rank_rows(lr_rows, metric, reverse)
        write_csv(output / f"ranking_{name}.csv", [dict(rank=i, learning_rate=row["learning_rate"], value=row[metric])
                  for i, row in enumerate(ranked, 1)], fields=["rank", "learning_rate", "value"])
    write_csv(output / "stability.csv", [row for row in lr_rows if row["status"] == "diverged"],
              fields=["run_id", "learning_rate", "status", "diverged_step", "completed_steps", "epsilon_spent_at_divergence", "last_finite_accuracy", "last_finite_loss"])
    history = combined_history(c, lr_rows, summaries)
    write_csv(output / "combined_lr_history.csv", history,
              fields=["learning_rate", "source", "status", "final_accuracy", "accuracy_auc", "final_test_loss", "steps_to_acc_90"])
    analysis = dict(experiment="expv1c", seed=42, single_seed_descriptive_only=True, no_significance_tests=True,
                    learning_rates=list(FISHER_LRS), dp_sgd_anchor_learning_rate=.1,
                    combined_history_available=bool(history), **boundary_analysis(lr_rows))
    save_json(output / "summary.json", analysis)
    print(f"Summary tables generated: {output}")
    return analysis


if __name__ == "__main__":
    summarize(*cli())
