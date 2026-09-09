"""Paired C-B utility and step-weighted mechanism summaries for ExpV4b."""
import argparse
import json
from pathlib import Path

import pandas as pd

from expv3.common import save_json
from expv4b.train_expv4b import ARMS

UTILITY = ["accuracy_auc", "final_accuracy", "late_mean_accuracy", "final_test_loss", "T80", "T85", "T90"]
MECHANISM = [
    "gamma_model", "gamma_oracle", "model_to_oracle_ratio", "model_multiplicative_error",
    "signal_retention", "signal_retention_after_gamma", "noise_retention",
    "noise_retention_after_gamma", "cosine_filtered", "snr_gain_db", "relmse_filtered",
    "compensated_gradient_norm", "gamma_update_ratio", "relmse_after_gamma", "cosine_after_gamma",
]


def summarize(config, runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    utility, mechanism, ranges = [], [], []
    for seed in config["seeds"]:
        for arm in ARMS:
            root = Path(runs) / f"seed{seed}" / arm
            summary = json.loads((root / "summary.json").read_text())
            utility.append({k: summary[k] for k in
                            ["seed", "run_id", "status", "divergence_stage", *UTILITY]})
            if arm != ARMS[0]:
                mechanism.append(pd.read_csv(root / "layer_metrics.csv"))
                refresh = pd.read_csv(root / "gamma_refresh_metrics.csv")
                ranges.append(dict(seed=seed, run_id=arm,
                                   gamma_min=float(refresh.gamma_model.min()),
                                   gamma_max_observed=float(refresh.gamma_model.max()),
                                   status=summary["status"], divergence_stage=summary["divergence_stage"]))
    utility = pd.DataFrame(utility)
    utility[UTILITY] = utility[UTILITY].apply(pd.to_numeric)
    utility.to_csv(output / "utility_by_seed.csv", index=False)
    utility.groupby("run_id")[UTILITY].agg(["mean", "median", "count"]).to_csv(output / "utility_summary.csv")
    paired = []
    for seed in config["seeds"]:
        rows = utility[utility.seed == seed].set_index("run_id")
        paired.append(dict(seed=seed, **{metric: rows.loc[ARMS[2], metric] - rows.loc[ARMS[1], metric]
                                       for metric in UTILITY}))
    paired = pd.DataFrame(paired)
    paired.to_csv(output / "paired_C_minus_B.csv", index=False)
    auc = paired.accuracy_auc.dropna()
    primary = dict(comparison="C-B", endpoint="accuracy_auc", planned_pairs=len(config["seeds"]),
                   completed_pairs=len(auc), positive_pairs=int((auc > 0).sum()),
                   mean_paired_difference=float(auc.mean()) if len(auc) else None,
                   median_paired_difference=float(auc.median()) if len(auc) else None,
                   smoke_functional_only=bool(config["smoke"]))
    save_json(output / "primary_endpoint.json", primary)
    mechanisms = pd.concat(mechanism, ignore_index=True)
    stats = []
    for arm, frame in mechanisms.groupby("run_id"):
        for layer, values in [("overall", frame), *list(frame.groupby("layer"))]:
            for metric in MECHANISM:
                series = values[metric].dropna()
                stats.append(dict(run_id=arm, layer=layer, metric=metric, count=len(series),
                                  median=series.median(), q10=series.quantile(.1), q90=series.quantile(.9)))
    pd.DataFrame(stats).to_csv(output / "mechanism_summary.csv", index=False)
    pd.DataFrame(ranges).to_csv(output / "gamma_ranges.csv", index=False)
    title = "Functional smoke only; no scientific conclusion." if config["smoke"] else "ExpV4b paired training comparison."
    report = [title, "", json.dumps(primary, indent=2), "",
              "A/B/C utility (missing thresholds mean not reached; counts exclude missing/diverged endpoints):",
              utility.to_string(index=False), "", "Raw gamma ranges:", pd.DataFrame(ranges).to_string(index=False), "",
              "Mechanism quantiles pool layer-step observations; overall is not a whole-model scalar gamma.",
              "The filtered metrics describe Wy before gamma. *_after_gamma describes the actual compensated update.",
              "Compare signal_retention_after_gamma and compensated_gradient_norm alongside cosine/SNR/relMSE",
              "to distinguish spectral filtering gains from update magnitude. See mechanism_summary.csv.",
              "Diverged arms retain diagnostics but have no completed primary endpoint; paired differences stay missing."]
    (output / "summary.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return primary


def main():
    parser = argparse.ArgumentParser(__doc__)
    for arg in ("config", "runs", "output"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    summarize(json.loads(Path(args.config).read_text()), args.runs, args.output)


if __name__ == "__main__":
    main()
