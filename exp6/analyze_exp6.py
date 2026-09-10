"""Write paired Exp6 endpoint comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp6.common import METHODS, ROOT, fingerprint, read_config, save_json

CONTRASTS = (("syn_adam", "dp_adam"),
             ("syn_adam_ema", "syn_adam"),
             ("syn_adam_ema", "dp_adam"))
METRICS = ("final_accuracy", "best_accuracy", "late_mean_accuracy",
           "final_test_loss", "epsilon_spent", "final_update_norm",
           "final_first_moment_norm")


def load_runs(config, runs):
    records = []
    for seed in config["seeds"]:
        for method in METHODS:
            path = Path(runs) / f"seed{seed}" / method
            meta = json.loads((path / "metadata.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
            train = pd.read_csv(path / "train_metrics.csv")
            records.append((path, meta, summary, train))
    return records


def analyze(config, runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = load_runs(config, runs)
    rows = [dict(seed=meta["seed"], method=meta["method"],
                 **{metric: summary[metric] for metric in METRICS})
            for _, meta, summary, _ in records]
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(output / "summary_per_seed.csv", index=False)

    across = []
    for method in METHODS:
        frame = per_seed[per_seed.method == method]
        for metric in METRICS:
            values = frame[metric].to_numpy(dtype=float)
            across.append(dict(method=method, metric=metric,
                               mean=float(values.mean()),
                               std=float(values.std(ddof=1)) if len(values) > 1 else None,
                               n=len(values)))
    across = pd.DataFrame(across)
    across.to_csv(output / "summary_across_seeds.csv", index=False)

    paired = []
    for left, right in CONTRASTS:
        for metric in METRICS:
            differences = []
            for seed in config["seeds"]:
                a = per_seed[(per_seed.seed == seed) & (per_seed.method == left)][metric].iloc[0]
                b = per_seed[(per_seed.seed == seed) & (per_seed.method == right)][metric].iloc[0]
                delta = float(a - b)
                differences.append(delta)
                paired.append(dict(contrast=f"{left} - {right}", metric=metric,
                                   seed=seed, difference=delta, mean=None, std=None, n=None))
            paired.append(dict(contrast=f"{left} - {right}", metric=metric,
                               seed="all", difference=None,
                               mean=float(np.mean(differences)),
                               std=float(np.std(differences, ddof=1)) if len(differences) > 1 else None,
                               n=len(differences)))
    paired = pd.DataFrame(paired)
    paired.to_csv(output / "paired_differences.csv", index=False)

    lines = ["# Exp6 paired comparisons", "",
             "Differences are paired within seed; no significance claims are made.", "",
             "| Contrast | Metric | Mean | Sample SD | N |",
             "| --- | --- | ---: | ---: | ---: |"]
    for row in paired[paired.seed == "all"].itertuples():
        std = "N/A" if pd.isna(row.std) else f"{row.std:.6g}"
        lines.append(f"| {row.contrast} | {row.metric} | {row.mean:.6g} | {std} | {int(row.n)} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    result = dict(passed=True, runs=len(records), seeds=config["seeds"],
                  methods=list(METHODS), contrasts=[f"{a} - {b}" for a, b in CONTRASTS],
                  fingerprint=fingerprint(config))
    save_json(output / "analysis.json", result)
    print(f"Analyzed {len(records)} runs; wrote paired differences.")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = read_config(args.config)
    analyze(config, args.runs, args.output or args.runs)


if __name__ == "__main__":
    main()
