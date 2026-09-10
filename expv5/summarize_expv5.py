"""Per-seed paired Fisher and adaptive-beta effects, plus layer diagnostics."""
import argparse
import json
from pathlib import Path
import pandas as pd
from expv3.common import save_json
from expv5.train_expv5 import ARMS

UTILITY = ["accuracy_auc", "final_accuracy", "late_mean_accuracy", "final_test_loss",
           "T50", "T70", "T80", "T85", "T90"]
MECHANISM = ["filtered_gradient_norm", "filter_gradient_norm_ratio", "parameter_update_norm",
             "gradient_norm_into_adam", "adam_first_moment_norm", "adam_second_moment_sqrt_norm",
             "adam_normalized_update_norm", "signal_retention", "noise_retention",
             "cosine_filtered", "snr_gain_db", "relmse_filtered", "beta_train", "gamma_model_raw"]


def summarize(config, runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    utility, mechanism = [], []
    for seed in config["seeds"]:
        for arm in ARMS:
            root = Path(runs) / f"seed{seed}" / arm
            summary = json.loads((root / "summary.json").read_text())
            utility.append({k: summary[k] for k in ["seed", "run_id", "status", *UTILITY]})
            mechanism.append(pd.read_csv(root / "layer_metrics.csv"))
    utility = pd.DataFrame(utility)
    utility[UTILITY] = utility[UTILITY].apply(pd.to_numeric)
    utility.to_csv(output / "utility_by_seed.csv", index=False)
    utility.groupby("run_id")[UTILITY].agg(["mean", "count"]).to_csv(output / "utility_summary.csv")
    effects = {}
    for name, left, right in [("B_minus_A", ARMS[1], ARMS[0]), ("C_minus_B", ARMS[2], ARMS[1])]:
        paired = []
        for seed in config["seeds"]:
            values = utility[utility.seed == seed].set_index("run_id")
            paired.append(dict(seed=seed, **{k: values.loc[left, k] - values.loc[right, k] for k in UTILITY}))
        paired = pd.DataFrame(paired)
        paired.to_csv(output / f"paired_{name}.csv", index=False)
        effects[name] = {k: dict(mean=float(paired[k].mean()) if paired[k].notna().any() else None,
                               pairs=int(paired[k].count())) for k in UTILITY}
    mechanisms = pd.concat(mechanism, ignore_index=True)
    by_seed_layer = mechanisms.groupby(["seed", "run_id", "layer"])[MECHANISM].mean()
    by_seed_layer.to_csv(output / "mechanism_by_seed_layer.csv")
    updates = by_seed_layer["parameter_update_norm"].unstack("run_id")
    (updates[ARMS[1]] / updates[ARMS[0]]).rename("parameter_update_ratio").to_csv(
        output / "adam_scale_B_vs_A.csv")
    mechanisms.groupby(["run_id", "layer"])[MECHANISM].agg(["mean", "median", "min", "max"]).to_csv(output / "mechanism_summary.csv")
    result = dict(smoke_functional_only=config["smoke"], effects=effects)
    save_json(output / "summary.json", result)
    report = ("Functional smoke only; no scientific conclusion." if config["smoke"] else "ExpV5 paired Adam comparison.")
    report += "\n" + utility.to_string(index=False) + "\n" + json.dumps(effects, indent=2)
    report += "\nMissing thresholds mean not reached; their paired differences stay missing.\n"
    report += "Compare filter_gradient_norm_ratio with actual parameter_update_norm in mechanism_by_seed_layer.csv.\n"
    report += "Mean parameter update norm B/A by seed and layer: adam_scale_B_vs_A.csv.\n"
    (output / "summary.txt").write_text(report)
    print(report)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for arg in ("config", "runs", "output"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    summarize(json.loads(Path(args.config).read_text()), args.runs, args.output)


if __name__ == "__main__":
    main()
