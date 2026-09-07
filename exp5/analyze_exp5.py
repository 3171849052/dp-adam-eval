"""Validate Exp5 runs and write paired contrasts and a concise report."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp5.common import (METHODS, ROOT, SYNTHETIC_METHODS, fingerprint,
                         provenance, read_config, save_json)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_runs(c, runs):
    root = Path(runs)
    expected = {f"seed{s}" for s in c["seeds"]}
    discovered = {p.name for p in root.glob("seed*") if p.is_dir()}
    require(discovered == expected, "Missing or unexpected seed directories")
    result = []
    source = None
    for seed in c["seeds"]:
        pairs = {}
        for method in METHODS:
            path = root / f"seed{seed}" / method
            meta = json.loads((path / "metadata.json").read_text())
            config = json.loads((path / "config.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
            require(meta.get("complete") and meta["seed"] == seed and meta["method"] == method, f"Identity: {path}")
            require(config == c and meta["fingerprint"] == fingerprint(c) and summary["fingerprint"] == fingerprint(c),
                    f"Configuration mismatch: {path}")
            if source is None:
                source = meta["provenance"]
            require(source == meta["provenance"], "Implementation provenance differs between runs")
            train = pd.read_csv(path / "train_metrics.csv")
            evals = pd.read_csv(path / "eval_metrics.csv")
            audit = json.loads((path / "pairing.json").read_text())
            total = meta["total_steps"]
            require(summary["completed_steps"] == total and len(train) == total, f"Incomplete steps: {path}")
            require(train.step.tolist() == list(range(1, total + 1)), f"Step sequence: {path}")
            require(train.seed.eq(seed).all() and train.method.eq(method).all(), f"CSV identity: {path}")
            require(len(audit["private"]) == total, f"Private audit length: {path}")
            require(audit["oracle_indices"] == meta["oracle_indices"], f"Oracle audit: {path}")
            expected_syn = list(range(0, total, c["K"])) if method in SYNTHETIC_METHODS else []
            require([x["step"] for x in audit["synthetic"]] == expected_syn, f"Refresh schedule: {path}")
            require(train.syn_refresh.tolist() == [s % c["K"] == 0 and method in SYNTHETIC_METHODS for s in range(total)],
                    f"CSV refresh schedule: {path}")
            for index, item in enumerate(audit["private"]):
                require(item["step"] == index + 1 and item["batch_hash"] == train.iloc[index].batch_hash,
                        f"Batch audit: {path}")
                require(item["noise_hash"] == train.iloc[index].noise_hash, f"Noise audit: {path}")
            pairs[method] = (meta, summary, train, evals, audit)
            result.append((path, meta, summary, train, evals, audit))
        reference = pairs["dp_adam"]
        for method, pair in pairs.items():
            for left, right in zip(reference[4]["private"], pair[4]["private"]):
                for key in ("step", "batch_hash", "noise_hash", "noise_rng_before", "noise_rng_after"):
                    require(left[key] == right[key], f"Private pairing mismatch {seed}/{method}/{key}")
        for method in SYNTHETIC_METHODS:
            a = pairs[method][4]["synthetic"]
            for other in SYNTHETIC_METHODS:
                b = pairs[other][4]["synthetic"]
                for left, right in zip(a, b):
                    for key in ("step", "count", "samples_hash", "labels_hash", "rng_before", "rng_after"):
                        require(left[key] == right[key], f"Synthetic pairing mismatch {seed}/{method}/{other}/{key}")
    return result, source


def late_median(frame, column):
    values = frame.loc[frame.step > frame.step.max() / 2, column].dropna()
    return None if values.empty else float(values.median())


def summarize(runs):
    rows = []
    for path, meta, summary, train, evals, audit in runs:
        row = dict(method=meta["method"], seed=meta["seed"], **{
            key: summary.get(key) for key in ("final_accuracy", "best_accuracy", "late_mean_accuracy", "final_test_loss",
                                               "mean_refresh_time", "total_refresh_time", "core_wall_time",
                                               "diagnostic_seconds", "wall_time", "peak_cuda_memory_core",
                                               "peak_cuda_memory_overall", "peak_cuda_memory_allocated",
                                               "peak_cuda_memory_reserved", "preconditioner_state_bytes",
                                               "temporal_state_bytes", "optimizer_state_bytes",
                                               "total_algorithm_state_bytes")})
        for column in ("aggregate_cosine", "clipping_shape_error", "coefficient_cv", "expected_noise_norm",
                       "actual_noise_norm", "diagnostic_snr", "update_norm", "method_clip_cosine", "noise_degradation_method",
                       "cos_method_adam_current_noise_off", "cos_method_adam_noisy"):
            if column in train:
                row[f"late_{column}"] = late_median(train, column)
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    summary_rows = []
    for method in METHODS:
        for metric in per_seed.columns.difference(["method", "seed"]):
            values = per_seed.loc[per_seed.method == method, metric].dropna().astype(float).to_numpy()
            summary_rows.append(dict(method=method, metric=metric, mean=float(values.mean()) if len(values) else None,
                                     std=float(values.std(ddof=1)) if len(values) > 1 else None, n=len(values)))
    return per_seed, pd.DataFrame(summary_rows)


def factorial_effects(per_seed):
    """Compute paired 2x2 effects per seed and across seeds."""
    metrics = [c for c in per_seed.columns if c not in ("method", "seed")]
    rows = []
    for metric in metrics:
        effects = []
        for seed in sorted(per_seed.seed.unique()):
            values = {method: per_seed[(per_seed.method == method) & (per_seed.seed == seed)][metric].iloc[0]
                      for method in ("syn_diag", "syn_diag_beta1", "syn_diag_beta2", "syn_diag_beta12")}
            if any(pd.isna(value) for value in values.values()):
                effect = (None, None, None)
            else:
                effect = (float(values["syn_diag_beta1"] - values["syn_diag"]),
                          float(values["syn_diag_beta2"] - values["syn_diag"]),
                          float(values["syn_diag_beta12"] - values["syn_diag_beta1"] - values["syn_diag_beta2"] + values["syn_diag"]))
            rows.append(dict(metric=metric, seed=int(seed), beta1_effect=effect[0], beta2_effect=effect[1],
                             interaction=effect[2], beta1_mean=None, beta1_std=None, beta1_n=None,
                             beta2_mean=None, beta2_std=None, beta2_n=None,
                             interaction_mean=None, interaction_std=None, interaction_n=None))
            effects.append(effect)
        for index, name in enumerate(("beta1", "beta2", "interaction")):
            values = [effect[index] for effect in effects if effect[index] is not None]
            stats = (float(np.mean(values)) if values else None,
                     float(np.std(values, ddof=1)) if len(values) > 1 else None, len(values))
            if index == 0:
                stats0 = stats
            elif index == 1:
                stats1 = stats
            else:
                stats2 = stats
        rows.append(dict(metric=metric, seed="all", beta1_effect=None, beta2_effect=None, interaction=None,
                         beta1_mean=stats0[0], beta1_std=stats0[1], beta1_n=stats0[2],
                         beta2_mean=stats1[0], beta2_std=stats1[1], beta2_n=stats1[2],
                         interaction_mean=stats2[0], interaction_std=stats2[1], interaction_n=stats2[2]))
    return pd.DataFrame(rows)


def paired_contrasts(per_seed):
    contrasts = [("syn_diag_beta12", "syn_diag"), ("syn_diag_beta12", "syn_diag_beta1"),
                 ("syn_diag_beta12", "syn_diag_beta2"), ("syn_diag_beta12", "dp_adam"),
                 ("syn_diag_beta12", "dp_kfc_adam")]
    metrics = [c for c in per_seed.columns if c not in ("method", "seed")]
    rows = []
    for left, right in contrasts:
        for metric in metrics:
            deltas = []
            for seed in sorted(per_seed.seed.unique()):
                a = per_seed[(per_seed.method == left) & (per_seed.seed == seed)][metric].iloc[0]
                b = per_seed[(per_seed.method == right) & (per_seed.seed == seed)][metric].iloc[0]
                delta = None if pd.isna(a) or pd.isna(b) else float(a - b)
                rows.append(dict(contrast=f"{left} - {right}", metric=metric, seed=int(seed), delta=delta,
                                 mean=None, std=None, n=None))
                if delta is not None:
                    deltas.append(delta)
            rows.append(dict(contrast=f"{left} - {right}", metric=metric, seed="all", delta=None,
                             mean=float(np.mean(deltas)) if deltas else None,
                             std=float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None, n=len(deltas)))
    return pd.DataFrame(rows)


def report(per_seed, contrasts, c):
    lines = ["# Exp5: Adam-style EMA SynDiag", "",
             "All contrasts are paired within seed; the final row for each contrast reports mean +/- sample SD (ddof=1).",
             "No p-values or significance claims are reported.", ""]
    if c["smoke"]:
        lines += ["Smoke validation only; this output is not a formal experimental conclusion.", ""]
    lines += ["| Method | Final accuracy | Best accuracy | Late mean accuracy | Final test loss |", "| --- | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        f = per_seed[per_seed.method == method]
        lines.append(f"| {method} | {f.final_accuracy.mean():.4f} | {f.best_accuracy.mean():.4f} | {f.late_mean_accuracy.mean():.4f} | {f.final_test_loss.mean():.4f} |")
    lines += ["", "## Cost and Algorithm State", "",
              "| Method | Wall | Core wall | Diagnostics | Refresh | Preconditioner bytes | Temporal bytes | Optimizer bytes | Total state bytes |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        f = per_seed[per_seed.method == method].iloc[0]
        lines.append("| {} | {:.4g} | {:.4g} | {:.4g} | {:.4g} | {} | {} | {} | {} |".format(
            method, f.wall_time, f.core_wall_time, f.diagnostic_seconds, f.total_refresh_time,
            int(f.preconditioner_state_bytes), int(f.temporal_state_bytes), int(f.optimizer_state_bytes),
            int(f.total_algorithm_state_bytes)))
    lines += ["", "## Paired Contrasts", "", "| Contrast | Metric | Mean | Sample SD | N |", "| --- | --- | ---: | ---: | ---: |"]
    for _, row in contrasts[contrasts.seed == "all"].iterrows():
        mean = "N/A" if pd.isna(row["mean"]) else f"{row['mean']:.6g}"
        std = "N/A" if pd.isna(row["std"]) else f"{row['std']:.6g}"
        lines.append(f"| {row['contrast']} | {row['metric']} | {mean} | {std} | {int(row['n'])} |")
    lines += ["", "## SynDiag Factorial Effects", "",
              "Y00=syn_diag, Y10=syn_diag_beta1, Y01=syn_diag_beta2, Y11=syn_diag_beta12. Values are paired within seed; no significance claims.", "",
              "| Metric | beta1 mean +/- SD | beta2 mean +/- SD | interaction mean +/- SD | N |",
              "| --- | ---: | ---: | ---: | ---: |"]
    # The caller attaches the factorial table after report construction.
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    c = read_config(args.config)
    runs, source = load_runs(c, args.runs)
    out = Path(args.output or args.runs)
    per_seed, summary = summarize(runs)
    contrasts = paired_contrasts(per_seed)
    factorial = factorial_effects(per_seed)
    per_seed.to_csv(out / "summary_per_seed.csv", index=False)
    summary.to_csv(out / "summary_across_seeds.csv", index=False)
    contrasts.to_csv(out / "paired_contrasts.csv", index=False)
    factorial.to_csv(out / "factorial_effects.csv", index=False)
    report_text = report(per_seed, contrasts, c)
    lines = report_text.rstrip().splitlines()
    def format_stat(mean, std):
        mean_text = "N/A" if pd.isna(mean) else f"{mean:.6g}"
        std_text = "N/A" if pd.isna(std) else f"{std:.6g}"
        return f"{mean_text} +/- {std_text}"

    lines.extend([f"| {row.metric} | {format_stat(row.beta1_mean, row.beta1_std)} | "
                  f"{format_stat(row.beta2_mean, row.beta2_std)} | "
                  f"{format_stat(row.interaction_mean, row.interaction_std)} | {int(row.beta1_n)} |"
                  for _, row in factorial[factorial.seed == "all"].iterrows()])
    (out / "report.md").write_text("\n".join(lines) + "\n")
    current_source = source == provenance()
    save_json(out / "analysis.json", dict(passed=True, runs=len(runs), seeds=c["seeds"],
                                           methods=list(METHODS), source_matches_current_checkout=current_source,
                                           archived_source_matches_current_checkout=current_source))
    print(f"Analyzed {len(runs)} runs; wrote paired contrasts and report.")


if __name__ == "__main__":
    main()
