"""Analyze Exp5b runs and write the preregistered paired summaries."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp5b.common import METHODS, MOMENTUM_METHODS, ROOT, SYNTHETIC_METHODS, fingerprint, provenance, read_config, save_json

PRIMARY_CONTRASTS = (
    ("syn_diag_beta12", "syn_diag", "Full - SynDiag"),
    ("syn_diag_beta12", "syn_diag_beta1", "Full - syn_diag_beta1"),
    ("syn_diag_beta12", "syn_diag_beta2", "Full - syn_diag_beta2"),
    ("syn_diag_beta12", "dp_sgd_momentum", "Full - dp_sgd_momentum"),
    ("syn_diag_beta12", "dp_kfc_momentum", "Full - dp_kfc_momentum"),
)


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
        seed_root = root / f"seed{seed}"
        discovered_methods = {p.name for p in seed_root.iterdir() if p.is_dir()}
        require(discovered_methods == set(METHODS), f"Method set mismatch: {seed_root}")
        for method in METHODS:
            path = seed_root / method
            meta = json.loads((path / "metadata.json").read_text())
            config = json.loads((path / "config.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
            require(meta.get("complete") and meta["seed"] == seed and meta["method"] == method,
                    f"Identity: {path}")
            require(config == c and meta["fingerprint"] == fingerprint(c) and summary["fingerprint"] == fingerprint(c),
                    f"Configuration mismatch: {path}")
            if source is None:
                source = meta["provenance"]
            require(source == meta["provenance"], "Implementation provenance differs between runs")
            train = pd.read_csv(path / "train_metrics.csv")
            evals = pd.read_csv(path / "eval_metrics.csv")
            audit = json.loads((path / "pairing.json").read_text())
            total = meta["total_steps"]
            require(total == c["total_private_steps"], f"Configured step count mismatch: {path}")
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
        reference = pairs["syn_diag"][4]["private"]
        for method, pair in pairs.items():
            for left, right in zip(reference, pair[4]["private"]):
                for key in ("step", "batch_indices", "batch_hash", "noise_hash", "noise_rng_before", "noise_rng_after"):
                    require(left[key] == right[key], f"Private pairing mismatch {seed}/{method}/{key}")
        reference_syn = pairs["syn_diag"][4]["synthetic"]
        for method in SYNTHETIC_METHODS:
            current_syn = pairs[method][4]["synthetic"]
            for left, right in zip(reference_syn, current_syn):
                for key in ("step", "count", "samples_hash", "labels_hash", "rng_before", "rng_after"):
                    require(left[key] == right[key], f"Synthetic pairing mismatch {seed}/{method}/{key}")
    return result, source


def late_median(frame, column):
    values = frame.loc[frame.step > frame.step.max() / 2, column].dropna()
    return None if values.empty else float(values.median())


def summarize(runs):
    rows = []
    summary_fields = (
        "final_accuracy", "best_accuracy", "late_mean_accuracy", "final_test_loss",
        "mean_refresh_time", "total_refresh_time", "core_wall_time", "diagnostic_seconds",
        "wall_time", "peak_cuda_memory_core", "peak_cuda_memory_overall", "peak_cuda_memory_allocated",
        "peak_cuda_memory_reserved", "preconditioner_state_bytes", "first_moment_state_bytes",
        "second_moment_state_bytes", "temporal_state_bytes",
        "optimizer_state_bytes", "total_algorithm_state_bytes",
    )
    train_fields = (
        "aggregate_cosine", "relative_distortion", "clipping_shape_error", "coefficient_cv",
        "expected_noise_norm", "actual_noise_norm", "diagnostic_snr", "update_norm",
        "method_update_norm", "method_clip_cosine", "noise_degradation_method",
        "cos_method_adam_current_noise_off", "cos_method_adam_noisy",
    )
    for path, meta, summary, train, evals, audit in runs:
        row = dict(method=meta["method"], seed=meta["seed"], **{key: summary.get(key) for key in summary_fields})
        for column in train_fields:
            if column in train:
                row[f"late_{column}"] = late_median(train, column)
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    summary_rows = []
    for method in METHODS:
        for metric in per_seed.columns.difference(["method", "seed"]):
            values = per_seed.loc[per_seed.method == method, metric].dropna().astype(float).to_numpy()
            summary_rows.append(dict(
                method=method, metric=metric,
                mean=float(values.mean()) if len(values) else None,
                std=float(values.std(ddof=1)) if len(values) > 1 else None,
                n=len(values),
            ))
    return per_seed, pd.DataFrame(summary_rows)


def _sample_stats(values):
    values = [float(value) for value in values if value is not None and not pd.isna(value)]
    return (
        float(np.mean(values)) if values else None,
        float(np.std(values, ddof=1)) if len(values) > 1 else None,
        len(values),
    )


def factorial_effects(per_seed):
    """Compute the preregistered paired 2x2 beta1/beta2 effects."""
    metrics = ["final_accuracy", "final_test_loss", "late_mean_accuracy"]
    rows = []
    for metric in metrics:
        effects = []
        for seed in sorted(per_seed.seed.unique()):
            values = {
                method: per_seed[(per_seed.method == method) & (per_seed.seed == seed)][metric].iloc[0]
                for method in ("syn_diag", "syn_diag_beta1", "syn_diag_beta2", "syn_diag_beta12")
            }
            if any(pd.isna(value) for value in values.values()):
                effect = (None, None, None)
            else:
                effect = (
                    float(values["syn_diag_beta1"] - values["syn_diag"]),
                    float(values["syn_diag_beta2"] - values["syn_diag"]),
                    float(values["syn_diag_beta12"] - values["syn_diag_beta1"] - values["syn_diag_beta2"] + values["syn_diag"]),
                )
            effects.append(effect)
            rows.append(dict(
                metric=metric, seed=int(seed), beta1_effect=effect[0], beta2_effect=effect[1], interaction=effect[2],
                beta1_effect_mean=None, beta1_effect_std=None, beta1_effect_n=None,
                beta2_effect_mean=None, beta2_effect_std=None, beta2_effect_n=None,
                interaction_mean=None, interaction_std=None, interaction_n=None,
            ))
        b1 = _sample_stats([effect[0] for effect in effects])
        b2 = _sample_stats([effect[1] for effect in effects])
        inter = _sample_stats([effect[2] for effect in effects])
        rows.append(dict(
            metric=metric, seed="all", beta1_effect=None, beta2_effect=None, interaction=None,
            beta1_effect_mean=b1[0], beta1_effect_std=b1[1], beta1_effect_n=b1[2],
            beta2_effect_mean=b2[0], beta2_effect_std=b2[1], beta2_effect_n=b2[2],
            interaction_mean=inter[0], interaction_std=inter[1], interaction_n=inter[2],
        ))
    return pd.DataFrame(rows)


def paired_contrasts(per_seed):
    metrics = ["final_accuracy", "final_test_loss", "late_mean_accuracy"]
    rows = []
    for left, right, label in PRIMARY_CONTRASTS:
        for metric in metrics:
            deltas = []
            for seed in sorted(per_seed.seed.unique()):
                a = per_seed[(per_seed.method == left) & (per_seed.seed == seed)][metric].iloc[0]
                b = per_seed[(per_seed.method == right) & (per_seed.seed == seed)][metric].iloc[0]
                delta = None if pd.isna(a) or pd.isna(b) else float(a - b)
                rows.append(dict(contrast=label, metric=metric, seed=int(seed), delta=delta, mean=None, std=None, n=None))
                if delta is not None:
                    deltas.append(delta)
            stats = _sample_stats(deltas)
            rows.append(dict(contrast=label, metric=metric, seed="all", delta=None,
                             mean=stats[0], std=stats[1], n=stats[2]))
    return pd.DataFrame(rows)


def _stat(frame, field, integer=False):
    if field not in frame:
        return "N/A"
    values = frame[field].dropna().astype(float).to_numpy()
    if not len(values):
        return "N/A"
    mean = float(values.mean())
    if integer and len(np.unique(values)) == 1:
        return str(int(values[0]))
    std = float(values.std(ddof=1)) if len(values) > 1 else None
    return f"{mean:.4g}" if std is None else f"{mean:.4g} +/- {std:.4g}"


def report(per_seed, contrasts, factorial, c):
    lines = [
        "# Exp5b: Matched SGD-scale temporal optimization",
        "",
        "Exp5b removes Adam training baselines and sets learning_rate=0.1 for all six methods.",
        "All four momentum-enabled methods use the same post-DP, bias-corrected FirstMomentState.",
        "The clean Adam direction is a mechanistic diagnostic reference only, not an Exp5b competing optimizer.",
        "All contrasts are paired within seed and report mean +/- sample SD (ddof=1); no p-values or significance claims are reported.",
        "",
    ]
    if c["smoke"]:
        lines += ["Smoke validation only; this output is not a formal experimental conclusion.", ""]
    lines += [
        "## Main results",
        "",
        "| Method | Final Acc ↑ | Final Loss ↓ | Late Acc ↑ | Core Time ↓ | Refresh Time ↓ | Algorithm State Bytes ↓ | Peak CUDA Memory Core ↓ | Best Acc | Mean Refresh Time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        frame = per_seed[per_seed.method == method]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            method, _stat(frame, "final_accuracy"), _stat(frame, "final_test_loss"),
            _stat(frame, "late_mean_accuracy"), _stat(frame, "core_wall_time"),
            _stat(frame, "total_refresh_time"), _stat(frame, "total_algorithm_state_bytes", integer=True),
            _stat(frame, "peak_cuda_memory_core", integer=True), _stat(frame, "best_accuracy"),
            _stat(frame, "mean_refresh_time"),
        ))
    lines += [
        "", "## Paired contrasts", "",
        "| Contrast | Metric | Mean | Sample SD | N |", "| --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in contrasts[contrasts.seed == "all"].iterrows():
        mean = "N/A" if pd.isna(row["mean"]) else f"{row['mean']:.6g}"
        std = "N/A" if pd.isna(row["std"]) else f"{row['std']:.6g}"
        lines.append(f"| {row['contrast']} | {row['metric']} | {mean} | {std} | {int(row['n'])} |")
    lines += [
        "", "## SynDiag factorial effects", "",
        "Y00=SynDiag, Y10=SynDiag+β1, Y01=SynDiag+β2, Y11=Full. Values are paired within seed; no significance claims.",
        "", "| Metric | β1 effect mean +/- SD | β2 effect mean +/- SD | Interaction mean +/- SD | N |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in factorial[factorial.seed == "all"].iterrows():
        def fmt(mean, std):
            return "N/A" if pd.isna(mean) else f"{mean:.6g} +/- {'N/A' if pd.isna(std) else f'{std:.6g}'}"
        lines.append(f"| {row['metric']} | {fmt(row['beta1_effect_mean'], row['beta1_effect_std'])} | "
                     f"{fmt(row['beta2_effect_mean'], row['beta2_effect_std'])} | "
                     f"{fmt(row['interaction_mean'], row['interaction_std'])} | {int(row['beta1_effect_n'])} |")
    lines += [
        "", "## Research questions", "",
        "- RQ1: Does post-DP first-moment EMA improve SynDiag utility under an SGD-scale learning rate?",
        "- RQ2: Does true-private-time synthetic β2 smoothing improve SynDiag utility?",
        "- RQ3: Do β1 and β2 interact constructively?",
        "- RQ4: How does Full compare with matched DP-SGD-Momentum and DP-KFC-Momentum in utility, geometry, and efficiency?",
        "", "Results are reported as observed; no outcome is hardcoded or treated as statistically significant.", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    c = read_config(args.config)
    runs, source = load_runs(c, args.runs)
    out = Path(args.output or args.runs)
    out.mkdir(parents=True, exist_ok=True)
    per_seed, summary = summarize(runs)
    contrasts = paired_contrasts(per_seed)
    factorial = factorial_effects(per_seed)
    per_seed.to_csv(out / "summary_per_seed.csv", index=False)
    summary.to_csv(out / "summary_across_seeds.csv", index=False)
    contrasts.to_csv(out / "paired_contrasts.csv", index=False)
    factorial.to_csv(out / "factorial_effects.csv", index=False)
    (out / "report.md").write_text(report(per_seed, contrasts, factorial, c))
    current_source = source == provenance()
    save_json(out / "analysis.json", dict(
        passed=True, runs=len(runs), seeds=c["seeds"], methods=list(METHODS),
        primary_contrasts=[label for _, _, label in PRIMARY_CONTRASTS],
        source_matches_current_checkout=current_source,
    ))
    print(f"Analyzed {len(runs)} runs; wrote paired contrasts, factorial effects, and report.")


if __name__ == "__main__":
    main()
