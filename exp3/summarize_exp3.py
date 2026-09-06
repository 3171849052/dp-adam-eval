"""Seed-level late summaries, sample SD, and strictly paired deltas."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from exp3.common import ROOT, METHODS, LAYERS, read_config, fingerprint, write_csv
from exp3.audit_upstream import require_clean

LATE = {"late_norm_cv": "norm_cv", "late_coefficient_cv": "coefficient_cv",
        "late_aggregate_cosine": "aggregate_cosine", "late_shape_error": "clipping_shape_error", "late_snr": "diagnostic_snr"}


def late_stats(frame, column, total):
    x = frame.loc[frame.step > total/2, column].dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return dict(median=None, q25=None, q75=None, iqr=None)
    q25, median, q75 = np.quantile(x, [.25, .5, .75])
    return dict(median=float(median), q25=float(q25), q75=float(q75), iqr=float(q75-q25))


def geometric_mean(values):
    x = np.array(values, dtype=float)
    if not len(x) or not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError("Geometric mean requires finite positive values")
    return float(np.exp(np.log(x).mean()))


def load_runs(root, c):
    discovered = {p.name for p in Path(root).glob("seed*") if p.is_dir()}
    if discovered != {f"seed{s}" for s in c["seeds"]}:
        raise ValueError("Missing or unexpected seed directories")
    runs = []
    for seed in c["seeds"]:
        for method in METHODS:
            path = Path(root) / f"seed{seed}" / method
            meta = json.loads((path / "metadata.json").read_text())
            require_clean(meta["provenance"], c["smoke"])
            for key in ("upstream_repo", "upstream_git_commit", "upstream_git_dirty", "upstream_git_remote"):
                if meta[key] != meta["provenance"][key]:
                    raise ValueError(f"Inconsistent upstream provenance: {path}/{key}")
            summary = json.loads((path / "summary.json").read_text())
            config = json.loads((path / "config.json").read_text())
            if config != c or meta["fingerprint"] != fingerprint(c) or summary["fingerprint"] != fingerprint(c):
                raise ValueError(f"Mixed configuration: {path}")
            if meta["seed"] != seed or meta["method"] != method or not meta.get("complete"):
                raise ValueError(f"Incomplete/mismatched run: {path}")
            frames = {n: pd.read_csv(path / f"{n}_metrics.csv") for n in ("train", "oracle", "refresh")}
            for f in frames.values():
                if len(f) and (set(f.seed) != {seed} or set(f.method) != {method} or set(f.config_fingerprint) != {fingerprint(c)}):
                    raise ValueError(f"Mixed metric rows: {path}")
            runs.append((path, meta, summary, frames))
    if len({json.dumps(m["provenance"], sort_keys=True) for _, m, _, _ in runs}) != 1:
        raise ValueError("Mixed implementation provenance")
    return runs


def paired_deltas(frame, seeds):
    pairs = [("syn_diag", "dp_sgd"), ("dp_kfc", "dp_sgd"), ("syn_diag", "dp_kfc")]
    result = []
    for a, b in pairs:
        for metric in frame.columns.difference(["seed", "method"]):
            values = []
            for seed in seeds:
                av = frame[(frame.seed == seed) & (frame.method == a)][metric].iloc[0]
                bv = frame[(frame.seed == seed) & (frame.method == b)][metric].iloc[0]
                value = None if pd.isna(av) or pd.isna(bv) else float(av-bv)
                result.append(dict(pair=f"{a} - {b}", metric=metric, seed=seed, delta=value, mean=None, std=None, n=None))
                if value is not None:
                    values.append(value)
            result.append(dict(pair=f"{a} - {b}", metric=metric, seed="all", delta=None,
                               mean=float(np.mean(values)) if values else None,
                               std=float(np.std(values, ddof=1)) if len(values)>1 else None, n=len(values)))
    return result


def summarize(root, c, output):
    rows = []
    for path, meta, summary, frames in load_runs(root, c):
        row = {k: summary[k] for k in ("method", "seed", "mean_refresh_time", "total_refresh_time", "wall_time", "core_wall_time", "diagnostic_seconds", "number_of_refreshes",
               "peak_cuda_memory_core", "peak_cuda_memory_overall", "peak_cuda_memory_allocated", "peak_cuda_memory_reserved", "preconditioner_state_bytes", "final_test_loss", "final_accuracy", "best_accuracy", "late_mean_accuracy")}
        row["peak_cuda_memory"] = row["peak_cuda_memory_core"]
        total = meta["total_steps"]
        for name, column in LATE.items():
            stats = late_stats(frames["train"], column, total)
            row[name] = stats["median"]
            row.update({f"{name}_{q}": stats[q] for q in ("q25", "q75", "iqr")})
        for category, frame, columns in (
            ("oracle", frames["oracle"], ["R_diag_new", "R_diag_old", "R_full_new", "R_full_old", "A_diag_raw", "A_diag_new", "A_diag_old", "S_full_raw", "S_full_new", "S_full_old", "private_delta_stale_diag", "private_delta_stale_full"]),
            ("refresh", frames["refresh"], ["A_old_diag", "A_new_diag", "delta_stale_diag", "S_old_full", "S_new_full", "delta_stale_full"])):
            for layer in LAYERS:
                for col in columns:
                    row[f"{category}_{col}_{layer}"] = late_stats(frame[frame.layer == layer], col, total)["median"] if len(frame) else None
        for kind in ("diag", "full"):
            for age in ("new", "old"):
                values = [row[f"oracle_R_{kind}_{age}_{layer}"] for layer in LAYERS]
                row[f"G_{kind}_{age}"] = geometric_mean(values) if all(v is not None for v in values) else None
            row[f"G_{kind}"] = row[f"G_{kind}_new"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    means = []
    for method in METHODS:
        for metric in frame.columns.difference(["seed", "method"]):
            values = frame.loc[frame.method == method, metric].dropna().astype(float).to_numpy()
            mean = float(values.mean()) if len(values) else None
            std = float(values.std(ddof=1)) if len(values)>1 else None
            means.append(dict(method=method, metric=metric, mean=mean, std=std, n=len(values),
                              mean_pm_std=f"{mean:.6g} ± {std:.6g}" if std is not None else (str(mean) if mean is not None else "N/A")))
    output = Path(output)
    write_csv(output / "summary_by_seed.csv", rows)
    write_csv(output / "summary_mean_std.csv", means)
    write_csv(output / "paired_deltas.csv", paired_deltas(frame, c["seeds"]))
    return frame


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/full.json"))
    p.add_argument("--runs", default=str(ROOT / "runs"))
    p.add_argument("--output", default=str(ROOT))
    a = p.parse_args()
    summarize(a.runs, read_config(a.config), a.output)
    print(f"Summary and paired deltas written to {a.output}")


if __name__ == "__main__":
    main()
