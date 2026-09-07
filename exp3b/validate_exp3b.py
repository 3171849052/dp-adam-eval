"""Validate Exp3b outputs and prove that the Exp3 source was not mutated."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp3b.common import (
    CLIPPING_METRICS,
    CONTRIB_COLUMNS,
    DEFAULT_RESULTS,
    DEFAULT_SOURCE,
    ENERGY_METRICS,
    GEOMETRY_METRICS,
    LAYERS,
    METHODS,
    SEEDS,
    add_derived_train_metrics,
    derive_geometry,
    require,
    source_manifest,
    source_manifest_digest,
    validate_contributions,
    validate_no_inferred_layer_snr,
    validate_source,
    write_json,
)


def _read_csv(results, name):
    path = Path(results) / name
    require(path.exists(), f"Missing Exp3b output: {path}")
    return pd.read_csv(path)


def validate(source=DEFAULT_SOURCE, results=DEFAULT_RESULTS):
    source = Path(source).resolve()
    results = Path(results).resolve()
    before = source_manifest(source)
    bundle = validate_source(source)
    metadata_path = results / "analysis_metadata.json"
    require(metadata_path.exists(), "Missing Exp3b analysis_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    require(metadata["source_config_fingerprint"] == bundle["fingerprint"], "Analysis fingerprint mismatch")
    require(metadata["source_manifest_sha256"] == source_manifest_digest(before), "Analysis source manifest mismatch")

    trajectory = _read_csv(results, "trajectory_metrics.csv")
    geometry = _read_csv(results, "geometry_summary.csv")
    window_seed = _read_csv(results, "window_summary_by_seed.csv")
    window_mean = _read_csv(results, "window_summary_mean_std.csv")
    main = _read_csv(results, "main_mechanism_table.csv")
    _read_csv(results, "paired_deltas.csv")
    _read_csv(results, "temporal_correlations.csv")
    early_late = _read_csv(results, "early_late_geometry.csv")

    require(len(trajectory) == 9 * 1170, "Trajectory row count")
    require(set(trajectory.method) == set(METHODS) and set(trajectory.seed) == set(SEEDS), "Trajectory run labels")
    validate_no_inferred_layer_snr(trajectory.columns)
    for (method, seed), group in trajectory.groupby(["method", "seed"], sort=True):
        require(group.step.tolist() == list(range(1, 1171)), f"Trajectory steps: {method}/{seed}")
        validate_contributions(group)
        require(np.isfinite(group[ENERGY_METRICS + CLIPPING_METRICS].to_numpy(dtype=float)).all(),
                f"Nonfinite trajectory derived metrics: {method}/{seed}")
        require((group["alpha_over_mean_coeff"] >= 0).all(), "Invalid alpha ratio")

    require(len(geometry) == 9 * 24, "Geometry row count")
    require(set(GEOMETRY_METRICS) <= set(geometry.columns), "Geometry metrics missing")
    require(np.isfinite(geometry[GEOMETRY_METRICS].to_numpy(dtype=float)).all(), "Nonfinite geometry output")
    require((geometry[["G_diag_new", "G_full_new", "G_diag_conv", "G_diag_fc", "G_full_conv", "G_full_fc",
                      "diag_fc_over_conv", "full_fc_over_conv"]] > 0).all().all(), "Nonpositive geometry output")
    require(set(geometry.step) == set(range(0, 1170, 50)), "Geometry step alignment")
    for (method, seed), group in geometry.groupby(["method", "seed"], sort=True):
        require(set(group.step) == set(range(0, 1170, 50)), f"Geometry steps: {method}/{seed}")

    require(set(main.method) == set(METHODS) and len(main) == 3, "Main mechanism table")
    require(set(early_late.method) == set(METHODS) and set(early_late.seed) == set(SEEDS),
            "Early/late geometry labels")
    require(len(early_late) == 9, "Early/late geometry row count")
    require(np.isfinite(early_late.iloc[:, 2:].to_numpy(dtype=float)).all(), "Nonfinite early/late geometry")
    require(len(window_seed) > 0 and len(window_mean) > 0, "Empty window summaries")
    require(set(window_seed.window) == {"early", "mid", "late", "all"}, "Window labels")
    require(np.isfinite(window_seed[["median", "q25", "q75", "iqr"]].to_numpy(dtype=float)).all(),
            "Nonfinite window summaries")

    after = source_manifest(source)
    require(before == after, "Exp3 source changed during Exp3b validation")
    result = {
        "passed": True,
        "source": str(source),
        "results": str(results),
        "runs": 9,
        "steps_per_run": 1170,
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "source_config_fingerprint": bundle["fingerprint"],
        "source_manifest_sha256": source_manifest_digest(before),
        "upstream_git_commit": bundle["provenance"].get("upstream_git_commit"),
        "upstream_git_dirty": bundle["provenance"].get("upstream_git_dirty"),
        "upstream_origin": bundle["provenance"].get("upstream_origin"),
        "no_source_mutation": True,
        "no_layerwise_snr_inferred": True,
        "auc_definition": "mean over discrete oracle-point log ratios; no continuous integration",
    }
    write_json(results / "validation.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()
    validate(args.source, args.results)


if __name__ == "__main__":
    main()
