"""Run the pure offline Exp3b mechanism analysis."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from exp3b.common import (
    CLIPPING_METRICS,
    ENERGY_METRICS,
    GEOMETRY_METRICS,
    DEFAULT_RESULTS,
    DEFAULT_SOURCE,
    METHODS,
    SEEDS,
    ORACLE_WINDOWS,
    across_seed_mean_std,
    alpha_negative_summary,
    add_derived_train_metrics,
    derive_geometry,
    paired_deltas,
    require,
    validate_source,
    window_lookup,
    window_summary,
    correlation_summary,
    geometry_auc_summary,
    write_json,
    source_manifest_digest,
    source_manifest,
    WINDOWS,
)


MAIN_METRICS = [
    "early_G_full", "late_G_full", "early_G_diag", "late_G_diag",
    "early_G_diag_conv", "early_G_diag_fc", "late_G_diag_conv", "late_G_diag_fc",
    "early_diag_fc_over_conv", "late_diag_fc_over_conv", "early_contrib_mass", "late_contrib_mass",
    "early_nearzero_mass_deficit", "late_nearzero_mass_deficit", "early_fc_share", "late_fc_share",
    "early_alpha_negative_fraction", "late_alpha_negative_fraction",
    "early_layer_entropy", "late_layer_entropy", "late_norm_cv", "late_coefficient_cv",
    "late_alpha_star", "late_alpha_over_mean_coeff", "late_aggregate_cosine",
    "late_shape_error", "late_clipped_aggregate_norm", "late_snr",
    "final_accuracy", "late_mean_accuracy",
]

PAIRED_METRICS = [
    "final_accuracy", "late_mean_accuracy",
    "early_G_full", "late_G_full", "early_G_diag", "late_G_diag",
    "early_G_diag_fc", "mid_G_diag_fc", "late_G_diag_fc",
    "early_G_diag_conv", "mid_G_diag_conv", "late_G_diag_conv",
    "early_diag_fc_over_conv", "mid_diag_fc_over_conv", "late_diag_fc_over_conv",
    "early_contrib_mass", "late_contrib_mass", "early_nearzero_mass_deficit", "late_nearzero_mass_deficit",
    "early_alpha_negative_fraction", "late_alpha_negative_fraction",
    "early_fc_share", "mid_fc_share", "late_fc_share",
    "early_layer_entropy", "mid_layer_entropy", "late_layer_entropy",
    "late_coefficient_cv", "late_alpha_over_mean_coeff", "late_aggregate_cosine",
    "late_shape_error", "late_clipped_aggregate_norm", "late_diagnostic_snr",
]

EARLY_LATE_COLUMNS = [
    "early_G_full", "late_G_full", "delta_G_full", "early_G_diag", "late_G_diag",
    "early_G_diag_conv", "late_G_diag_conv", "early_G_diag_fc", "late_G_diag_fc",
    "early_diag_fc_over_conv", "late_diag_fc_over_conv", "early_fc_share", "late_fc_share",
    "early_layer_entropy", "late_layer_entropy",
]


def save_csv(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _seed_metrics(trajectories, geometry, energy_summary, geometry_summary, alpha_summary, runs):
    rows = []
    for run in runs:
        method, seed = run["method"], run["seed"]
        row = {"method": method, "seed": seed,
               "final_accuracy": float(run["summary"]["final_accuracy"]),
               "late_mean_accuracy": float(run["summary"]["late_mean_accuracy"])}
        for metric in ENERGY_METRICS + CLIPPING_METRICS:
            for window in ("early", "mid", "late", "all"):
                row[f"{window}_{metric}"] = window_lookup(energy_summary, method, seed, window, metric)
                if metric == "clipping_alpha_star":
                    row[f"{window}_alpha_star"] = row[f"{window}_{metric}"]
                if metric == "diagnostic_snr":
                    row[f"{window}_snr"] = row[f"{window}_{metric}"]
                if metric == "clipping_shape_error":
                    row[f"{window}_shape_error"] = row[f"{window}_{metric}"]
        for metric in GEOMETRY_METRICS:
            for window in ("early", "mid", "late", "all"):
                row[f"{window}_{metric}"] = window_lookup(geometry_summary, method, seed, window, metric)
        for window in ("early", "mid", "late", "all"):
            row[f"{window}_G_full"] = row[f"{window}_G_full_new"]
            row[f"{window}_G_diag"] = row[f"{window}_G_diag_new"]
        for metric in ("full_beneficial_fraction", "full_log_auc", "diag_log_auc"):
            for window in ("early", "mid", "late", "all"):
                row[f"{window}_{metric}"] = window_lookup(geometry_summary, method, seed, window, metric)
        for window in ("early", "mid", "late", "all"):
            row[f"{window}_alpha_negative_fraction"] = float(
                alpha_summary[(alpha_summary.method == method) & (alpha_summary.seed == seed) &
                              (alpha_summary.window == window)]["alpha_negative_fraction"].iloc[0]
            )
        row["early_delta_G_full"] = row["late_G_full"] - row["early_G_full"]
        row["early_delta_G_diag"] = row["late_G_diag"] - row["early_G_diag"]
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["method", "seed"]).reset_index(drop=True)
    require(len(result) == 9, "Seed summary must contain 9 rows")
    return result


def _main_table(seed_metrics):
    rows = []
    for method in METHODS:
        group = seed_metrics[seed_metrics.method == method]
        row = {"method": method}
        for metric in MAIN_METRICS:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_n"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def _early_late(seed_metrics):
    rows = []
    for _, source in seed_metrics.iterrows():
        row = {"method": source["method"], "seed": int(source["seed"])}
        for column in EARLY_LATE_COLUMNS:
            if column == "delta_G_full":
                row[column] = float(source["late_G_full"] - source["early_G_full"])
            else:
                row[column] = float(source[column])
        rows.append(row)
    return pd.DataFrame(rows)


def _early_late_mean_std(frame):
    rows = []
    for method, group in frame.groupby("method", sort=True):
        for column in frame.columns[2:]:
            values = group[column].to_numpy(dtype=float)
            rows.append({"method": method, "metric": column, "mean": float(values.mean()),
                         "std": float(values.std(ddof=1)), "n": len(values)})
    return pd.DataFrame(rows)


def _source_mass_diagnostic(trajectories):
    rows = []
    for method, group in trajectories.groupby("method", sort=True):
        late = group[group.step >= 586]
        rows.append({"method": method,
                     "min_contrib_mass": float(group.contrib_mass.min()),
                     "median_contrib_mass": float(group.contrib_mass.median()),
                     "late_median_contrib_mass": float(late.contrib_mass.median()),
                     "max_nearzero_mass_deficit": float(group.nearzero_mass_deficit.max())})
    return rows


def analyze(source=DEFAULT_SOURCE, output=DEFAULT_RESULTS):
    source = Path(source).resolve()
    output = Path(output).resolve()
    before_manifest = source_manifest(source)
    bundle = validate_source(source)
    trajectories, geometries = [], []
    for run in bundle["runs"]:
        trajectory = add_derived_train_metrics(run["train"])
        geometry = derive_geometry(run["oracle"])
        trajectories.append(trajectory)
        geometries.append(geometry)
    trajectories = pd.concat(trajectories, ignore_index=True).sort_values(["method", "seed", "step"]).reset_index(drop=True)
    geometries = pd.concat(geometries, ignore_index=True).sort_values(["method", "seed", "step"]).reset_index(drop=True)
    energy_summary = window_summary(trajectories, ENERGY_METRICS + CLIPPING_METRICS, oracle=False)
    geometry_summary = window_summary(geometries, GEOMETRY_METRICS, oracle=True)
    geometry_auc = geometry_auc_summary(geometries)
    alpha_summary = alpha_negative_summary(trajectories)
    alpha_window_summary = alpha_summary.assign(
        metric="alpha_negative_fraction", median=alpha_summary["alpha_negative_fraction"],
        q25=alpha_summary["alpha_negative_fraction"], q75=alpha_summary["alpha_negative_fraction"],
        iqr=0., n=alpha_summary["alpha_negative_count"].astype(int),
    )[["method", "seed", "window", "metric", "median", "q25", "q75", "iqr", "n"]]
    window_by_seed = pd.concat([energy_summary, geometry_summary, geometry_auc, alpha_window_summary], ignore_index=True)
    window_mean_std = across_seed_mean_std(window_by_seed)
    seed_metrics = _seed_metrics(trajectories, geometries, energy_summary,
                                 pd.concat([geometry_summary, geometry_auc], ignore_index=True),
                                 alpha_summary, bundle["runs"])
    main_table = _main_table(seed_metrics)
    early_late = _early_late(seed_metrics)
    early_late_mean_std = _early_late_mean_std(early_late)
    correlations = correlation_summary(
        trajectories,
        ("fc_share", "fc_to_conv", "layer_entropy", "nearzero_mass_deficit"),
        ("coefficient_cv", "coefficient_mean", "clipping_alpha_star", "alpha_over_mean_coeff",
         "aggregate_cosine", "clipping_shape_error", "clipped_aggregate_norm", "diagnostic_snr"),
    )
    paired = paired_deltas(seed_metrics, PAIRED_METRICS)

    output.mkdir(parents=True, exist_ok=True)
    save_csv(trajectories, output / "trajectory_metrics.csv")
    save_csv(window_by_seed, output / "window_summary_by_seed.csv")
    save_csv(window_mean_std, output / "window_summary_mean_std.csv")
    save_csv(paired, output / "paired_deltas.csv")
    save_csv(correlations, output / "temporal_correlations.csv")
    save_csv(alpha_summary, output / "alpha_negative_summary.csv")
    save_csv(geometries, output / "geometry_summary.csv")
    save_csv(main_table, output / "main_mechanism_table.csv")
    save_csv(early_late, output / "early_late_geometry.csv")
    save_csv(early_late_mean_std, output / "early_late_geometry_mean_std.csv")
    save_csv(seed_metrics, output / "seed_metrics.csv")
    after_manifest = source_manifest(source)
    require(before_manifest == after_manifest, "Exp3 source changed during analysis")
    write_json(output / "analysis_metadata.json", {
        "source": str(source), "source_manifest_sha256": source_manifest_digest(before_manifest),
        "source_validation": bundle["validation"], "source_config_fingerprint": bundle["fingerprint"],
        "source_provenance": bundle["provenance"], "windows": {k: list(v) for k, v in WINDOWS.items()},
        "oracle_windows": {k: list(v) for k, v in ORACLE_WINDOWS.items()},
        "auc_definition": "mean over discrete oracle-point log ratios; no continuous integration",
        "energy_definition": "mean per-example squared-norm fraction, not norm of an aggregated layer gradient",
        "composition_definition": "epsilon-weighted normalized layer composition after dividing raw contrib columns by contrib_mass",
        "nearzero_mass_definition": "1 - mean_i[||h_i||^2 / max(||h_i||^2, eps)], not a near-zero sample fraction",
        "alpha_definition": "unconstrained least-squares scalar projection; alpha_star may be negative",
        "source_contrib_mass_diagnostic": _source_mass_diagnostic(trajectories),
    })
    print(f"Exp3b analysis written to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()
    analyze(args.source, args.output)


if __name__ == "__main__":
    main()
