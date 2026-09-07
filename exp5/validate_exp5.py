"""Strict Exp5 pairing, finite-metric, state, and schedule validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp5.analyze_exp5 import load_runs
from exp5.common import LAYERS, METHODS, ROOT, SYNTHETIC_METHODS, provenance, read_config, save_json


def validate(c, runs, output):
    loaded, source = load_runs(c, runs)
    if not c["smoke"] and len(loaded) != len(c["seeds"]) * len(METHODS):
        raise ValueError("Formal experiment has the wrong run count")
    provenance_values = {json.dumps(meta["provenance"], sort_keys=True)
                         for _, meta, _, _, _, _ in loaded}
    if len(provenance_values) != 1:
        raise ValueError("Provenance differs within experiment")
    by_seed_method = {(meta["seed"], meta["method"]): (path, meta, summary, train, evals, audit)
                      for path, meta, summary, train, evals, audit in loaded}

    for path, meta, summary, train, evals, audit in loaded:
        if not meta.get("complete") or summary.get("fingerprint") != meta["fingerprint"]:
            raise ValueError(f"Incomplete or fingerprint-mismatched run: {path}")
        for column in train.columns:
            if column in ("method", "seed", "batch_hash", "noise_hash"):
                continue
            if not np.isfinite(train[column].dropna().to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite metric: {path}/{column}")
        total = meta["total_steps"]
        if train.step.tolist() != list(range(1, total + 1)):
            raise ValueError(f"Step sequence mismatch: {path}")
        expected_eval = (train.step % c["eval_interval"] == 0) | (train.step == total)
        if not train.test_accuracy.notna().equals(expected_eval) or not train.test_loss.notna().equals(expected_eval):
            raise ValueError(f"Evaluation schedule mismatch: {path}")
        if len(audit["private"]) != total:
            raise ValueError(f"Private audit length mismatch: {path}")
        for index, item in enumerate(audit["private"]):
            if item["step"] != index + 1 or item["batch_hash"] != train.iloc[index].batch_hash:
                raise ValueError(f"Private audit mismatch: {path}")
        required_state = ("preconditioner_state_bytes", "temporal_state_bytes",
                          "optimizer_state_bytes", "total_algorithm_state_bytes")
        if any(key not in summary or summary[key] < 0 for key in required_state):
            raise ValueError(f"Invalid state byte fields: {path}")
        if summary["total_algorithm_state_bytes"] != sum(summary[key] for key in required_state[:3]):
            raise ValueError(f"State byte sum mismatch: {path}")
        for key in ("wall_time", "core_wall_time", "diagnostic_seconds", "total_refresh_time", "mean_refresh_time",
                    "peak_cuda_memory_core", "peak_cuda_memory_overall", "peak_cuda_memory_allocated",
                    "peak_cuda_memory_reserved"):
            if key not in summary or not np.isfinite(summary[key]) or summary[key] < 0:
                raise ValueError(f"Invalid cost field: {path}/{key}")
        if not np.isclose(summary["core_wall_time"], summary["wall_time"] - summary["diagnostic_seconds"]):
            raise ValueError(f"Core wall accounting mismatch: {path}")

        if meta["method"] in SYNTHETIC_METHODS:
            for item in audit["synthetic"]:
                if item["count"] != c["M_syn"]:
                    raise ValueError(f"Synthetic budget mismatch: {path}")
        elif audit["synthetic"]:
            raise ValueError(f"dp_adam produced synthetic refreshes: {path}")

        refresh = pd.read_csv(Path(path) / "refresh_metrics.csv")
        oracle = pd.read_csv(Path(path) / "oracle_metrics.csv")
        for frame, name in ((refresh, "refresh"), (oracle, "oracle")):
            for column in frame.select_dtypes(include=[np.number]).columns:
                values = frame[column].dropna().to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite {name} metric: {path}/{column}")
        expected_refresh = list(range(0, total, c["K"])) if meta["method"] in SYNTHETIC_METHODS else []
        if sorted(refresh.step.unique().tolist()) != expected_refresh:
            raise ValueError(f"Refresh metrics schedule mismatch: {path}")
        if len(refresh) != len(expected_refresh) * len(LAYERS):
            raise ValueError(f"Refresh metrics row count mismatch: {path}")
        if expected_refresh and set(refresh.layer) != set(LAYERS):
            raise ValueError(f"Refresh metric layers mismatch: {path}")
        if expected_refresh and not refresh.groupby("step").layer.nunique().eq(len(LAYERS)).all():
            raise ValueError(f"Refresh metric layer coverage mismatch: {path}")
        expected_oracle = list(range(0, total, c["K"])) if c["oracle_enabled"] else []
        if sorted(oracle.step.unique().tolist()) != expected_oracle or len(oracle) != len(expected_oracle) * len(LAYERS):
            raise ValueError(f"Oracle metrics schedule mismatch: {path}")
        if expected_oracle and set(oracle.layer) != set(LAYERS):
            raise ValueError(f"Oracle metric layers mismatch: {path}")
        if expected_oracle and not oracle.groupby("step").layer.nunique().eq(len(LAYERS)).all():
            raise ValueError(f"Oracle metric layer coverage mismatch: {path}")

        beta2_on = meta["method"] in ("syn_diag_beta2", "syn_diag_beta12")
        beta2_columns = ["beta2_delta_t", "beta2_D_innovation", "beta2_D_ema"]
        if beta2_on:
            if any(column not in refresh.columns for column in beta2_columns):
                raise ValueError(f"Missing beta2 columns: {path}")
            refresh_steps = sorted(refresh.step.unique().tolist())
            first = refresh[refresh.step == refresh_steps[0]]
            if not first[beta2_columns].isna().all().all():
                raise ValueError(f"First beta2 diagnostics must be empty: {path}")
            for previous, current in zip(refresh_steps, refresh_steps[1:]):
                frame = refresh[refresh.step == current]
                if not (frame.beta2_delta_t == current - previous).all():
                    raise ValueError(f"beta2 delta_t mismatch: {path}")
                if frame[["beta2_D_innovation", "beta2_D_ema"]].isna().any().any():
                    raise ValueError(f"Missing beta2 diagnostics: {path}")
            if len(refresh_steps) > 1:
                later = refresh[refresh.step != refresh_steps[0]]
                if later[beta2_columns].isna().any().any():
                    raise ValueError(f"Missing later beta2 diagnostics: {path}")
            for item in audit["synthetic"]:
                if not item.get("q_hash") or not item.get("v_hash"):
                    raise ValueError(f"Missing beta2 q/v hash: {path}")
        elif not refresh.empty and any(column in refresh.columns for column in beta2_columns):
            if not refresh[beta2_columns].isna().all().all():
                raise ValueError(f"beta2-off run contains diagnostics: {path}")

    for seed in c["seeds"]:
        reference = by_seed_method[seed, "dp_adam"][5]["private"]
        for method in METHODS:
            current = by_seed_method[seed, method][5]["private"]
            for left, right in zip(reference, current):
                keys = ("step", "batch_hash", "batch_indices", "noise_hash", "noise_rng_before", "noise_rng_after")
                if any(left[key] != right[key] for key in keys):
                    raise ValueError(f"Private pairing mismatch: seed={seed}, method={method}")
        reference_syn = by_seed_method[seed, "syn_diag"][5]["synthetic"]
        for method in SYNTHETIC_METHODS:
            current_syn = by_seed_method[seed, method][5]["synthetic"]
            for left, right in zip(reference_syn, current_syn):
                keys = ("step", "count", "samples_hash", "labels_hash", "rng_before", "rng_after")
                if any(left[key] != right[key] for key in keys):
                    raise ValueError(f"Synthetic pairing mismatch: seed={seed}, method={method}")

    result = dict(passed=True, runs=len(loaded), methods=list(METHODS), seeds=c["seeds"],
                  steps_per_run=loaded[0][1]["total_steps"],
                  pairing="batch indices/hash, noise hash/RNG, and synthetic sample/label hashes",
                  source_matches_current_checkout=source == provenance())
    save_json(Path(output) / "validation.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    validate(read_config(args.config), args.runs, args.output or args.runs)


if __name__ == "__main__":
    main()
