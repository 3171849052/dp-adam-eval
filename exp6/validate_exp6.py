"""Validate only Exp6's core protocol and pairing invariants."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp6.analyze_exp6 import load_runs
from exp6.common import METHODS, SYNTHETIC_METHODS, read_config, save_json


def validate(config, runs, output):
    records = load_runs(config, runs)
    expected = len(config["seeds"]) * len(METHODS)
    if len(records) != expected:
        raise ValueError(f"Expected {expected} runs, found {len(records)}")
    by_seed = {}
    for path, meta, summary, train in records:
        if meta["seed"] not in config["seeds"] or meta["method"] not in METHODS:
            raise ValueError(f"Run identity mismatch: {path}")
        if summary["status"] != "completed" or summary["completed_steps"] != meta["total_steps"]:
            raise ValueError(f"Incomplete run: {path}")
        if len(train) != meta["total_steps"] or train.step.tolist() != list(range(1, meta["total_steps"] + 1)):
            raise ValueError(f"Step sequence mismatch: {path}")
        numeric = train.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.dropna().to_numpy()).all():
            raise ValueError(f"Nonfinite training metric: {path}")
        pairing = json.loads((path / "pairing.json").read_text())
        expected_refresh = list(range(0, meta["total_steps"], config["K"])) \
            if meta["method"] in SYNTHETIC_METHODS else []
        if [item["step"] for item in pairing["synthetic"]] != expected_refresh:
            raise ValueError(f"Synthetic refresh schedule mismatch: {path}")
        refresh = pd.read_csv(path / "refresh_metrics.csv")
        if sorted(refresh.step.unique().tolist()) != expected_refresh:
            raise ValueError(f"Refresh metrics mismatch: {path}")
        by_seed.setdefault(meta["seed"], []).append((meta, pairing))

    for seed, arms in by_seed.items():
        initial_hashes = {meta["initial_model_hash"] for meta, _ in arms}
        if len(initial_hashes) != 1:
            raise ValueError(f"Initial model pairing mismatch: seed={seed}")
        reference = next(pairing for meta, pairing in arms if meta["method"] == "dp_adam")
        for meta, pairing in arms:
            keys = ("step", "batch_indices", "batch_hash", "noise_hash",
                    "noise_rng_before", "noise_rng_after")
            if [[item[key] for key in keys] for item in pairing["private"]] != \
               [[item[key] for key in keys] for item in reference["private"]]:
                raise ValueError(f"Private pairing mismatch: seed={seed}, method={meta['method']}")
        synthetic = [pairing for meta, pairing in arms if meta["method"] in SYNTHETIC_METHODS]
        for pairing in synthetic[1:]:
            keys = ("step", "count", "samples_hash", "labels_hash", "rng_before", "rng_after")
            if [[item[key] for key in keys] for item in pairing["synthetic"]] != \
               [[item[key] for key in keys] for item in synthetic[0]["synthetic"]]:
                raise ValueError(f"Synthetic pairing mismatch: seed={seed}")
    result = dict(passed=True, pairing_passed=True, runs=len(records),
                  seeds=config["seeds"], methods=list(METHODS))
    save_json(Path(output) / "validation.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    validate(read_config(args.config), args.runs, args.output or args.runs)


if __name__ == "__main__":
    main()
