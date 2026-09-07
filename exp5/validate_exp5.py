"""Strict Exp5 pairing and finite-metric validation."""
import argparse
import json
from pathlib import Path

import numpy as np

from exp5.analyze_exp5 import load_runs
from exp5.common import METHODS, ROOT, read_config, save_json


def validate(c, runs, output):
    loaded, _ = load_runs(c, runs)
    if not c["smoke"]:
        expected = len(c["seeds"]) * len(METHODS)
        if len(loaded) != expected:
            raise ValueError(f"Formal experiment requires {expected} runs")
    for path, meta, summary, train, evals, audit in loaded:
        for column in train.columns:
            if column in ("method", "seed", "batch_hash", "noise_hash"):
                continue
            values = train[column].dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite metric {path}/{column}")
        total = meta["total_steps"]
        expected_eval = (train.step % c["eval_interval"] == 0) | (train.step == total)
        if not train.test_accuracy.notna().equals(expected_eval):
            raise ValueError(f"Evaluation schedule mismatch: {path}")
        if meta["method"] in ("syn_diag", "syn_diag_beta1", "syn_diag_beta2", "syn_diag_beta12", "dp_kfc_adam"):
            for item in audit["synthetic"]:
                if item["count"] != c["M_syn"]:
                    raise ValueError(f"Synthetic budget mismatch: {path}")
        oracle = Path(path) / "oracle_metrics.csv"
        if not oracle.exists():
            raise ValueError(f"Missing oracle diagnostics: {path}")
    result = dict(passed=True, runs=len(loaded), methods=list(METHODS), seeds=c["seeds"],
                  steps_per_run=loaded[0][1]["total_steps"], pairing="batch/noise and synthetic sample/label hashes")
    save_json(Path(output) / "validation.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/full.json"))
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    c = read_config(args.config)
    validate(c, args.runs, args.output or args.runs)


if __name__ == "__main__":
    main()
