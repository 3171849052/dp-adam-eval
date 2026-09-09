"""Validate the small ExpV4a artifact and certificate surface."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from expv3.common import save_json

from expv4a.gamma_estimators import certificate_holds
from expv4a.train_expv4a import check_config, run_specs


def _finite(frame, columns, nullable=()):
    for column in columns:
        assert column in frame.columns, column
        values = pd.to_numeric(frame[column], errors="coerce")
        if column not in nullable:
            assert not values.isna().any(), column
        assert np.isfinite(values.dropna().to_numpy(dtype=float)).all(), column


def _validate_gamma(root, spec):
    refresh = pd.read_csv(root / "gamma_refresh_metrics.csv")
    intervals = pd.read_csv(root / "gamma_interval_metrics.csv")
    assert len(refresh) > 0 and len(intervals) > 0
    assert len(refresh) == len(pd.read_csv(root / "refresh_metrics.csv")) * 4
    assert len(intervals) == len(pd.read_csv(root / "beta_interval_metrics.csv"))
    assert not refresh.duplicated(["refresh_index", "layer"]).any()
    assert not intervals.duplicated(["interval_index", "layer"]).any()
    assert set(refresh.layer) == {"conv1", "conv2", "fc1", "fc2"}
    assert set(intervals.layer) == {"conv1", "conv2", "fc1", "fc2"}
    _finite(refresh, [
        "step", "refresh_index", "interval_index", "beta_train", "trace_F", "sum_a",
        "sum_a_H2", "H_mean", "H_q50", "H_q90", "mean_H2", "sum_H2",
        "model_signal_retention",
        "gamma_model_raw", "model_noise_retention_before",
        "model_noise_retention_after_raw",
    ])
    _finite(intervals, [
        "interval_index", "n_steps", "gamma_model", "gamma_oracle",
        "model_to_oracle_ratio", "model_multiplicative_error",
        "clean_signal_energy_in", "clean_signal_energy_filtered",
        "signal_retention_oracle", "signal_retention_after_model_gamma",
        "noise_energy_in", "noise_energy_filtered", "noise_retention_oracle",
        "noise_retention_after_model_gamma", "dp_signal_energy_in",
        "dp_signal_energy_out",
    ], nullable=("gamma_dp_raw",))
    for _, row in refresh.iterrows():
        assert certificate_holds(row.gamma_model_raw, row.mean_H2)
        assert float(row.gamma_model_raw) >= 1.0 - 1e-8
        assert float(row.model_noise_retention_after_raw) <= 1.0 + 1e-8
    assert intervals.gamma_oracle.gt(0).all()
    assert intervals.gamma_model.gt(0).all()
    valid = intervals.gamma_dp_valid.astype(str).str.lower().isin(("true", "1"))
    assert valid.equals(intervals.gamma_dp_raw.notna())
    assert set(intervals.seed) == {spec["seed"]}
    assert set(intervals.run_id) == {spec["run_id"]}


def validate(config, runs, output, require_tests=True):
    check_config(config)
    runs, output = Path(runs), Path(output)
    records = []
    for seed in config["seeds"]:
        for spec in run_specs(config):
            v3spec = dict(spec, seed=seed)
            root = runs / f"seed{seed}" / spec["run_id"]
            assert (root / "v4a_config.json").exists(), root
            assert json.loads((root / "v4a_config.json").read_text()) == config
            for filename in (
                "summary.json", "metadata.json", "pairing.json",
                "refresh_metrics.csv", "layer_metrics.csv",
                "beta_step_metrics.csv", "beta_interval_metrics.csv",
                "gamma_refresh_metrics.csv", "gamma_interval_metrics.csv",
            ):
                assert (root / filename).exists(), filename
            metadata = json.loads((root / "metadata.json").read_text())
            summary = json.loads((root / "summary.json").read_text())
            assert summary["status"] == "completed"
            assert int(summary["completed_steps"]) == int(summary["privacy_steps"])
            assert int(summary["completed_steps"]) > 0
            assert summary["final_model_hash"]
            assert metadata["gamma_diagnostic_only"] is True
            assert metadata["gamma_influences_training"] is False
            assert summary["gamma_diagnostic_only"] is True
            assert summary["gamma_influences_training"] is False
            _validate_gamma(root, v3spec)
            records.append({"seed": seed, "run_id": spec["run_id"], "passed": True})
    result = {"passed": True, "runs": records, "certificate_tolerance": 1e-8}
    save_json(output / "validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = check_config(json.loads(Path(args.config).read_text()))
    validate(config, args.runs, args.output)
    print("ExpV4a validation passed", flush=True)


if __name__ == "__main__":
    main()
