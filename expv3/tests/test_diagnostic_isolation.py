import json
import math

import pandas as pd

from expv3.common import DP_METHOD, FISHER_METHOD
from expv3.train_expv3 import train


def _normalize(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def _trajectory(root):
    run = next(root.glob("seed42/*"))
    pairing = json.loads((run / "pairing.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    steps = pd.read_csv(run / "beta_step_metrics.csv")
    controller = pd.read_csv(run / "beta_controller_metrics.csv")
    step_fields = ["step", "layer", "r", "dimension", "trace_A", "trace_G", "trace_F",
                   "noisy_gradient_energy", "expected_noise_energy",
                   "noise_debiased_energy_raw", "beta_dp_step_raw",
                   "beta_dp_step_positive", "beta_dp_step_negative", "refresh_index",
                   "active_beta_train", "active_beta_source_interval"]
    controller_fields = ["interval_index", "layer", "beta_train", "beta_source_interval",
                         "beta_raw_previous", "beta_update_accepted", "beta_fallback_used",
                         "beta_fallback_reason", "H_hash", "H_beta1_hash"]
    return {
        "private": _normalize(pairing["private"]),
        "synthetic": _normalize(pairing["synthetic"]),
        "steps": _normalize(steps[step_fields].to_dict("records")),
        "controller": _normalize(controller[controller_fields].to_dict("records")),
        "model": [row["model_hash"] for row in pairing["private"]],
        "final": summary["final_model_hash"],
    }


def test_oracle_training_trajectory_isolation(config, tiny_data, tmp_path):
    run = "dp_fisher_wiener_adaptive_beta_lr0p50"
    train(config, 42, run, tmp_path / "on", tiny_data, oracle_diagnostics=True)
    train(config, 42, run, tmp_path / "off", tiny_data, oracle_diagnostics=False)
    assert _trajectory(tmp_path / "on") == _trajectory(tmp_path / "off")
    off_steps = pd.read_csv(tmp_path / "off" / "seed42" / run / "beta_step_metrics.csv")
    assert off_steps.clean_signal_energy.isna().all()


def test_oracle_mutation_does_not_change_training(config, tiny_data, tmp_path, monkeypatch):
    import expv3.train_expv3 as trainer

    run = "dp_fisher_wiener_adaptive_beta_lr0p50"
    train(config, 42, run, tmp_path / "clean", tiny_data)

    def poisoned(_model, _trace_state):
        return {
            layer: {"clean_signal_energy": value, "beta_oracle_step": value}
            for layer, value in zip(("conv1", "conv2", "fc1", "fc2"),
                                    (1e-30, 1e30, float("nan"), float("inf")))
        }

    monkeypatch.setattr(trainer, "capture_oracle_beta_diagnostics", poisoned)
    train(config, 42, run, tmp_path / "poisoned", tiny_data)
    clean, mutated = _trajectory(tmp_path / "clean"), _trajectory(tmp_path / "poisoned")
    assert clean["steps"] == mutated["steps"]
    assert clean["controller"] == mutated["controller"]
    assert clean["model"] == mutated["model"]
    assert clean["final"] == mutated["final"]


def test_dp_sgd_measurement_training_isolation(config, tiny_data, tmp_path):
    run = "dp_sgd_lr0p50"
    train(config, 42, run, tmp_path / "on", tiny_data, beta_measurement=True)
    train(config, 42, run, tmp_path / "off", tiny_data, beta_measurement=False)
    on, off = _trajectory(tmp_path / "on"), _trajectory(tmp_path / "off")
    assert on["private"] == off["private"]
    assert on["synthetic"] == off["synthetic"]
    assert on["model"] == off["model"]
    assert on["final"] == off["final"]
    assert pd.read_csv(tmp_path / "off" / "seed42" / run / "beta_step_metrics.csv").empty


def test_dp_sgd_method_identity():
    assert DP_METHOD == "dp_sgd"
    assert FISHER_METHOD != DP_METHOD
