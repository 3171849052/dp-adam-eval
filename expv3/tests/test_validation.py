import json

import pandas as pd
import pytest
import torch

from expv3.adaptive_fisher_wiener import h_hash, selected_stats_digest, stats_digest
from expv3.common import FISHER_METHOD, LAYERS, fingerprint, provenance, run_specs
from expv3.summarize_expv3 import controller_decision_rates
from expv3.train_expv3 import train
from expv3.validate_expv3 import _validate_controller, _validate_run, validate


def _controller_frame():
    rows = []
    for interval in (0, 1):
        for layer in LAYERS:
            h = .5 if interval == 0 else 1. / 3.
            stats = stats_digest({"H_mean": h, "H_std": 0., "H_min": h, "H_max": h,
                                  "H_q10": h, "H_q25": h, "H_q50": h, "H_q75": h, "H_q90": h})
            compact = selected_stats_digest({"H_mean": .5, "H_q10": .5, "H_q50": .5, "H_q90": .5})
            rows.append({
                "refresh_step": interval * 2, "interval_index": interval, "layer": layer,
                "beta_train": 1.0 if interval == 0 else .5,
                "beta_source_interval": None if interval == 0 else 0,
                "beta_raw_previous": None if interval == 0 else .5,
                "beta_update_accepted": interval == 1,
                "beta_fallback_used": False, "beta_fallback_reason": "initial" if interval == 0 else "none",
                "previous_beta_train": 1.0 if interval == 0 else 1.0,
                "numerator_previous": None if interval == 0 else 1.,
                "denominator_previous": None if interval == 0 else 2.,
                "previous_interval_n_steps": 0 if interval == 0 else 2,
                "accepted_update_count": 0 if interval == 0 else 1, "fallback_count": 0,
                "trace_A": 1., "trace_G": 1., "trace_F": 1.,
                "beta_scaled_trace_F": 1. if interval == 0 else .5,
                "H_mean": h, "H_std": 0., "H_min": h, "H_max": h,
                "H_q10": h, "H_q25": h, "H_q50": h, "H_q75": h, "H_q90": h,
                "H_beta1_mean": .5, "H_beta1_q10": .5, "H_beta1_q50": .5, "H_beta1_q90": .5,
                "H_fro_ratio_vs_beta1": 1. if interval == 0 else 2. / 3.,
                "H_hash": h_hash(torch.tensor([[h]])),
                "H_hash_copy": h_hash(torch.tensor([[h]])),
                "H_beta1_hash": h_hash(torch.tensor([[.5]])),
                "H_beta1_hash_copy": h_hash(torch.tensor([[.5]])),
                "H_stats_digest": stats, "H_beta1_stats_digest": compact,
                "beta_algorithm_active": True,
            })
    return pd.DataFrame(rows)


def _valid_inputs():
    frame = _controller_frame()
    interval = pd.DataFrame([{"interval_index": i, "layer": layer, "beta_dp_raw": .5,
                              "noise_debiased_energy_sum": 1., "trace_F": 2., "n_steps": 2}
                             for i in (0, 1) for layer in LAYERS])
    refresh = pd.DataFrame([{"refresh_step": 0, "refresh_index": 0, "measurement_only": False},
                            {"refresh_step": 2, "refresh_index": 1, "measurement_only": False}])
    certificates = [{"interval_index": i, "layer": layer,
                     "H_hash": h_hash(torch.tensor([[.5 if i == 0 else 1. / 3.]])),
                     "H_beta1_hash": h_hash(torch.tensor([[.5]])),
                     "lambda_A": [1.], "lambda_G": [1.], "r": 1.,
                     "beta_train": 1. if i == 0 else .5}
                    for i in (0, 1) for layer in LAYERS]
    return frame, interval, refresh, certificates


def test_validator_lag_corruption():
    frame, interval, refresh, certificates = _valid_inputs()
    _validate_controller({}, frame, interval, FISHER_METHOD, refresh, certificates)
    frame.loc[(frame.interval_index == 1), "beta_source_interval"] = 1
    with pytest.raises(AssertionError):
        _validate_controller({}, frame, interval, FISHER_METHOD, refresh, certificates)


def test_validator_h_corruption():
    frame, interval, refresh, certificates = _valid_inputs()
    _validate_controller({}, frame, interval, FISHER_METHOD, refresh, certificates)
    frame.loc[0, "H_hash"] = "corrupt"
    with pytest.raises(AssertionError):
        _validate_controller({}, frame, interval, FISHER_METHOD, refresh, certificates)


@pytest.fixture
def valid_adaptive_artifact(config, tiny_data, tmp_path):
    run = "dp_fisher_wiener_adaptive_beta_lr0p50"
    train(config, 42, run, tmp_path, tiny_data)
    return tmp_path, config, run


@pytest.fixture
def valid_smoke_artifact(config, tiny_data, tmp_path):
    for spec in run_specs(config):
        train(config, 42, spec["run_id"], tmp_path, tiny_data)
    return tmp_path, config


def _validate_target(artifact, config, run):
    spec = next(spec for spec in run_specs(config) if spec["run_id"] == run)
    return _validate_run(config, artifact, dict(spec, seed=42), fingerprint(config), provenance())


def _mutate_csv(root, name, mutate):
    path = root / name
    frame = pd.read_csv(path)
    mutate(frame)
    frame.to_csv(path, index=False)


def _blank_step_source(frame):
    index = frame.index[(frame.step == 3) & (frame.layer == "conv1")][0]
    frame.loc[index, "active_beta_source_interval"] = None


def _bad_active_beta(frame):
    frame.loc[0, "active_beta_train"] = 123.0


def _bad_positive(frame):
    frame.loc[0, "beta_dp_step_positive"] = 3.0


def _bad_interval_raw(frame):
    frame.loc[0, "beta_dp_raw"] += 1.0


def _bad_controller_field(column):
    def mutate(frame):
        frame.loc[frame.interval_index == 1, column] += 1.0
    return mutate


def test_validator_rejects_step_beta_source_corruption(valid_adaptive_artifact):
    artifact, config, run = valid_adaptive_artifact
    _mutate_csv(artifact / "seed42" / run, "beta_step_metrics.csv", _blank_step_source)
    with pytest.raises(AssertionError):
        _validate_target(artifact, config, run)


def test_validator_rejects_active_beta_mismatch(valid_adaptive_artifact):
    artifact, config, run = valid_adaptive_artifact
    _mutate_csv(artifact / "seed42" / run, "beta_step_metrics.csv", _bad_active_beta)
    with pytest.raises(AssertionError):
        _validate_target(artifact, config, run)


def test_validator_rejects_nonfinite_beta_when_expected_finite(valid_adaptive_artifact):
    artifact, config, run = valid_adaptive_artifact
    def mutate(frame):
        frame.loc[0, "beta_dp_step_raw"] = float("inf")
        frame.loc[0, "diagnostic_valid"] = False
    _mutate_csv(artifact / "seed42" / run, "beta_step_metrics.csv", mutate)
    with pytest.raises(AssertionError):
        _validate_target(artifact, config, run)


def test_validator_rejects_nonfinite_oracle_beta_when_expected_finite(valid_adaptive_artifact):
    artifact, config, run = valid_adaptive_artifact
    path = artifact / "seed42" / run / "beta_step_metrics.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "beta_oracle_step"] = float("nan")
    frame.loc[0, "diagnostic_valid"] = False
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        _validate_target(artifact, config, run)


def test_validator_rejects_incorrect_positive_beta(valid_adaptive_artifact):
    artifact, config, run = valid_adaptive_artifact
    _mutate_csv(artifact / "seed42" / run, "beta_step_metrics.csv", _bad_positive)
    with pytest.raises(AssertionError):
        _validate_target(artifact, config, run)


def test_validator_rejects_interval_beta_corruption(valid_adaptive_artifact):
    artifact, config, run = valid_adaptive_artifact
    _mutate_csv(artifact / "seed42" / run, "beta_interval_metrics.csv", _bad_interval_raw)
    with pytest.raises(AssertionError):
        _validate_target(artifact, config, run)


def test_validator_rejects_beta_raw_previous_corruption(valid_adaptive_artifact):
    artifact, config, run = valid_adaptive_artifact
    _mutate_csv(artifact / "seed42" / run, "beta_controller_metrics.csv",
                _bad_controller_field("beta_raw_previous"))
    with pytest.raises(AssertionError):
        _validate_target(artifact, config, run)


def test_validator_rejects_controller_numerator_corruption(valid_adaptive_artifact):
    artifact, config, run = valid_adaptive_artifact
    _mutate_csv(artifact / "seed42" / run, "beta_controller_metrics.csv",
                _bad_controller_field("numerator_previous"))
    with pytest.raises(AssertionError):
        _validate_target(artifact, config, run)


def test_validator_rejects_controller_denominator_corruption(valid_adaptive_artifact):
    artifact, config, run = valid_adaptive_artifact
    _mutate_csv(artifact / "seed42" / run, "beta_controller_metrics.csv",
                _bad_controller_field("denominator_previous"))
    with pytest.raises(AssertionError):
        _validate_target(artifact, config, run)


def test_validator_rejects_previous_beta_corruption(valid_adaptive_artifact):
    artifact, config, run = valid_adaptive_artifact
    _mutate_csv(artifact / "seed42" / run, "beta_controller_metrics.csv",
                _bad_controller_field("previous_beta_train"))
    with pytest.raises(AssertionError):
        _validate_target(artifact, config, run)


def test_validator_rejects_controller_count_corruption(valid_adaptive_artifact):
    artifact, config, run = valid_adaptive_artifact
    _mutate_csv(artifact / "seed42" / run, "beta_controller_metrics.csv",
                _bad_controller_field("accepted_update_count"))
    with pytest.raises(AssertionError):
        _validate_target(artifact, config, run)


def test_fallback_rate_excludes_initial_interval():
    frame = pd.DataFrame({
        "method": [FISHER_METHOD] * 3,
        "interval_index": [0, 1, 2],
        "beta_fallback_used": [False, True, False],
        "beta_update_accepted": [False, False, True],
    })
    fallback, accepted = controller_decision_rates(frame)
    assert fallback == 0.5 and accepted == 0.5


def test_validator_rejects_extra_seed_directory(valid_smoke_artifact):
    artifact, config = valid_smoke_artifact
    (artifact / "figures").mkdir()
    assert validate(config, artifact, artifact, require_tests=False)["passed"]
    (artifact / "seed999").mkdir()
    with pytest.raises(AssertionError):
        validate(config, artifact, artifact, require_tests=False)


def test_adaptive_beta_observation_counts_as_core_runtime(valid_smoke_artifact):
    artifact, config = valid_smoke_artifact
    adaptive = json.loads((artifact / "seed42" / "dp_fisher_wiener_adaptive_beta_lr0p50" / "summary.json").read_text())
    dp = json.loads((artifact / "seed42" / "dp_sgd_lr0p50" / "summary.json").read_text())
    assert adaptive["total_beta_observation_time"] >= 0
    assert dp["total_beta_observation_time"] >= 0
    for summary in (adaptive, dp):
        assert summary["core_training_runtime"] >= 0
        assert summary["research_overhead_seconds"] >= 0
    assert adaptive["research_overhead_seconds"] == pytest.approx(
        adaptive["total_oracle_diagnostic_time"]
        + adaptive["total_reconstruction_diagnostic_time"]
        + adaptive["total_diagnostic_spectrum_time"]
    )
    assert dp["research_overhead_seconds"] == pytest.approx(
        dp["total_oracle_diagnostic_time"]
        + dp["total_reconstruction_diagnostic_time"]
        + dp["total_diagnostic_spectrum_time"]
        + dp["total_beta_observation_time"]
        + dp["total_measurement_only_refresh_time"]
    )
