import os
import time

import pytest
import torch
from torch.utils.data import TensorDataset

from expv3.common import ROOT, FULL_CONFIG, SMOKE_CONFIG, provenance, save_json

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def pytest_configure(config):
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(ROOT / "runs" / f"_pytest_{time.time_ns()}")
    config.cache._cachedir = ROOT / "runs" / "_pytest_cache"


@pytest.fixture
def config():
    value = dict(SMOKE_CONFIG)
    value["device"] = "cpu"
    return value


@pytest.fixture
def tiny_data():
    generator = torch.Generator().manual_seed(123)
    data = TensorDataset(
        torch.randn(32, 1, 28, 28, generator=generator),
        torch.randint(0, 10, (32,), generator=generator),
    )
    return data, data


REQUIRED_NODEIDS = {
    "expv3/tests/test_config.py::test_config_exactness",
    "expv3/tests/test_beta_controller.py::test_initial_beta_is_one",
    "expv3/tests/test_beta_controller.py::test_positive_previous_interval_updates_next_beta",
    "expv3/tests/test_beta_controller.py::test_negative_previous_interval_holds_beta",
    "expv3/tests/test_beta_controller.py::test_nonfinite_previous_interval_holds_beta",
    "expv3/tests/test_beta_controller.py::test_ratio_of_sums_not_mean_of_ratios",
    "expv3/tests/test_beta_controller.py::test_partial_interval_finalize",
    "expv3/tests/test_adaptive_gain.py::test_beta1_gain_equivalence",
    "expv3/tests/test_adaptive_gain.py::test_beta_monotonicity",
    "expv3/tests/test_adaptive_gain.py::test_layerwise_beta",
    "expv3/tests/test_lag_semantics.py::test_no_current_batch_leakage",
    "expv3/tests/test_lag_semantics.py::test_observation_only_affects_future_interval",
    "expv3/tests/test_privacy_order.py::test_pre_wiener_y",
    "expv3/tests/test_privacy_order.py::test_actual_noise_excluded",
    "expv3/tests/test_diagnostic_isolation.py::test_oracle_training_trajectory_isolation",
    "expv3/tests/test_diagnostic_isolation.py::test_dp_sgd_measurement_training_isolation",
    "expv3/tests/test_diagnostic_isolation.py::test_oracle_mutation_does_not_change_training",
    "expv3/tests/test_rng_pairing.py::test_paired_rng",
    "expv3/tests/test_rng_pairing.py::test_determinism",
    "expv3/tests/test_rng_pairing.py::test_training_determinism",
    "expv3/tests/test_divergence.py::test_divergence_semantics",
    "expv3/tests/test_divergence.py::test_partial_interval_not_marked_applied_without_private_step",
    "expv3/tests/test_divergence.py::test_research_diagnostic_nan_does_not_diverge",
    "expv3/tests/test_validation.py::test_validator_lag_corruption",
    "expv3/tests/test_validation.py::test_validator_h_corruption",
    "expv3/tests/test_validation.py::test_validator_rejects_step_beta_source_corruption",
    "expv3/tests/test_validation.py::test_validator_rejects_active_beta_mismatch",
    "expv3/tests/test_validation.py::test_validator_rejects_nonfinite_beta_when_expected_finite",
    "expv3/tests/test_validation.py::test_validator_rejects_nonfinite_oracle_beta_when_expected_finite",
    "expv3/tests/test_validation.py::test_validator_rejects_incorrect_positive_beta",
    "expv3/tests/test_validation.py::test_validator_rejects_interval_beta_corruption",
    "expv3/tests/test_validation.py::test_validator_rejects_beta_raw_previous_corruption",
    "expv3/tests/test_validation.py::test_validator_rejects_controller_numerator_corruption",
    "expv3/tests/test_validation.py::test_validator_rejects_controller_denominator_corruption",
    "expv3/tests/test_validation.py::test_validator_rejects_previous_beta_corruption",
    "expv3/tests/test_validation.py::test_validator_rejects_controller_count_corruption",
    "expv3/tests/test_validation.py::test_fallback_rate_excludes_initial_interval",
    "expv3/tests/test_validation.py::test_validator_rejects_extra_seed_directory",
    "expv3/tests/test_validation.py::test_adaptive_beta_observation_counts_as_core_runtime",
    "expv3/tests/test_pipeline.py::test_pipeline",
}
REQUIRED_NODEIDS.update({
    'expv3/tests/test_hardening.py::test_diverged_epsilon_matches_privacy_steps',
    'expv3/tests/test_hardening.py::test_dp_sgd_synthetic_measurement_training_isolation',
    'expv3/tests/test_hardening.py::test_filtered_gradient_divergence_consumes_privacy_step',
    'expv3/tests/test_hardening.py::test_filtered_gradient_divergence_preserves_beta_observation',
    'expv3/tests/test_hardening.py::test_parameter_divergence_preserves_consumed_events',
    'expv3/tests/test_hardening.py::test_unexpected_optimizer_exception_propagates',
    'expv3/tests/test_hardening.py::test_formal_diverged_anchor_regression',
    'expv3/tests/test_hardening.py::test_loss_divergence_does_not_consume_privacy_step[0]',
    'expv3/tests/test_hardening.py::test_loss_divergence_does_not_consume_privacy_step[2]',
    'expv3/tests/test_hardening.py::test_noisy_gradient_divergence_consumes_privacy_step',
    'expv3/tests/test_hardening.py::test_oracle_exception_does_not_change_training',
    'expv3/tests/test_hardening.py::test_raw_beta_statistics_use_interval_rows',
})
REQUIRED_NODEIDS.update({
    'expv3/tests/test_final_hardening.py::test_validator_rejects_loss_divergence_full_dp_claim',
    'expv3/tests/test_final_hardening.py::test_filtered_divergence_runtime_includes_failed_step',
    'expv3/tests/test_final_hardening.py::test_mean_wiener_filter_time_includes_failed_filter_attempt',
    'expv3/tests/test_final_hardening.py::test_loss_divergence_has_no_failed_private_timing_row',
    'expv3/tests/test_final_hardening.py::test_oracle_disabled_beta_finite_semantics_consistent',
    'expv3/tests/test_final_hardening.py::test_aggregate_step_metric_does_not_connect_seeds',
    'expv3/tests/test_final_hardening.py::test_formal_multiseed_plot_uses_unique_step_grid',
    'expv3/tests/test_final_hardening.py::test_beta_train_aggregate_uses_same_interval_across_seeds',
    'expv3/tests/test_final_hardening.py::test_beta_train_oracle_ratio_uses_current_interval_oracle',
    'expv3/tests/test_final_hardening.py::test_beta_raw_oracle_ratio_keeps_layers_separate',
    'expv3/tests/test_final_hardening.py::test_mechanism_plot_uses_equal_seed_weight',
    'expv3/tests/test_final_hardening.py::test_fallback_plot_uses_equal_seed_weight',
    'expv3/tests/test_final_hardening.py::test_h_adaptive_vs_beta1_uses_counterfactual_field',
    'expv3/tests/test_final_hardening.py::test_summary_rows_include_seed_and_status',
    'expv3/tests/test_final_hardening.py::test_lr_summary_reports_completed_denominator',
    'expv3/tests/test_final_hardening.py::test_paired_summary_reports_valid_pair_count',
})
PASSED = set()


def pytest_runtest_logreport(report):
    if report.when == "call" and report.passed:
        PASSED.add(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0 and REQUIRED_NODEIDS <= PASSED:
        save_json(ROOT / "runs" / "test_evidence.json", {
            "passed": True,
            "tests": session.testscollected,
            "required_nodeids": sorted(REQUIRED_NODEIDS),
            "passed_nodeids": sorted(PASSED),
            "provenance": provenance(),
            "isolation": "controller consumes past DP-safe scalars; oracle and measurement paths are trajectory isolated",
        })
