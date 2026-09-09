import os

import pytest
import torch
from torch.utils.data import TensorDataset

from expv3.common import ROOT, FULL_CONFIG, SMOKE_CONFIG, provenance, save_json

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def pytest_configure(config):
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(ROOT / "runs" / "_pytest_tmp")
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
    "expv3/tests/test_diagnostic_isolation.py::test_oracle_isolation",
    "expv3/tests/test_diagnostic_isolation.py::test_dp_sgd_measurement_isolation",
    "expv3/tests/test_rng_pairing.py::test_paired_rng",
    "expv3/tests/test_rng_pairing.py::test_determinism",
    "expv3/tests/test_divergence.py::test_divergence_semantics",
    "expv3/tests/test_validation.py::test_validator_lag_corruption",
    "expv3/tests/test_validation.py::test_validator_h_corruption",
    "expv3/tests/test_pipeline.py::test_pipeline",
}
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

