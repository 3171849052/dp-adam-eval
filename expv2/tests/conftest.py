import os

import pytest
import torch
from torch.utils.data import TensorDataset

from expv2.common import ROOT, provenance, read_config, save_json

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def pytest_configure(config):
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(ROOT / "runs" / "_pytest_tmp")
    config.cache._cachedir = ROOT / "runs" / "_pytest_cache"


@pytest.fixture
def config():
    value = read_config(ROOT / "configs" / "smoke.json")
    value["device"] = "cpu"
    return value


@pytest.fixture
def tiny_data():
    generator = torch.Generator().manual_seed(123)
    dataset = TensorDataset(
        torch.randn(32, 1, 28, 28, generator=generator),
        torch.randint(0, 10, (32,), generator=generator),
    )
    return dataset, dataset


REQUIRED_NODEIDS = {
    "expv2/tests/test_config.py::test_config_exactness",
    "expv2/tests/test_beta_math.py::test_beta_math",
    "expv2/tests/test_noise_debiasing.py::test_noise_debiasing",
    "expv2/tests/test_noise_debiasing.py::test_negative_beta_preserved",
    "expv2/tests/test_pooling.py::test_pooled_ratio_of_sums",
    "expv2/tests/test_pooling.py::test_partial_final_window",
    "expv2/tests/test_beta_math.py::test_theoretical_variance",
    "expv2/tests/test_diagnostic_isolation.py::test_diagnostic_isolation",
    "expv2/tests/test_diagnostic_isolation.py::test_beta_mutation_isolation",
    "expv2/tests/test_diagnostic_isolation.py::test_dp_sgd_synthetic_measurement_isolation",
    "expv2/tests/test_diagnostic_isolation.py::test_pre_filter_y",
    "expv2/tests/test_diagnostic_isolation.py::test_packed_dimension",
    "expv2/tests/test_pairing.py::test_paired_rng",
    "expv2/tests/test_regression.py::test_regression",
    "expv2/tests/test_pipeline.py::test_pipeline",
    "expv2/tests/test_validation_edges.py::test_validator_accepts_completed_run_with_nonfinite_beta_diagnostic",
    "expv2/tests/test_validation_edges.py::test_validator_rejects_incorrect_beta_diagnostic_flag",
    "expv2/tests/test_validation_edges.py::test_validator_rejects_inconsistent_beta_summary",
    "expv2/tests/test_validation_edges.py::test_validation_is_idempotent_after_plot",
    "expv2/tests/test_validation_edges.py::test_beta_step_cartesian_grid",
    "expv2/tests/test_validation_edges.py::test_trace_f_algebra_validation",
    "expv2/tests/test_validation_edges.py::test_refresh_index_validation",
    "expv2/tests/test_validation_edges.py::test_refresh_interval_trace_consistency",
    "expv2/tests/test_validation_edges.py::test_dp_sgd_measurement_runtime_accounting",
    "expv2/tests/test_validation_edges.py::test_invalid_seed_run_combination_rejected",
}
PASSED = set()


def pytest_runtest_logreport(report):
    if report.when == "call" and report.passed:
        PASSED.add(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0 and REQUIRED_NODEIDS <= PASSED:
        save_json(
            ROOT / "runs" / "test_evidence.json",
            dict(
                passed=True,
                tests=session.testscollected,
                required_nodeids=sorted(REQUIRED_NODEIDS),
                passed_nodeids=sorted(PASSED),
                provenance=provenance(),
                isolation="beta diagnostics and DP-SGD synthetic measurement are post-hoc and trajectory isolated",
            ),
        )
