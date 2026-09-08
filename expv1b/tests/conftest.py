import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from expv1b.common import ROOT, provenance, read_config, save_json

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
    "expv1b/tests/test_lr_sweep.py::test_lr_grid_exact",
    "expv1b/tests/test_pairing.py::test_paired_rng",
    "expv1b/tests/test_lr_sweep.py::test_lr_only_optimizer",
    "expv1b/tests/test_pipeline.py::test_diagnostic_isolation",
    "expv1b/tests/test_lr_sweep.py::test_effective_signal_lr",
    "expv1b/tests/test_regression.py::test_regression",
    "expv1b/tests/test_pipeline.py::test_pipeline",
    "expv1b/tests/test_validation_edges.py::test_formal_nonanchor_hash_is_not_fixed",
    "expv1b/tests/test_validation_edges.py::test_formal_nonanchor_wrongly_none_regression",
    "expv1b/tests/test_validation_edges.py::test_diverged_fisher_synthetic_pairing_uses_common_prefix",
    "expv1b/tests/test_validation_edges.py::test_completed_fisher_synthetic_pairing_requires_full_length",
    "expv1b/tests/test_validation_edges.py::test_divergence_on_refresh_step_accepts_existing_refresh",
    "expv1b/tests/test_validation_edges.py::test_completed_refresh_schedule",
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
                isolation="all canonical runs use paired private/synthetic/noise streams; diagnostics are post-hoc",
            ),
        )
