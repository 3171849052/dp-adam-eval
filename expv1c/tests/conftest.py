import json
import os
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from expv1c.common import ROOT, provenance, read_config, save_json

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


from expv1c.test_requirements import REQUIRED_NODEIDS
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
