import os
from pathlib import Path
import pytest
from expv1.common import ROOT, read_config, provenance, save_json

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def pytest_configure(config):
    config.option.basetemp = str(ROOT/'runs/_pytest_tmp')
    config.cache._cachedir = ROOT/'runs/_pytest_cache'


@pytest.fixture
def config():
    c=read_config(ROOT/'configs/smoke.json')
    c['device']='cpu'
    return c


PASSED = set()


def pytest_runtest_logreport(report):
    if report.when == 'call' and report.passed:
        PASSED.add(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    required = {f'expv1/tests/test_pipeline.py::test_diagnostic_trajectory_isolation[{m}]'
                for m in ('dp_sgd', 'dp_scalar_wiener', 'dp_fisher_wiener')}
    if exitstatus == 0 and required <= PASSED and session.testscollected >= 23:
        save_json(ROOT/'runs/test_evidence.json',dict(passed=True,tests=session.testscollected,
            provenance=provenance(), passed_nodeids=sorted(PASSED), isolation='all methods diagnostics on/off and eigen budget trajectories'))
