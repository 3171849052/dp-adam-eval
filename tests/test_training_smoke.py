import csv
import json
import yaml
import pytest
from dp_wiener_mnist import trainer
from dp_wiener_mnist.run_logging import RunLog


@pytest.mark.parametrize("algorithm", ["dp_sgd", "dp_fisher_wiener"])
def test_training(smoke, tiny_data, monkeypatch, algorithm):
    smoke.algorithm = algorithm
    if algorithm == "dp_sgd":

        def forbidden(*a, **kw):
            raise AssertionError("Wiener path reached by DP-SGD")

        for name in [
            "synthetic_samples",
            "build_covariances",
            "build_fisher_state",
            "apply_fisher_wiener",
        ]:
            monkeypatch.setattr(trainer, name, forbidden)
    log = RunLog(smoke, yaml.safe_dump(smoke.to_dict()))
    try:
        summary = trainer.train(smoke, log, tiny_data)
    finally:
        log.close()
    assert summary["status"] == "completed" and summary["completed_steps"] == 4
    for name in [
        "config.yaml",
        "resolved_config.yaml",
        "metrics.csv",
        "summary.json",
        "train.log",
    ]:
        assert (log.root / name).stat().st_size > 0
    rows = list(csv.DictReader((log.root / "metrics.csv").open()))
    assert len(rows) == 4
    assert [r["filter_refresh"] for r in rows] == (
        ["True", "False", "True", "False"]
        if algorithm == "dp_fisher_wiener"
        else ["False"] * 4
    )
    if algorithm == "dp_fisher_wiener":
        assert summary["number_of_refreshes"] == 2
    assert json.loads((log.root / "summary.json").read_text()) == summary


def test_divergence_retains_partial_metrics(smoke, tiny_data, monkeypatch):
    original = trainer.private_update
    calls = 0

    def diverge(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise FloatingPointError("injected non-finite gradient")
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "private_update", diverge)
    log = RunLog(smoke, yaml.safe_dump(smoke.to_dict()))
    try:
        summary = trainer.train(smoke, log, tiny_data)
    finally:
        log.close()
    assert summary["status"] == "diverged" and summary["diverged_step"] == 2
    assert summary["completed_steps"] == 2
    assert len(list(csv.DictReader((log.root / "metrics.csv").open()))) == 2


def test_failure_summary(smoke, monkeypatch):
    def fail(*a, **kw):
        raise RuntimeError("dataset unavailable")

    monkeypatch.setattr(trainer, "build_data", fail)
    log = RunLog(smoke, yaml.safe_dump(smoke.to_dict()))
    try:
        with pytest.raises(RuntimeError):
            trainer.train(smoke, log)
    finally:
        log.close()
    summary = json.loads((log.root / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["completed_steps"] == 0
