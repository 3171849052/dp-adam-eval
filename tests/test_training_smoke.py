import csv
import json
import yaml
import pytest
import torch
from opacus import GradSampleModule
from dp_wiener_mnist import trainer
from dp_wiener_mnist.run_logging import RunLog
from dp_wiener_mnist.model import SimpleCNN
from dp_wiener_mnist.utils import RNGStream


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
    assert (
        summary["status"] == "completed"
        and summary["planned_steps"] == 4
        and summary["privacy_steps"] == 4
        and summary["completed_steps"] == 4
    )
    for name in [
        "config.yaml",
        "resolved_config.yaml",
        "metrics.csv",
        "summary.json",
        "train.log",
    ]:
        assert (log.root / name).stat().st_size > 0
    rows = list(csv.DictReader((log.root / "metrics.csv").open()))
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1"
    assert rows[0]["global_step"] == "4"
    assert rows[0]["privacy_steps"] == "4"
    if algorithm == "dp_fisher_wiener":
        assert summary["number_of_refreshes"] == 2
        assert rows[0]["refresh_count"] == "2"
    else:
        assert rows[0]["refresh_count"] == "0"
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
    assert summary["privacy_steps"] == 2
    assert summary["completed_steps"] == 2
    assert summary["epochs_completed"] == 0
    assert len(list(csv.DictReader((log.root / "metrics.csv").open()))) == 0


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


def test_accountant_counts_noise_before_fisher_failure(monkeypatch):
    model = GradSampleModule(SimpleCNN(), loss_reduction="sum")
    x = torch.randn(2, 1, 28, 28, generator=torch.Generator().manual_seed(7))
    y = torch.tensor([1, 2])
    F = torch.nn.functional
    F.cross_entropy(model(x), y, reduction="sum").backward()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accountant = trainer.RDPAccountant()
    privacy_steps = 0

    def account():
        nonlocal privacy_steps
        accountant.step(noise_multiplier=2.0, sample_rate=0.5)
        privacy_steps += 1

    def fail(*args, **kwargs):
        raise FloatingPointError("injected Fisher failure")

    monkeypatch.setattr(trainer, "apply_fisher_wiener", fail)
    try:
        with pytest.raises(FloatingPointError):
            trainer.private_update(
                model,
                active={},
                optimizer=optimizer,
                rng=RNGStream(46, "cpu"),
                sigma=2.0,
                max_grad_norm=1.0,
                sample_count=2,
                expected_batch_size=4,
                algorithm="dp_fisher_wiener",
                on_private_mechanism=account,
            )
    finally:
        model.remove_hooks()
    assert privacy_steps == 1
    assert accountant.get_epsilon(1e-5) > 0


def test_training_summary_keeps_post_noise_privacy_cost(smoke, tiny_data, monkeypatch):
    smoke.algorithm = "dp_fisher_wiener"

    def fail(*args, **kwargs):
        raise FloatingPointError("injected Fisher failure")

    monkeypatch.setattr(trainer, "apply_fisher_wiener", fail)
    log = RunLog(smoke, yaml.safe_dump(smoke.to_dict()))
    try:
        summary = trainer.train(smoke, log, tiny_data)
    finally:
        log.close()
    assert summary["status"] == "diverged"
    assert summary["privacy_steps"] == summary["completed_steps"] + 1
    assert summary["epsilon_spent"] > 0


def test_training_executes_empty_poisson_mechanism(tmp_path, tiny_data, monkeypatch):
    c = trainer.Config()
    c.seed = 0  # seed+1=1 yields an empty first N=4, q=.25 draw
    c.runtime.device = "cpu"
    c.data.batch_size = 1
    c.data.eval_batch_size = 1
    c.data.train_subset = 4
    c.data.test_subset = 1
    c.training.epochs = 1
    c.output.root = str(tmp_path)
    observed = []
    original = trainer.clip_and_noise_gradients

    def capture(*args, **kwargs):
        observed.append(kwargs["sample_count"])
        return original(*args, **kwargs)

    # The four-sample override is intentionally used despite the fixture's
    # larger test set; build_data only consumes the matching leading samples.
    from torch.utils.data import TensorDataset

    data = (
        TensorDataset(tiny_data[0].tensors[0][:4], tiny_data[0].tensors[1][:4]),
        TensorDataset(tiny_data[1].tensors[0][:1], tiny_data[1].tensors[1][:1]),
    )
    monkeypatch.setattr(trainer, "clip_and_noise_gradients", capture)
    log = RunLog(c, yaml.safe_dump(c.to_dict()))
    try:
        summary = trainer.train(c, log, data)
    finally:
        log.close()
    assert observed[0] == 0
    assert summary["privacy_steps"] == summary["completed_steps"] == 4
