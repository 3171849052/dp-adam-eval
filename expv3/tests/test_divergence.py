import pytest
import pandas as pd

from expv3.common import run_specs, threshold_step
from expv3.train_expv3 import DivergenceError, train
from expv3.validate_expv3 import validate


def test_divergence_semantics():
    error = DivergenceError(7, "filtered gradient")
    assert error.step == 7 and error.stage == "filtered gradient"
    assert threshold_step([{"step": 0, "test_accuracy": 0.5}], 0.5) == 1
    with pytest.raises(ValueError):
        raise ValueError("research diagnostics must not be divergence")


def test_partial_interval_not_marked_applied_without_private_step(config, tiny_data, tmp_path, monkeypatch):
    import expv3.train_expv3 as trainer

    original = trainer.private_update

    def fail_at_boundary(*args, **kwargs):
        if kwargs.get("step") == config["K"]:
            raise DivergenceError(config["K"], "loss")
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "private_update", fail_at_boundary)
    for spec in run_specs(config):
        train(config, 42, spec["run_id"], tmp_path, tiny_data)
    assert validate(config, tmp_path, tmp_path, require_tests=False)["passed"]
    root = tmp_path / "seed42" / "dp_fisher_wiener_adaptive_beta_lr0p50"
    summary = pd.read_json(root / "summary.json", typ="series")
    assert summary["status"] == "diverged" and summary["diverged_step"] == config["K"]
    intervals = pd.read_csv(root / "beta_interval_metrics.csv")
    zero = intervals[intervals.interval_index == 0]
    assert not zero.was_used_by_next_interval.any()
    assert not zero.applied_to_training.any()


def test_research_diagnostic_nan_does_not_diverge(config, tiny_data, tmp_path, monkeypatch):
    import expv3.train_expv3 as trainer

    def nan_diagnose(*_args, **_kwargs):
        row = {"signal_retention": float("nan"),
               "filtered_gradient_norm": float("nan"),
               "clean_clipped_norm": float("nan")}
        return row, [dict(row, layer=layer) for layer in ("conv1", "conv2", "fc1", "fc2")], []

    monkeypatch.setattr(trainer, "diagnose", nan_diagnose)
    run = "dp_fisher_wiener_adaptive_beta_lr0p50"
    train(config, 42, run, tmp_path, tiny_data)
    summary = pd.read_json(tmp_path / "seed42" / run / "summary.json", typ="series")
    assert summary["status"] == "completed"
