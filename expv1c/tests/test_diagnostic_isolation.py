import json

import pandas as pd
import pytest

from expv1c.common import RUN_IDS
from expv1c.plot_expv1c import plot
from expv1c.summarize_expv1c import summarize
from expv1b import train_expv1b as train_module
from expv1c.train_expv1c import train
from expv1c.validate_expv1c import validate


def test_diagnostic_isolation(config, tiny_data, tmp_path):
    train(config, "dp_fisher_wiener_lr1p50", tmp_path / "on", tiny_data, diagnostics=True)
    train(config, "dp_fisher_wiener_lr1p50", tmp_path / "off", tiny_data, diagnostics=False)
    on_root = tmp_path / "on" / "seed42" / "dp_fisher_wiener_lr1p50"
    off_root = tmp_path / "off" / "seed42" / "dp_fisher_wiener_lr1p50"
    on_meta = json.loads((on_root / "summary.json").read_text())
    off_meta = json.loads((off_root / "summary.json").read_text())
    assert on_meta["final_model_hash"] == off_meta["final_model_hash"]
    assert json.loads((on_root / "pairing.json").read_text()) == json.loads((off_root / "pairing.json").read_text())


def _nonfinite_diagnose(original):
    def patched(*args, **kwargs):
        row, layers, bins = original(*args, **kwargs)
        row["relmse_filtered"] = float("inf")
        return row, layers, bins
    return patched


def test_nonfinite_diagnostics_do_not_stop_training(config, tiny_data, tmp_path, monkeypatch):
    original = train_module.diagnose
    monkeypatch.setattr(train_module, "diagnose", _nonfinite_diagnose(original))
    train(config, "dp_fisher_wiener_lr1p50", tmp_path, tiny_data)
    root = tmp_path / "seed42" / "dp_fisher_wiener_lr1p50"
    summary = json.loads((root / "summary.json").read_text())
    train_metrics = pd.read_csv(root / "train_metrics.csv")
    assert summary["status"] == "completed"
    assert summary["completed_steps"] == summary["planned_steps"] == 4
    assert isinstance(summary["final_model_hash"], str)
    assert summary["diagnostics_all_finite"] is False
    assert summary["diagnostic_nonfinite_count"] == 4
    assert summary["diagnostic_nonfinite_steps"] == [0, 1, 2, 3]
    assert train_metrics["diagnostics_finite"].tolist() == [False] * 4


def test_nonfinite_diagnostics_preserve_trajectory(config, tiny_data, tmp_path, monkeypatch):
    original = train_module.diagnose
    monkeypatch.setattr(train_module, "diagnose", _nonfinite_diagnose(original))
    train(config, "dp_fisher_wiener_lr1p50", tmp_path / "on", tiny_data, diagnostics=True)
    train(config, "dp_fisher_wiener_lr1p50", tmp_path / "off", tiny_data, diagnostics=False)
    on_root = tmp_path / "on" / "seed42" / "dp_fisher_wiener_lr1p50"
    off_root = tmp_path / "off" / "seed42" / "dp_fisher_wiener_lr1p50"
    on_summary = json.loads((on_root / "summary.json").read_text())
    off_summary = json.loads((off_root / "summary.json").read_text())
    assert on_summary["final_model_hash"] == off_summary["final_model_hash"]
    assert json.loads((on_root / "pairing.json").read_text()) == json.loads((off_root / "pairing.json").read_text())
    assert on_summary["diagnostics_all_finite"] is False
    assert off_summary["diagnostics_all_finite"] is True


def test_nonfinite_filtered_gradient_still_diverges(config, tiny_data, tmp_path, monkeypatch):
    original = train_module.apply_fisher_wiener

    def poisoned(model, active):
        original(model, active)
        next(model.parameters()).grad.fill_(float("nan"))

    monkeypatch.setattr(train_module, "apply_fisher_wiener", poisoned)
    train(config, "dp_fisher_wiener_lr1p50", tmp_path, tiny_data)
    root = tmp_path / "seed42" / "dp_fisher_wiener_lr1p50"
    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "diverged"
    assert summary["completed_steps"] == 0
    assert summary["diverged_step"] == 0


