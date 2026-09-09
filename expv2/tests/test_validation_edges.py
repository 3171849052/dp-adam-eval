import json
import math
import shutil

import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from expv2.common import ROOT, read_config, run_specs_for_seed
from expv2.plot_expv2 import plot
from expv2.summarize_expv2 import summarize
from expv2.train_expv2 import selected_run_specs, train
from expv2.validate_expv2 import validate


@pytest.fixture(scope="module")
def completed_runs(tmp_path_factory):
    config = read_config(ROOT / "configs" / "smoke.json")
    config["device"] = "cpu"
    generator = torch.Generator().manual_seed(123)
    dataset = TensorDataset(
        torch.randn(32, 1, 28, 28, generator=generator),
        torch.randint(0, 10, (32,), generator=generator),
    )
    runs = tmp_path_factory.mktemp("validation_edges")
    for spec in run_specs_for_seed(config, 42):
        train(config, 42, spec["run_id"], runs, (dataset, dataset))
    assert validate(config, runs, runs, require_tests=False)["passed"]
    return config, runs, tmp_path_factory


def _clone(completed_runs, name):
    _, source, factory = completed_runs
    destination = factory.mktemp(name) / "artifact"
    shutil.copytree(source, destination)
    for run_root in (destination / "seed42").iterdir():
        metadata_path = run_root / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["output"] = str(run_root.resolve())
        _save_json(metadata_path, metadata)
    return destination


def _run_root(runs, run_name="dp_sgd_lr0p10"):
    return runs / "seed42" / run_name


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _mark_nonfinite_beta(runs):
    root = _run_root(runs)
    beta = pd.read_csv(root / "beta_step_metrics.csv")
    beta.loc[0, "beta_dp_step_raw"] = float("inf")
    beta.loc[0, "diagnostic_valid"] = False
    beta.to_csv(root / "beta_step_metrics.csv", index=False)

    train_metrics = pd.read_csv(root / "train_metrics.csv")
    train_metrics.loc[train_metrics.step == 0, "beta_diagnostics_finite"] = False
    train_metrics.to_csv(root / "train_metrics.csv", index=False)

    for filename in ("summary.json", "metadata.json"):
        path = root / filename
        value = json.loads(path.read_text())
        value["beta_diagnostics_all_finite"] = False
        value["beta_diagnostic_nonfinite_count"] = 1
        value["beta_diagnostic_nonfinite_steps"] = [0]
        _save_json(path, value)


def test_validator_accepts_completed_run_with_nonfinite_beta_diagnostic(completed_runs):
    runs = _clone(completed_runs, "nonfinite_beta")
    _mark_nonfinite_beta(runs)
    result = validate(completed_runs[0], runs, runs, require_tests=False)
    assert result["passed"]


def test_validator_rejects_incorrect_beta_diagnostic_flag(completed_runs):
    runs = _clone(completed_runs, "incorrect_beta_flag")
    _mark_nonfinite_beta(runs)
    root = _run_root(runs)
    beta = pd.read_csv(root / "beta_step_metrics.csv")
    beta.loc[0, "diagnostic_valid"] = True
    beta.to_csv(root / "beta_step_metrics.csv", index=False)
    with pytest.raises(AssertionError):
        validate(completed_runs[0], runs, runs, require_tests=False)


def test_validator_rejects_inconsistent_beta_summary(completed_runs):
    runs = _clone(completed_runs, "incorrect_beta_summary")
    _mark_nonfinite_beta(runs)
    root = _run_root(runs)
    for filename in ("summary.json", "metadata.json"):
        path = root / filename
        value = json.loads(path.read_text())
        value["beta_diagnostics_all_finite"] = True
        _save_json(path, value)
    with pytest.raises(AssertionError):
        validate(completed_runs[0], runs, runs, require_tests=False)


def test_validation_is_idempotent_after_plot(completed_runs):
    config, source, factory = completed_runs
    runs = _clone(completed_runs, "plot_idempotence")
    assert validate(config, runs, runs, require_tests=False)["passed"]
    summarize(config, runs, runs, require_tests=False)
    plot(config, runs, runs, require_tests=False)
    assert validate(config, runs, runs, require_tests=False)["passed"]
    assert len(list((runs / "figures").glob("*.png"))) == 7
    assert len(list((runs / "figures").glob("*.pdf"))) == 7


def test_beta_step_cartesian_grid(completed_runs):
    runs = _clone(completed_runs, "missing_beta_row")
    root = _run_root(runs)
    beta = pd.read_csv(root / "beta_step_metrics.csv").iloc[1:]
    beta.to_csv(root / "beta_step_metrics.csv", index=False)
    with pytest.raises(AssertionError):
        validate(completed_runs[0], runs, runs, require_tests=False)


def test_trace_f_algebra_validation(completed_runs):
    runs = _clone(completed_runs, "bad_trace_f")
    root = _run_root(runs)
    beta = pd.read_csv(root / "beta_step_metrics.csv")
    beta.loc[0, "trace_F"] *= 2
    beta.to_csv(root / "beta_step_metrics.csv", index=False)
    with pytest.raises(AssertionError):
        validate(completed_runs[0], runs, runs, require_tests=False)


def test_refresh_index_validation(completed_runs):
    runs = _clone(completed_runs, "bad_refresh_index")
    root = _run_root(runs)
    beta = pd.read_csv(root / "beta_step_metrics.csv")
    beta.loc[0, "refresh_index"] = 1
    beta.to_csv(root / "beta_step_metrics.csv", index=False)
    with pytest.raises(AssertionError):
        validate(completed_runs[0], runs, runs, require_tests=False)


def test_refresh_interval_trace_consistency(completed_runs):
    runs = _clone(completed_runs, "bad_refresh_trace")
    root = _run_root(runs)
    beta = pd.read_csv(root / "beta_step_metrics.csv")
    beta.loc[1, "trace_A"] *= 2
    beta.loc[1, "trace_F"] = beta.loc[1, "trace_A"] * beta.loc[1, "trace_G"]
    beta.to_csv(root / "beta_step_metrics.csv", index=False)
    with pytest.raises(AssertionError):
        validate(completed_runs[0], runs, runs, require_tests=False)


def test_dp_sgd_measurement_runtime_accounting(completed_runs):
    config, runs, _ = completed_runs
    root = _run_root(runs)
    summary = json.loads((root / "summary.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text())
    refresh = pd.read_csv(root / "refresh_metrics.csv")
    for field in (
        "total_measurement_only_refresh_time", "total_beta_trace_time",
        "research_overhead_seconds", "core_training_runtime",
    ):
        assert field in summary and math.isfinite(summary[field])
        assert field in metadata and math.isfinite(metadata[field])
    assert refresh["measurement_only"].all()
    assert (refresh["beta_trace_time"] >= 0).all()
    assert summary["total_measurement_only_refresh_time"] >= 0
    assert summary["research_overhead_seconds"] >= summary["diagnostic_seconds"]
    assert summary["core_training_runtime"] <= summary["wall_time"]
    assert validate(config, runs, runs, require_tests=False)["passed"]


def test_invalid_seed_run_combination_rejected():
    config = read_config(ROOT / "configs" / "full.json")
    with pytest.raises(ValueError, match="seed 7.*dp_fisher_wiener_lr0p80"):
        selected_run_specs(config, [7], "dp_fisher_wiener_lr0p80")
    selected = selected_run_specs(config, [42], "dp_fisher_wiener_lr0p80")
    assert [(item["seed"], item["run_id"]) for item in selected] == [(42, "dp_fisher_wiener_lr0p80")]
