"""Regression tests for the final ExpV3 artifact/reporting hardening pass."""

import json

import pandas as pd
import pytest
import torch

import expv3.plot_expv3 as plotting
import expv3.train_expv3 as trainer
from expv3.common import FISHER_METHOD, LAYERS, fingerprint, provenance, run_specs
from expv3.summarize_expv3 import build_paired_utility_rows, summarize
from expv3.validate_expv3 import _validate_run


RUN = "dp_fisher_wiener_adaptive_beta_lr0p50"


def _validate_one(config, root):
    spec = next(item for item in run_specs(config) if item["run_id"] == RUN)
    return _validate_run(config, root.parent.parent, dict(spec, seed=42),
                        fingerprint(config), provenance())


def _loss_divergence(config, tiny_data, output, monkeypatch):
    original = trainer.F.cross_entropy
    calls = 0

    def loss(*args, **kwargs):
        nonlocal calls
        value = original(*args, **kwargs)
        if kwargs.get("reduction") == "sum" and torch.is_grad_enabled():
            if calls == 0:
                value = value * float("nan")
            calls += 1
        return value

    monkeypatch.setattr(trainer.F, "cross_entropy", loss)
    trainer.train(config, 42, RUN, output, tiny_data)
    return output / "seed42" / RUN


def test_validator_rejects_loss_divergence_full_dp_claim(config, tiny_data, tmp_path, monkeypatch):
    root = _loss_divergence(config, tiny_data, tmp_path, monkeypatch)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["accounted_dp_mechanisms_valid"] is True
    assert summary["end_to_end_dp_accounting_complete"] is False
    assert pd.read_csv(root / "failed_step_metrics.csv").empty
    for name in ("summary.json", "metadata.json"):
        path = root / name
        value = json.loads(path.read_text())
        value["end_to_end_dp_accounting_complete"] = True
        path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        _validate_one(config, root)


def test_filtered_divergence_runtime_includes_failed_step(config, tiny_data, tmp_path, monkeypatch):
    original = trainer.apply_fisher_wiener
    calls = 0

    def fail(model, active):
        nonlocal calls
        original(model, active)
        if calls == config["K"]:
            next(model.parameters()).grad.fill_(float("nan"))
        calls += 1

    monkeypatch.setattr(trainer, "apply_fisher_wiener", fail)
    summary = trainer.train(config, 42, RUN, tmp_path, tiny_data)
    root = tmp_path / "seed42" / RUN
    assert summary["accounted_dp_mechanisms_valid"] is True
    assert summary["end_to_end_dp_accounting_complete"] is True
    failed = pd.read_csv(root / "failed_step_metrics.csv")
    assert len(failed) == 1
    for field in trainer.FAILED_STEP_FIELDS[6:]:
        assert float(failed.iloc[0][field]) >= 0
    train = pd.read_csv(root / "train_metrics.csv")
    refresh = pd.read_csv(root / "refresh_metrics.csv")
    assert summary["total_beta_observation_time"] == pytest.approx(
        train.beta_observation_time.sum() + failed.beta_observation_time.sum())
    assert summary["total_wiener_filter_time"] == pytest.approx(
        train.wiener_filter_time.sum() + failed.wiener_filter_time.sum())
    assert summary["total_beta_controller_time"] == pytest.approx(
        train.beta_controller_time.sum() + failed.beta_controller_time.sum()
        + refresh.beta_controller_time.sum())
    _validate_one(config, root)


def test_loss_divergence_has_no_failed_private_timing_row(config, tiny_data, tmp_path, monkeypatch):
    root = _loss_divergence(config, tiny_data, tmp_path, monkeypatch)
    failed = pd.read_csv(root / "failed_step_metrics.csv")
    assert failed.empty
    assert list(failed.columns)[:len(trainer.FAILED_STEP_FIELDS)] == trainer.FAILED_STEP_FIELDS


def test_oracle_disabled_preserves_deployable_interval_artifact(config, tiny_data, tmp_path):
    trainer.train(config, 42, RUN, tmp_path / "oracle", tiny_data, oracle_diagnostics=True)
    off = trainer.train(config, 42, RUN, tmp_path / "off", tiny_data,
                        oracle_diagnostics=False)
    normal_root = tmp_path / "oracle" / "seed42" / RUN
    off_root = tmp_path / "off" / "seed42" / RUN
    intervals = pd.read_csv(off_root / "beta_interval_metrics.csv")
    assert off["status"] == "completed" and len(intervals) > 0
    assert intervals.beta_dp_raw.notna().all()
    assert intervals.beta_oracle.isna().all()
    assert off["final_model_hash"] == json.loads((normal_root / "summary.json").read_text())["final_model_hash"]
    _validate_one(config, off_root)


def test_aggregate_step_metric_does_not_connect_seeds():
    result = plotting.aggregate_step_metric([
        pd.DataFrame({"step": [0, 1, 2], "test_loss": [1., 2., 3.]}),
        pd.DataFrame({"step": [0, 1, 2], "test_loss": [3., 4., 5.]}),
        pd.DataFrame({"step": [0, 1, 2], "test_loss": [5., 6., 7.]}),
    ], "test_loss")
    assert result.step.tolist() == [0, 1, 2]
    assert result["mean"].tolist() == [3., 4., 5.]
    assert result["count"].tolist() == [3, 3, 3]


def test_formal_multiseed_plot_uses_unique_step_grid():
    frames = [
        pd.DataFrame({"step": [0, 1, 2], "test_accuracy": [0.1, 0.2, 0.3]}),
        pd.DataFrame({"step": [0, 1, 2], "test_accuracy": [0.2, 0.3, 0.4]}),
        pd.DataFrame({"step": [0, 1, 2], "test_accuracy": [0.3, 0.4, 0.5]}),
    ]
    aggregate = plotting.aggregate_step_metric(frames, "test_accuracy")
    assert aggregate.step.is_monotonic_increasing
    assert aggregate.step.is_unique
    assert aggregate.step.tolist() == [0, 1, 2]


def test_beta_train_aggregate_uses_same_interval_across_seeds():
    result = plotting.aggregate_beta_train([
        pd.DataFrame({"interval_index": [0, 1], "beta_train": [1., 2.]}),
        pd.DataFrame({"interval_index": [0, 1], "beta_train": [1., 4.]}),
    ])
    assert result.interval_index.tolist() == [0, 1]
    assert result["median"].tolist() == [1., 3.]


def test_beta_train_oracle_ratio_uses_current_interval_oracle():
    controller = pd.DataFrame({
        "seed": [42, 42], "method": [FISHER_METHOD] * 2,
        "learning_rate": [.5, .5], "layer": ["fc2", "fc2"],
        "interval_index": [0, 1], "beta_train": [1., .2],
    })
    intervals = pd.DataFrame({
        "seed": [42, 42], "method": [FISHER_METHOD] * 2,
        "learning_rate": [.5, .5], "layer": ["fc2", "fc2"],
        "interval_index": [0, 1], "beta_oracle": [.5, .1],
    })
    result = plotting.beta_train_oracle_ratio(controller, intervals)
    assert result.loc[result.interval_index == 1, "ratio"].iloc[0] == pytest.approx(2.)


def test_h_adaptive_vs_beta1_uses_counterfactual_field():
    controller = pd.DataFrame({
        "layer": ["fc2", "fc2"], "H_q50": [.1, .1],
        "H_beta1_q50": [.4, .4],
    })
    result = plotting.aggregate_h_q50(controller)
    row = result[result.layer == "fc2"].iloc[0]
    assert row.adaptive == pytest.approx(.1)
    assert row.counterfactual == pytest.approx(.4)


def test_summary_rows_include_seed_and_status(config, tiny_data, tmp_path):
    for spec in run_specs(config):
        trainer.train(config, 42, spec["run_id"], tmp_path, tiny_data)
    summarize(config, tmp_path, tmp_path, require_tests=False)
    for name in ("summary_mechanism.csv", "summary_beta_layers.csv"):
        frame = pd.read_csv(tmp_path / name)
        assert {"seed", "run_id", "status", "completed_steps", "metric_scope"} <= set(frame)
        assert frame.status.eq("completed").all() and frame.metric_scope.eq("full").all()


def test_lr_summary_reports_completed_denominator(config, tiny_data, tmp_path):
    for spec in run_specs(config):
        trainer.train(config, 42, spec["run_id"], tmp_path, tiny_data)
    summarize(config, tmp_path, tmp_path, require_tests=False)
    frame = pd.read_csv(tmp_path / "summary_learning_rate.csv")
    assert {"n_runs", "n_completed", "n_diverged", "utility_aggregate_scope"} <= set(frame)
    assert (frame.n_runs == frame.n_completed + frame.n_diverged).all()
    assert frame.utility_aggregate_scope.eq("completed_only").all()
    assert frame.final_accuracy_mean.equals(frame.final_accuracy_mean_completed)


def test_paired_summary_reports_valid_pair_count():
    summaries = [
        {"seed": 42, "run_id": "dp_sgd_lr0p50", "status": "completed", "final_accuracy": .5,
         "accuracy_auc": .4, "late_mean_accuracy": .5},
        {"seed": 42, "run_id": RUN, "status": "completed", "final_accuracy": .6,
         "accuracy_auc": .5, "late_mean_accuracy": .6},
        {"seed": 7, "run_id": "dp_sgd_lr0p50", "status": "completed", "final_accuracy": .5,
         "accuracy_auc": .4, "late_mean_accuracy": .5},
        {"seed": 7, "run_id": RUN, "status": "completed", "final_accuracy": .7,
         "accuracy_auc": .6, "late_mean_accuracy": .7},
        {"seed": 91, "run_id": "dp_sgd_lr0p50", "status": "completed", "final_accuracy": .5,
         "accuracy_auc": .4, "late_mean_accuracy": .5},
        {"seed": 91, "run_id": RUN, "status": "diverged", "final_accuracy": None,
         "accuracy_auc": None, "late_mean_accuracy": None},
    ]
    rows = build_paired_utility_rows(summaries, [42, 7, 91])
    aggregate = [row for row in rows if row["seed"] == "aggregate"]
    assert aggregate and all(row["n_total_pairs"] == 3 and row["n_valid_pairs"] == 2
                             for row in aggregate)
