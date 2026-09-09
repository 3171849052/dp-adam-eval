"""Strict ExpV3 artifact validator, including lag and pairing invariants."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from opacus.accountants import RDPAccountant

from expv3.common import build_beta_interval_rows as build_interval_rows

from expv3.adaptive_fisher_wiener import selected_stats_digest, stats_digest
from expv3.common import (
    DP_METHOD,
    EXP1B_REFERENCE_ROOT,
    EXP1_REFERENCE_ROOT,
    EXP2_REFERENCE_ROOT,
    FISHER_METHOD,
    LAYERS,
    ROOT,
    check_config,
    expected_total_steps,
    fingerprint,
    output_path,
    provenance,
    read_config,
    require_pinned,
    run_specs,
    save_json,
)


def load(path):
    return json.loads(Path(path).read_text())


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return read_config(args.config), output_path(args.runs), output_path(args.output)


def _expected_epsilon(privacy_steps, noise_multiplier, sample_rate, delta):
    accountant = RDPAccountant()
    for _ in range(privacy_steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return accountant.get_epsilon(delta)


def _validate_privacy_budget(config, summary):
    if not config["smoke"]:
        assert math.isclose(summary["noise_multiplier"], 1.068115234375, rel_tol=0.0, abs_tol=1e-12)
        if summary["status"] == "completed":
            assert summary["privacy_steps"] == 1170
            assert math.isclose(summary["epsilon_spent"], 0.995693195331761, rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(summary["sample_rate"], config["batch_size"] / (config["train_subset"] or 60000))
    assert 0 <= summary["epsilon_spent"] <= config["epsilon"] + .02
    expected_epsilon = _expected_epsilon(summary["privacy_steps"], summary["noise_multiplier"],
                                         summary["sample_rate"], config["delta"])
    assert math.isclose(summary["epsilon_spent"], expected_epsilon, rel_tol=1e-9, abs_tol=1e-10)


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in ("true", "1"):
            return True
        if value.lower() in ("false", "0"):
            return False
    raise AssertionError((value, "not boolean"))


def _assert_finite_columns(frame, columns, nullable=()):
    for column in columns:
        assert column in frame.columns, column
        values = pd.to_numeric(frame[column], errors="coerce")
        missing = values.isna()
        assert column in nullable or not missing.any(), (column, "missing")
        finite = np.isfinite(values[~missing].to_numpy(dtype=float))
        assert finite.all(), (column, "non-finite")


def _assert_close(left, right, rtol=1e-6, atol=1e-10):
    assert np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float),
                       rtol=rtol, atol=atol, equal_nan=False)


def _float_class(value):
    if value is None or (isinstance(value, str) and not value.strip()):
        return "missing"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "invalid"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return "finite"


def _assert_derived(actual, expected, name, rtol=1e-6, atol=1e-10):
    assert _float_class(actual) == _float_class(expected), (name, actual, expected)
    if _float_class(expected) == "finite":
        _assert_close([actual], [expected], rtol=rtol, atol=atol)


def _compare_pairing(left, right, keys):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        for key in keys:
            assert a[key] == b[key], (key, a.get("step"))


def _expected_refresh_steps(status, completed_steps, diverged_step, total, K):
    if status == "completed":
        return list(range(0, total, K))
    assert diverged_step is not None and 0 <= diverged_step < total
    assert completed_steps in (diverged_step, diverged_step + 1)
    return list(range(0, diverged_step + 1, K))


def _validate_beta_steps(config, frame, seed, run_id, completed,
                         method=None, controller=None, oracle_enabled=True,
                         measurement_enabled=True):
    expected_pairs = ({(step, layer) for step in range(completed) for layer in LAYERS}
                      if measurement_enabled else set())
    actual_pairs = {(int(row.step), row.layer) for _, row in frame.iterrows()}
    assert len(frame) == len(expected_pairs) and actual_pairs == expected_pairs
    assert not frame.duplicated(["step", "layer"]).any()
    assert set(frame.layer) <= set(LAYERS)
    if len(frame):
        assert set(frame.seed) == {seed} and set(frame.run_id) == {run_id}
    core = ["step", "r", "dimension", "trace_A", "trace_G", "trace_F", "refresh_index",
            "noisy_gradient_energy", "expected_noise_energy",
            "noise_debiased_energy_raw"]
    _assert_finite_columns(frame, core, nullable=() if oracle_enabled else ("clean_signal_energy",))
    if len(frame):
        assert frame.trace_F.gt(0).all() and frame.dimension.gt(0).all() and frame.r.gt(0).all()
        _assert_close(frame.trace_F, frame.trace_A * frame.trace_G)
        _assert_close(frame.expected_noise_energy, frame.dimension * frame.r)
        _assert_close(frame.noise_debiased_energy_raw,
                      frame.noisy_gradient_energy - frame.expected_noise_energy)
        for _, group in frame.groupby(["refresh_index", "layer"]):
            _assert_close(group.trace_A, [group.trace_A.iloc[0]] * len(group))
            _assert_close(group.trace_G, [group.trace_G.iloc[0]] * len(group))
            _assert_close(group.trace_F, [group.trace_F.iloc[0]] * len(group))
    controller_by_key = {}
    if controller is not None:
        controller_by_key = {(int(row.interval_index), row.layer): row
                             for _, row in controller.iterrows()}
    for _, row in frame.iterrows():
        expected_raw = float(row.noise_debiased_energy_raw) / float(row.trace_F)
        expected_positive = max(expected_raw, 0.0)
        _assert_derived(row.beta_dp_step_raw, expected_raw, "beta_dp_step_raw")
        _assert_derived(row.beta_dp_step_positive, expected_positive, "beta_dp_step_positive")
        if oracle_enabled and pd.isna(row.oracle_diagnostic_error):
            assert _finite(row.clean_signal_energy)
            _assert_derived(row.beta_oracle_step,
                            float(row.clean_signal_energy) / float(row.trace_F),
                            "beta_oracle_step")
        assert _bool(row.beta_dp_step_negative) is (expected_raw < 0)
        deployable_fields = (
            "noisy_gradient_energy", "expected_noise_energy",
            "noise_debiased_energy_raw", "beta_dp_step_raw",
            "beta_dp_step_positive",
        )
        oracle_fields = ("clean_signal_energy", "beta_oracle_step")
        expected_valid = all(_finite(row[field]) for field in deployable_fields)
        if oracle_enabled:
            expected_valid = expected_valid and all(_finite(row[field]) for field in oracle_fields)
        assert _bool(row.diagnostic_valid) is expected_valid
        refresh_index = int(row.step) // int(config["K"])
        assert int(row.refresh_index) == refresh_index
        expected_source = None if method == DP_METHOD or refresh_index == 0 else refresh_index - 1
        actual_source = None if pd.isna(row.active_beta_source_interval) else int(row.active_beta_source_interval)
        assert actual_source == expected_source
        if method == DP_METHOD:
            assert pd.isna(row.active_beta_train)
        else:
            assert _finite(row.active_beta_train) and float(row.active_beta_train) > 0
            if controller is not None:
                control = controller_by_key[(refresh_index, row.layer)]
                _assert_close([row.active_beta_train], [control.beta_train])


def _validate_intervals(config, steps, intervals, method, successful_steps=None):
    expected = build_interval_rows(steps.to_dict("records"), config["beta_window"])
    assert len(intervals) == len(expected)
    if not expected:
        return
    assert not intervals.duplicated(["layer", "interval_index"]).any()
    left = intervals.sort_values(["layer", "interval_index"]).reset_index(drop=True)
    right = pd.DataFrame(expected).sort_values(["layer", "interval_index"]).reset_index(drop=True)
    derived = ("trace_F", "oracle_signal_energy_sum", "noisy_energy_sum",
               "expected_noise_energy_sum", "noise_debiased_energy_sum", "beta_oracle",
               "beta_dp_raw", "beta_dp_positive", "conditional_variance_sum",
               "estimated_signal_energy_positive", "estimated_energy_to_noise_ratio",
               "oracle_estimator_sd", "oracle_observability_snr")
    for column in derived:
        assert column in left.columns and column in right.columns
        for actual, authoritative in zip(left[column], right[column]):
            _assert_derived(actual, authoritative, column)
    for column in ("seed", "run_id", "method", "learning_rate", "interval_index",
                   "start_step", "end_step", "n_steps", "layer", "sample_count"):
        if len(left):
            assert left[column].tolist() == right[column].tolist(), column
    used = set()
    successful_steps = steps if successful_steps is None else successful_steps
    if method == FISHER_METHOD and len(successful_steps):
        used = {int(step) // int(config["K"]) - 1
                for step in successful_steps.step.unique() if int(step) // int(config["K"]) > 0}
    for _, row in left.iterrows():
        raw = row.beta_dp_raw
        assert _bool(row.raw_finite) is (_float_class(raw) == "finite")
        assert _bool(row.raw_positive) is (_finite(raw) and float(raw) > 0)
        assert _bool(row.beta_dp_negative) is (_finite(raw) and float(raw) < 0)
        assert _bool(row.was_used_by_next_interval) is (int(row.interval_index) in used)
        assert _bool(row.applied_to_training) is (int(row.interval_index) in used)
        if int(row.interval_index) not in used:
            assert pd.isna(row.next_beta_train)
        _assert_derived(row.beta_oracle_interval, row.beta_oracle, "beta_oracle_interval")
        _assert_derived(row.beta_dp_raw_interval, row.beta_dp_raw, "beta_dp_raw_interval")
    if method == DP_METHOD:
        assert not intervals.was_used_by_next_interval.any()


def _validate_controller(config, frame, interval, method, refresh, certificates):
    expected_rows = len(refresh) * len(LAYERS)
    assert len(frame) == expected_rows
    assert not frame.duplicated(["refresh_step", "layer"]).any()
    refresh_step_values = refresh["step"] if "step" in refresh else refresh["refresh_step"]
    assert {(int(row.refresh_step), row.layer) for _, row in frame.iterrows()} == {
        (int(step), layer) for step in refresh_step_values for layer in LAYERS
    }
    assert set(frame.layer) == (set(LAYERS) if expected_rows else set())
    _assert_finite_columns(frame, ["refresh_step", "interval_index", "previous_interval_n_steps",
                                  "trace_A", "trace_G", "trace_F"], nullable=())
    if "beta_algorithm_active" in frame:
        assert frame.beta_algorithm_active.map(_bool).eq(method == FISHER_METHOD).all()
    cert = {(int(row["interval_index"]), row["layer"]): row for row in certificates}
    interval_by_key = {(int(row.interval_index), row.layer): row
                       for _, row in interval.iterrows()}
    by_key = {(int(row.interval_index), row.layer): row for _, row in frame.iterrows()}
    if method == DP_METHOD:
        for _, row in frame.iterrows():
            assert pd.isna(row.beta_train) and pd.isna(row.beta_source_interval)
            assert pd.isna(row.beta_raw_previous) and pd.isna(row.numerator_previous)
            assert pd.isna(row.denominator_previous)
            assert not _bool(row.beta_update_accepted) and not _bool(row.beta_fallback_used)
            assert str(row.beta_fallback_reason) == "not_applicable"
            assert int(row.previous_interval_n_steps) == 0
            for field in ("beta_scaled_trace_F", "H_mean", "H_std", "H_min", "H_max",
                          "H_q10", "H_q25", "H_q50", "H_q75", "H_q90",
                          "H_beta1_mean", "H_beta1_q10", "H_beta1_q50", "H_beta1_q90",
                          "H_fro_ratio_vs_beta1", "H_hash", "H_hash_copy",
                          "H_beta1_hash", "H_beta1_hash_copy", "H_stats_digest",
                          "H_beta1_stats_digest"):
                assert pd.isna(row[field])
        return

    assert frame.beta_train.notna().all()
    assert frame.beta_train.gt(0).all() and np.isfinite(frame.beta_train).all()
    for _, row in frame.iterrows():
        index, layer = int(row.interval_index), row.layer
        expected_source = None if index == 0 else index - 1
        actual_source = None if pd.isna(row.beta_source_interval) else int(row.beta_source_interval)
        assert actual_source == expected_source
        if index == 0:
            assert float(row.beta_train) == 1.0
            assert pd.isna(row.beta_raw_previous) and pd.isna(row.numerator_previous)
            assert pd.isna(row.denominator_previous)
            assert int(row.previous_interval_n_steps) == 0
            assert str(row.beta_fallback_reason) == "initial"
            assert not _bool(row.beta_update_accepted) and not _bool(row.beta_fallback_used)
            assert float(row.previous_beta_train) == 1.0
            assert int(row.accepted_update_count) == 0 and int(row.fallback_count) == 0
        else:
            previous = interval_by_key[(index - 1, layer)]
            previous_controller = by_key[(index - 1, layer)]
            raw = float(previous.beta_dp_raw)
            accepted = math.isfinite(raw) and raw > 0
            expected_beta = raw if accepted else float(previous_controller.beta_train)
            _assert_close([row.beta_train], [expected_beta])
            _assert_derived(row.beta_raw_previous, previous.beta_dp_raw, "beta_raw_previous")
            _assert_derived(row.numerator_previous, previous.noise_debiased_energy_sum,
                            "numerator_previous")
            _assert_derived(row.denominator_previous, previous.trace_F, "denominator_previous")
            _assert_derived(row.previous_beta_train, previous_controller.beta_train,
                            "previous_beta_train")
            assert int(row.previous_interval_n_steps) == int(previous.n_steps)
            assert _bool(row.beta_update_accepted) is accepted
            assert _bool(row.beta_fallback_used) is (not accepted)
            expected_reason = "none" if accepted else ("negative" if math.isfinite(raw) else "nonfinite")
            assert str(row.beta_fallback_reason) == expected_reason
            accepted_count = int(previous_controller.accepted_update_count) + int(accepted)
            fallback_count = int(previous_controller.fallback_count) + int(not accepted)
            assert int(row.accepted_update_count) == accepted_count
            assert int(row.fallback_count) == fallback_count
        expected_cert = cert[(index, layer)]
        _assert_derived(row.beta_train, expected_cert["beta_train"], "certificate beta_train")
        assert row.H_hash == expected_cert["H_hash"]
        assert row.H_beta1_hash == expected_cert["H_beta1_hash"]
        lambda_A = torch.tensor(expected_cert["lambda_A"], dtype=torch.float32)
        lambda_G = torch.tensor(expected_cert["lambda_G"], dtype=torch.float32)
        lambda_F = lambda_G[:, None] * lambda_A[None, :]
        r = float(expected_cert["r"])
        beta = float(expected_cert["beta_train"])
        scaled = beta * lambda_F
        expected_H = torch.where(scaled + r > 0, scaled / (scaled + r), torch.zeros_like(scaled))
        beta1_H = torch.where(lambda_F + r > 0, lambda_F / (lambda_F + r), torch.zeros_like(lambda_F))
        from expv3.adaptive_fisher_wiener import h_hash, h_stats
        assert h_hash(expected_H) == row.H_hash
        assert h_hash(beta1_H) == row.H_beta1_hash
        if beta == 1.0:
            assert row.H_hash == row.H_beta1_hash
        expected_stats = h_stats(expected_H)
        for key, value in expected_stats.items():
            _assert_derived(row[key], value, key, rtol=1e-6, atol=1e-7)
        expected_beta1_stats = h_stats(beta1_H)
        for key, value in (("H_beta1_mean", expected_beta1_stats["H_mean"]),
                           ("H_beta1_q10", expected_beta1_stats["H_q10"]),
                           ("H_beta1_q50", expected_beta1_stats["H_q50"]),
                           ("H_beta1_q90", expected_beta1_stats["H_q90"])):
            _assert_derived(row[key], value, key, rtol=1e-6, atol=1e-7)
        expected_ratio = float(expected_H.double().norm()) / (float(beta1_H.double().norm()) + 1e-12)
        _assert_derived(row.H_fro_ratio_vs_beta1, expected_ratio, "H_fro_ratio_vs_beta1")
        assert str(row.H_hash) == str(row.H_hash_copy)
        assert str(row.H_beta1_hash) == str(row.H_beta1_hash_copy)
        current_stats = {f"H_{key}": row[f"H_{key}"] for key in (
            "mean", "std", "min", "max", "q10", "q25", "q50", "q75", "q90")}
        assert row.H_stats_digest == stats_digest(current_stats)
        compact = {"H_mean": row.H_beta1_mean, "H_q10": row.H_beta1_q10,
                   "H_q50": row.H_beta1_q50, "H_q90": row.H_beta1_q90}
        assert row.H_beta1_stats_digest == selected_stats_digest(compact)


def _validate_run(config, runs, spec, current_fp, current_provenance):
    seed = spec["seed"]
    root = runs / f"seed{seed}" / spec["run_id"]
    assert root.is_dir(), root
    required_files = ("config.json", "metadata.json", "summary.json", "pairing.json",
                      "train_metrics.csv", "layer_metrics.csv", "refresh_metrics.csv",
                      "beta_step_metrics.csv", "beta_interval_metrics.csv",
                      "beta_controller_metrics.csv", "failed_step_metrics.csv",
                      "eigenbin_metrics.csv", "h_certificates.json")
    assert all((root / name).exists() for name in required_files)
    cfg, meta, summary, pairing = [load(root / name) for name in ("config.json", "metadata.json", "summary.json", "pairing.json")]
    assert cfg == config and meta["provenance"] == current_provenance
    assert meta["fingerprint"] == summary["fingerprint"] == current_fp
    assert meta["run_id"] == summary["run_id"] == spec["run_id"]
    assert meta["method"] == summary["method"] == spec["method"]
    assert meta["learning_rate"] == summary["learning_rate"] == spec["learning_rate"]
    assert meta["complete"] is (summary["status"] == "completed")
    assert meta["actual_noise_saved"] is False and meta["private_fisher_used"] is False
    assert meta["oracle_is_research_only"] is True
    assert meta["release_safe_under_dp"] is False
    assert meta["deployable_beta_controller_is_postprocessing"] is True
    assert meta["deployable_beta_estimator_is_postprocessing"] is True
    assert meta["deployable_beta_path_reads_clean_gradient"] is False
    assert meta["oracle_path_separate_from_controller"] is True
    assert summary["deployable_beta_path_reads_clean_gradient"] is False
    assert summary["oracle_path_separate_from_controller"] is True
    assert meta["accounted_dp_mechanisms_valid"] == summary["accounted_dp_mechanisms_valid"] is True
    expected_complete = summary["status"] == "completed" or summary.get("divergence_stage") in {
        "noisy_gradient", "filtered_gradient", "optimizer_exception", "parameters"
    }
    assert meta["end_to_end_dp_accounting_complete"] == summary["end_to_end_dp_accounting_complete"]
    assert summary["end_to_end_dp_accounting_complete"] is expected_complete
    oracle_enabled = bool(meta.get("oracle_diagnostics_enabled", True))
    beta_measurement_enabled = bool(meta.get("beta_measurement_enabled", True))
    assert meta["contains_non_dp_oracle"] is oracle_enabled
    assert summary["contains_non_dp_oracle"] is oracle_enabled
    assert summary["beta_algorithm_active"] is (spec["method"] == FISHER_METHOD)
    assert summary["beta_measurement_enabled"] is beta_measurement_enabled
    total = expected_total_steps(config)
    if not config["smoke"]:
        assert total == 1170
    status, completed = summary["status"], int(summary["completed_steps"])
    assert status in ("completed", "diverged")
    assert summary["planned_steps"] == meta["total_steps"] == total
    if status == "completed":
        assert completed == total and summary["parameters_finite"]
    else:
        assert 0 <= completed <= total
        assert completed in (summary["diverged_step"], summary["diverged_step"] + 1)
    privacy = summary["privacy_steps"]
    observations = summary["beta_observation_steps"]
    stage = summary["divergence_stage"]
    for key in ("privacy_steps", "beta_observation_steps", "divergence_stage", "planned_privacy_steps"):
        assert meta[key] == summary[key]
    assert summary["planned_privacy_steps"] == total
    assert type(privacy) is int and type(observations) is int
    if status == "completed":
        assert stage is None and privacy == completed == total
    else:
        assert completed == summary["diverged_step"] < total
        assert stage in ("loss", "noisy_gradient", "filtered_gradient", "optimizer_exception", "parameters")
        assert privacy == completed + int(stage != "loss")
    expected_observations = completed + int(stage in ("filtered_gradient", "optimizer_exception", "parameters"))
    assert observations == (expected_observations if beta_measurement_enabled else 0)
    assert 0 <= observations <= privacy <= total
    assert [row["step"] for row in pairing["privacy"]] == list(range(privacy))
    keys = ("step", "batch_indices", "noise_rng_before", "noise_rng_after")
    _compare_pairing(pairing["private"], pairing["privacy"][:completed], keys)
    for index, event in enumerate(pairing["privacy"]):
        assert len(event["batch_indices"]) == config["batch_size"]
        assert len(set(event["batch_indices"])) == config["batch_size"]
        assert all(0 <= value < (config["train_subset"] or 60000) for value in event["batch_indices"])
        assert event["noise_rng_before"] != event["noise_rng_after"]
        if index:
            assert event["noise_rng_before"] == pairing["privacy"][index-1]["noise_rng_after"]
    assert all(row["parameters_finite"] for row in pairing["private"])
    if stage != "optimizer_exception":
        assert summary["parameters_finite"] is (stage != "parameters")
    if status == "completed" and completed:
        assert summary["final_model_hash"] == pairing["private"][-1]["model_hash"]
    frames = {name: pd.read_csv(root / f"{name}.csv") for name in (
        "train_metrics", "layer_metrics", "refresh_metrics", "beta_step_metrics",
        "beta_interval_metrics", "beta_controller_metrics", "failed_step_metrics",
        "eigenbin_metrics")}
    train, layer, refresh, beta_step, interval, controller, failed, eigenbin = [frames[name] for name in frames]
    assert train.step.tolist() == list(range(completed))
    expected_layer_rows = completed * len(LAYERS) if oracle_enabled else 0
    assert len(layer) == expected_layer_rows
    assert eigenbin.empty
    _validate_beta_steps(config, beta_step, seed, spec["run_id"], observations,
                         method=spec["method"], controller=controller,
                         oracle_enabled=oracle_enabled,
                         measurement_enabled=beta_measurement_enabled)
    if beta_measurement_enabled:
        _validate_intervals(config, beta_step, interval, spec["method"], train)
    else:
        assert interval.empty
    expected_failed_rows = 0 if status == "completed" or stage == "loss" else 1
    assert len(failed) == expected_failed_rows
    if expected_failed_rows:
        failed_row = failed.iloc[0]
        assert int(failed_row.step) == int(summary["diverged_step"])
        assert str(failed_row.divergence_stage) == str(stage)
        assert int(failed_row.seed) == seed and str(failed_row.run_id) == spec["run_id"]
        assert str(failed_row.method) == spec["method"]
        assert float(failed_row.learning_rate) == float(spec["learning_rate"])
    _assert_finite_columns(failed, [
        "step", "beta_observation_time", "beta_controller_time",
        "oracle_diagnostic_time", "reconstruction_diagnostic_time", "wiener_filter_time",
    ])
    if not failed.empty:
        assert (failed[[
            "beta_observation_time", "beta_controller_time", "oracle_diagnostic_time",
            "reconstruction_diagnostic_time", "wiener_filter_time",
        ]] >= 0).all().all()
        assert failed.dp_mechanism_executed.map(_bool).all()
        expected_observation_recorded = (
            beta_measurement_enabled
            and stage in ("filtered_gradient", "optimizer_exception", "parameters")
        )
        assert failed.beta_observation_recorded.map(_bool).eq(expected_observation_recorded).all()
    assert beta_step.dp_observation_recorded.map(_bool).all()
    assert beta_step.optimizer_step_completed.map(_bool).tolist() == [int(value) < completed for value in beta_step.step]
    invalid_steps = sorted(set(int(row.step) for _, row in beta_step.iterrows() if not _bool(row.diagnostic_valid)))
    assert summary["beta_diagnostic_nonfinite_steps"] == invalid_steps
    assert summary["beta_diagnostics_all_finite"] is (not invalid_steps)
    errors = sorted({int(row.step) for frame in (train, beta_step)
                     for _, row in frame.iterrows() if pd.notna(row.oracle_diagnostic_error)})
    assert summary["oracle_diagnostic_error_steps"] == errors
    if status == "diverged":
        for key in ("final_accuracy", "accuracy_auc", "late_mean_accuracy", "T50", "T70", "T80", "T85", "T90"):
            assert summary[key] is None
    expected_refresh = _expected_refresh_steps(status, completed, summary["diverged_step"], total, config["K"])
    if not meta.get("synthetic_measurement_enabled", True):
        assert spec["method"] == DP_METHOD and not beta_measurement_enabled
        expected_refresh = []
    assert refresh.step.tolist() == expected_refresh
    assert (refresh.refresh_index.to_numpy() == refresh.step.to_numpy() // config["K"]).all()
    assert [row["step"] for row in pairing["private"]] == list(range(completed))
    assert [row["step"] for row in pairing["synthetic"]] == expected_refresh
    assert refresh.step.tolist() == [row["step"] for row in pairing["synthetic"]]
    _validate_controller(config, controller, interval, spec["method"], refresh, load(root / "h_certificates.json"))
    for _, row in interval.iterrows():
        if _bool(row.applied_to_training):
            next_control = controller[(controller.interval_index == int(row.interval_index) + 1)
                                      & (controller.layer == row.layer)]
            assert len(next_control) == 1
            _assert_derived(row.next_beta_train, next_control.iloc[0].beta_train, "next_beta_train")
    assert summary["number_of_refreshes"] == len(expected_refresh)
    assert summary["noise_multiplier"] == meta["noise_multiplier"]
    _validate_privacy_budget(config, summary)
    if spec["method"] == DP_METHOD:
        assert summary["active_state_bytes"] == 0 and not meta["beta_algorithm_active"]
        assert refresh.measurement_only.all()
        assert controller.beta_train.isna().all()
        if not train.empty:
            assert (train.signal_retention == 1).all()
            assert (train.noise_retention == 1).all()
            assert (train.snr_gain_db == 0).all()
            assert (train.mse_reduction == 0).all()
    else:
        assert summary["active_state_bytes"] > 0 and meta["beta_algorithm_active"]
        assert (~refresh.measurement_only).all()
    # Numeric research diagnostics may be non-finite, but core DP fields and all
    # persisted state/counters remain finite for completed runs.
    _assert_finite_columns(train, ["step", "epoch", "train_loss", "learning_rate",
                                   "beta_observation_time", "oracle_diagnostic_time",
                                   "reconstruction_diagnostic_time", "wiener_filter_time",
                                   "beta_controller_time"],
                           nullable=("test_loss", "test_accuracy", "clip_rate"))
    _assert_finite_columns(refresh, ["step", "refresh_index", "refresh_time", "diagnostic_spectrum_time", "beta_controller_time", "active_state_bytes"])
    assert (refresh.refresh_time >= 0).all() and (refresh.beta_controller_time >= 0).all()
    assert refresh.measurement_only.map(_bool).eq(spec["method"] == DP_METHOD).all()
    assert math.isclose(summary["total_refresh_time"], refresh.refresh_time.sum(), rel_tol=1e-6, abs_tol=1e-10)
    assert math.isclose(summary["total_measurement_only_refresh_time"], refresh.loc[refresh.measurement_only, "refresh_time"].sum(), rel_tol=1e-6, abs_tol=1e-10)
    controller_time = (refresh.beta_controller_time.sum() + train.beta_controller_time.sum()
                       + failed.beta_controller_time.sum())
    assert math.isclose(summary["total_beta_controller_time"], controller_time, rel_tol=1e-6, abs_tol=1e-10)
    filter_time = train.wiener_filter_time.sum() + failed.wiener_filter_time.sum()
    spectrum_time = refresh.diagnostic_spectrum_time.sum()
    assert math.isclose(summary["total_wiener_filter_time"], filter_time,
                        rel_tol=1e-6, abs_tol=1e-10)
    assert math.isclose(summary["total_diagnostic_spectrum_time"], spectrum_time,
                        rel_tol=1e-6, abs_tol=1e-10)
    beta_observation_time = train.beta_observation_time.sum() + failed.beta_observation_time.sum()
    oracle_time = train.oracle_diagnostic_time.sum() + failed.oracle_diagnostic_time.sum()
    reconstruction_time = (train.reconstruction_diagnostic_time.sum()
                           + failed.reconstruction_diagnostic_time.sum())
    assert math.isclose(summary["total_beta_observation_time"], beta_observation_time,
                        rel_tol=1e-6, abs_tol=1e-10)
    assert math.isclose(summary["total_oracle_diagnostic_time"], oracle_time,
                        rel_tol=1e-6, abs_tol=1e-10)
    assert math.isclose(summary["total_reconstruction_diagnostic_time"], reconstruction_time,
                        rel_tol=1e-6, abs_tol=1e-10)
    assert math.isclose(summary["diagnostic_seconds"],
                        oracle_time + reconstruction_time + refresh.diagnostic_spectrum_time.sum(),
                        rel_tol=1e-6, abs_tol=1e-10)
    if spec["method"] == FISHER_METHOD:
        expected_research = oracle_time + reconstruction_time + refresh.diagnostic_spectrum_time.sum()
    else:
        expected_research = (oracle_time + reconstruction_time
                             + refresh.diagnostic_spectrum_time.sum()
                             + beta_observation_time + refresh.loc[refresh.measurement_only, "refresh_time"].sum())
    assert math.isclose(summary["research_overhead_seconds"], expected_research,
                        rel_tol=1e-6, abs_tol=1e-10)
    assert _finite(summary["core_training_runtime"]) and summary["core_training_runtime"] >= 0
    assert math.isclose(summary["core_training_runtime"],
                        summary["wall_time"] - summary["research_overhead_seconds"],
                        rel_tol=1e-6, abs_tol=1e-10)
    if status == "completed":
        expected_eval = [step for step in range(total) if (step + 1) % config["eval_interval"] == 0 or step + 1 == total]
        assert train.loc[train.test_accuracy.notna(), "step"].tolist() == expected_eval
        assert summary["final_accuracy"] is not None and summary["accuracy_auc"] is not None
    return meta, summary, pairing, train, interval


def validate(config, runs, output, require_tests=True):
    check_config(config)
    runs, output = output_path(runs), output_path(output)
    current = provenance()
    require_pinned(current, config["smoke"])
    if require_tests:
        evidence = load(ROOT / "runs" / "test_evidence.json")
        required = set(evidence.get("required_nodeids", []))
        assert evidence.get("passed") is True and evidence.get("provenance") == current
        assert required <= set(evidence.get("passed_nodeids", []))
    fp = fingerprint(config)
    specs = [dict(spec, seed=seed) for seed in config["seeds"] for spec in run_specs(config)]
    expected_seed_dirs = {f"seed{seed}" for seed in config["seeds"]}
    actual_seed_dirs = {
        path.name for path in runs.iterdir()
        if path.is_dir() and path.name.startswith("seed") and path.name[4:].isdigit()
    }
    assert actual_seed_dirs == expected_seed_dirs, (actual_seed_dirs, expected_seed_dirs)
    for seed in config["seeds"]:
        actual = sorted(path.name for path in (runs / f"seed{seed}").iterdir() if path.is_dir())
        expected = sorted(spec["run_id"] for spec in run_specs(config))
        assert actual == expected, (seed, actual, expected)
    records = []
    for spec in specs:
        records.append(_validate_run(config, runs, spec, fp, current))
    for seed in config["seeds"]:
        same_seed = [records[i] for i, spec in enumerate(specs) if spec["seed"] == seed]
        base = same_seed[0][2]
        for record in same_seed[1:]:
            pair = record[2]
            common_privacy = min(len(base["privacy"]), len(pair["privacy"]))
            _compare_pairing(base["privacy"][:common_privacy], pair["privacy"][:common_privacy],
                             ("step", "batch_indices", "noise_rng_before", "noise_rng_after"))
            common = min(len(base["private"]), len(pair["private"]))
            _compare_pairing(base["private"][:common], pair["private"][:common],
                             ("step", "batch_indices", "noise_rng_before", "noise_rng_after"))
        synthetic = same_seed[0][2]["synthetic"]
        for record in same_seed[1:]:
            common = min(len(synthetic), len(record[2]["synthetic"]))
            _compare_pairing(synthetic[:common], record[2]["synthetic"][:common],
                             ("step", "samples_hash", "labels_hash", "rng_before", "rng_after"))
    result = {
        "passed": True, "runs": len(specs), "fingerprint": fp, "provenance": current,
        "reference_artifact_available": {
            "expv1": EXP1_REFERENCE_ROOT.exists(), "expv1b": EXP1B_REFERENCE_ROOT.exists(),
            "expv2_formal": EXP2_REFERENCE_ROOT.exists(),
        },
        "single_seed_descriptive_only": len(config["seeds"]) == 1,
    }
    save_json(output / "validation.json", result)
    return result


if __name__ == "__main__":
    config, runs, output = cli()
    try:
        result = validate(config, runs, output)
    except Exception as exc:
        save_json(output / "validation.json", {"passed": False, "error": str(exc), "provenance": provenance()})
        raise
    print(f"Validation passed: {result['runs']} runs")
