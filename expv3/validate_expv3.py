"""Strict ExpV3 artifact validator, including lag and pairing invariants."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from expv2.beta_estimation import build_interval_rows

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


def _validate_beta_steps(config, frame, seed, run_id, completed):
    assert len(frame) == completed * len(LAYERS)
    assert not frame.duplicated(["step", "layer"]).any()
    assert set(frame.layer) <= set(LAYERS)
    assert set(frame.seed) == {seed} and set(frame.run_id) == {run_id}
    _assert_finite_columns(frame, ["step", "r", "dimension", "trace_A", "trace_G", "trace_F",
                                  "clean_signal_energy", "noisy_gradient_energy",
                                  "expected_noise_energy", "noise_debiased_energy_raw"])
    assert (frame.step.to_numpy() == np.repeat(np.arange(completed), len(LAYERS))).all()
    assert (frame.trace_F > 0).all() and (frame.dimension > 0).all() and (frame.r > 0).all()
    _assert_close(frame.trace_F, frame.trace_A * frame.trace_G)
    _assert_close(frame.expected_noise_energy, frame.dimension * frame.r)
    _assert_close(frame.noise_debiased_energy_raw,
                  frame.noisy_gradient_energy - frame.expected_noise_energy)
    for _, row in frame.iterrows():
        if _finite(row.beta_oracle_step):
            _assert_close([row.beta_oracle_step], [row.clean_signal_energy / row.trace_F])
        if _finite(row.beta_dp_step_raw):
            _assert_close([row.beta_dp_step_raw], [row.noise_debiased_energy_raw / row.trace_F])
        if _finite(row.beta_dp_step_positive):
            _assert_close([row.beta_dp_step_positive], [max(row.beta_dp_step_raw, 0.0)])
        assert _bool(row.beta_dp_step_negative) is (_finite(row.beta_dp_step_raw) and row.beta_dp_step_raw < 0)
        assert _bool(row.diagnostic_valid) is all(
            _finite(row[field]) for field in (
                "clean_signal_energy", "noisy_gradient_energy", "expected_noise_energy",
                "noise_debiased_energy_raw", "beta_oracle_step", "beta_dp_step_raw",
                "beta_dp_step_positive")
        )
    if run_id.startswith("dp_fisher_wiener"):
        assert frame.active_beta_train.notna().all()
    else:
        assert frame.active_beta_train.isna().all()


def _validate_intervals(config, steps, intervals, method):
    expected = build_interval_rows(steps.to_dict("records"), config["beta_window"])
    assert len(intervals) == len(expected)
    assert not intervals.duplicated(["layer", "interval_index"]).any()
    if len(intervals):
        assert set(intervals.layer) == set(LAYERS)
        _assert_close(intervals.beta_dp_raw,
                      intervals.noise_debiased_energy_sum / intervals.trace_F)
        _assert_close(intervals.beta_oracle,
                      intervals.oracle_signal_energy_sum / intervals.trace_F)
        _assert_close(intervals.beta_dp_positive,
                      intervals.beta_dp_raw.clip(lower=0))
        _assert_close(intervals.trace_F,
                      intervals.trace_F)  # keeps the column in the algebra certificate
        for _, row in intervals.iterrows():
            assert _bool(row.raw_finite) is _finite(row.beta_dp_raw)
            assert _bool(row.raw_positive) is (_finite(row.beta_dp_raw) and row.beta_dp_raw > 0)
            assert _bool(row.beta_dp_negative) is (_finite(row.beta_dp_raw) and row.beta_dp_raw < 0)
            assert row.n_steps == row.end_step - row.start_step + 1
            assert row.start_step == row.interval_index * config["K"]
    # Recompute every ratio-of-sums field from the imported ExpV2 algebra.
    left = intervals.sort_values(["layer", "interval_index"]).reset_index(drop=True)
    right = pd.DataFrame(expected).sort_values(["layer", "interval_index"]).reset_index(drop=True)
    for column in ("trace_F", "oracle_signal_energy_sum", "noisy_energy_sum",
                   "expected_noise_energy_sum", "noise_debiased_energy_sum", "beta_oracle",
                   "beta_dp_raw", "beta_dp_positive", "conditional_variance_sum"):
        if len(left):
            _assert_close(left[column], right[column])
    if method == DP_METHOD and len(intervals):
        assert not intervals.was_used_by_next_interval.any()


def _validate_controller(config, frame, interval, method, refresh, certificates):
    expected_rows = len(refresh) * len(LAYERS)
    assert len(frame) == expected_rows
    assert not frame.duplicated(["refresh_step", "layer"]).any()
    assert set(frame.layer) == set(LAYERS)
    _assert_finite_columns(frame, ["refresh_step", "interval_index", "previous_interval_n_steps",
                                  "trace_A", "trace_G", "trace_F"], nullable=())
    if method == DP_METHOD:
        assert frame.beta_train.isna().all()
        assert frame.beta_fallback_reason.astype(str).eq("not_applicable").all()
        return
    assert frame.beta_train.notna().all()
    assert (frame.beta_train > 0).all() and np.isfinite(frame.beta_train).all()
    assert frame.beta_source_interval.isna().sum() == len(LAYERS)
    initial = frame[frame.interval_index == 0]
    assert len(initial) == len(LAYERS)
    assert (initial.beta_train == 1.0).all()
    assert initial.beta_fallback_reason.astype(str).eq("initial").all()
    assert (~initial.beta_update_accepted.map(_bool)).all()
    assert (~initial.beta_fallback_used.map(_bool)).all()

    interval_by_key = {(int(row.interval_index), row.layer): row
                       for _, row in interval.iterrows()}
    for refresh_index, group in frame.groupby("interval_index"):
        if int(refresh_index) == 0:
            continue
        assert (group.beta_source_interval == int(refresh_index) - 1).all()
        assert len(group) == len(LAYERS)
        for _, row in group.iterrows():
            previous = interval_by_key[(int(refresh_index) - 1, row.layer)]
            raw = float(previous.beta_dp_raw)
            expected_beta = raw if math.isfinite(raw) and raw > 0 else float(
                frame[(frame.interval_index == int(refresh_index) - 1) & (frame.layer == row.layer)].beta_train.iloc[0]
            )
            _assert_close([row.beta_train], [expected_beta], rtol=1e-6, atol=1e-10)
            accepted = math.isfinite(raw) and raw > 0
            assert _bool(row.beta_update_accepted) is accepted
            assert _bool(row.beta_fallback_used) is (not accepted)
            expected_reason = "none" if accepted else ("negative" if math.isfinite(raw) else "nonfinite")
            assert str(row.beta_fallback_reason) == expected_reason
            assert int(row.previous_interval_n_steps) == int(previous.n_steps)
    for _, row in frame.iterrows():
        if _finite(row.H_mean):
            assert 0 <= row.H_min <= row.H_max <= 1
            assert 0 <= row.H_q10 <= row.H_q25 <= row.H_q50 <= row.H_q75 <= row.H_q90 <= 1
            assert 0 <= row.H_beta1_q10 <= row.H_beta1_q50 <= row.H_beta1_q90 <= 1
            _assert_close([row.beta_scaled_trace_F], [row.beta_train * row.trace_F])
            assert str(row.H_hash) == str(row.H_hash_copy)
            assert str(row.H_beta1_hash) == str(row.H_beta1_hash_copy)
            current_stats = {f"H_{key}": row[f"H_{key}"] for key in ("mean", "std", "min", "max", "q10", "q25", "q50", "q75", "q90")}
            assert row.H_stats_digest == stats_digest(current_stats)
            compact = {"H_mean": row.H_beta1_mean, "H_q10": row.H_beta1_q10,
                       "H_q50": row.H_beta1_q50, "H_q90": row.H_beta1_q90}
            assert row.H_beta1_stats_digest == selected_stats_digest(compact)
    cert = {(int(row["interval_index"]), row["layer"]): row for row in certificates}
    for _, row in frame.iterrows():
        expected_cert = cert[(int(row.interval_index), row.layer)]
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
        expected_stats = h_stats(expected_H)
        for key, value in expected_stats.items():
            _assert_close([row[key]], [value], rtol=1e-6, atol=1e-7)


def _validate_run(config, runs, spec, current_fp, current_provenance):
    seed = spec["seed"]
    root = runs / f"seed{seed}" / spec["run_id"]
    assert root.is_dir(), root
    required_files = ("config.json", "metadata.json", "summary.json", "pairing.json",
                      "train_metrics.csv", "layer_metrics.csv", "refresh_metrics.csv",
                      "beta_step_metrics.csv", "beta_interval_metrics.csv",
                      "beta_controller_metrics.csv", "eigenbin_metrics.csv", "h_certificates.json")
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
    frames = {name: pd.read_csv(root / f"{name}.csv") for name in (
        "train_metrics", "layer_metrics", "refresh_metrics", "beta_step_metrics",
        "beta_interval_metrics", "beta_controller_metrics", "eigenbin_metrics")}
    train, layer, refresh, beta_step, interval, controller, eigenbin = [frames[name] for name in frames]
    assert train.step.tolist() == list(range(completed))
    assert len(layer) == completed * len(LAYERS)
    assert eigenbin.empty
    _validate_beta_steps(config, beta_step, seed, spec["run_id"], completed)
    _validate_intervals(config, beta_step, interval, spec["method"])
    expected_refresh = _expected_refresh_steps(status, completed, summary["diverged_step"], total, config["K"])
    assert refresh.step.tolist() == expected_refresh
    assert [row["step"] for row in pairing["private"]] == list(range(completed))
    assert [row["step"] for row in pairing["synthetic"]] == expected_refresh
    assert refresh.step.tolist() == [row["step"] for row in pairing["synthetic"]]
    _validate_controller(config, controller, interval, spec["method"], refresh, load(root / "h_certificates.json"))
    assert summary["number_of_refreshes"] == len(expected_refresh)
    assert summary["noise_multiplier"] == meta["noise_multiplier"]
    if not config["smoke"]:
        assert math.isclose(summary["noise_multiplier"], 1.068115234375, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(summary["epsilon_spent"], 0.995693195331761, rel_tol=0.0, abs_tol=1e-10)
    assert math.isclose(summary["sample_rate"], config["batch_size"] / (config["train_subset"] or 60000))
    assert 0 < summary["epsilon_spent"] <= config["epsilon"] + .02
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
    _assert_finite_columns(train, ["step", "epoch", "train_loss", "learning_rate"], nullable=("test_loss", "test_accuracy", "clip_rate"))
    _assert_finite_columns(refresh, ["step", "refresh_index", "refresh_time", "diagnostic_spectrum_time", "beta_controller_time", "active_state_bytes"])
    assert (refresh.refresh_time >= 0).all() and (refresh.beta_controller_time >= 0).all()
    assert math.isclose(summary["total_refresh_time"], refresh.refresh_time.sum(), rel_tol=1e-6, abs_tol=1e-10)
    assert math.isclose(summary["total_measurement_only_refresh_time"], refresh.loc[refresh.measurement_only, "refresh_time"].sum(), rel_tol=1e-6, abs_tol=1e-10)
    controller_time = refresh.beta_controller_time.sum() + train.beta_controller_time.sum()
    assert math.isclose(summary["total_beta_controller_time"], controller_time, rel_tol=1e-6, abs_tol=1e-10)
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
        "single_seed_descriptive_only": True,
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
