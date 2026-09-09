"""Strict ExpV2 artifact, algebra, pairing, and fixed-reference validation."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from expv2.beta_estimation import (
    beta_step_diagnostic_is_finite,
    build_hold_predictor,
    build_interval_rows,
    build_lagged_rows,
    window_sensitivity,
)
from expv2.common import (
    EXP1B_REFERENCE_ROOT,
    EXP1_REFERENCE_ROOT,
    LAYERS,
    ROOT,
    check_config,
    expected_total_steps,
    fingerprint,
    output_path,
    provenance,
    require_pinned,
    run_specs_for_seed,
    save_json,
)


ANCHORS = {
    (42, "dp_sgd_lr0p10"): "9e9c9e76e84680f6fcb1c34ded5047f73f36b54c8616fb14293a470270fce1b2",
    (42, "dp_fisher_wiener_lr0p10"): "c04595c804fe9f04bd3421385e6e75e920bc288e9920c486fa0929c009ed6aa7",
    (42, "dp_fisher_wiener_lr0p80"): "38c1dfe4ca6ad65b49598e8a327edaf81f72873ff5a1df28f5884d34028a1b33",
}

RELEASE_FLAGS = {
    "contains_non_dp_oracle": True,
    "release_safe_under_dp": False,
    "deployable_beta_estimator_is_postprocessing": True,
    "oracle_is_research_only": True,
}

BETA_CORE_FIELDS = (
    "step", "r", "dimension", "trace_A", "trace_G", "trace_F",
    "clean_signal_energy", "noisy_gradient_energy", "expected_noise_energy",
    "noise_debiased_energy_raw",
)
INTERVAL_CORE_FIELDS = (
    "interval_index", "start_step", "end_step", "n_steps", "trace_F",
    "oracle_signal_energy_sum", "noisy_energy_sum", "expected_noise_energy_sum",
    "noise_debiased_energy_sum", "conditional_variance_sum", "sample_count",
)


def load(path):
    return json.loads(Path(path).read_text())


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from expv2.common import read_config
    return read_config(args.config), output_path(args.runs), output_path(args.output)


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _assert_finite(frame, allowed_missing=()):
    """Require numeric columns to be finite, with explicit nullable columns."""
    for column in frame.select_dtypes(include="number"):
        missing = frame[column].isna()
        assert column in allowed_missing or not missing.any(), (column, "unexpected missing")
        assert frame.loc[~missing, column].map(math.isfinite).all(), (column, "non-finite")


def _assert_core_finite(frame, columns):
    for column in columns:
        assert column in frame.columns
        values = pd.to_numeric(frame[column], errors="coerce")
        assert not values.isna().any(), (column, "unexpected missing")
        assert np.isfinite(values.to_numpy(dtype=float)).all(), (column, "non-finite")


def _assert_integer(frame, columns):
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        assert not values.isna().any(), (column, "unexpected missing")
        numeric = values.to_numpy(dtype=float)
        assert np.equal(numeric, np.floor(numeric)).all(), (column, "not integral")


def _assert_boolean(frame, column):
    assert column in frame.columns
    assert not frame[column].isna().any(), (column, "unexpected missing")
    assert frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).all(), (
        column, "expected boolean"
    )


def _assert_close(actual, expected, *, rtol=1e-6, atol=1e-10):
    assert np.allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
        rtol=rtol, atol=atol, equal_nan=False,
    )


def _compare_optional_numeric(actual, expected, column, *, rtol=1e-5, atol=1e-5):
    """Compare finite derived values and permit a non-finite actual diagnostic."""
    left = pd.to_numeric(actual[column], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float)
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    # A finite artifact value cannot replace a non-finite value expected from
    # the recomputation. Non-finite research values themselves are permitted.
    assert not (left_finite & ~right_finite).any(), column
    if left_finite.any():
        assert right_finite[left_finite].all(), column
        _assert_close(left[left_finite], right[left_finite], rtol=rtol, atol=atol)


def _compare_expected_rows(actual, expected_rows, sort_columns, exact_columns, numeric_columns):
    expected = pd.DataFrame(expected_rows)
    if expected.empty and not expected_rows:
        expected = pd.DataFrame(columns=actual.columns)
    assert len(actual) == len(expected)
    left = actual.sort_values(sort_columns).reset_index(drop=True)
    right = expected.sort_values(sort_columns).reset_index(drop=True)
    assert left[exact_columns].to_dict("records") == right[exact_columns].to_dict("records")
    for column in numeric_columns:
        _compare_optional_numeric(left, right, column)


def _compare_pairing(reference, actual, keys):
    assert len(reference) == len(actual)
    for left, right in zip(reference, actual):
        for key in keys:
            assert left[key] == right[key], (key, left.get("step"))


def _reference_root(seed, run_name):
    if run_name == "dp_sgd_lr0p10":
        return EXP1_REFERENCE_ROOT / f"seed{seed}" / "dp_sgd"
    if run_name == "dp_fisher_wiener_lr0p10":
        return EXP1_REFERENCE_ROOT / f"seed{seed}" / "dp_fisher_wiener"
    if run_name == "dp_fisher_wiener_lr0p80":
        return EXP1B_REFERENCE_ROOT / "seed42" / run_name
    raise ValueError(run_name)


def _validate_fixed_reference(config, runs):
    result = {"checked": False, "references": {}}
    if config["smoke"]:
        result["available"] = all(_reference_root(42, name).exists() for name in (
            "dp_sgd_lr0p10", "dp_fisher_wiener_lr0p10", "dp_fisher_wiener_lr0p80"))
        return result
    for seed in config["seeds"]:
        for spec in run_specs_for_seed(config, seed):
            name = spec["run_id"]
            reference = _reference_root(seed, name)
            assert reference.exists(), f"missing fixed reference: {reference}"
            actual = runs / f"seed{seed}" / name
            reference_summary = load(reference / "summary.json")
            actual_summary = load(actual / "summary.json")
            assert actual_summary["final_model_hash"] == reference_summary["final_model_hash"]
            assert actual_summary["completed_steps"] == reference_summary["completed_steps"]
            actual_pair = load(actual / "pairing.json")
            reference_pair = load(reference / "pairing.json")
            _compare_pairing(
                reference_pair["private"], actual_pair["private"],
                ("step", "batch_indices", "noise_rng_before", "noise_rng_after", "model_hash"),
            )
            if spec["method"] == "dp_fisher_wiener":
                _compare_pairing(
                    reference_pair["synthetic"], actual_pair["synthetic"],
                    ("step", "samples_hash", "labels_hash", "rng_before", "rng_after"),
                )
            result["references"][f"seed{seed}/{name}"] = str(reference)
    result["checked"] = True
    return result


def _validate_beta_step(config, frame, seed, run_name, total):
    required = {
        "step", "layer", "r", "dimension", "trace_A", "trace_G", "trace_F",
        "clean_signal_energy", "noisy_gradient_energy", "expected_noise_energy",
        "noise_debiased_energy_raw", "beta_oracle_step", "beta_dp_step_raw",
        "beta_dp_step_positive", "beta_dp_step_negative", "refresh_index", "diagnostic_valid",
    }
    assert required <= set(frame.columns)
    assert len(frame) == total * len(LAYERS)
    _assert_integer(frame, ("step", "dimension", "refresh_index"))
    _assert_boolean(frame, "beta_dp_step_negative")
    _assert_boolean(frame, "diagnostic_valid")

    expected_pairs = {(step, layer) for step in range(total) for layer in LAYERS}
    actual_pairs = {(int(row.step), row.layer) for row in frame.itertuples()}
    assert actual_pairs == expected_pairs
    assert not frame.duplicated(["step", "layer"]).any()
    assert set(frame.layer) == set(LAYERS)
    assert set(frame.seed) == {seed} and set(frame.run_id) == {run_name}
    assert (frame.dimension > 0).all() and (frame.r > 0).all()

    _assert_core_finite(frame, (
        "step", "r", "dimension", "trace_A", "trace_G", "trace_F",
        "clean_signal_energy", "noisy_gradient_energy", "expected_noise_energy",
        "noise_debiased_energy_raw",
    ))
    assert (frame.trace_F > 0).all()
    _assert_close(frame.trace_F, frame.trace_A * frame.trace_G)
    _assert_close(frame.expected_noise_energy, frame.dimension * frame.r, atol=1e-10)
    _assert_close(
        frame.noise_debiased_energy_raw,
        frame.noisy_gradient_energy - frame.expected_noise_energy,
    )
    assert (frame.refresh_index == frame.step // config["K"]).all()

    for row in frame.to_dict("records"):
        expected_valid = beta_step_diagnostic_is_finite(row)
        assert bool(row["diagnostic_valid"]) is expected_valid
        if _finite(row["beta_oracle_step"]):
            _assert_close(
                [row["beta_oracle_step"]],
                [row["clean_signal_energy"] / row["trace_F"]],
            )
        if _finite(row["beta_dp_step_raw"]):
            _assert_close(
                [row["beta_dp_step_raw"]],
                [row["noise_debiased_energy_raw"] / row["trace_F"]],
            )
            if _finite(row["beta_dp_step_positive"]):
                _assert_close(
                    [row["beta_dp_step_positive"]],
                    [max(row["beta_dp_step_raw"], 0.0)],
                )
        if _finite(row["beta_dp_step_raw"]):
            assert bool(row["beta_dp_step_negative"]) is (float(row["beta_dp_step_raw"]) < 0)

    for (layer, refresh_index), group in frame.groupby(["layer", "refresh_index"]):
        for column in ("trace_A", "trace_G", "trace_F"):
            _assert_close(group[column], [group[column].iloc[0]] * len(group))


def _validate_intervals(config, step_frame, interval_frame, lagged_frame, hold_frame):
    expected = build_interval_rows(step_frame.to_dict("records"), config["beta_primary_window"])
    assert len(interval_frame) == len(expected)
    assert not interval_frame.duplicated(["layer", "interval_index"]).any()
    _assert_integer(interval_frame, ("interval_index", "start_step", "end_step", "n_steps", "sample_count"))
    _assert_boolean(interval_frame, "beta_dp_negative")
    _assert_core_finite(interval_frame, (
        "interval_index", "start_step", "end_step", "n_steps", "trace_F",
        "oracle_signal_energy_sum", "noisy_energy_sum", "expected_noise_energy_sum",
        "noise_debiased_energy_sum", "conditional_variance_sum", "sample_count",
    ))

    _compare_expected_rows(
        interval_frame, expected, ["layer", "interval_index"],
        ["seed", "run_id", "method", "learning_rate", "interval_index", "start_step",
         "end_step", "n_steps", "layer", "sample_count", "beta_dp_negative"],
        ["trace_F", "oracle_signal_energy_sum", "noisy_energy_sum", "expected_noise_energy_sum",
         "noise_debiased_energy_sum", "beta_oracle", "beta_dp_raw", "beta_dp_positive",
         "estimated_signal_energy_positive", "estimated_energy_to_noise_ratio",
         "oracle_estimator_sd", "oracle_observability_snr", "conditional_variance_sum"],
    )

    total = int(step_frame.step.max()) + 1
    assert config["beta_primary_window"] == config["K"]
    for layer, group in interval_frame.groupby("layer"):
        group = group.sort_values("interval_index")
        for row in group.itertuples():
            start = int(row.interval_index) * config["K"]
            n_steps = min(config["K"], total - start)
            assert row.start_step == start
            assert row.end_step == start + n_steps - 1
            assert row.n_steps == n_steps

    expected_lag = build_lagged_rows(expected)
    expected_hold = build_hold_predictor(expected)
    assert len(lagged_frame) == len(expected_lag) and len(hold_frame) == len(expected_hold)
    _assert_integer(lagged_frame, ("source_interval", "target_interval"))
    _assert_integer(hold_frame, ("source_interval", "target_interval"))
    _assert_boolean(lagged_frame, "lag_prediction_positive")
    _assert_boolean(hold_frame, "hold_prediction_positive")
    assert (lagged_frame.target_interval - lagged_frame.source_interval == 1).all()
    assert (hold_frame.target_interval - hold_frame.source_interval == 1).all()
    _compare_expected_rows(
        lagged_frame, expected_lag, ["layer", "target_interval"],
        ["seed", "run_id", "method", "learning_rate", "layer", "source_interval",
         "target_interval", "lag_prediction_positive"],
        ["beta_dp_previous_raw", "beta_oracle_current", "lag_ratio", "lag_log10_abs_ratio_error"],
    )
    _compare_expected_rows(
        hold_frame, expected_hold, ["layer", "target_interval"],
        ["seed", "run_id", "method", "learning_rate", "layer", "source_interval",
         "target_interval", "hold_prediction_positive"],
        ["beta_hold_predictor", "beta_oracle_current", "hold_ratio", "hold_log10_abs_ratio_error"],
    )


def _validate_window_sensitivity(config, step_frame, window_frame):
    expected = window_sensitivity(step_frame.to_dict("records"), config["beta_window_sensitivity"])
    assert len(window_frame) == len(expected)
    assert set(window_frame.window_size) == set(config["beta_window_sensitivity"])
    assert not window_frame.duplicated(["window_size", "layer", "window_index"]).any()
    _assert_integer(window_frame, ("window_size", "window_index", "n_steps"))
    _assert_boolean(window_frame, "negative")
    _assert_core_finite(window_frame, ("trace_F",))
    _compare_expected_rows(
        window_frame, expected, ["window_size", "layer", "window_index"],
        ["window_size", "seed", "run_id", "method", "learning_rate", "layer",
         "window_index", "n_steps", "negative"],
        ["beta_oracle", "beta_dp_raw", "log_error", "trace_F", "oracle_observability_snr"],
    )


def _validate_runtime(summary, refresh):
    _assert_core_finite(
        pd.DataFrame([summary]),
        ("wall_time", "diagnostic_seconds", "research_overhead_seconds",
         "core_training_runtime", "total_refresh_time", "total_measurement_only_refresh_time",
         "total_beta_trace_time", "fisher_beta_trace_time"),
    )
    refresh_time = float(refresh.refresh_time.sum())
    measurement_only = float(refresh.loc[refresh.measurement_only, "refresh_time"].sum())
    beta_trace = float(refresh.beta_trace_time.sum())
    fisher_trace = float(refresh.loc[~refresh.measurement_only, "beta_trace_time"].sum())
    assert math.isclose(summary["total_refresh_time"], refresh_time, rel_tol=1e-6, abs_tol=1e-10)
    assert math.isclose(summary["total_measurement_only_refresh_time"], measurement_only, rel_tol=1e-6, abs_tol=1e-10)
    assert math.isclose(summary["total_beta_trace_time"], beta_trace, rel_tol=1e-6, abs_tol=1e-10)
    assert math.isclose(summary["fisher_beta_trace_time"], fisher_trace, rel_tol=1e-6, abs_tol=1e-10)
    expected_overhead = summary["diagnostic_seconds"] + measurement_only + fisher_trace
    assert math.isclose(summary["research_overhead_seconds"], expected_overhead, rel_tol=1e-6, abs_tol=1e-10)
    assert summary["research_overhead_seconds"] >= summary["diagnostic_seconds"]
    assert summary["core_training_runtime"] <= summary["wall_time"] + 1e-8
    assert summary["core_training_runtime"] >= -1e-8


def _validate_beta_summary(summary, metadata, train, beta_step):
    _assert_boolean(train, "beta_diagnostics_finite")
    invalid_from_beta = sorted({
        int(row["step"]) for row in beta_step.to_dict("records")
        if not bool(row["diagnostic_valid"])
    })
    invalid_from_train = sorted({
        int(row.step) for row in train.itertuples() if not bool(row.beta_diagnostics_finite)
    })
    assert invalid_from_train == invalid_from_beta
    assert summary["beta_diagnostic_nonfinite_steps"] == invalid_from_beta
    assert summary["beta_diagnostic_nonfinite_count"] == len(invalid_from_beta)
    assert summary["beta_diagnostics_all_finite"] is (len(invalid_from_beta) == 0)
    for field in (
        "beta_diagnostics_all_finite", "beta_diagnostic_nonfinite_count",
        "beta_diagnostic_nonfinite_steps",
    ):
        assert metadata[field] == summary[field]


def validate(config, runs, output, require_tests=True):
    check_config(config)
    runs, output = output_path(runs), output_path(output)
    current = provenance()
    require_pinned(current, config["smoke"])
    fp = fingerprint(config)
    evidence_path = ROOT / "runs" / "test_evidence.json"
    if require_tests:
        proof = load(evidence_path)
        required = set(proof.get("required_nodeids", []))
        assert proof.get("passed") is True and proof.get("provenance") == current
        assert required <= set(proof.get("passed_nodeids", [])), "required tests did not pass"

    expected_seeds = [int(seed) for seed in config["seeds"]]
    actual_seed_dirs = {
        path.name for path in runs.iterdir()
        if path.is_dir() and re.fullmatch(r"seed[0-9]+", path.name)
    }
    assert actual_seed_dirs == {f"seed{seed}" for seed in expected_seeds}
    records = {}
    for seed in expected_seeds:
        expected_specs = run_specs_for_seed(config, seed)
        seed_root = runs / f"seed{seed}"
        assert sorted(path.name for path in seed_root.iterdir() if path.is_dir()) == sorted(
            spec["run_id"] for spec in expected_specs
        )
        for spec0 in expected_specs:
            spec = dict(spec0, seed=seed)
            name = spec["run_id"]
            root = seed_root / name
            config_file, metadata, summary, pairing = [
                load(root / f"{n}.json") for n in ("config", "metadata", "summary", "pairing")
            ]
            assert config_file == config
            assert metadata["provenance"] == current
            assert metadata["complete"] == (summary["status"] == "completed")
            assert metadata["run_id"] == name and metadata["method"] == spec["method"]
            assert metadata["learning_rate"] == summary["learning_rate"] == spec["learning_rate"]
            assert metadata["fingerprint"] == summary["fingerprint"] == fp
            assert metadata["optimizer"] == "SGD" and metadata["momentum"] == 0
            assert metadata["beta_train"] == 1 and metadata["no_beta_feedback"] is True
            assert metadata["filter_position"] == "after global clipping and DP Gaussian noise"
            assert output_path(metadata["output"]) == root.resolve()
            for field, value in RELEASE_FLAGS.items():
                assert metadata.get(field) is value
                assert summary.get(field) is value
            total = expected_total_steps(config)
            assert summary["planned_steps"] == metadata["total_steps"] == total
            assert summary["status"] == "completed" and summary["completed_steps"] == total
            assert summary["parameters_finite"] is True
            if not config["smoke"]:
                assert summary["final_model_hash"] == ANCHORS.get((seed, name), summary["final_model_hash"])
            for value in summary.values():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    assert math.isfinite(float(value))
            assert math.isclose(summary["noise_multiplier"], 1.068115234375, rel_tol=0, abs_tol=1e-15) if not config["smoke"] else summary["noise_multiplier"] > 0
            assert math.isclose(summary["epsilon_spent"], 0.995693195331761, rel_tol=0, abs_tol=1e-12) if not config["smoke"] else summary["epsilon_spent"] > 0

            train = pd.read_csv(root / "train_metrics.csv")
            layer = pd.read_csv(root / "layer_metrics.csv")
            refresh = pd.read_csv(root / "refresh_metrics.csv")
            eigenbin = pd.read_csv(root / "eigenbin_metrics.csv")
            beta_step = pd.read_csv(root / "beta_step_metrics.csv")
            beta_interval = pd.read_csv(root / "beta_interval_metrics.csv")
            beta_lagged = pd.read_csv(root / "beta_lagged_metrics.csv")
            beta_hold = pd.read_csv(root / "beta_hold_predictor.csv")
            beta_window = pd.read_csv(root / "beta_window_sensitivity.csv")

            _assert_integer(train, ("step", "epoch"))
            assert train.step.tolist() == list(range(total))
            _assert_core_finite(train, ("step", "epoch", "train_loss", "learning_rate"))
            assert len(layer) == total * len(LAYERS)
            assert {(int(row.step), row.layer) for row in layer.itertuples()} == {
                (step, layer_name) for step in range(total) for layer_name in LAYERS
            }
            assert len(eigenbin) == 0

            _assert_integer(refresh, ("step", "refresh_index"))
            _assert_boolean(refresh, "measurement_only")
            _assert_core_finite(
                refresh,
                ("step", "refresh_index", "refresh_time", "beta_trace_time",
                 "diagnostic_spectrum_time", "active_state_bytes"),
            )
            assert list(refresh.step) == list(range(0, total, config["K"]))
            assert (refresh.refresh_index == refresh.step // config["K"]).all()
            assert set(refresh.measurement_only) == {name.startswith("dp_sgd_")}
            _validate_runtime(summary, refresh)

            assert [item["step"] for item in pairing["private"]] == list(range(total))
            assert [item["step"] for item in pairing["synthetic"]] == list(range(0, total, config["K"]))
            assert list(refresh.step) == [item["step"] for item in pairing["synthetic"]]
            assert len(pairing["private"]) == total
            assert set(pairing) == {"private", "synthetic"}

            _validate_beta_step(config, beta_step, seed, name, total)
            assert len(beta_step) == total * len(LAYERS)
            _validate_beta_summary(summary, metadata, train, beta_step)
            _validate_intervals(config, beta_step, beta_interval, beta_lagged, beta_hold)
            _validate_window_sensitivity(config, beta_step, beta_window)
            records[(seed, name)] = (spec, metadata, summary, pairing)

    for seed in expected_seeds:
        names = [spec["run_id"] for spec in run_specs_for_seed(config, seed)]
        base_pair = records[(seed, names[0])][3]["private"]
        for name in names[1:]:
            pair = records[(seed, name)][3]["private"]
            _compare_pairing(base_pair, pair, ("step", "batch_indices", "noise_rng_before", "noise_rng_after"))
        synthetic_base = records[(seed, names[0])][3]["synthetic"]
        for name in names[1:]:
            _compare_pairing(
                synthetic_base, records[(seed, name)][3]["synthetic"],
                ("step", "samples_hash", "labels_hash", "rng_before", "rng_after"),
            )

    regression = _validate_fixed_reference(config, runs)
    result = dict(
        passed=True, experiment="expv2", runs=len(records), fingerprint=fp,
        provenance=current, regression=regression,
        beta_primary_window=config["beta_primary_window"],
        no_success_thresholds=True, formal_experiment_launched=not config["smoke"],
    )
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
