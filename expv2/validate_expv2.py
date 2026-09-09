"""Strict ExpV2 artifact, algebra, pairing, and fixed-reference validation."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from expv2.beta_estimation import build_interval_rows, build_lagged_rows, build_hold_predictor
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
    return value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))


def _assert_finite(frame, allowed_missing=()):
    for column in frame.select_dtypes(include="number"):
        missing = frame[column].isna()
        assert column in allowed_missing or not missing.any(), (column, "unexpected missing")
        assert frame.loc[~missing, column].map(math.isfinite).all(), (column, "non-finite")


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
            # ExpV1 has no measurement stream for DP-SGD. Fisher synthetic
            # probes are still canonical and must match the fixed references.
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
    assert frame.step.tolist() == sorted(frame.step.tolist()) or True
    assert len(frame) == total * 4
    assert set(frame.layer) == set(LAYERS)
    assert not frame.duplicated(["step", "layer"]).any()
    assert frame.step.min() == 0 and frame.step.max() == total - 1
    _assert_finite(frame)
    assert (frame.dimension > 0).all() and (frame.r > 0).all()
    assert (frame.trace_F > 0).all()
    assert (frame.expected_noise_energy - frame.dimension * frame.r).abs().max() < 1e-12
    assert (frame.noise_debiased_energy_raw - (frame.noisy_gradient_energy - frame.expected_noise_energy)).abs().max() < 1e-10
    assert (frame.beta_oracle_step - frame.clean_signal_energy / frame.trace_F).abs().max() < 1e-6
    assert (frame.beta_dp_step_raw - frame.noise_debiased_energy_raw / frame.trace_F).abs().max() < 1e-6
    assert (frame.beta_dp_step_positive - frame.beta_dp_step_raw.clip(lower=0)).abs().max() < 1e-6
    assert (frame.beta_dp_step_negative == (frame.beta_dp_step_raw < 0)).all()
    assert set(frame.seed) == {seed} and set(frame.run_id) == {run_name}


def _validate_intervals(config, step_frame, interval_frame, lagged_frame, hold_frame):
    expected = build_interval_rows(step_frame.to_dict("records"), config["beta_primary_window"])
    assert len(interval_frame) == len(expected)
    actual = interval_frame.sort_values(["layer", "interval_index"]).reset_index(drop=True)
    expected_df = pd.DataFrame(expected).sort_values(["layer", "interval_index"]).reset_index(drop=True)
    assert actual[["layer", "interval_index", "start_step", "end_step", "n_steps"]].to_dict("records") == expected_df[["layer", "interval_index", "start_step", "end_step", "n_steps"]].to_dict("records")
    for column in ("trace_F", "oracle_signal_energy_sum", "noisy_energy_sum", "expected_noise_energy_sum",
                   "noise_debiased_energy_sum", "beta_oracle", "beta_dp_raw", "oracle_estimator_sd",
                   "oracle_observability_snr"):
        assert (actual[column] - expected_df[column]).abs().max() < 1e-5, column
    assert not actual.duplicated(["layer", "interval_index"]).any()
    _assert_finite(interval_frame)
    for layer, group in interval_frame.groupby("layer"):
        group = group.sort_values("interval_index")
        assert group.start_step.tolist()[0] == 0
        assert group.end_step.tolist()[-1] == int(step_frame.step.max())
    expected_lag = build_lagged_rows(expected)
    expected_hold = build_hold_predictor(expected)
    assert len(lagged_frame) == len(expected_lag) and len(hold_frame) == len(expected_hold)
    assert lagged_frame.source_interval.add(lagged_frame.target_interval * 0 + 1).tolist() == lagged_frame.target_interval.tolist()
    _assert_finite(lagged_frame, ("lag_ratio", "lag_log10_abs_ratio_error"))
    _assert_finite(hold_frame, ("hold_ratio", "hold_log10_abs_ratio_error"))
    assert (lagged_frame.target_interval - lagged_frame.source_interval == 1).all()
    assert (hold_frame.target_interval - hold_frame.source_interval == 1).all()
    for frame, expected_rows, key in ((lagged_frame, expected_lag, "lag_log10_abs_ratio_error"),
                                      (hold_frame, expected_hold, "hold_log10_abs_ratio_error")):
        left = frame.sort_values(["layer", "target_interval"]).reset_index(drop=True)
        right = pd.DataFrame(expected_rows).sort_values(["layer", "target_interval"]).reset_index(drop=True)
        pd.testing.assert_series_equal(
            left[key].fillna(-999), right[key].fillna(-999),
            check_exact=False, rtol=1e-5, atol=1e-5, check_names=False,
        )


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
    assert sorted(path.name for path in runs.iterdir() if path.is_dir()) == sorted(f"seed{seed}" for seed in expected_seeds)
    records = {}
    for seed in expected_seeds:
        expected_specs = run_specs_for_seed(config, seed)
        seed_root = runs / f"seed{seed}"
        assert sorted(path.name for path in seed_root.iterdir() if path.is_dir()) == sorted(spec["run_id"] for spec in expected_specs)
        for spec0 in expected_specs:
            spec = dict(spec0, seed=seed)
            name = spec["run_id"]
            root = seed_root / name
            config_file, metadata, summary, pairing = [load(root / f"{n}.json") for n in ("config", "metadata", "summary", "pairing")]
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
            assert train.step.tolist() == list(range(total))
            assert len(layer) == total * 4 and not layer.duplicated(["step", "layer"]).any()
            assert len(eigenbin) == 0
            assert list(refresh.step) == list(range(0, total, config["K"]))
            assert [item["step"] for item in pairing["private"]] == list(range(total))
            assert [item["step"] for item in pairing["synthetic"]] == list(range(0, total, config["K"]))
            assert list(refresh.step) == [item["step"] for item in pairing["synthetic"]]
            assert len(beta_step) == total * 4
            _validate_beta_step(config, beta_step, seed, name, total)
            _validate_intervals(config, beta_step, beta_interval, beta_lagged, beta_hold)
            assert set(beta_window.window_size) == set(config["beta_window_sensitivity"])
            assert not beta_window.duplicated(["window_size", "layer", "window_index"]).any()
            _assert_finite(beta_window, ("log_error",))
            assert set(train.seed) == {seed} and set(layer.seed) == {seed}
            assert all(math.isfinite(float(value)) for value in refresh.select_dtypes(include="number").stack())
            assert set(pairing) == {"private", "synthetic"}
            assert len(pairing["private"]) == total
            records[(seed, name)] = (spec, metadata, summary, pairing)

    # Private batch and Gaussian streams are paired across methods for each seed.
    for seed in expected_seeds:
        names = [spec["run_id"] for spec in run_specs_for_seed(config, seed)]
        base_pair = records[(seed, names[0])][3]["private"]
        for name in names[1:]:
            pair = records[(seed, name)][3]["private"]
            _compare_pairing(base_pair, pair, ("step", "batch_indices", "noise_rng_before", "noise_rng_after"))
        synthetic_base = records[(seed, names[0])][3]["synthetic"]
        for name in names[1:]:
            _compare_pairing(synthetic_base, records[(seed, name)][3]["synthetic"],
                             ("step", "samples_hash", "labels_hash", "rng_before", "rng_after"))

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
