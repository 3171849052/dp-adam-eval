"""Strict ExpV1b artifact, privacy, pairing, and ExpV1 regression validation."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from expv1b.common import (
    FISHER_LRS,
    LAYERS,
    METHODS,
    REFERENCE_ROOT,
    ROOT,
    check_config,
    expected_total_steps,
    fingerprint,
    output_path,
    provenance,
    require_pinned,
    run_specs,
    save_json,
)


ANCHORS = {
    "dp_sgd_lr0p10": "9e9c9e76e84680f6fcb1c34ded5047f73f36b54c8616fb14293a470270fce1b2",
    "dp_fisher_wiener_lr0p10": "c04595c804fe9f04bd3421385e6e75e920bc288e9920c486fa0929c009ed6aa7",
}


def load(path):
    return json.loads(Path(path).read_text())


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return load(args.config), output_path(args.runs), output_path(args.output)


def _assert_finite_frame(frame, allowed_missing=()):
    for column in frame.select_dtypes(include="number"):
        missing = frame[column].isna()
        assert column in allowed_missing or not missing.any(), (column, "unexpected missing values")
        values = frame.loc[~missing, column]
        assert values.map(math.isfinite).all(), (column, "non-finite value")


def _compare_pairing(reference, actual, keys):
    assert len(reference) == len(actual)
    for ref, got in zip(reference, actual):
        for key in keys:
            assert ref[key] == got[key], (key, ref.get("step"))


def _validate_regression(c, runs):
    """Validate full trajectories when the immutable ExpV1 artifact is present."""
    available = all(
        (REFERENCE_ROOT / "seed42" / method / f"{name}.json").exists()
        for method, name in (("dp_sgd", "summary"), ("dp_fisher_wiener", "summary"))
    )
    result = {"reference_artifact_available": available, "checked": False}
    if not c["smoke"]:
        for run_id, expected in ANCHORS.items():
            actual = load(runs / "seed42" / run_id / "summary.json")
            assert actual["final_model_hash"] == expected, (run_id, actual["final_model_hash"])
        result["checked"] = True
        if available:
            for run_id, method in (("dp_sgd_lr0p10", "dp_sgd"), ("dp_fisher_wiener_lr0p10", "dp_fisher_wiener")):
                reference_root = REFERENCE_ROOT / "seed42" / method
                actual_root = runs / "seed42" / run_id
                reference_pair = load(reference_root / "pairing.json")
                actual_pair = load(actual_root / "pairing.json")
                _compare_pairing(
                    reference_pair["private"], actual_pair["private"],
                    ("step", "batch_indices", "noise_rng_before", "noise_rng_after", "model_hash"),
                )
                if method == "dp_fisher_wiener":
                    _compare_pairing(
                        reference_pair["synthetic"], actual_pair["synthetic"],
                        ("step", "samples_hash", "labels_hash", "rng_before", "rng_after"),
                    )
            result["trajectory_checked"] = True
        else:
            result["trajectory_checked"] = False
    return result


def validate(c, runs, output, require_tests=True):
    check_config(c)
    runs, output = output_path(runs), output_path(output)
    current = provenance()
    require_pinned(current, c["smoke"])
    fp = fingerprint(c)
    evidence_path = ROOT / "runs" / "test_evidence.json"
    if require_tests:
        proof = load(evidence_path)
        required = set(proof.get("required_nodeids", []))
        assert proof.get("passed") is True and proof.get("provenance") == current
        assert required <= set(proof.get("passed_nodeids", [])), "required tests did not pass"

    specs = run_specs(c)
    expected_ids = [spec["run_id"] for spec in specs]
    actual_ids = sorted(path.name for path in (runs / "seed42").iterdir() if path.is_dir())
    assert actual_ids == sorted(expected_ids), (actual_ids, expected_ids)
    records = {}
    for spec in specs:
        run_id, method, learning_rate = spec["run_id"], spec["method"], spec["learning_rate"]
        root = runs / "seed42" / run_id
        config, meta, summary, pairing = [load(root / f"{name}.json") for name in ("config", "metadata", "summary", "pairing")]
        assert config == c
        assert meta["run_id"] == run_id and meta["method"] == method
        assert meta["learning_rate"] == learning_rate and summary["learning_rate"] == learning_rate
        assert meta["fingerprint"] == summary["fingerprint"] == fp
        assert meta["provenance"] == current and meta["complete"] == (summary["status"] == "completed")
        assert meta["eigenmode_diagnostics"] is False and meta["eigen_budget"] == 0
        assert meta["optimizer"] == "SGD" and meta["momentum"] == 0 and meta["beta"] == 1
        assert meta["target_epsilon"] == c["epsilon"] and meta["target_delta"] == c["delta"]
        assert meta["accountant"] == "rdp" and meta["max_grad_norm"] == c["max_grad_norm"]
        assert output_path(meta["output"]) == root.resolve()

        planned = expected_total_steps(c)
        status = summary["status"]
        assert status in ("completed", "diverged")
        assert summary["planned_steps"] == planned and meta["total_steps"] == planned
        completed_steps = summary["completed_steps"]
        assert 0 <= completed_steps <= planned
        if status == "completed":
            assert completed_steps == planned and summary["parameters_finite"]
            assert summary["final_model_hash"] == (ANCHORS.get(run_id) if not c["smoke"] else summary["final_model_hash"])
        else:
            assert summary["diverged_step"] is not None
            assert summary["diverged_step"] < planned

        train = pd.read_csv(root / "train_metrics.csv")
        layer = pd.read_csv(root / "layer_metrics.csv")
        refresh = pd.read_csv(root / "refresh_metrics.csv")
        eigenbin = pd.read_csv(root / "eigenbin_metrics.csv")
        _assert_finite_frame(train, {"test_loss", "test_accuracy"})
        layer_missing = {"kappa"}
        if method == "dp_sgd":
            layer_missing |= {
                "trace_A", "trace_G", "trace_F", "lambdaF_mean", "lambdaF_median",
                "lambdaF_q10", "lambdaF_q90",
            }
        _assert_finite_frame(layer, layer_missing)
        _assert_finite_frame(refresh)
        assert eigenbin.empty
        if status == "completed":
            assert train.step.tolist() == list(range(planned))
            assert len(layer) == planned * 4
        else:
            assert train.step.tolist() == list(range(completed_steps))
            assert len(layer) == completed_steps * 4
        assert set(layer.layer) <= set(LAYERS)
        assert not layer.duplicated(["step", "layer"]).any()
        assert set(train.config_fingerprint) <= {fp}
        assert set(layer.config_fingerprint) <= {fp}
        assert set(refresh.config_fingerprint) <= {fp}
        expected_refresh = list(range(0, completed_steps, c["K"])) if method == "dp_fisher_wiener" else []
        assert refresh.step.tolist() == expected_refresh
        assert [item["step"] for item in pairing["private"]] == list(range(completed_steps))
        assert [item["step"] for item in pairing["synthetic"]] == expected_refresh
        if method == "dp_sgd":
            assert not pairing["synthetic"] and summary["active_state_bytes"] == 0
        else:
            assert summary["active_state_bytes"] > 0

        for frame in (train, layer, refresh):
            if not frame.empty:
                assert set(frame.seed) == {42}
                assert set(frame.method) == {method}
                assert set(frame.config_fingerprint) == {fp}
        expected_evaluations = [
            step for step in range(completed_steps)
            if (step + 1) % c["eval_interval"] == 0 or step + 1 == planned
        ]
        assert train.loc[train.test_accuracy.notna(), "step"].tolist() == expected_evaluations
        assert train.loc[train.test_loss.notna(), "step"].tolist() == expected_evaluations
        assert summary["number_of_refreshes"] == len(expected_refresh)
        for frame in (train, layer):
            if frame.empty:
                continue
            assert (frame.signal_amplitude_retention >= 0).all()
            assert (frame.effective_signal_lr >= 0).all()
            expected_eff = frame.learning_rate * frame.signal_retention.clip(lower=0).pow(0.5)
            assert (frame.effective_signal_lr - expected_eff).abs().max() < 2e-6
            assert (frame.optimizer_update_norm >= 0).all()
        if method == "dp_sgd" and not train.empty:
            assert (train.signal_retention == 1).all()
            assert (train.effective_signal_lr == learning_rate).all()
            assert (train.snr_gain_db == 0).all() and (train.mse_reduction == 0).all()
        assert meta["noise_multiplier"] == summary["noise_multiplier"]
        assert meta["sample_rate"] == summary["sample_rate"] == c["batch_size"] / (c["train_subset"] or 60000)
        assert summary["epsilon_spent"] >= 0 and math.isfinite(summary["epsilon_spent"])
        records[run_id] = (spec, meta, summary, pairing, train, layer)

    # Pairing is checked across the full set for normal runs, and over the
    # common prefix if a stability probe has explicitly diverged.
    base = records[expected_ids[0]]
    for run_id in expected_ids[1:]:
        current_record = records[run_id]
        for key in ("noise_multiplier", "sample_rate", "total_steps", "initial_model_hash"):
            assert current_record[1][key] == base[1][key]
        common = min(len(base[3]["private"]), len(current_record[3]["private"]))
        _compare_pairing(
            base[3]["private"][:common], current_record[3]["private"][:common],
            ("step", "batch_indices", "noise_rng_before", "noise_rng_after"),
        )
    complete_records = [records[run_id] for run_id in expected_ids if records[run_id][2]["status"] == "completed"]
    if complete_records:
        epsilon_values = [record[2]["epsilon_spent"] for record in complete_records]
        assert all(math.isclose(value, epsilon_values[0], rel_tol=0.0, abs_tol=1e-12) for value in epsilon_values)
    fisher_pairs = [records[run_id][3]["synthetic"] for run_id in expected_ids if records[run_id][0]["method"] == "dp_fisher_wiener"]
    for pair in fisher_pairs[1:]:
        _compare_pairing(fisher_pairs[0], pair, ("step", "samples_hash", "labels_hash", "rng_before", "rng_after"))
    # The preceding common pairing comparison intentionally ignores model
    # hashes across methods: filters make their trajectories differ.  Explicit
    # same-method LR pairing is the regression-sensitive comparison.
    fisher_base = records["dp_fisher_wiener_lr0p10"][3]["private"]
    for run_id in expected_ids:
        if records[run_id][0]["method"] != "dp_fisher_wiener":
            continue
        pair = records[run_id][3]["private"]
        common = min(len(fisher_base), len(pair))
        _compare_pairing(fisher_base[:common], pair[:common], ("step", "batch_indices", "noise_rng_before", "noise_rng_after"))

    regression = _validate_regression(c, runs)
    result = dict(
        passed=True, runs=len(specs), fingerprint=fp, provenance=current,
        reference_artifact_available=regression["reference_artifact_available"],
        regression=regression,
        single_seed_descriptive_only=True,
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
