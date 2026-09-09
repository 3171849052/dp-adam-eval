"""Strict ExpV1c artifact, privacy, pairing, and ExpV1 regression validation."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from expv1c.common import (
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


from expv1b.validate_expv1b import (
    load, _assert_finite_frame, _assert_finite_columns,
    TRAIN_CORE_COLUMNS, LAYER_CORE_COLUMNS, _bool_value,
    _validate_diagnostic_finiteness, _validate_divergence_progress,
    _compare_pairing, _compare_pairing_prefix, expected_refresh_steps,
)

ANCHORS = {
    "dp_sgd_lr0p10": "9e9c9e76e84680f6fcb1c34ded5047f73f36b54c8616fb14293a470270fce1b2",
    "dp_fisher_wiener_lr0p80": "38c1dfe4ca6ad65b49598e8a327edaf81f72873ff5a1df28f5884d34028a1b33",
}


def _should_check_anchor_hash(c, run_id):
    """Only the DP-SGD .10 and ExpV1b Fisher .80 regression runs have fixed hashes."""
    return not c["smoke"] and run_id in ANCHORS


def _validate_final_hash(c, run_id, summary):
    """Validate hash shape for every completed run and value only for anchors."""
    value = summary.get("final_model_hash")
    assert isinstance(value, str) and len(value) == 64
    try:
        int(value, 16)
    except (TypeError, ValueError) as exc:
        raise AssertionError("final_model_hash must be a 64-character hex string") from exc
    if _should_check_anchor_hash(c, run_id):
        assert value == ANCHORS[run_id], (run_id, value)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return load(args.config), output_path(args.runs), output_path(args.output)


def _validate_regression(c, runs):
    available = all((REFERENCE_ROOT / "seed42" / run_id / "pairing.json").exists() for run_id in ANCHORS)
    result = dict(reference_artifact_available=available, checked=False, trajectory_checked=False)
    if c["smoke"]:
        return result
    assert available, "Pinned ExpV1b reference artifact is required for formal validation"
    for run_id, expected in ANCHORS.items():
        ref = REFERENCE_ROOT / "seed42" / run_id
        actual = runs / "seed42" / run_id
        reference_summary, summary = load(ref / "summary.json"), load(actual / "summary.json")
        assert reference_summary["final_model_hash"] == summary["final_model_hash"] == expected
        assert summary["status"] == "completed"
        for metric in ("final_accuracy", "accuracy_auc"):
            assert reference_summary[metric] == summary[metric]
        reference_pair, pair = load(ref / "pairing.json"), load(actual / "pairing.json")
        _compare_pairing(reference_pair["private"], pair["private"],
                         ("step", "batch_indices", "noise_rng_before", "noise_rng_after", "model_hash"))
        _compare_pairing(reference_pair["synthetic"], pair["synthetic"],
                         ("step", "samples_hash", "labels_hash", "rng_before", "rng_after"))
    result.update(checked=True, trajectory_checked=True)
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
        from expv1c.test_requirements import REQUIRED_NODEIDS
        required = set(REQUIRED_NODEIDS)
        assert set(proof.get("required_nodeids", [])) == required
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
        for key in ("diagnostics_all_finite", "diagnostic_nonfinite_count", "diagnostic_nonfinite_steps"):
            assert meta[key] == summary[key]
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
            _validate_final_hash(c, run_id, summary)
        else:
            _validate_divergence_progress(completed_steps, summary["diverged_step"], planned)

        train = pd.read_csv(root / "train_metrics.csv")
        layer = pd.read_csv(root / "layer_metrics.csv")
        refresh = pd.read_csv(root / "refresh_metrics.csv")
        eigenbin = pd.read_csv(root / "eigenbin_metrics.csv")
        _assert_finite_columns(
            train, TRAIN_CORE_COLUMNS,
            {"test_loss", "test_accuracy", "clip_rate"},
        )
        _assert_finite_columns(layer, LAYER_CORE_COLUMNS)
        _assert_finite_frame(refresh)
        assert eigenbin.empty
        _validate_diagnostic_finiteness(
            train, layer, meta["diagnostics_enabled"], summary,
        )
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
        expected_refresh = expected_refresh_steps(
            method=method, status=status, completed_steps=completed_steps,
            diverged_step=summary["diverged_step"], planned_steps=planned, K=c["K"],
        )
        assert refresh.step.tolist() == expected_refresh
        assert [item["step"] for item in pairing["private"]] == list(range(completed_steps))
        assert [item["step"] for item in pairing["synthetic"]] == expected_refresh
        assert refresh.step.tolist() == [item["step"] for item in pairing["synthetic"]]
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
            if ((step + 1) % c["eval_interval"] == 0 or step + 1 == planned)
            and pairing["private"][step]["parameters_finite"]
        ]
        assert train.loc[train.test_accuracy.notna(), "step"].tolist() == expected_evaluations
        assert train.loc[train.test_loss.notna(), "step"].tolist() == expected_evaluations
        assert summary["number_of_refreshes"] == len(expected_refresh)
        finite_steps = train.loc[train.diagnostics_finite.map(_bool_value), "step"]
        finite_train = train[train.step.isin(finite_steps)]
        finite_layer = layer[layer.step.isin(finite_steps)]
        for frame in (finite_train, finite_layer):
            if frame.empty:
                continue
            assert (frame.signal_retention >= 0).all()
            assert (frame.signal_amplitude_retention - frame.signal_retention.pow(0.5)).abs().max() < 2e-6
            assert (frame.signal_amplitude_retention >= 0).all()
            assert (frame.effective_signal_lr >= 0).all()
            expected_eff = frame.learning_rate * frame.signal_retention.clip(lower=0).pow(0.5)
            assert (frame.effective_signal_lr - expected_eff).abs().max() < 2e-6
            assert (frame.optimizer_update_norm >= 0).all()
        if method == "dp_sgd" and not finite_train.empty:
            assert (finite_train.signal_retention == 1).all()
            assert (finite_train.effective_signal_lr == learning_rate).all()
            assert (finite_train.snr_gain_db == 0).all() and (finite_train.mse_reduction == 0).all()
        assert meta["noise_multiplier"] == summary["noise_multiplier"]
        assert meta["sample_rate"] == summary["sample_rate"] == c["batch_size"] / (c["train_subset"] or 60000)
        assert summary["epsilon_spent"] >= 0 and math.isfinite(summary["epsilon_spent"])
        from opacus.accountants import RDPAccountant
        from opacus.accountants.utils import get_noise_multiplier
        sigma = get_noise_multiplier(target_epsilon=c["epsilon"], target_delta=c["delta"],
                                     sample_rate=summary["sample_rate"], steps=planned, accountant="rdp")
        assert summary["noise_multiplier"] == sigma
        accountant = RDPAccountant()
        for _ in range(completed_steps):
            accountant.step(noise_multiplier=sigma, sample_rate=summary["sample_rate"])
        assert math.isclose(summary["epsilon_spent"], accountant.get_epsilon(c["delta"]), abs_tol=1e-12)
        if not c["smoke"]:
            assert sigma == 1.068115234375
        from expv1c.common import threshold_steps
        evaluations = train[train.test_accuracy.notna()].to_dict("records")
        assert summary["steps_to_acc_90"] == (threshold_steps(evaluations, .9) if status == "completed" else None)
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
    fisher_ids = [run_id for run_id in expected_ids if records[run_id][0]["method"] == "dp_fisher_wiener"]
    fisher_base_id = fisher_ids[0]
    fisher_base_pair = records[fisher_base_id][3]["synthetic"]
    for run_id in fisher_ids[1:]:
        pair = records[run_id][3]["synthetic"]
        keys = ("step", "samples_hash", "labels_hash", "rng_before", "rng_after")
        if records[fisher_base_id][2]["status"] == "completed" and records[run_id][2]["status"] == "completed":
            _compare_pairing(fisher_base_pair, pair, keys)
        else:
            _compare_pairing_prefix(fisher_base_pair, pair, keys)
    # The preceding common pairing comparison intentionally ignores model
    # hashes across methods: filters make their trajectories differ.  Explicit
    # same-method LR pairing is the regression-sensitive comparison.
    fisher_base = records["dp_fisher_wiener_lr0p80"][3]["private"]
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
