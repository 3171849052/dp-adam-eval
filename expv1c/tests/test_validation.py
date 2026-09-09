import pytest

from expv1c.common import read_config
from expv1c.validate_expv1c import (
    ANCHORS,
    _compare_pairing,
    _compare_pairing_prefix,
    _validate_final_hash,
    _validate_divergence_progress,
    expected_refresh_steps,
)


def _synthetic_items(steps):
    return [
        {
            "step": step,
            "samples_hash": f"samples-{step}",
            "labels_hash": f"labels-{step}",
            "rng_before": f"before-{step}",
            "rng_after": f"after-{step}",
        }
        for step in steps
    ]


def test_formal_nonanchor_hash_is_not_fixed():
    config = read_config("expv1c/configs/full.json")
    arbitrary_hash = "a" * 64
    _validate_final_hash(config, "dp_fisher_wiener_lr1p50", {"final_model_hash": arbitrary_hash})
    with pytest.raises(AssertionError):
        _validate_final_hash(config, "dp_fisher_wiener_lr1p50", {"final_model_hash": "z" * 64})


def test_formal_nonanchor_wrongly_none_regression():
    config = read_config("expv1c/configs/full.json")
    assert ANCHORS.get("dp_fisher_wiener_lr1p50") is None
    _validate_final_hash(config, "dp_fisher_wiener_lr1p50", {"final_model_hash": "b" * 64})
    _validate_final_hash(config, "dp_sgd_lr0p10", {"final_model_hash": ANCHORS["dp_sgd_lr0p10"]})
    _validate_final_hash(config, "dp_fisher_wiener_lr0p80", {"final_model_hash": ANCHORS["dp_fisher_wiener_lr0p80"]})


def test_diverged_fisher_synthetic_pairing_uses_common_prefix():
    reference = _synthetic_items([0, 50, 100, 150])
    actual = _synthetic_items([0, 50, 100])
    keys = ("step", "samples_hash", "labels_hash", "rng_before", "rng_after")
    _compare_pairing_prefix(reference, actual, keys)
    bad = [dict(item) for item in actual]
    bad[1]["samples_hash"] = "wrong"
    with pytest.raises(AssertionError):
        _compare_pairing_prefix(reference, bad, keys)
    gap = _synthetic_items([0, 50, 150])
    with pytest.raises(AssertionError):
        _compare_pairing_prefix(reference, gap, keys)


def test_completed_fisher_synthetic_pairing_requires_full_length():
    reference = _synthetic_items([0, 50, 100, 150])
    actual = _synthetic_items([0, 50, 100])
    with pytest.raises(AssertionError):
        _compare_pairing(reference, actual, ("step", "samples_hash", "labels_hash", "rng_before", "rng_after"))


def test_divergence_on_refresh_step_accepts_existing_refresh():
    steps = expected_refresh_steps(
        method="dp_fisher_wiener", status="diverged", completed_steps=600,
        diverged_step=600, planned_steps=1170, K=50,
    )
    assert steps == list(range(0, 601, 50))
    assert steps[-1] == 600


def test_divergence_between_refresh_steps():
    steps = expected_refresh_steps(
        method="dp_fisher_wiener", status="diverged", completed_steps=623,
        diverged_step=623, planned_steps=1170, K=50,
    )
    assert steps[-1] == 600
    assert 650 not in steps


def test_divergence_completed_steps_invariant():
    for completed in (600, 601):
        _validate_divergence_progress(completed, 600, 1170)
    for completed in (599, 602):
        with pytest.raises(AssertionError):
            _validate_divergence_progress(completed, 600, 1170)


def test_completed_refresh_schedule():
    steps = expected_refresh_steps(
        method="dp_fisher_wiener", status="completed", completed_steps=1170,
        diverged_step=None, planned_steps=1170, K=50,
    )
    assert len(steps) == 24 and steps[0] == 0 and steps[-1] == 1150
    assert expected_refresh_steps(
        method="dp_sgd", status="completed", completed_steps=1170,
        diverged_step=None, planned_steps=1170, K=50,
    ) == []


def test_validator_accepts_completed_run_with_nonfinite_research_diagnostic(config, tiny_data, tmp_path):
    import pandas as pd
    from expv1c.common import RUN_IDS, save_json
    from expv1c.train_expv1c import train
    from expv1c.validate_expv1c import validate, load
    from expv1c.summarize_expv1c import summarize
    from expv1c.plot_expv1c import plot
    for run_id in RUN_IDS:
        train(config, run_id, tmp_path, tiny_data)
    root = tmp_path / "seed42" / "dp_fisher_wiener_lr1p50"
    frame = pd.read_csv(root / "train_metrics.csv")
    frame.loc[3, "relmse_filtered"] = float("inf")
    # Exercise the formerly unconditional mechanism assertions too.
    frame.loc[3, "signal_retention"] = float("nan")
    frame.loc[3, "effective_signal_lr"] = float("inf")
    frame.loc[3, "diagnostics_finite"] = False
    frame.to_csv(root / "train_metrics.csv", index=False)
    summary = load(root / "summary.json")
    summary.update(diagnostics_all_finite=False, diagnostic_nonfinite_count=1, diagnostic_nonfinite_steps=[3])
    save_json(root / "summary.json", summary)
    meta = load(root / "metadata.json")
    meta.update(summary)
    save_json(root / "metadata.json", meta)
    assert summary["status"] == "completed"
    assert validate(config, tmp_path, tmp_path, require_tests=False)["passed"]
    summarize(config, tmp_path, tmp_path, require_tests=False)
    plot(config, tmp_path, tmp_path, require_tests=False)
    frame.loc[3, "diagnostics_finite"] = True
    frame.to_csv(root / "train_metrics.csv", index=False)
    with pytest.raises(AssertionError):
        validate(config, tmp_path, tmp_path, require_tests=False)


def test_boundary_ranking_ignores_divergence_and_nonfinite():
    from expv1c.summarize_expv1c import boundary_analysis
    rows = [dict(learning_rate=lr, status="completed", accuracy_auc=score, final_accuracy=score, final_test_loss=1-score)
            for lr, score in [(.8, .8), (1., .9), (1.2, .95), (1.5, .92)]]
    assert boundary_analysis(rows)["peak_bracketed"]
    rows[-1].update(accuracy_auc=.98)
    assert boundary_analysis(rows)["boundary_best"]
    assert not boundary_analysis(rows)["peak_bracketed"]
    rows[-1].update(status="diverged", accuracy_auc=float("inf"))
    result = boundary_analysis(rows)
    assert result["highest_accuracy_auc_lr"] == 1.2
    assert result["stability_boundary_observed"] and not result["peak_bracketed"]


@pytest.mark.parametrize("stage", ["filtered_gradient", "updated_parameters"])
def test_full_validator_accepts_algorithmic_divergence(config, tiny_data, tmp_path, monkeypatch, stage):
    from expv1c.common import RUN_IDS
    from expv1c.train_expv1c import train
    from expv1c.validate_expv1c import validate, load
    from expv1b import train_expv1b as helpers
    original = helpers.apply_fisher_wiener
    original_step = helpers.torch.optim.SGD.step
    count = 0
    def poison_filter(model, active):
        nonlocal count
        original(model, active)
        if count == 2:
            next(model.parameters()).grad.fill_(float("nan"))
        count += 1
    def poison_step(optimizer, *args, **kwargs):
        nonlocal count
        result = original_step(optimizer, *args, **kwargs)
        if count == 2:
            with helpers.torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("nan"))
        count += 1
        return result
    for run_id in RUN_IDS[:-1]:
        train(config, run_id, tmp_path, tiny_data)
    with monkeypatch.context() as patch:
        if stage == "filtered_gradient":
            patch.setattr(helpers, "apply_fisher_wiener", poison_filter)
        else:
            patch.setattr(helpers.torch.optim.SGD, "step", poison_step)
        train(config, RUN_IDS[-1], tmp_path, tiny_data)
    summary = load(tmp_path / "seed42" / RUN_IDS[-1] / "summary.json")
    assert summary["status"] == "diverged" and summary["diverged_step"] == 2
    assert summary["completed_steps"] == (2 if stage == "filtered_gradient" else 3)
    assert validate(config, tmp_path, tmp_path, require_tests=False)["passed"]
