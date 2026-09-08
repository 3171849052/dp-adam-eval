import pytest

from expv1b.common import read_config
from expv1b.validate_expv1b import (
    ANCHORS,
    _compare_pairing,
    _compare_pairing_prefix,
    _validate_final_hash,
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
    config = read_config("expv1b/configs/full.json")
    arbitrary_hash = "a" * 64
    _validate_final_hash(config, "dp_fisher_wiener_lr0p30", {"final_model_hash": arbitrary_hash})
    with pytest.raises(AssertionError):
        _validate_final_hash(config, "dp_fisher_wiener_lr0p30", {"final_model_hash": "z" * 64})


def test_formal_nonanchor_wrongly_none_regression():
    config = read_config("expv1b/configs/full.json")
    assert ANCHORS.get("dp_fisher_wiener_lr0p30") is None
    _validate_final_hash(config, "dp_fisher_wiener_lr0p30", {"final_model_hash": "b" * 64})
    _validate_final_hash(config, "dp_sgd_lr0p10", {"final_model_hash": ANCHORS["dp_sgd_lr0p10"]})
    _validate_final_hash(config, "dp_fisher_wiener_lr0p10", {"final_model_hash": ANCHORS["dp_fisher_wiener_lr0p10"]})


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

