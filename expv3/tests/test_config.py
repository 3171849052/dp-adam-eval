import pytest

from expv3.common import FULL_CONFIG, FISHER_LRS, check_config, read_config, run_specs


def test_config_exactness():
    assert read_config("expv3/configs/full.json") == FULL_CONFIG
    assert [spec["run_id"] for spec in run_specs(FULL_CONFIG)] == [
        "dp_sgd_lr0p50",
        "dp_fisher_wiener_adaptive_beta_lr0p50",
        "dp_fisher_wiener_adaptive_beta_lr1p00",
        "dp_fisher_wiener_adaptive_beta_lr5p00",
    ]
    assert FULL_CONFIG["fisher_learning_rates"] == list(FISHER_LRS)


def test_config_rejects_undocumented_controls():
    value = dict(FULL_CONFIG, gamma=1.0)
    with pytest.raises(ValueError):
        check_config(value)

