import copy

import pytest

from expv2.common import check_config, read_config, run_specs, run_specs_for_seed


def test_config_exactness():
    full = read_config("expv2/configs/full.json")
    assert full["experiment"] == "expv2"
    assert full["seeds"] == [42, 7, 91]
    assert full["base_learning_rate"] == 0.1
    assert full["stress_learning_rate"] == 0.8 and full["stress_seed"] == 42
    assert full["beta_train"] == 1 and full["beta_primary_window"] == 50
    assert full["beta_window_sensitivity"] == [10, 25, 50, 100]
    assert full["optimizer"] == "SGD" and full["momentum"] == 0
    assert full["K"] == 50 and full["M_syn"] == 2560
    assert [s["run_id"] for s in run_specs(full)] == [
        "dp_sgd_lr0p10", "dp_fisher_wiener_lr0p10", "dp_fisher_wiener_lr0p80"
    ]
    assert [s["run_id"] for s in run_specs_for_seed(full, 7)] == [
        "dp_sgd_lr0p10", "dp_fisher_wiener_lr0p10"
    ]
    smoke = read_config("expv2/configs/smoke.json")
    assert smoke["seeds"] == [42] and smoke["K"] == 2 and smoke["M_syn"] == 8
    assert smoke["beta_primary_window"] == 2


@pytest.mark.parametrize("key,value", [
    ("base_learning_rate", 0.2), ("stress_learning_rate", 0.7),
    ("stress_seed", 7), ("beta_train", 2), ("beta_primary_window", 25),
    ("optimizer", "Adam"), ("momentum", 0.9), ("K", 10), ("M_syn", 16),
])
def test_config_rejects_protocol_drift(key, value):
    config = read_config("expv2/configs/full.json")
    config[key] = value
    with pytest.raises(ValueError):
        check_config(config)

