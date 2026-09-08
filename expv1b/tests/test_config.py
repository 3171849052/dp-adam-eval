from expv1b.common import FISHER_LRS, RUN_IDS, read_config, synthetic_batches


def test_config_protocol():
    full = read_config("expv1b/configs/full.json")
    assert full["seed"] == 42
    assert full["fisher_learning_rates"] == list(FISHER_LRS)
    assert full["dp_sgd_learning_rate"] == 0.1
    assert full["beta"] == 1
    assert full["momentum"] == 0
    assert full["optimizer"] == "SGD"
    assert full["K"] == 50 and full["M_syn"] == 2560
    assert synthetic_batches(full) == 10
    assert not any("scheduler" in key or "calibration" in key for key in full)


def test_smoke_has_full_grid_and_seven_runs():
    smoke = read_config("expv1b/configs/smoke.json")
    assert smoke["seed"] == 42
    assert smoke["fisher_learning_rates"] == list(FISHER_LRS)
    assert synthetic_batches(smoke) == 2
    assert len(RUN_IDS) == 7
    assert all("scalar" not in run_id for run_id in RUN_IDS)

