from expv1c.common import FISHER_LRS, RUN_IDS, read_config, synthetic_batches


def test_config_protocol():
    full = read_config("expv1c/configs/full.json")
    assert full["seed"] == 42
    assert full["fisher_learning_rates"] == list(FISHER_LRS)
    assert full["dp_sgd_learning_rate"] == 0.1
    assert full["beta"] == 1
    assert full["momentum"] == 0
    assert full["optimizer"] == "SGD"
    assert full["K"] == 50 and full["M_syn"] == 2560
    assert synthetic_batches(full) == 10
    assert not any("scheduler" in key or "calibration" in key for key in full)


def test_smoke_has_full_grid_and_five_runs():
    smoke = read_config("expv1c/configs/smoke.json")
    assert smoke["seed"] == 42
    assert smoke["fisher_learning_rates"] == list(FISHER_LRS)
    assert synthetic_batches(smoke) == 2
    assert len(RUN_IDS) == 5
    assert all("scalar" not in run_id for run_id in RUN_IDS)



def test_protocol_rejects_drift():
    import pytest
    from expv1c.common import check_config, output_path
    full = read_config("expv1c/configs/full.json")
    for key, value in [("seed", 43), ("dp_sgd_learning_rate", .2), ("fisher_learning_rates", [.8, 1., 1.2, 1.5, 2.]),
                       ("beta", 2), ("optimizer", "Adam"), ("momentum", .9), ("K", 25), ("M_syn", 1280),
                       ("scheduler", None), ("beta_calibration", False), ("epochs", 6), ("epsilon", 2)]:
        with pytest.raises(ValueError):
            check_config(dict(full, **{key: value}))
    with pytest.raises(ValueError):
        output_path("expv1b/runs/forbidden")
