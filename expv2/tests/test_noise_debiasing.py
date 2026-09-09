from expv2.beta_estimation import single_step_beta


def test_noise_debiasing():
    value = single_step_beta(5, 6, 100, 10, 0.01)
    assert value["expected_noise_energy"] == 1
    assert value["noise_debiased_energy_raw"] == 5
    assert value["beta_dp_step_raw"] == 0.5


def test_negative_beta_preserved():
    value = single_step_beta(5, 0.5, 100, 10, 0.01)
    assert value["noise_debiased_energy_raw"] == -0.5
    assert value["beta_dp_step_raw"] < 0
    assert value["beta_dp_step_positive"] == 0
    assert value["beta_dp_step_negative"] is True

