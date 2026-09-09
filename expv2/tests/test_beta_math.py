import math

from expv2.beta_estimation import noise_variance, single_step_beta, theoretical_conditional_variance


def test_beta_math():
    value = single_step_beta(5, 6, 100, 10, 0.01)
    assert value["beta_oracle_step"] == 0.5
    assert value["beta_dp_step_raw"] == 0.5
    assert value["beta_dp_step_positive"] == 0.5


def test_noise_variance():
    sigma = 1.068115234375
    assert noise_variance(sigma, 1, 256) == (sigma / 256) ** 2


def test_theoretical_variance():
    assert theoretical_conditional_variance(0.01, 5, 100) == 4 * 0.01 * 5 + 2 * 100 * 0.01 ** 2

