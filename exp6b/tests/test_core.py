import math

import torch

from exp6b.optimizers import SecondMomentState, beta2_bias_correct, beta2_update
from exp6b.train_exp6b import _refresh_metrics


def test_noise_variance_floor_formula_and_zero_q_denominator():
    sigma, clip, batch, eps = 3.5, 1.0, 256, 1e-8
    noise_std = sigma * clip / batch
    floor = noise_std ** 2
    assert floor == (sigma * clip / batch) ** 2
    assert math.isclose((torch.sqrt(torch.tensor(floor)) + eps).item(),
                        noise_std + eps, rel_tol=1e-6, abs_tol=1e-8)
    config = {"M_syn": 1, "adam_eps": eps, "eps_num": 1e-12}
    row = _refresh_metrics({"x": torch.zeros(1)}, config, floor, None, 0)[0]
    assert math.isclose(row["denominator_q50"], noise_std + eps,
                        rel_tol=1e-6, abs_tol=1e-8)


def test_constant_q_ema_bias_correction_still_equals_q():
    beta = .999
    q = {"x": torch.tensor([3., 8.])}
    previous = {"x": torch.zeros(2)}
    for step in range(1, 6):
        previous = beta2_update(previous, q, beta)
        corrected = beta2_bias_correct(previous, beta, step)
        torch.testing.assert_close(corrected["x"], q["x"])


def test_second_moment_state_does_not_contain_floor():
    state = SecondMomentState(.999)
    q = {"x": torch.tensor([2.])}
    corrected = state.update(q)
    torch.testing.assert_close(corrected["x"], q["x"])
    torch.testing.assert_close(state.v["x"], torch.tensor([.002]))


def test_dp_adam_state_matches_exp6_state():
    from exp6.optimizers import ExplicitAdamState as Exp6AdamState
    from exp6b.optimizers import ExplicitAdamState as Exp6bAdamState

    torch.manual_seed(9)
    gradients = [[torch.randn(3), torch.randn(2)] for _ in range(3)]
    old = Exp6AdamState([torch.zeros(3), torch.zeros(2)], .9, .999, 1e-8)
    new = Exp6bAdamState([torch.zeros(3), torch.zeros(2)], .9, .999, 1e-8)
    for values in gradients:
        old_result = old.update(values)
        new_result = new.update(values)
        for old_values, new_values in zip(old_result, new_result):
            if isinstance(old_values, list):
                for old_value, new_value in zip(old_values, new_values):
                    torch.testing.assert_close(old_value, new_value)
            else:
                assert old_values == new_values
