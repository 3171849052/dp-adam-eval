import torch

from exp6.optimizers import beta2_bias_correct, beta2_update


def test_constant_q_hand_formula():
    beta = .999
    previous = {"x": torch.zeros(2)}
    q = {"x": torch.tensor([3., 8.])}
    for step in range(1, 6):
        previous = beta2_update(previous, q, beta)
        corrected = beta2_bias_correct(previous, beta, step)
        torch.testing.assert_close(corrected["x"], q["x"])
