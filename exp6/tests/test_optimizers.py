import copy

import torch

from exp6.optimizers import ExplicitAdamState, FirstMomentState, SecondMomentState


def test_constant_q_bias_correction():
    q = {"weight": torch.tensor([2.0, 5.0]), "bias": torch.tensor([7.0])}
    state = SecondMomentState(beta2=.999)
    for _ in range(5):
        corrected = state.update(q)
        for name in q:
            torch.testing.assert_close(corrected[name], q[name])


def test_explicit_dp_adam_matches_torch_adam_each_step():
    torch.manual_seed(9)
    params = [torch.nn.Parameter(torch.randn(3)), torch.nn.Parameter(torch.randn(2))]
    reference = [torch.nn.Parameter(value.detach().clone()) for value in params]
    actual = ExplicitAdamState(params, .9, .999, 1e-8)
    optimizer = torch.optim.Adam(reference, lr=.001, betas=(.9, .999),
                                 eps=1e-8, weight_decay=0.)
    for _ in range(4):
        gradients = [torch.randn_like(param) for param in params]
        direction, _, _, _ = actual.update(gradients)
        with torch.no_grad():
            for param, update in zip(params, direction):
                param.add_(update, alpha=-.001)
        for param, gradient in zip(reference, gradients):
            param.grad = gradient.clone()
        optimizer.step()
        for param, expected in zip(params, reference):
            torch.testing.assert_close(param, expected, rtol=2e-6, atol=2e-7)


def test_first_moment_bias_correction_uses_one_based_step():
    state = FirstMomentState([torch.zeros(1)], .9)
    first = state.update([torch.tensor([4.])])[0]
    torch.testing.assert_close(first, torch.tensor([4.]))
