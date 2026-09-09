import copy
import math

import torch

from expv4a.gamma_estimators import (
    certificate_holds,
    model_gamma_from_eigenvalues,
    model_gamma_from_state,
)


def _state(h):
    return {
        "lambda_A": torch.tensor([1.0, 2.0]),
        "lambda_G": torch.tensor([3.0, 4.0]),
        "H": torch.full((2, 2), h),
    }


def test_identity_gain_has_unit_gamma():
    result = model_gamma_from_state(_state(1.0), 1.0)
    assert result["gamma_model_raw"] == 1.0


def test_constant_gain_has_inverse_gamma():
    result = model_gamma_from_state(_state(0.25), 1.0)
    assert result["gamma_model_raw"] == 4.0


def test_model_certificate_for_random_positive_gains():
    h = torch.tensor([[.1, .3], [.6, .9]])
    result = model_gamma_from_state({**_state(1.0), "H": h}, 2.0)
    assert certificate_holds(result["gamma_model_raw"], result["mean_H2"])


def test_gamma_does_not_modify_fisher_state():
    state = _state(.4)
    before = copy.deepcopy(state)
    model_gamma_from_state(state, 3.0)
    for key in state:
        assert torch.equal(state[key], before[key])


def test_eigenvalue_gamma_matches_adaptive_h_reference():
    lambda_a = torch.tensor([0.4, 1.3], dtype=torch.float64)
    lambda_g = torch.tensor([0.7, 2.1, 3.2], dtype=torch.float64)
    beta, r = 3.5, 0.2
    lambda_f = lambda_g[:, None] * lambda_a[None, :]
    scaled = beta * lambda_f
    h = scaled / (scaled + r)
    state = {
        "lambda_A": lambda_a,
        "lambda_G": lambda_g,
        "H": h,
    }
    expected = model_gamma_from_state(state, beta)
    actual = model_gamma_from_eigenvalues(lambda_a, lambda_g, beta, r)
    for field in (
        "gamma_model_raw", "model_signal_retention", "mean_H2",
        "model_noise_retention_after_raw",
    ):
        assert math.isclose(actual[field], expected[field], rel_tol=1e-12, abs_tol=1e-12)


def test_eigenvalue_gamma_responds_to_beta():
    lambda_a = torch.tensor([0.4, 1.3], dtype=torch.float64)
    lambda_g = torch.tensor([0.7, 2.1, 3.2], dtype=torch.float64)
    small = model_gamma_from_eigenvalues(lambda_a, lambda_g, 0.25, 0.2)
    large = model_gamma_from_eigenvalues(lambda_a, lambda_g, 4.0, 0.2)
    assert not math.isclose(
        small["gamma_model_raw"], large["gamma_model_raw"], rel_tol=1e-6, abs_tol=1e-12
    )
