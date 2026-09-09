import copy

import torch

from expv4a.gamma_estimators import (
    certificate_holds,
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
