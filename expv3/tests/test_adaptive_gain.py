import torch

from expv3.adaptive_fisher_wiener import rebuild_H_with_beta


def _state():
    la = torch.tensor([0.2, 1.0], dtype=torch.float32)
    lg = torch.tensor([0.5, 2.0], dtype=torch.float32)
    lf = lg[:, None] * la[None, :]
    h = lf / (lf + 0.25)
    return {"layer": {"Q_A": torch.eye(2), "lambda_A": la, "Q_G": torch.eye(2),
                       "lambda_G": lg, "H": h}}


def test_beta1_gain_equivalence():
    state = _state()
    result = rebuild_H_with_beta(state, {"layer": 1.0}, 0.25)
    assert torch.equal(result["layer"]["H"], state["layer"]["H"])


def test_beta_monotonicity():
    state = _state()
    low = rebuild_H_with_beta(state, {"layer": 0.1}, 0.25)["layer"]["H"]
    one = rebuild_H_with_beta(state, {"layer": 1.0}, 0.25)["layer"]["H"]
    high = rebuild_H_with_beta(state, {"layer": 10.0}, 0.25)["layer"]["H"]
    assert torch.all(low <= one) and torch.all(one <= high)
    assert torch.isfinite(high).all() and (high >= 0).all() and (high <= 1).all()


def test_layerwise_beta():
    state = {"a": _state()["layer"], "b": _state()["layer"]}
    result = rebuild_H_with_beta(state, {"a": 0.1, "b": 0.5}, 0.25)
    assert torch.all(result["a"]["H"] < result["b"]["H"])

