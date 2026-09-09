import pytest
import torch
from dp_wiener_mnist.fisher_wiener import (
    build_fisher_state,
    apply_fisher_matrix,
    pack_layer_gradient,
    unpack_layer_gradient,
)
from dp_wiener_mnist.types import CovariancePair


@pytest.mark.parametrize("sigma", [0.0, 2.0])
def test_gain(sigma):
    a = torch.diag(torch.tensor([-1.0, 0.0, 2.0]))
    g = torch.diag(torch.tensor([0.0, 3.0]))
    state = build_fisher_state(CovariancePair({"l": a}, {"l": g}), sigma, 1.0, 4)["l"]
    lf = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])
    r = (sigma / 4) ** 2
    expected = torch.where(lf + r > 0, lf / (lf + r), torch.zeros_like(lf))
    torch.testing.assert_close(state["H"], expected, rtol=0, atol=0)
    assert all(t.dtype == torch.float32 for t in state.values())


def test_explicit_full_fisher():
    a = torch.tensor([[2.0, 0.3], [0.3, 1.0]])
    g = torch.tensor([[3.0, 0.1], [0.1, 2.0]])
    m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state = build_fisher_state(CovariancePair({"l": a}, {"l": g}), 2.0, 1.0, 2)["l"]
    f = torch.kron(a, g)
    expected = (
        ((f @ torch.linalg.inv(f + torch.eye(4))) @ m.T.contiguous().flatten())
        .reshape(2, 2)
        .T
    )
    torch.testing.assert_close(
        apply_fisher_matrix(m, state), expected, rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize(
    "layer",
    [
        torch.nn.Conv2d(2, 3, 3),
        torch.nn.Linear(3, 2),
        torch.nn.Linear(3, 2, bias=False),
    ],
)
def test_pack_roundtrip(layer):
    for p in layer.parameters():
        p.grad = torch.randn_like(p)
    before = [p.grad.clone() for p in layer.parameters()]
    matrix = pack_layer_gradient(layer)
    unpack_layer_gradient(layer, matrix)
    for p, expected in zip(layer.parameters(), before):
        assert torch.equal(p.grad, expected)
