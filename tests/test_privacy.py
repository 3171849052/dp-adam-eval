import pytest
import torch
from dp_wiener_mnist.privacy import clip_and_noise_gradients
from dp_wiener_mnist.utils import RNGStream


def setup():
    model = torch.nn.Linear(2, 1)
    model.weight.grad_sample = torch.tensor([[[3.0, 0.0]], [[0.0, 0.0]]])
    model.bias.grad_sample = torch.tensor([[4.0], [2.0]])
    return model


@pytest.mark.parametrize("sigma", [0.0, 2.0])
def test_global_clipping_and_summed_noise(sigma):
    m = setup()
    rng = RNGStream(51, "cpu")
    ref = RNGStream(51, "cpu")
    factors = torch.tensor([1 / (5 + 1e-6), 1 / (2 + 1e-6)])
    expected = []
    with ref.use():
        for p in m.parameters():
            summed = (p.grad_sample.reshape(2, -1) * factors[:, None]).sum(0)
            noise = torch.randn_like(summed) * sigma * 1.0
            expected.append(((summed + noise) / 2).view_as(p))
    with rng.use():
        clip_and_noise_gradients(m, sigma, 1.0, 2)
    for p, e in zip(m.parameters(), expected):
        torch.testing.assert_close(p.grad, e, rtol=0, atol=0)
    assert rng.audit() == ref.audit()


def test_invalid_gradients():
    m = setup()
    m.bias.grad_sample[0] = float("nan")
    with pytest.raises(FloatingPointError):
        clip_and_noise_gradients(m, 1, 1, 2)
