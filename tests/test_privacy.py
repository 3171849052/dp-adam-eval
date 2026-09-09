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


def test_expected_batch_size_is_not_actual_sample_count():
    m = setup()
    clip_and_noise_gradients(
        m,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        sample_count=2,
        expected_batch_size=4,
    )
    expected_weight = torch.tensor([[3.0 / (5 + 1e-6), 0.0]]) / 4
    expected_bias = torch.tensor([4.0 / (5 + 1e-6) + 2.0 / (2 + 1e-6)]) / 4
    torch.testing.assert_close(m.weight.grad, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(m.bias.grad, expected_bias, rtol=0, atol=0)


def test_empty_batch_is_pure_noise_and_uses_expected_denominator():
    m = torch.nn.Linear(2, 1)
    actual = RNGStream(51, "cpu")
    reference = RNGStream(51, "cpu")
    with reference.use():
        expected_weight = torch.randn_like(m.weight) / 4
        expected_bias = torch.randn_like(m.bias) / 4
    with actual.use():
        clip_and_noise_gradients(
            m,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            sample_count=0,
            expected_batch_size=4,
        )
    torch.testing.assert_close(m.weight.grad, expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(m.bias.grad, expected_bias, rtol=0, atol=0)
    assert actual.audit() == reference.audit()
