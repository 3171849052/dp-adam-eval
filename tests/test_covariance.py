import torch
from dp_wiener_mnist.covariance import (
    compute_linear_covariances,
    compute_conv2d_covariances,
    accumulate_covariances,
)
from dp_wiener_mnist.types import CovariancePair


def test_linear_bias_and_mean():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    g = torch.tensor([[2.0], [4.0]])
    a, b = compute_linear_covariances(x, g)
    augmented = torch.cat([x, torch.ones(2, 1)], 1)
    torch.testing.assert_close(
        a, augmented.T @ augmented / 2 + 1e-5 * torch.eye(3), rtol=0, atol=0
    )
    torch.testing.assert_close(b, g.T @ g / 2 + 1e-5 * torch.eye(1), rtol=0, atol=0)
    avg = accumulate_covariances(
        [CovariancePair({"x": a}, {"x": b}), CovariancePair({"x": a * 3}, {"x": b * 3})]
    )
    assert torch.equal(avg.A["x"], torch.stack([a, a * 3]).mean(0))


def test_conv_patches():
    x = torch.arange(9.0).reshape(1, 1, 3, 3)
    g = torch.arange(4.0).reshape(1, 1, 2, 2)
    a, b = compute_conv2d_covariances(x, g, 2, 1, 0)
    patches = torch.tensor(
        [
            [0.0, 1.0, 3.0, 4.0, 1.0],
            [1.0, 2.0, 4.0, 5.0, 1.0],
            [3.0, 4.0, 6.0, 7.0, 1.0],
            [4.0, 5.0, 7.0, 8.0, 1.0],
        ]
    )
    torch.testing.assert_close(
        a, patches.T @ patches / 4 + 1e-5 * torch.eye(5), rtol=0, atol=0
    )
    torch.testing.assert_close(b, torch.tensor([[3.5 + 1e-5]]), rtol=0, atol=0)
