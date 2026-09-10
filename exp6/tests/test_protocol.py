import torch
from opacus import GradSampleModule

from exp6.common import DEFAULT, SimpleCNN, RNGStream
from exp6.train_exp6 import clip_and_noise_gradients, private_update


def test_synthetic_state_does_not_change_private_grad_sample_before_clip(monkeypatch):
    config = dict(DEFAULT, smoke=True, device="cpu", threads=1, batch_size=2)
    model = GradSampleModule(SimpleCNN(), loss_reduction="sum")
    x = torch.randn(2, 1, 28, 28)
    y = torch.tensor([1, 2])
    model.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(x), y, reduction="sum").backward()
    before = [param.grad_sample.detach().clone() for param in model.parameters()]
    observed = []
    original = clip_and_noise_gradients

    def capture(*args, **kwargs):
        observed.extend(param.grad_sample.detach().clone() for param in model.parameters())
        return original(*args, **kwargs)

    monkeypatch.setattr("exp6.train_exp6.clip_and_noise_gradients", capture)
    private_update(model, RNGStream(46, "cpu"), 1., config, 2)
    for actual, expected in zip(observed, before):
        torch.testing.assert_close(actual, expected)
    model.remove_hooks()
