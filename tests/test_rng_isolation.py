import inspect
import torch
from opacus import GradSampleModule
from dp_wiener_mnist.fisher_wiener import synthetic_samples, build_covariances
from dp_wiener_mnist.model import SimpleCNN
from dp_wiener_mnist.utils import RNGStream, digest


def test_real_refresh_isolation():
    torch.manual_seed(42)
    model = GradSampleModule(SimpleCNN(), loss_reduction="sum")
    x = torch.randn(4, 1, 28, 28)
    y = torch.arange(4)
    torch.nn.functional.cross_entropy(model(x), y, reduction="sum").backward()
    old = digest(model.parameters())
    grads = digest(p.grad for p in model.parameters())
    samples_before = digest(p.grad_sample for p in model.parameters())
    noise = RNGStream(46, "cpu")
    control = RNGStream(46, "cpu")
    synthetic = RNGStream(45, "cpu")
    global_before = torch.get_rng_state().clone()
    c = {"batch_size": 4, "M_syn": 8}
    probes, _ = synthetic_samples(c, torch.device("cpu"), synthetic)
    cov = build_covariances(model._module.state_dict(), probes, c, torch.device("cpu"))
    assert len(cov.A) == 4
    assert torch.equal(global_before, torch.get_rng_state())
    with noise.use():
        actual = torch.randn(100)
    with control.use():
        expected = torch.randn(100)
    assert torch.equal(actual, expected)
    assert old == digest(model.parameters()) and grads == digest(
        p.grad for p in model.parameters()
    )
    assert samples_before == digest(p.grad_sample for p in model.parameters())
    assert list(inspect.signature(build_covariances).parameters) == [
        "model_state",
        "samples",
        "c",
        "dev",
    ]
    model.remove_hooks()
