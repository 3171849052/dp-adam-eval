"""Core math and optimizer integration only; the CLI smoke checks the run artifacts."""
import pytest
import torch
import torch.nn.functional as F
from opacus import GradSampleModule

from expv3 import train_expv3 as v3
from expv3.adaptive_fisher_wiener import rebuild_H_with_beta
from expv3.beta_controller import AdaptiveBetaController
from expv3.common import FISHER_METHOD, LAYERS, RNGStream, SimpleCNN
from expv4b import train_expv4b as v4b


@pytest.mark.parametrize("gamma", [1.0, 1e6])
def test_optimizer_receives_uncapped_post_wiener_gradient(monkeypatch, gamma):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = GradSampleModule(SimpleCNN(), loss_reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    before = [p.detach().clone() for p in model.parameters()]
    filtered = []

    def filter_gradient(model, active):
        for p in model.parameters():
            p.grad.mul_(0.25)
            filtered.append(p.grad.clone())

    monkeypatch.setattr(v3, "apply_fisher_wiener", filter_gradient)
    F.cross_entropy(model(torch.randn(2, 1, 28, 28)), torch.tensor([1, 2]), reduction="sum").backward()
    v3.private_update(
        model, {}, optimizer, RNGStream(46, torch.device("cpu")), 1.0,
        v4b.SMOKE_CONFIG, 2, FISHER_METHOD, 0.5,
        diagnostics=False, beta_measurement=False, oracle_diagnostics=False,
        gamma={name: gamma for name in LAYERS}, active_beta={name: 1.0 for name in LAYERS},
        gamma_experiment=v4b)
    for p, initial, wy in zip(model.parameters(), before, filtered):
        torch.testing.assert_close(p.grad, gamma * wy)
        torch.testing.assert_close(p, initial - 0.5 * gamma * wy)
    model.remove_hooks()


def test_current_interval_gamma_and_one_interval_lag():
    base = {name: dict(lambda_A=torch.tensor([1., 2.]), lambda_G=torch.tensor([3.]),
                       H=torch.tensor([[3./7., 6./10.]])) for name in LAYERS}
    for adaptive in (False, True):
        controller = AdaptiveBetaController(LAYERS)
        for interval, expected_beta in enumerate((1., 2. if adaptive else 1.)):
            betas = {name: controller.active_beta(name) for name in LAYERS}
            assert set(betas.values()) == {expected_beta}
            active = rebuild_H_with_beta(base, betas, 4.)
            gammas, _ = v4b.refresh_gamma(active, betas, interval*2, interval,
                                         dict(seed=42, run_id="test"))
            for name in LAYERS:
                a = expected_beta * torch.tensor([[3., 6.]], dtype=torch.float64)
                h = active[name]["H"].double()
                assert gammas[name] == pytest.approx(float((a.sum()/(a*h*h).sum()).sqrt()))
                controller.observe(name, 20., 2., 9.)
                assert controller.active_beta(name) == expected_beta
            controller.finalize_all(interval, apply=adaptive)
