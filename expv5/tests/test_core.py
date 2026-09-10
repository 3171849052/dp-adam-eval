"""Only optimizer input, real updates, and beta timing; CLI smoke covers artifacts."""
import json
from pathlib import Path
import pytest
import torch
from opacus import GradSampleModule
from expv1.fisher_wiener import pack_layer_gradient, unpack_layer_gradient
from expv3 import train_expv3 as v3
from expv3.adaptive_fisher_wiener import rebuild_H_with_beta
from expv3.beta_controller import AdaptiveBetaController
from expv3.common import LAYERS, RNGStream
from expv5 import train_expv5 as v5


@pytest.mark.parametrize("arm", v5.ARMS)
def test_actual_adam_input_moments_and_parameter_delta(monkeypatch, arm):
    torch.set_num_threads(2)
    torch.manual_seed(123)
    config = json.loads(Path("expv5/configs/smoke.json").read_text())
    spec = next(s for s in v5.run_specs(config) if s["run_id"] == arm)
    assert not spec["gamma"]
    base_model = torch.nn.Sequential()
    for name in LAYERS:
        base_model.add_module(name, torch.nn.Linear(3, 3))
    model = GradSampleModule(base_model)
    optimizer = v5.make_optimizer(model, spec["learning_rate"], config)
    group = optimizer.param_groups[0]
    assert (group["lr"], group["betas"], group["eps"], group["weight_decay"]) == (.001, (.9, .999), 1e-8, 0)
    reference = torch.nn.Sequential()
    for name in LAYERS:
        reference.add_module(name, torch.nn.Linear(3, 3))
    reference.load_state_dict(model._module.state_dict())
    reference_adam = torch.optim.Adam(reference.parameters(), lr=.001, betas=(.9, .999), eps=1e-8)
    base = {name: dict(Q_A=torch.eye(4), Q_G=torch.eye(3),
                       lambda_A=torch.arange(1., 5.), lambda_G=torch.arange(1., 4.)) for name in LAYERS}
    for state in base.values():
        lf = state["lambda_G"][:, None] * state["lambda_A"][None, :]
        state["H"] = lf / (lf + 4.)
    beta = 3. if spec["adaptive_beta"] else 1.
    active = rebuild_H_with_beta(base, {name: beta for name in LAYERS}, 4.)
    captured = {}
    real_dp = v3.clip_and_noise_gradients
    def capture(*args, **kwargs):
        real_dp(*args, **kwargs)
        captured.update({name: pack_layer_gradient(getattr(model._module, name)).clone() for name in LAYERS})
    monkeypatch.setattr(v3, "clip_and_noise_gradients", capture)
    for step in range(2):
        before = {name: [p.detach().clone() for p in getattr(model._module, name).parameters()] for name in LAYERS}
        for p in model.parameters():
            p.grad_sample = torch.randn(2, *p.shape)
            p.grad = torch.zeros_like(p)
        v3.private_update(model, active, optimizer, RNGStream(127 + step, torch.device("cpu")),
                          1., config, 2, spec["method"], .001,
                          diagnostics=False, beta_measurement=False, oracle_diagnostics=False)
        for name in LAYERS:
            expected = captured[name]
            if arm != v5.ARMS[0]:
                lf = base[name]["lambda_G"][:, None] * base[name]["lambda_A"][None, :]
                expected = expected * (beta * lf / (beta * lf + 4.))
            torch.testing.assert_close(pack_layer_gradient(getattr(model._module, name)), expected)
            unpack_layer_gradient(getattr(reference, name), expected)
            actual = v5.norm([p - old for p, old in zip(getattr(model._module, name).parameters(), before[name])])
            assert optimizer.layer_metrics[name]["parameter_update_norm"] == pytest.approx(actual)
        reference_adam.step()
        for p, ref in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, ref)
            for field in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[p][field], reference_adam.state[ref][field])
    model.remove_hooks()


def test_beta_lag_and_hold_last_positive():
    controller = AdaptiveBetaController(LAYERS)
    for raw, expected in [(3., 3.), (-2., 3.)]:
        previous = {name: controller.active_beta(name) for name in LAYERS}
        for _ in range(2):
            for name in LAYERS:
                controller.observe(name, 10 + raw * 2, 10, 2)
                assert controller.active_beta(name) == previous[name]
        controller.finalize_all()
        assert all(controller.active_beta(name) == expected for name in LAYERS)
