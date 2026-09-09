import json

import torch
from opacus import GradSampleModule

from expv2 import beta_estimation
from expv2.common import LAYERS, SimpleCNN, RNGStream
from expv2.train_expv2 import make_optimizer, private_update, train


def test_diagnostic_isolation(config, tiny_data, tmp_path):
    train(config, 42, "dp_fisher_wiener_lr0p10", tmp_path / "on", tiny_data, diagnostics=True)
    train(config, 42, "dp_fisher_wiener_lr0p10", tmp_path / "off", tiny_data, diagnostics=False)
    on = tmp_path / "on" / "seed42" / "dp_fisher_wiener_lr0p10"
    off = tmp_path / "off" / "seed42" / "dp_fisher_wiener_lr0p10"
    assert json.loads((on / "summary.json").read_text())["final_model_hash"] == json.loads((off / "summary.json").read_text())["final_model_hash"]
    assert json.loads((on / "pairing.json").read_text()) == json.loads((off / "pairing.json").read_text())


def test_beta_mutation_isolation(config, tiny_data, tmp_path, monkeypatch):
    original = beta_estimation.single_step_beta

    def poisoned(*args, **kwargs):
        result = original(*args, **kwargs)
        result["beta_oracle_step"] = float("nan")
        result["beta_dp_step_raw"] = float("inf")
        return result

    monkeypatch.setattr(beta_estimation, "single_step_beta", poisoned)
    train(config, 42, "dp_sgd_lr0p10", tmp_path / "poisoned", tiny_data)
    monkeypatch.setattr(beta_estimation, "single_step_beta", original)
    train(config, 42, "dp_sgd_lr0p10", tmp_path / "clean", tiny_data)
    poisoned_root = tmp_path / "poisoned" / "seed42" / "dp_sgd_lr0p10"
    clean_root = tmp_path / "clean" / "seed42" / "dp_sgd_lr0p10"
    assert json.loads((poisoned_root / "summary.json").read_text())["final_model_hash"] == json.loads((clean_root / "summary.json").read_text())["final_model_hash"]
    assert json.loads((poisoned_root / "pairing.json").read_text()) == json.loads((clean_root / "pairing.json").read_text())


def test_dp_sgd_synthetic_measurement_isolation(config, tiny_data, tmp_path):
    train(config, 42, "dp_sgd_lr0p10", tmp_path / "measurement", tiny_data, beta_measurement=True)
    train(config, 42, "dp_sgd_lr0p10", tmp_path / "none", tiny_data, beta_measurement=False)
    measured = tmp_path / "measurement" / "seed42" / "dp_sgd_lr0p10"
    none = tmp_path / "none" / "seed42" / "dp_sgd_lr0p10"
    assert json.loads((measured / "summary.json").read_text())["final_model_hash"] == json.loads((none / "summary.json").read_text())["final_model_hash"]
    assert json.loads((measured / "pairing.json").read_text())["private"] == json.loads((none / "pairing.json").read_text())["private"]


def _active_zero(model):
    state = {}
    for name in LAYERS:
        layer = getattr(model._module, name)
        a, g = layer.weight[0].numel() + 1, layer.weight.shape[0]
        state[name] = {"Q_A": torch.eye(a), "Q_G": torch.eye(g), "H": torch.zeros(g, a)}
    return state


def test_pre_filter_y(config, monkeypatch):
    model = GradSampleModule(SimpleCNN(), loss_reduction="sum")
    x = torch.randn(4, 1, 28, 28, generator=torch.Generator().manual_seed(5))
    y = torch.tensor([0, 1, 2, 3])
    torch.nn.functional.cross_entropy(model(x), y, reduction="sum").backward()
    trace = {name: {"trace_A": 1.0, "trace_G": 1.0, "trace_F": 1.0} for name in LAYERS}
    seen = []
    original = beta_estimation.single_step_beta

    def spy(*args, **kwargs):
        seen.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(beta_estimation, "single_step_beta", spy)
    opt = make_optimizer(model, 0.1, config)
    private_update(model, _active_zero(model), opt, RNGStream(9, "cpu"), 1.0, config, 4,
                   "dp_fisher_wiener", 0.1, trace_state=trace, run_id="toy")
    assert seen and any(value > 0 for value in seen)
    assert all(torch.equal(getattr(model._module, name).weight.grad, torch.zeros_like(getattr(model._module, name).weight.grad)) for name in LAYERS)
    model.remove_hooks()


def test_packed_dimension(config):
    model = GradSampleModule(SimpleCNN(), loss_reduction="sum")
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    from expv1.fisher_wiener import pack_layer_gradient
    for name in LAYERS:
        layer = getattr(model._module, name)
        assert pack_layer_gradient(layer).numel() == layer.weight.numel() + layer.bias.numel()
    model.remove_hooks()

