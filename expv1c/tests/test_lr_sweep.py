import copy
import inspect

import torch
from dp_kfac.types import CovariancePair
from opacus import GradSampleModule

from expv1c.common import LAYERS, FISHER_LRS, add_effective_signal_metrics
from expv1c.train_expv1c import make_optimizer, private_update
from expv1c import train_expv1c as train_module
from expv1c import common
from expv1.fisher_wiener import build_fisher_state


def test_lr_grid_exact():
    assert list(FISHER_LRS) == [0.8, 1.0, 1.2, 1.5]
    assert list(common.RUN_SPEC_BY_ID) == [
        "dp_sgd_lr0p10", "dp_fisher_wiener_lr0p80", "dp_fisher_wiener_lr1p00",
        "dp_fisher_wiener_lr1p20", "dp_fisher_wiener_lr1p50",
    ]


def test_filter_state_is_lr_independent():
    generator = torch.Generator().manual_seed(11)
    A = torch.randn(4, 4, generator=generator)
    G = torch.randn(3, 3, generator=generator)
    cov = CovariancePair(A={"layer": A @ A.T + torch.eye(4)}, G={"layer": G @ G.T + torch.eye(3)})
    state_a = build_fisher_state(cov, sigma=2.0, max_grad_norm=1.0, batch_size=4)
    state_b = build_fisher_state(cov, sigma=2.0, max_grad_norm=1.0, batch_size=4)
    for key in ("Q_A", "Q_G", "lambda_A", "lambda_G", "H"):
        torch.testing.assert_close(state_a["layer"][key], state_b["layer"][key], rtol=0, atol=0)
    assert "learning_rate" not in inspect.signature(build_fisher_state).parameters


def _state(model):
    result = {}
    for name in LAYERS:
        layer = getattr(model._module, name)
        a, g = layer.weight[0].numel() + 1, layer.weight.shape[0]
        result[name] = {
            "Q_A": torch.eye(a), "Q_G": torch.eye(g), "H": torch.ones(g, a),
            "lambda_A": torch.ones(a), "lambda_G": torch.ones(g),
        }
    return result


def _prepared(seed, x, y):
    torch.manual_seed(seed)
    model = GradSampleModule(__import__("expv1.common", fromlist=["SimpleCNN"]).SimpleCNN(), loss_reduction="sum")
    torch.nn.functional.cross_entropy(model(x), y, reduction="sum").backward()
    return model


def test_lr_only_optimizer(config):
    x = torch.randn(4, 1, 28, 28, generator=torch.Generator().manual_seed(90))
    y = torch.tensor([0, 1, 2, 3])
    a, b = _prepared(91, x, y), _prepared(91, x, y)
    before = [parameter.detach().clone() for parameter in a.parameters()]
    opt_a = make_optimizer(a, 0.8, config)
    opt_b = make_optimizer(b, 1.2, config)
    private_update(a, _state(a), opt_a, common.RNGStream(99, "cpu"), 2.0, config, 4,
                   "dp_fisher_wiener", 0.8, diagnostics=False)
    private_update(b, _state(b), opt_b, common.RNGStream(99, "cpu"), 2.0, config, 4,
                   "dp_fisher_wiener", 1.2, diagnostics=False)
    for old, pa, pb in zip(before, a.parameters(), b.parameters()):
        delta_a = old - pa
        delta_b = old - pb
        torch.testing.assert_close(delta_b, 1.5 * delta_a, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)
    a.remove_hooks(); b.remove_hooks()


def test_effective_signal_lr():
    row = {"signal_retention": 0.25, "filtered_gradient_norm": 2.0, "clean_clipped_norm": 1.0}
    add_effective_signal_metrics(row, 0.3)
    assert row["signal_amplitude_retention"] == 0.5
    assert row["effective_signal_lr"] == 0.15
    identity = {"signal_retention": 1.0, "filtered_gradient_norm": 2.0, "clean_clipped_norm": 1.0}
    add_effective_signal_metrics(identity, 0.1)
    assert identity["effective_signal_lr"] == 0.1


def test_threshold_steps():
    from expv1c.common import threshold_steps
    evaluations = [
        {"step": 99, "test_accuracy": 0.55}, {"step": 199, "test_accuracy": 0.72},
        {"step": 299, "test_accuracy": 0.81}, {"step": 399, "test_accuracy": 0.84},
        {"step": 499, "test_accuracy": 0.86},
    ]
    assert [threshold_steps(evaluations, x) for x in (0.5, 0.7, 0.8, 0.85)] == [100, 200, 300, 500]
    assert threshold_steps(evaluations, 0.9) is None


    evaluations.append({"step": 599, "test_accuracy": .90})
    assert threshold_steps(evaluations, .9) == 600
