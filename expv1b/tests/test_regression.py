import json

import torch
from opacus import GradSampleModule

from expv1.common import LAYERS, SimpleCNN, RNGStream, read_config
from expv1.train_expv1 import private_update as expv1_private_update, make_optimizer as expv1_optimizer
from expv1b.train_expv1b import make_optimizer, private_update
from expv1b.validate_expv1b import ANCHORS


def _active(model):
    state = {}
    for name in LAYERS:
        layer = getattr(model._module, name)
        a, g = layer.weight[0].numel() + 1, layer.weight.shape[0]
        state[name] = {"Q_A": torch.eye(a), "Q_G": torch.eye(g), "H": torch.ones(g, a)}
    return state


def _model_with_grad(seed, x, y):
    torch.manual_seed(seed)
    model = GradSampleModule(SimpleCNN(), loss_reduction="sum")
    torch.nn.functional.cross_entropy(model(x), y, reduction="sum").backward()
    return model


def test_regression(config):
    x = torch.randn(4, 1, 28, 28, generator=torch.Generator().manual_seed(301))
    y = torch.tensor([1, 2, 3, 4])
    for method in ("dp_sgd", "dp_fisher_wiener"):
        reference = _model_with_grad(302, x, y)
        actual = _model_with_grad(302, x, y)
        state = None if method == "dp_sgd" else _active(reference)
        reference_state = None if method == "dp_sgd" else _active(actual)
        ref_opt = expv1_optimizer(reference, read_config("expv1/configs/smoke.json"))
        new_opt = make_optimizer(actual, 0.1, config)
        ref = expv1_private_update(reference, state, ref_opt, RNGStream(306, "cpu"), 2.0, config, 4, method, diagnostics=False)
        got = private_update(actual, reference_state, new_opt, RNGStream(306, "cpu"), 2.0, config, 4, method, 0.1, diagnostics=False)
        assert ref[3] == got[3]
        for left, right in zip(reference.parameters(), actual.parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        reference.remove_hooks(); actual.remove_hooks()
    formal = "expv1/runs/formal_20260908_182603"
    for run_id, expected in ANCHORS.items():
        from pathlib import Path
        summary = json.loads((Path(formal) / "seed42" / ("dp_sgd" if run_id.startswith("dp_sgd") else "dp_fisher_wiener") / "summary.json").read_text())
        assert summary["final_model_hash"] == expected
