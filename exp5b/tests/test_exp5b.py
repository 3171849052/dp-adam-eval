import inspect
import json
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

from exp5b.common import DEFAULT, METHODS, MOMENTUM_METHODS
from exp5b.optimizers import FirstMomentState, SecondMomentState, beta2_update, state_bytes


def test_first_moment_state_equivalence_across_all_momentum_methods():
    params = [torch.zeros(3), torch.zeros(2)]
    states = [FirstMomentState(params, .9) for _ in range(4)]
    gradients = [
        [torch.tensor([1., -2., 3.]), torch.tensor([4., 5.])],
        [torch.tensor([-2., 1., 0.5]), torch.tensor([3., -1.])],
        [torch.tensor([4., 2., -1.]), torch.tensor([-2., 7.])],
    ]
    for gradient in gradients:
        outputs = [state.update(gradient) for state in states]
        for output in outputs[1:]:
            for expected, actual in zip(outputs[0], output):
                assert torch.equal(expected, actual)
        deltas = [[-.1 * value for value in output] for output in outputs]
        for delta in deltas[1:]:
            for expected, actual in zip(deltas[0], delta):
                assert torch.equal(expected, actual)


def test_first_moment_formula_and_bias_correction():
    state = FirstMomentState([torch.zeros(2)], .9)
    first = state.update([torch.tensor([2., 4.])])[0]
    second = state.update([torch.tensor([4., 8.])])[0]
    expected_m = .9 * (.1 * torch.tensor([2., 4.])) + .1 * torch.tensor([4., 8.])
    torch.testing.assert_close(first, torch.tensor([2., 4.]))
    torch.testing.assert_close(second, expected_m / (1 - .9 ** 2))


def test_matched_momentum_is_not_raw_torch_sgd_momentum():
    source = inspect.getsource(FirstMomentState)
    assert "torch.optim.SGD" not in source
    state = FirstMomentState([torch.zeros(1)], .9)
    state.update([torch.tensor([2.])])
    actual = state.update([torch.tensor([4.])])[0]
    expected = (.9 * (.1 * torch.tensor([2.])) + .1 * torch.tensor([4.])) / (1 - .9 ** 2)
    torch.testing.assert_close(actual, expected)
    raw_sgd_style = (.9 * torch.tensor([2.]) + torch.tensor([4.]))
    assert not torch.allclose(actual, raw_sgd_style)


def test_all_six_configs_use_exact_learning_rate():
    assert set(METHODS) == {
        "syn_diag", "syn_diag_beta1", "syn_diag_beta2", "syn_diag_beta12",
        "dp_sgd_momentum", "dp_kfc_momentum",
    }
    assert DEFAULT["learning_rate"] == .1
    for method in METHODS:
        assert DEFAULT["learning_rate"] == .1
        assert method not in {"dp_adam", "dp_kfc_adam", "dp_sgd", "dp_kfc"}


def test_beta2_uses_true_private_step_delta():
    q0, q1, q2 = torch.tensor([2.]), torch.tensor([6.]), torch.tensor([10.])
    state = SecondMomentState(.999)
    v0, previous, delta = state.update({"x": q0}, 0)
    assert previous is None and delta is None
    v1, _, delta = state.update({"x": q1}, 37)
    assert delta == 37
    torch.testing.assert_close(v0["x"], q0)
    torch.testing.assert_close(v1["x"], beta2_update(q0, q1, .999, 37))
    v2, _, delta = state.update({"x": q2}, 87)
    assert delta == 50
    torch.testing.assert_close(v2["x"], beta2_update(v1["x"], q2, .999, 50))


def _tiny_config(**updates):
    config = dict(
        DEFAULT, smoke=True, seeds=[42], device="cpu", threads=1,
        total_private_steps=1, batch_size=4, epochs=1, M_syn=4, K=1,
        M_oracle=4, M_stale=4, analysis_batch_size=4, train_subset=4,
        test_subset=4, eval_interval=1, diagnostic_interval=1, oracle_enabled=False,
    )
    config.update(updates)
    return config


def _tiny_data():
    torch.manual_seed(123)
    return TensorDataset(torch.randn(4, 1, 28, 28), torch.arange(4) % 10)


@pytest.mark.parametrize("method", ["syn_diag_beta12", "dp_sgd_momentum", "dp_kfc_momentum"])
def test_diagnostics_on_off_preserves_training_pairing(tmp_path, method):
    from exp5b.train_exp5b import train

    data = (_tiny_data(), _tiny_data())
    on = train(_tiny_config(), 42, method, tmp_path / "on", data, diagnostics=True)
    off = train(_tiny_config(), 42, method, tmp_path / "off", data, diagnostics=False)
    assert on["final_model_hash"] == off["final_model_hash"]
    on_audit = json.loads((tmp_path / "on" / "seed42" / method / "pairing.json").read_text())
    off_audit = json.loads((tmp_path / "off" / "seed42" / method / "pairing.json").read_text())
    assert on_audit["private"] == off_audit["private"]
    if method in {"syn_diag_beta12", "dp_kfc_momentum"}:
        assert on_audit["synthetic"] == off_audit["synthetic"]


@pytest.mark.parametrize("method,expected", [
    ("syn_diag", ["refresh_start", "refresh_end", "before_private_batch"]),
    ("syn_diag_beta12", ["refresh_start", "beta2_core_a_end", "beta2_diag_end",
                          "beta2_core_b_start", "beta2_core_b_end", "refresh_end", "before_private_batch"]),
    ("dp_sgd_momentum", ["before_private_batch"]),
    ("dp_kfc_momentum", ["refresh_start", "refresh_end", "before_private_batch"]),
])
def test_refresh_event_order(tmp_path, method, expected):
    from exp5b.train_exp5b import train

    events = []
    train(_tiny_config(), 42, method, tmp_path, (_tiny_data(), _tiny_data()),
          diagnostics=False, event_hook=lambda kind, step, active: events.append(kind))
    assert events == expected


def test_all_six_tiny_end_to_end_validation_analysis_and_state_bytes(tmp_path, monkeypatch):
    from exp5b.analyze_exp5b import main as analyze_main
    from exp5b.train_exp5b import train
    from exp5b.validate_exp5b import main as validate_main

    config = _tiny_config()
    data = (_tiny_data(), _tiny_data())
    for method in METHODS:
        train(config, 42, method, tmp_path / "runs", data, diagnostics=True)

    for method in METHODS:
        root = tmp_path / "runs" / "seed42" / method
        checkpoint = torch.load(root / "final_state.pt", map_location="cpu")
        summary = json.loads((root / "summary.json").read_text())
        p = state_bytes(checkpoint["preconditioner"])
        m = state_bytes(checkpoint["first_moment"]["m"]) if checkpoint["first_moment"] else 0
        v = state_bytes(checkpoint["second_moment"]["v"]) if checkpoint["second_moment"] else 0
        expected = {
            "syn_diag": p, "syn_diag_beta1": p + m, "syn_diag_beta2": p + v,
            "syn_diag_beta12": p + m + v, "dp_sgd_momentum": m, "dp_kfc_momentum": p + m,
        }[method]
        assert summary["total_algorithm_state_bytes"] == expected
        assert summary["completed_steps"] == 1
        assert summary["optimizer_state_bytes"] == 0

    monkeypatch.setattr(sys, "argv", ["validate_exp5b", "--config", str(tmp_path / "config.json"),
                                        "--runs", str(tmp_path / "runs"), "--output", str(tmp_path / "runs")])
    (tmp_path / "config.json").write_text(json.dumps(config))
    validate_main()
    assert json.loads((tmp_path / "runs" / "validation.json").read_text())["passed"]

    monkeypatch.setattr(sys, "argv", ["analyze_exp5b", "--config", str(tmp_path / "config.json"),
                                        "--runs", str(tmp_path / "runs"), "--output", str(tmp_path / "runs")])
    analyze_main()
    for name in ("summary_per_seed.csv", "summary_across_seeds.csv", "paired_contrasts.csv",
                 "factorial_effects.csv", "report.md", "analysis.json"):
        assert (tmp_path / "runs" / name).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tiny_paths(tmp_path):
    from exp5b.train_exp5b import train

    config = _tiny_config(device="cuda")
    data = (_tiny_data(), _tiny_data())
    for method in ("syn_diag_beta12", "dp_sgd_momentum", "dp_kfc_momentum"):
        result = train(config, 42, method, tmp_path, data, diagnostics=False)
        assert result["completed_steps"] == 1
        assert torch.isfinite(torch.tensor(result["total_refresh_time"]))
        state = torch.load(tmp_path / "seed42" / method / "final_state.pt", map_location="cpu")
        assert all(torch.isfinite(value).all() for value in state["model"].values())
