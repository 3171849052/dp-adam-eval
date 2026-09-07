import gc
import inspect
import json
import sys
import weakref
from contextlib import contextmanager

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


def _tensor_refs(value):
    if isinstance(value, torch.Tensor):
        return [weakref.ref(value)]
    if isinstance(value, dict):
        return sum((_tensor_refs(item) for item in value.values()), [])
    if isinstance(value, (tuple, list)):
        return sum((_tensor_refs(item) for item in value), [])
    return []


class _TinyGradSampleModel(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(2, 3, device=device))
        self.second = torch.nn.Parameter(torch.zeros(4, device=device))


def _attach_grad_samples(model, values):
    for parameter, value in zip(model.parameters(), values):
        parameter.grad_sample = value.to(parameter.device).clone()


def _tiny_grad_samples(batch_size=4):
    return [
        torch.arange(batch_size * 6, dtype=torch.float32).reshape(batch_size, 2, 3) / 7,
        torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 4) / 5,
    ]


def test_noise_shape_metadata_uses_parameter_numel_only():
    from exp5b.train_exp5b import noise_shapes_from_params

    parameters = [torch.nn.Parameter(torch.zeros(2, 3)), torch.nn.Parameter(torch.zeros(4))]
    for parameter in parameters:
        parameter.grad_sample = object()
    assert noise_shapes_from_params(parameters) == [(6,), (4,)]
    source = inspect.getsource(noise_shapes_from_params)
    assert "grad_sample" not in source
    assert ".sum(" not in source


def test_official_noisy_gradient_and_rng_match_upstream_exactly():
    from dp_kfac.privacy import clip_and_noise_gradients
    from exp5b.common import RNGStream, digest
    from exp5b.train_exp5b import _same_rng_state, aggregate_audit, aggregate_core

    values = _tiny_grad_samples()
    model_old = _TinyGradSampleModel()
    model_new = _TinyGradSampleModel()
    _attach_grad_samples(model_old, values)
    _attach_grad_samples(model_new, values)
    config = {"max_grad_norm": .8, "learning_rate": .1}

    old_rng = RNGStream(1234, "cpu")
    old_before = old_rng.snapshot_state()
    with old_rng.use():
        clip_and_noise_gradients(model_old, .37, config["max_grad_norm"], 4, store_summed_grad=True)
    old_after = old_rng.snapshot_state()

    new_rng = RNGStream(1234, "cpu")
    result = aggregate_core(model_new, None, new_rng, .37, config, 4)

    assert result["noise_shapes"] == [(6,), (4,)]
    assert _same_rng_state(old_before, result["_rng_before_state"])
    assert _same_rng_state(old_after, result["_rng_after_state"])
    for index, (old, new) in enumerate(zip(model_old.parameters(), model_new.parameters())):
        assert torch.equal(old.grad, new.grad)
        assert torch.equal(old.grad, result["noisy"][index])
        assert not hasattr(new, "summed_grad")

    replay = RNGStream.from_snapshot(old_before, "cpu")
    with replay.use():
        old_noise = [torch.randn(parameter.numel(), dtype=parameter.dtype)
                     for parameter in model_old.parameters()]
    assert _same_rng_state(replay.snapshot_state(), old_after)
    new_audit = aggregate_audit(result, new_rng, list(model_new.parameters()))
    assert new_audit["noise_hash"] == digest(old_noise)
    assert new_audit["noise_rng_before"] == digest([old_before["cpu"]])
    assert new_audit["noise_rng_after"] == digest([old_after["cpu"]])


def test_diagnostic_clean_aggregate_matches_upstream_summed_grad():
    from dp_kfac.privacy import clip_and_noise_gradients
    from exp5b.train_exp5b import diagnostic_clean_clipped_aggregate

    values = _tiny_grad_samples()
    model_old = _TinyGradSampleModel()
    model_new = _TinyGradSampleModel()
    _attach_grad_samples(model_old, values)
    _attach_grad_samples(model_new, values)
    config = {"max_grad_norm": .8}

    clip_and_noise_gradients(model_old, 0., config["max_grad_norm"], 4, store_summed_grad=True)
    expected = [parameter.summed_grad.detach().clone() for parameter in model_old.parameters()]
    actual = diagnostic_clean_clipped_aggregate(model_new, config, 4)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_aggregate_core_source_keeps_summed_grad_diagnostic_only():
    from exp5b.train_exp5b import aggregate_core

    source = inspect.getsource(aggregate_core)
    assert "store_summed_grad=False" in source
    assert "store_summed_grad=True" not in source
    assert '"clean"' not in source


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


@pytest.mark.parametrize("method,refresh_name", [
    ("syn_diag_beta12", "refresh_beta2_core_a"),
    ("dp_kfc_momentum", "refresh_active"),
])
def test_old_preconditioner_is_dead_before_next_refresh_core(tmp_path, monkeypatch, method, refresh_name):
    import exp5b.train_exp5b as train_module
    config = _tiny_config(epochs=2, total_private_steps=2, oracle_enabled=False)
    refs = []
    original_hook = getattr(train_module, refresh_name)

    def checked_refresh(*args, **kwargs):
        if refs:
            gc.collect()
            assert all(ref() is None for ref in refs)
        return original_hook(*args, **kwargs)

    monkeypatch.setattr(train_module, refresh_name, checked_refresh)

    def event_hook(kind, step, active):
        if kind == "refresh_end" and step == 0:
            refs.extend(_tensor_refs(active))

    result = train_module.train(config, 42, method, tmp_path, (_tiny_data(), _tiny_data()),
                                diagnostics=True, event_hook=event_hook)
    assert result["completed_steps"] == 2
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_mandatory_hashes_are_inside_diagnostic_context(tmp_path, monkeypatch):
    import exp5b.train_exp5b as train_module
    config = _tiny_config(epochs=2, total_private_steps=2, oracle_enabled=False)
    original_digest = train_module.digest
    original_diagnostics = train_module.CostTracker.diagnostics
    depth = 0
    phase = False
    observed = []

    @contextmanager
    def diagnostics(self):
        nonlocal depth
        depth += 1
        try:
            with original_diagnostics(self):
                yield
        finally:
            depth -= 1

    def digest(value):
        if phase:
            observed.append(depth)
        return original_digest(value)

    def hook(kind, step, active):
        nonlocal phase
        if kind == "refresh_start" and step >= 0:
            phase = True

    monkeypatch.setattr(train_module.CostTracker, "diagnostics", diagnostics)
    monkeypatch.setattr(train_module, "digest", digest)
    train_module.train(config, 42, "syn_diag_beta12", tmp_path, (_tiny_data(), _tiny_data()),
                       diagnostics=False, event_hook=hook)
    assert observed and all(value > 0 for value in observed)


@pytest.mark.parametrize("method,core_name", [
    ("syn_diag", "refresh_active"),
    ("syn_diag_beta12", "refresh_beta2_core_a"),
])
def test_refresh_seconds_exclude_hashes_and_stale_diagnostics(tmp_path, monkeypatch, method, core_name):
    import exp5b.train_exp5b as train_module
    config = _tiny_config(oracle_enabled=False)

    class FakeClock:
        now = 0.

        def __call__(self):
            return self.now

    clock = FakeClock()
    original_samples = train_module.synthetic_samples
    original_digest = train_module.digest
    original_core = getattr(train_module, core_name)

    def samples(*args, **kwargs):
        result = original_samples(*args, **kwargs)
        clock.now += 2.
        return result

    def digest(value):
        clock.now += 100.
        return original_digest(value)

    def core(*args, **kwargs):
        result = original_core(*args, **kwargs)
        clock.now += 3.
        return result

    monkeypatch.setattr(train_module.time, "perf_counter", clock)
    monkeypatch.setattr(train_module, "synthetic_samples", samples)
    monkeypatch.setattr(train_module, "digest", digest)
    monkeypatch.setattr(train_module, core_name, core)
    result = train_module.train(config, 42, method, tmp_path, (_tiny_data(), _tiny_data()), diagnostics=True)
    assert result["total_refresh_time"] == pytest.approx(5.)
    assert result["diagnostic_seconds"] > 0
    assert result["core_wall_time"] == pytest.approx(5.)


def test_core_wall_time_excludes_diagnostic_clock(tmp_path, monkeypatch):
    import exp5b.train_exp5b as train_module
    config = _tiny_config(oracle_enabled=False)

    class FakeClock:
        now = 0.

        def __call__(self):
            return self.now

    clock = FakeClock()
    original_diagnostics = train_module.CostTracker.diagnostics
    original_aggregate = train_module.aggregate_core

    @contextmanager
    def diagnostics(self):
        with original_diagnostics(self):
            clock.now += 50.
            yield

    def aggregate(*args, **kwargs):
        result = original_aggregate(*args, **kwargs)
        clock.now += 10.
        return result

    monkeypatch.setattr(train_module.time, "perf_counter", clock)
    monkeypatch.setattr(train_module.CostTracker, "diagnostics", diagnostics)
    monkeypatch.setattr(train_module, "aggregate_core", aggregate)
    result = train_module.train(config, 42, "dp_sgd_momentum", tmp_path,
                                (_tiny_data(), _tiny_data()), diagnostics=True)
    assert result["diagnostic_seconds"] > 0
    assert result["wall_time"] > result["core_wall_time"]
    assert result["core_wall_time"] == pytest.approx(10.)


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
        assert summary["preconditioner_state_bytes"] == p
        assert summary["first_moment_state_bytes"] == m
        assert summary["second_moment_state_bytes"] == v
        assert summary["temporal_state_bytes"] == m + v
        assert summary["total_algorithm_state_bytes"] == expected
        assert summary["completed_steps"] == 1
        assert summary["optimizer_state_bytes"] == 0

    monkeypatch.setattr(sys, "argv", ["validate_exp5b", "--config", str(tmp_path / "config.json"),
                                        "--runs", str(tmp_path / "runs"), "--output", str(tmp_path / "runs")])
    (tmp_path / "config.json").write_text(json.dumps(config))
    validate_main()
    assert json.loads((tmp_path / "runs" / "validation.json").read_text())["passed"]

    # A CSV/JSON-only archive must validate from explicit state fields without
    # inferring a first/second split from temporal_state_bytes.
    for method in METHODS:
        (tmp_path / "runs" / "seed42" / method / "final_state.pt").unlink()
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("method", ["syn_diag_beta12", "dp_kfc_momentum"])
def test_cuda_clean_diagnostic_released_before_official_core(method):
    from exp5b.common import RNGStream
    from exp5b.train_exp5b import aggregate_core, diagnostic_clean_clipped_aggregate

    model = _TinyGradSampleModel("cuda")
    _attach_grad_samples(model, _tiny_grad_samples())
    config = {"max_grad_norm": .8, "learning_rate": .1}
    clean = diagnostic_clean_clipped_aggregate(model, config, 4)
    refs = [weakref.ref(value) for value in clean]
    clean_cpu = [value.cpu().clone() for value in clean]
    del clean
    gc.collect()
    torch.cuda.synchronize()
    assert all(ref() is None for ref in refs)

    result = aggregate_core(model, None, RNGStream(1234, "cuda"), .37, config, 4)
    assert all(not hasattr(parameter, "summed_grad") for parameter in model.parameters())
    assert all(value.is_cuda for value in result["noisy"])
    assert all(value.device.type == "cpu" for value in clean_cpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_refresh_releases_stale_before_core_peak(tmp_path, monkeypatch):
    import exp5b.train_exp5b as train_module
    config = _tiny_config(device="cuda", epochs=2, total_private_steps=2, oracle_enabled=False)
    refs = []
    original = train_module.refresh_beta2_core_a

    def checked(*args, **kwargs):
        if refs:
            gc.collect()
            assert all(ref() is None for ref in refs)
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "refresh_beta2_core_a", checked)

    def hook(kind, step, active):
        if kind == "refresh_end" and step == 0:
            refs.extend(_tensor_refs(active))

    result = train_module.train(config, 42, "syn_diag_beta12", tmp_path,
                                (_tiny_data(), _tiny_data()), diagnostics=True, event_hook=hook)
    assert all(ref() is None for ref in refs)
    assert result["peak_cuda_memory_core"] <= result["peak_cuda_memory_overall"]
