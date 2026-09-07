import copy
import json
from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from torch.utils.data import TensorDataset

from exp5.common import DEFAULT, METHODS, RNGStream, SimpleCNN, digest
from exp5.optimizers import (FirstMomentState, SecondMomentState,
                             adam_candidate, beta1_bias_correct, beta1_ema,
                             beta2_diagnostics, beta2_update, state_bytes)


def test_beta1_ema_and_bias_correction_match_hand_formula():
    beta = .9
    old = [torch.tensor([1., 2.]), torch.tensor([3.])]
    grad = [torch.tensor([5., 7.]), torch.tensor([11.])]
    moment = beta1_ema(old, grad, beta)
    assert torch.equal(moment[0], beta * old[0] + (1 - beta) * grad[0])
    corrected = beta1_bias_correct(moment, beta, 2)
    torch.testing.assert_close(corrected[0], moment[0] / (1 - beta ** 2))


def test_explicit_first_moment_state_uses_step_bias_correction():
    state = FirstMomentState([torch.zeros(2)], .9)
    first = state.update([torch.tensor([2., 4.])])[0]
    second = state.update([torch.tensor([4., 8.])])[0]
    torch.testing.assert_close(first, torch.tensor([2., 4.]))
    expected_m = .9 * (.1 * torch.tensor([2., 4.])) + .1 * torch.tensor([4., 8.])
    torch.testing.assert_close(second, expected_m / (1 - .9 ** 2))


def test_first_moment_peek_does_not_mutate_and_update_does():
    state = FirstMomentState([torch.zeros(2)], .9)
    before = state.state_dict()
    peek = state.peek([torch.tensor([2., 4.])])[0]
    assert state.step == before["step"] == 0
    assert torch.equal(state.m[0], before["m"][0])
    torch.testing.assert_close(peek, torch.tensor([2., 4.]))
    state.update([torch.tensor([2., 4.])])
    assert state.step == 1
    assert not torch.equal(state.m[0], before["m"][0])


def test_beta2_uses_actual_refresh_gap_and_initializes_v_from_q0():
    q0 = torch.tensor([2.])
    q1 = torch.tensor([6.])
    state = SecondMomentState(.999)
    v0, previous0, delta0 = state.update({"x": q0}, 0)
    assert delta0 is None
    assert previous0 is None
    torch.testing.assert_close(v0["x"], q0)
    old_v = v0
    v1, previous1, delta1 = state.update({"x": q1}, 37)
    assert delta1 == 37
    assert previous1["x"] is old_v["x"]
    assert not hasattr(state, "last_previous")
    torch.testing.assert_close(v1["x"], beta2_update(q0, q1, .999, 37))


def test_beta2_diagnostics_match_float64_log_rms_and_first_refresh_is_empty():
    q0 = {"conv1": torch.tensor([1., 2.]), "fc1": torch.tensor([4.])}
    q1 = {"conv1": torch.tensor([2., 8.]), "fc1": torch.tensor([1.])}
    first = beta2_diagnostics(None, q0, q0, 1e-12)
    assert first["conv1"]["beta2_D_innovation"] is None
    current = {name: beta2_update(q0[name], q1[name], .999, 7) for name in q0}
    actual = beta2_diagnostics(q0, q1, current, 1e-12)
    old = q0["conv1"].double()
    expected_innovation = ((q1["conv1"].double() + 1e-12).log() - (old + 1e-12).log()).square().mean().sqrt()
    expected_ema = ((current["conv1"].double() + 1e-12).log() - (old + 1e-12).log()).square().mean().sqrt()
    assert actual["conv1"]["beta2_D_innovation"] == pytest.approx(float(expected_innovation), rel=1e-12)
    assert actual["conv1"]["beta2_D_ema"] == pytest.approx(float(expected_ema), rel=1e-12)


def test_beta2_diagnostics_is_bitwise_pure_for_all_inputs():
    previous = {"conv1": torch.tensor([1., 2.]), "fc1": torch.tensor([4.])}
    q = {"conv1": torch.tensor([2., 8.]), "fc1": torch.tensor([1.])}
    current = {name: beta2_update(previous[name], q[name], .999, 7) for name in previous}
    before = ({name: value.clone() for name, value in previous.items()},
              {name: value.clone() for name, value in q.items()},
              {name: value.clone() for name, value in current.items()})
    beta2_diagnostics(previous, q, current, 1e-12)
    for actual, expected in zip((previous, q, current), before):
        for name in expected:
            assert torch.equal(actual[name], expected[name])


def test_state_bytes_uses_tensor_dtype_and_nested_state():
    value = {"fp32": torch.zeros(3, dtype=torch.float32), "fp64": [torch.zeros(2, dtype=torch.float64)]}
    assert state_bytes(value) == 3 * 4 + 2 * 8


@pytest.mark.parametrize("kind", ["syn_diag", "dp_kfc"])
def test_precondition_clip_noise_then_update_order(kind):
    from exp3.preconditioners import apply, refresh, synthetic_samples
    from exp4.dynamics import aggregate
    from exp5.optimizers import adam_optimizer, apply_gradients
    from exp5.train_exp5 import make_diagonal_p

    c = dict(DEFAULT, batch_size=2, M_syn=2, analysis_batch_size=2, K=2)
    device = torch.device("cpu")
    torch.manual_seed(12)
    base = SimpleCNN().to(device)
    m1 = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    m2 = GradSampleModule(SimpleCNN().to(device), loss_reduction="sum")
    m1._module.load_state_dict(base.state_dict())
    m2._module.load_state_dict(base.state_dict())
    if kind == "syn_diag":
        active = ("syn_diag", {name: make_diagonal_p({name: torch.ones(1)}, c)[name].expand(
            getattr(base, name).weight.numel() + getattr(base, name).bias.numel()).clone()
            for name in ("conv1", "conv2", "fc1", "fc2")})
    else:
        samples, _ = synthetic_samples(c, device, RNGStream(33, device))
        active = refresh(m1._module.state_dict(), samples, c, device, "dp_kfc")
    x = torch.randn(2, 1, 28, 28)
    y = torch.tensor([1, 2])
    for model in (m1, m2):
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y, reduction="sum").backward()
    rng1, rng2 = RNGStream(44, device), RNGStream(44, device)
    apply(m1, active)
    first = aggregate(m1, None, rng1, 1.3, c, 2)
    second = aggregate(m2, active, rng2, 1.3, c, 2)
    assert first["noise_hash"] == second["noise_hash"]
    for a, b in zip(first["noisy"], second["noisy"]):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    if kind == "syn_diag":
        apply_gradients(m1.parameters(), first["noisy"], .001)
        apply_gradients(m2.parameters(), second["noisy"], .001)
    else:
        o1, o2 = adam_optimizer(m1.parameters(), c), adam_optimizer(m2.parameters(), c)
        for model, opt, grads in ((m1, o1, first["noisy"]), (m2, o2, second["noisy"])):
            for p, g in zip(model.parameters(), grads):
                p.grad = g
            opt.step()
    for a, b in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    m1.remove_hooks()
    m2.remove_hooks()


def test_beta2_off_is_direct_current_q_and_beta1_off_is_direct_direction():
    from exp5.train_exp5 import make_diagonal_p
    q = torch.tensor([3.])
    state = SecondMomentState(.999)
    v, _, _ = state.update({"x": q}, 0)
    torch.testing.assert_close(v["x"], q)
    p = make_diagonal_p({"x": q}, {"lambda": 1e-3})["x"]
    torch.testing.assert_close(p, 1 / (q.sqrt() + 1e-3))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_adam_candidate_matches_torch_adam_and_does_not_mutate_state(device):
    dev = torch.device(device)
    params = [torch.nn.Parameter(torch.randn(3, device=dev)), torch.nn.Parameter(torch.randn(2, device=dev))]
    opt = torch.optim.Adam(params, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    grads = [torch.randn_like(p) for p in params]
    before = copy.deepcopy(opt.state_dict())
    rng = digest([torch.get_rng_state()])
    direction = adam_candidate(params, opt, grads, .001)
    assert before["param_groups"] == opt.state_dict()["param_groups"]
    assert rng == digest([torch.get_rng_state()])
    actual = [torch.nn.Parameter(p.detach().clone()) for p in params]
    ref = torch.optim.Adam(actual, lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    ref.load_state_dict(copy.deepcopy(before))
    for p, g in zip(actual, grads):
        p.grad = g.clone()
    ref.step()
    for p, old, u in zip(actual, params, direction):
        torch.testing.assert_close(p, old - .001 * u, rtol=1e-6, atol=2e-7)


def test_rng_stream_replay_isolated():
    stream = RNGStream(4, "cpu")
    before = stream.audit()
    with stream.use():
        torch.randn(5)
    after = stream.audit()
    assert before != after


def test_six_method_pairing_and_diagnostics_isolation(tmp_path):
    from exp5.analyze_exp5 import load_runs
    from exp5.train_exp5 import train

    c = dict(DEFAULT, smoke=True, seeds=[42], device="cpu", threads=2,
             batch_size=4, epochs=1, M_syn=8, K=2, M_oracle=8, M_stale=8,
             analysis_batch_size=4, train_subset=8, test_subset=8,
             eval_interval=2, diagnostic_interval=2)
    torch.manual_seed(99)
    data = TensorDataset(torch.randn(8, 1, 28, 28), torch.arange(8) % 10)
    for method in METHODS:
        train(c, 42, method, tmp_path / "paired", (data, data))
    loaded, _ = load_runs(c, tmp_path / "paired")
    assert len(loaded) == len(METHODS)
    by_method = {meta["method"]: audit for _, meta, _, _, _, audit in loaded}
    for method in METHODS:
        for a, b in zip(by_method["dp_adam"]["private"], by_method[method]["private"]):
            assert (a["batch_hash"], a["noise_hash"]) == (b["batch_hash"], b["noise_hash"])
    for method in ("syn_diag", "syn_diag_beta1", "syn_diag_beta2", "syn_diag_beta12", "dp_kfc_adam"):
        for a, b in zip(by_method["syn_diag"]["synthetic"], by_method[method]["synthetic"]):
            assert (a["samples_hash"], a["labels_hash"]) == (b["samples_hash"], b["labels_hash"])
    for method in METHODS:
        on = train(c, 42, method, tmp_path / "on", (data, data), diagnostics=True)
        off = train(c, 42, method, tmp_path / "off", (data, data), diagnostics=False)
        assert on["final_model_hash"] == off["final_model_hash"]
        on_audit = json.loads((tmp_path / "on" / "seed42" / method / "pairing.json").read_text())
        off_audit = json.loads((tmp_path / "off" / "seed42" / method / "pairing.json").read_text())
        assert [(x["batch_hash"], x["noise_hash"]) for x in on_audit["private"]] == [(x["batch_hash"], x["noise_hash"]) for x in off_audit["private"]]


def _tiny_train_config():
    return dict(DEFAULT, smoke=True, seeds=[42], device="cpu", threads=1,
                batch_size=4, epochs=1, M_syn=4, K=1, M_oracle=4, M_stale=4,
                analysis_batch_size=4, train_subset=4, test_subset=4,
                eval_interval=1, diagnostic_interval=1, oracle_enabled=False)


def _tiny_data():
    return TensorDataset(torch.randn(4, 1, 28, 28), torch.arange(4) % 10)


@pytest.mark.parametrize("method", ["syn_diag", "syn_diag_beta12", "dp_kfc_adam"])
def test_refresh_events_precede_before_batch(tmp_path, method):
    from exp5.train_exp5 import train

    events = []
    train(_tiny_train_config(), 42, method, tmp_path, (_tiny_data(), _tiny_data()),
          event_hook=lambda kind, step, active: events.append(kind))
    expected = ["refresh_start"]
    if method == "syn_diag_beta12":
        expected += ["beta2_core_a_end", "beta2_diag_end",
                     "beta2_core_b_start", "beta2_core_b_end"]
    expected += ["refresh_end", "before_batch"]
    assert events == expected


def test_dp_adam_has_no_refresh_events(tmp_path):
    from exp5.train_exp5 import train

    events = []
    train(_tiny_train_config(), 42, "dp_adam", tmp_path, (_tiny_data(), _tiny_data()),
          event_hook=lambda kind, step, active: events.append(kind))
    assert events == ["before_batch"]


def test_candidates_only_run_inside_diagnostics_and_not_when_disabled(tmp_path, monkeypatch):
    import exp5.train_exp5 as train_module

    data = (_tiny_data(), _tiny_data())
    inside = [False]
    calls = []
    original_context = train_module.CostTracker.diagnostics
    original_candidate = train_module.adam_candidate
    original_advance = train_module.AdamDirection.advance
    original_refresh = train_module.refresh_active
    original_beta2_diagnostics = train_module.beta2_diagnostics

    @contextmanager
    def marked_diagnostics(self):
        with original_context(self):
            inside[0] = True
            try:
                yield
            finally:
                inside[0] = False

    def marked_candidate(*args, **kwargs):
        calls.append(("candidate", inside[0]))
        return original_candidate(*args, **kwargs)

    def marked_advance(self, *args, **kwargs):
        calls.append(("adam_direction", inside[0]))
        return original_advance(self, *args, **kwargs)

    def marked_refresh(*args, **kwargs):
        calls.append(("refresh", inside[0]))
        return original_refresh(*args, **kwargs)

    def marked_beta2_diagnostics(*args, **kwargs):
        calls.append(("beta2_diagnostics", inside[0]))
        return original_beta2_diagnostics(*args, **kwargs)

    monkeypatch.setattr(train_module.CostTracker, "diagnostics", marked_diagnostics)
    monkeypatch.setattr(train_module, "adam_candidate", marked_candidate)
    monkeypatch.setattr(train_module.AdamDirection, "advance", marked_advance)
    monkeypatch.setattr(train_module, "refresh_active", marked_refresh)
    monkeypatch.setattr(train_module, "beta2_diagnostics", marked_beta2_diagnostics)

    off = train_module.train(_tiny_train_config(), 42, "dp_adam", tmp_path / "off", data,
                             diagnostics=False)
    assert not [kind for kind, _ in calls if kind in ("candidate", "adam_direction")]
    assert all(not in_context for kind, in_context in calls if kind == "refresh")

    calls.clear()
    on = train_module.train(_tiny_train_config(), 42, "syn_diag", tmp_path / "on", data,
                            diagnostics=True)
    assert [kind for kind, _ in calls if kind == "adam_direction"]
    assert all(in_context for kind, in_context in calls if kind == "adam_direction")
    assert all(not in_context for kind, in_context in calls if kind == "refresh")
    assert on["diagnostic_seconds"] > 0
    assert on["core_wall_time"] == pytest.approx(on["wall_time"] - on["diagnostic_seconds"])

    calls.clear()
    train_module.train(_tiny_train_config(), 42, "syn_diag_beta2", tmp_path / "beta2", data,
                       diagnostics=True)
    assert [kind for kind, _ in calls if kind == "beta2_diagnostics"]
    assert all(in_context for kind, in_context in calls if kind == "beta2_diagnostics")
    assert all(not in_context for kind, in_context in calls if kind == "refresh")
    checkpoint = torch.load(tmp_path / "beta2" / "seed42" / "syn_diag_beta2" / "final_state.pt",
                           map_location="cpu")
    assert set(checkpoint["second_moment"]) == {"beta2", "last_refresh_step", "v"}
    assert off["final_model_hash"] == train_module.train(
        _tiny_train_config(), 42, "dp_adam", tmp_path / "off_again", data, diagnostics=False
    )["final_model_hash"]


def test_beta2_core_b_has_no_diagnostic_only_tensor_references(tmp_path, monkeypatch):
    import gc
    import weakref
    import exp5.train_exp5 as train_module

    original_core_a = train_module.refresh_beta2_core_a
    original_state = train_module.SecondMomentState
    references = {"q": [], "previous": []}
    states = []

    def marked_core_a(*args, **kwargs):
        q, v, previous, delta_t = original_core_a(*args, **kwargs)
        references["q"].append([weakref.ref(value) for value in q.values()])
        if previous is not None:
            references["previous"].append([weakref.ref(value) for value in previous.values()])
        return q, v, previous, delta_t

    class TrackingSecondMomentState(original_state):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            states.append(self)

    monkeypatch.setattr(train_module, "refresh_beta2_core_a", marked_core_a)
    monkeypatch.setattr(train_module, "SecondMomentState", TrackingSecondMomentState)
    events = []

    def check_event(kind, step, active):
        events.append(kind)
        if kind == "beta2_core_b_start":
            gc.collect()
            assert states and set(vars(states[-1])) == {"beta2", "v", "last_refresh_step"}
            assert all(all(reference() is None for reference in refs)
                       for refs in references["q"] + references["previous"])

    config = dict(_tiny_train_config(), epochs=2, K=1)
    train_module.train(config, 42, "syn_diag_beta2", tmp_path, (_tiny_data(), _tiny_data()),
                       event_hook=check_event)

    assert events.count("beta2_core_b_start") == 2
    assert states and all(set(vars(state)) == {"beta2", "v", "last_refresh_step"}
                          for state in states)
    assert all(all(reference() is None for reference in refs)
               for refs in references["q"] + references["previous"])


def test_beta2_hashes_and_metrics_are_diagnostic_only(tmp_path, monkeypatch):
    import exp5.train_exp5 as train_module

    inside = [False]
    q_v_hash_contexts = []
    metric_contexts = []
    original_context = train_module.CostTracker.diagnostics
    original_digest = train_module.digest
    original_metrics = train_module.beta2_diagnostics
    dict_values_type = type({}.values())

    @contextmanager
    def marked_diagnostics(self):
        with original_context(self):
            inside[0] = True
            try:
                yield
            finally:
                inside[0] = False

    def marked_digest(values):
        if isinstance(values, dict_values_type):
            q_v_hash_contexts.append(inside[0])
        return original_digest(values)

    def marked_metrics(*args, **kwargs):
        metric_contexts.append(inside[0])
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(train_module.CostTracker, "diagnostics", marked_diagnostics)
    monkeypatch.setattr(train_module, "digest", marked_digest)
    monkeypatch.setattr(train_module, "beta2_diagnostics", marked_metrics)
    train_module.train(_tiny_train_config(), 42, "syn_diag_beta2", tmp_path / "on",
                       (_tiny_data(), _tiny_data()), diagnostics=True)
    train_module.train(_tiny_train_config(), 42, "syn_diag_beta2", tmp_path / "off",
                       (_tiny_data(), _tiny_data()), diagnostics=False)

    assert q_v_hash_contexts and all(q_v_hash_contexts)
    assert metric_contexts and all(metric_contexts)


def test_beta2_p_construction_is_core_after_payload_release(tmp_path, monkeypatch):
    import gc
    import weakref
    import exp5.train_exp5 as train_module

    inside = [False]
    q_references = []
    original_context = train_module.CostTracker.diagnostics
    original_core_a = train_module.refresh_beta2_core_a
    original_make_p = train_module.make_diagonal_p

    @contextmanager
    def marked_diagnostics(self):
        with original_context(self):
            inside[0] = True
            try:
                yield
            finally:
                inside[0] = False

    def marked_core_a(*args, **kwargs):
        q, v, previous, delta_t = original_core_a(*args, **kwargs)
        q_references.append([weakref.ref(value) for value in q.values()])
        return q, v, previous, delta_t

    calls = []

    def marked_make_p(q, c):
        gc.collect()
        calls.append(inside[0])
        assert not inside[0]
        assert all(reference() is None for refs in q_references for reference in refs)
        return original_make_p(q, c)

    monkeypatch.setattr(train_module.CostTracker, "diagnostics", marked_diagnostics)
    monkeypatch.setattr(train_module, "refresh_beta2_core_a", marked_core_a)
    monkeypatch.setattr(train_module, "make_diagonal_p", marked_make_p)
    train_module.train(_tiny_train_config(), 42, "syn_diag_beta2", tmp_path,
                       (_tiny_data(), _tiny_data()), diagnostics=True)
    assert calls == [False]


def test_beta2_refresh_seconds_are_core_a_plus_core_b(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import exp5.train_exp5 as train_module

    clock_values = iter([0., 0., 0., 2., 2., 5., 400.])

    @contextmanager
    def simulated_diagnostics(self):
        self.diagnostic_seconds += 100.
        yield

    monkeypatch.setattr(train_module.CostTracker, "diagnostics", simulated_diagnostics)
    monkeypatch.setattr(train_module.time, "perf_counter", lambda: next(clock_values))
    result = train_module.train(_tiny_train_config(), 42, "syn_diag_beta2", tmp_path,
                                (_tiny_data(), _tiny_data()), diagnostics=True)
    assert result["total_refresh_time"] == pytest.approx(5.)
    assert result["core_wall_time"] == pytest.approx(100.)


@pytest.mark.parametrize("method", ["syn_diag_beta2", "syn_diag_beta12"])
def test_beta2_diagnostics_on_off_preserve_pairing_state_and_hashes(tmp_path, method):
    from exp5.train_exp5 import train

    data = (_tiny_data(), _tiny_data())
    on = train(_tiny_train_config(), 42, method, tmp_path / "on", data, diagnostics=True)
    off = train(_tiny_train_config(), 42, method, tmp_path / "off", data, diagnostics=False)
    assert on["final_model_hash"] == off["final_model_hash"]
    on_pairing = json.loads((tmp_path / "on" / "seed42" / method / "pairing.json").read_text())
    off_pairing = json.loads((tmp_path / "off" / "seed42" / method / "pairing.json").read_text())
    assert on_pairing["synthetic"] == off_pairing["synthetic"]
    assert {key: on_pairing["synthetic"][0][key] for key in ("q_hash", "v_hash")} == {
        key: off_pairing["synthetic"][0][key] for key in ("q_hash", "v_hash")}
    on_state = torch.load(tmp_path / "on" / "seed42" / method / "final_state.pt", map_location="cpu")
    off_state = torch.load(tmp_path / "off" / "seed42" / method / "final_state.pt", map_location="cpu")
    for a, b in zip(on_state["second_moment"]["v"].values(), off_state["second_moment"]["v"].values()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("method", ["syn_diag_beta2", "syn_diag_beta12"])
def test_temporal_state_bytes_count_only_algorithm_moments(tmp_path, method):
    from exp5.train_exp5 import state_bytes as tensor_state_bytes, train

    train(_tiny_train_config(), 42, method, tmp_path, (_tiny_data(), _tiny_data()), diagnostics=True)
    root = tmp_path / "seed42" / method
    summary = json.loads((root / "summary.json").read_text())
    checkpoint = torch.load(root / "final_state.pt", map_location="cpu")
    expected = tensor_state_bytes(checkpoint["second_moment"]["v"])
    if method == "syn_diag_beta12":
        expected += tensor_state_bytes(checkpoint["first_moment"]["m"])
    assert summary["temporal_state_bytes"] == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tiny_beta12_training_covers_full_refresh_path(tmp_path):
    from exp5.train_exp5 import train

    result = train(dict(_tiny_train_config(), device="cuda"), 42, "syn_diag_beta12", tmp_path,
                   (_tiny_data(), _tiny_data()), diagnostics=True)
    assert result["completed_steps"] == 1
    assert result["total_refresh_time"] > 0 and torch.isfinite(torch.tensor(result["total_refresh_time"]))
    state = torch.load(tmp_path / "seed42" / "syn_diag_beta12" / "final_state.pt", map_location="cpu")
    assert all(torch.isfinite(value).all() for value in state["model"].values())


def test_report_aggregates_costs_across_seeds():
    from exp5.analyze_exp5 import report

    rows = []
    for seed in (1, 2, 3):
        for method in METHODS:
            rows.append(dict(method=method, seed=seed, final_accuracy=.5,
                             best_accuracy=.6, late_mean_accuracy=.55, final_test_loss=1.,
                             wall_time=float(seed), core_wall_time=float(seed) / 2,
                             diagnostic_seconds=float(seed) / 2, total_refresh_time=float(seed),
                             mean_refresh_time=float(seed) / 2, peak_cuda_memory_core=seed,
                             peak_cuda_memory_overall=seed, preconditioner_state_bytes=10,
                             temporal_state_bytes=20, optimizer_state_bytes=30,
                             total_algorithm_state_bytes=60))
    import pandas as pd
    per_seed = pd.DataFrame(rows)
    contrasts = pd.DataFrame(columns=["seed", "mean", "std", "n", "contrast", "metric"])
    text = report(per_seed, contrasts, {"smoke": False})
    assert "2 +/- 1" in text
    assert "Preconditioner bytes" in text
    assert "| 10 | 20 | 30 | 60 |" in text
