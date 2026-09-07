import copy
import json

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
    v0, delta0 = state.update({"x": q0}, 0)
    assert delta0 is None
    torch.testing.assert_close(v0["x"], q0)
    v1, delta1 = state.update({"x": q1}, 37)
    assert delta1 == 37
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
    v, _ = state.update({"x": q}, 0)
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
