import copy
import json

import pytest
import torch
from torch.utils.data import TensorDataset

from exp5.common import DEFAULT, METHODS, RNGStream, SimpleCNN, digest
from exp5.optimizers import (FirstMomentState, SecondMomentState,
                             adam_candidate, beta1_bias_correct, beta1_ema,
                             beta2_update)


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


def test_beta2_off_is_direct_current_q_and_beta1_off_is_direct_direction():
    q = torch.tensor([3.])
    state = SecondMomentState(.999)
    v, _ = state.update({"x": q}, 0)
    torch.testing.assert_close(v["x"], q)
    gradient = [torch.tensor([1., -2.])]
    assert all(torch.equal(a, b) for a, b in zip(gradient, gradient))


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
    on = train(c, 42, "dp_adam", tmp_path / "on", (data, data), diagnostics=True)
    off = train(c, 42, "dp_adam", tmp_path / "off", (data, data), diagnostics=False)
    assert on["final_model_hash"] == off["final_model_hash"]
