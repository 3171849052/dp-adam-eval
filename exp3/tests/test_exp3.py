import copy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset
from exp3.common import DEFAULT, ROOT, METHODS, RNGStream, read_config
from exp3.preconditioners import make_p, state_bytes, synthetic_samples, apply
from exp3.geometry import spread, gram_spectrum, positive_eigenvalues, compare, stale
from exp3.metrics import norm_metrics
from exp3.summarize_exp3 import geometric_mean, late_stats, paired_deltas
from exp3.train_exp3 import make_optimizer, private_update, train


@pytest.mark.parametrize("method", METHODS)
def test_plain_sgd(method):
    model = torch.nn.Linear(2, 1)
    optimizer = make_optimizer(model, DEFAULT)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == .1
    assert optimizer.param_groups[0]["momentum"] == 0


def test_syndiag_exact_formula():
    v = torch.tensor([0., 1e-6, 4.], dtype=torch.float64)
    p = make_p({"a": v}, .001)["a"]
    assert torch.equal(p, 1/(v.sqrt()+.001))
    assert not torch.allclose(p, 1/(v+.001).sqrt())


def test_precondition_before_clip():
    events = []
    def clip_noise(*args, **kwargs):
        events.extend(["clip", "noise"])
    class Optim:
        def step(self): events.append("sgd")
    model = torch.nn.Linear(1, 1)
    with patch("exp3.train_exp3.apply", side_effect=lambda *a: events.append("kfac")), \
         patch("exp3.train_exp3.before_clip", return_value={"clipped_aggregate_norm": 1}), \
         patch("exp3.train_exp3._compute_per_sample_norms_squared", return_value=torch.ones(2)), \
         patch("exp3.train_exp3.clip_and_noise_gradients", clip_noise), \
         patch("exp3.train_exp3.after_noise", return_value={"expected_noise_norm": 1}):
        private_update(model, ("dp_kfc", ({}, {})), Optim(), RNGStream(42, "cpu"), 1, DEFAULT, 2)
    assert events == ["kfac", "clip", "noise", "sgd"]


def test_matched_schedule_and_budgets():
    c = read_config(ROOT / "configs/full.json")
    assert c["K"] == 50 and c["M_syn"] == 2560
    assert list(range(0,1170,c["K"])) == [50*i for i in range(24)]
    assert c["damping"] == c["lambda"] == .001


def test_synthetic_pairing():
    c = dict(DEFAULT, batch_size=4, M_syn=8)
    a, b = RNGStream(45, "cpu"), RNGStream(45, "cpu")
    for _ in range(2):
        x, ax = synthetic_samples(c, torch.device("cpu"), a)
        y, ay = synthetic_samples(c, torch.device("cpu"), b)
        assert ax == ay and all(torch.equal(v,w) for v,w in zip(x,y))
    _, different = synthetic_samples(c, torch.device("cpu"), RNGStream(10,"cpu"))
    assert different["samples_hash"] != ax["samples_hash"]


def test_norm_cv_hand():
    result = norm_metrics(torch.tensor([1.,3.]), eps=0)
    assert result["norm_mean"] == 2 and result["norm_std"] == 1
    assert result["norm_cv"] == .5


def test_diagonal_ratio_hand():
    raw = torch.tensor([1.,100.,10000.],dtype=torch.float64)
    pre = raw.sqrt()
    assert spread(pre,0)/spread(raw,0) == pytest.approx(.5)


def test_gram_matches_direct_fisher():
    h = np.random.default_rng(2).normal(size=(5,9))
    eig, _ = gram_spectrum(h, 2)
    direct, _ = positive_eigenvalues(torch.tensor(h.T@h/5), 9)
    torch.testing.assert_close(eig, direct, atol=1e-12, rtol=1e-12)


def test_full_ratio_hand_and_numerical_zeros():
    h = np.diag([1.,10.,100.,0.])
    eig,_ = gram_spectrum(h)
    pre,_ = gram_spectrum(np.diag([1.,np.sqrt(10),10.,0.]))
    assert len(eig) == 3 and spread(pre,0)/spread(eig,0) == pytest.approx(.5)
    assert torch.isfinite(eig).all()


def test_staleness_old_new():
    new = dict(A=2., S=3.)
    first = stale(None,new)
    assert first["A_old_diag"] is None and first["delta_stale_full"] is None
    row = stale(dict(A=5., S=1.),new)
    assert row["delta_stale_diag"] == 3 and row["delta_stale_full"] == -2


def test_actual_state_bytes():
    assert state_bytes(None) == 0
    assert state_bytes(("syn_diag", {"a":torch.ones(7)})) == 28
    assert state_bytes(("dp_kfc", ({"a":torch.ones(2,2)}, {"a":torch.ones(3,3)}))) == 52


def test_late_median_iqr():
    f = pd.DataFrame(dict(step=[1,2,3,4], x=[999,999,2.,6.]))
    assert late_stats(f,"x",4) == dict(median=4.,q25=3.,q75=5.,iqr=2.)
    assert late_stats(f,"x",8)["median"] is None


def test_geometric_mean():
    assert geometric_mean([1,4,1,4]) == pytest.approx(2.)
    with pytest.raises(ValueError): geometric_mean([0,1])


def test_paired_delta():
    f = pd.DataFrame([dict(seed=s, method=m, x=s+v) for s in [42,7,91] for m,v in zip(METHODS,[1,3,6])])
    rows = paired_deltas(f,[42,7,91])
    means = {r["pair"]:r for r in rows if r["seed"] == "all"}
    assert means["syn_diag - dp_sgd"]["mean"] == 2
    assert means["dp_kfc - dp_sgd"]["mean"] == 5
    assert means["syn_diag - dp_kfc"]["mean"] == -3
    assert means["syn_diag - dp_sgd"]["std"] == 0


def test_identity_reference():
    model = torch.nn.Linear(2,1)
    for p in model.parameters(): p.grad_sample = torch.ones(3,*p.shape)
    old = [p.grad_sample.clone() for p in model.parameters()]
    apply(model,None)
    assert all(torch.equal(v,p.grad_sample) for v,p in zip(old,model.parameters()))
    geom = dict(A=2.,S=3.,rank=3,eigen_tolerance=1e-12)
    assert compare(geom,geom)["R_diag"] == compare(geom,geom)["R_full"] == 1


@pytest.mark.parametrize("method", METHODS)
def test_oracle_trajectory_isolation(tmp_path, method):
    c = read_config(ROOT / "configs/smoke.json")
    generator = torch.Generator().manual_seed(900)
    data = TensorDataset(torch.randn(32,1,28,28,generator=generator),torch.randint(0,10,(32,),generator=generator))
    train(c,42,method,tmp_path/"on",(data,data))
    train(dict(c,oracle_enabled=False),42,method,tmp_path/"off",(data,data))
    def read(part, filename):
        return json.loads((tmp_path/part/"seed42"/method/filename).read_text())
    assert read("on","pairing.json") == read("off","pairing.json")
    assert read("on","summary.json")["final_model_hash"] == read("off","summary.json")["final_model_hash"]
    oracle = pd.read_csv(tmp_path/"on"/"seed42"/method/"oracle_metrics.csv")
    assert oracle.loc[oracle.step == 0,"R_diag_old"].isna().all()
    assert np.isfinite(oracle.loc[oracle.step > 0,["R_diag_old","R_full_old"]]).all().all()
    # The independent diagnostic budget must not change training either.
    train(dict(c,oracle_enabled=False,M_stale=4),42,method,tmp_path/"small_probe",(data,data))
    for key in ("private","synthetic"):
        assert read("on","pairing.json")[key] == read("small_probe","pairing.json")[key]


def test_config_rejects_protocol_change(tmp_path):
    c = dict(DEFAULT, momentum=.9)
    path = tmp_path/"bad.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError): read_config(path)


def test_kfac_actual_transform_then_global_clip():
    from dp_kfac.privacy import clip_and_noise_gradients
    model = torch.nn.Sequential(torch.nn.Linear(1,1))
    model[0].weight.grad_sample = torch.tensor([[[3.]],[[0.]]])
    model[0].bias.grad_sample = torch.tensor([[0.],[4.]])
    apply(model,("dp_kfc",({"0":torch.eye(2)*2},{"0":torch.eye(1)})))
    assert model[0].weight.grad_sample[0,0,0] == 6
    assert model[0].bias.grad_sample[1,0] == 8
    clip_and_noise_gradients(model,0.,1.,2,store_summed_grad=True)
    assert model[0].weight.grad.item() == pytest.approx(6/(6+1e-6)/2)
    assert model[0].bias.grad.item() == pytest.approx(8/(8+1e-6)/2)


def test_gram_rank_deficient_and_scaled():
    h = np.array([[1.,2.,3.],[2.,4.,6.],[0.,0.,0.]])
    a,_ = gram_spectrum(h,1)
    b,_ = gram_spectrum(h*1e-7,2)
    assert len(a) == len(b) == 1
    torch.testing.assert_close(b,a*1e-14,atol=1e-25,rtol=1e-10)


def test_diagnostics_do_not_mutate_state_or_rng(tmp_path):
    from exp3.common import SimpleCNN, digest
    from exp3.geometry import diagnose
    c = read_config(ROOT/"configs/smoke.json")
    model = SimpleCNN()
    samples,_ = synthetic_samples(c,torch.device("cpu"),RNGStream(51,"cpu"))
    before = digest(model.parameters())
    rng = torch.get_rng_state().clone()
    result = diagnose(model.state_dict(),samples,{"identity":None},c,torch.device("cpu"),tmp_path)
    assert digest(model.parameters()) == before and torch.equal(torch.get_rng_state(),rng)
    assert set(result["identity"]) == {"conv1","conv2","fc1","fc2"}
    assert not list(tmp_path.iterdir())
