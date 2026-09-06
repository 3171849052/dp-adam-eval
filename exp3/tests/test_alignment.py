"""Independent upstream references: these tests reject the old SUM-loss bug."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from opacus import GradSampleModule
from exp3.common import DEFAULT, ROOT, LAYERS, SimpleCNN, RNGStream, read_config, provenance
from exp3.preconditioners import synthetic_samples, refresh, apply
from exp3.geometry import oracle_compare
from exp3.audit_upstream import checkout_info, require_clean
from exp3.cost import CostTracker, core_wall_time
from exp3.summarize_exp3 import summarize, load_runs
from dp_kfac.recorder import KFACRecorder
from dp_kfac.covariance import compute_covariances, accumulate_covariances, compute_inverse_sqrt
from dp_kfac.precondition import precondition_per_sample_gradients


def reference(state, samples, c, dev, reduction="mean"):
    """Deliberately independent of Exp3 model/refresh helpers."""
    model = GradSampleModule(SimpleCNN().to(dev), loss_reduction="sum")
    model._module.load_state_dict(state)
    recorder = KFACRecorder(model)
    recorder.enable()
    covariances = []
    x, y = samples
    try:
        for start in range(0, len(x), c["batch_size"]):
            model.zero_grad()
            output = model(x[start:start+c["batch_size"]])
            F.cross_entropy(output, y[start:start+c["batch_size"]], reduction=reduction).backward()
            covariances.append(compute_covariances(model, recorder.activations, recorder.backprops))
            recorder.clear()
        cov = accumulate_covariances(covariances)
        return cov, compute_inverse_sqrt(cov, damping=1e-3)
    finally:
        recorder.remove()
        model.remove_hooks()


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def equivalence(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dev = torch.device(request.param)
    # Match the training entrypoint's deterministic numerical settings before
    # either independent reference or Exp3 construction is evaluated.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    c = read_config(ROOT/"configs/smoke.json")
    torch.manual_seed(123)
    state = SimpleCNN().state_dict()
    samples, _ = synthetic_samples(c, dev, RNGStream(777, dev))
    ref_cov, ref_roots = reference(state, samples, c, dev)
    active, cov = refresh(state, samples, c, dev, "dp_kfc", return_covariances=True)
    wrong_cov, wrong_roots = reference(state, samples, c, dev, "sum")
    return dev, c, ref_cov, ref_roots, active, cov, wrong_cov, wrong_roots


def test_upstream_covariance_and_inverse_equivalence(equivalence):
    dev, _, ref_cov, ref_roots, active, cov, _, _ = equivalence
    errors = {}
    for label, actual, expected in [("cov_A", cov.A, ref_cov.A), ("cov_G", cov.G, ref_cov.G),
                                    ("inv_A", active[1][0], ref_roots[0]), ("inv_G", active[1][1], ref_roots[1])]:
        errors[label] = {}
        assert set(actual) == set(LAYERS)
        for layer in LAYERS:
            errors[label][layer] = float((actual[layer]-expected[layer]).abs().max())
            torch.testing.assert_close(actual[layer], expected[layer], atol=0, rtol=0)
    print("UPSTREAM_EQUIVALENCE", str(dev), json.dumps(errors, sort_keys=True))


def test_uncorrected_sum_loss_fails_mean_reference(equivalence):
    dev, c, ref_cov, ref_roots, _, _, wrong_cov, wrong_roots = equivalence
    errors = {}
    for n in LAYERS:
        ridge = torch.eye(ref_cov.G[n].shape[0], device=dev)*1e-5
        torch.testing.assert_close(wrong_cov.G[n]-ridge, (ref_cov.G[n]-ridge)*c["batch_size"]**2, atol=1e-8, rtol=1e-4)
        assert not torch.allclose(wrong_cov.G[n], ref_cov.G[n], atol=1e-8, rtol=1e-5)
        assert not torch.allclose(wrong_roots[1][n], ref_roots[1][n], atol=1e-7, rtol=1e-6)
        errors[n] = float((wrong_roots[1][n]-ref_roots[1][n]).abs().max())
    print("SUM_BUG_REJECTED", str(dev), json.dumps(errors, sort_keys=True))


def test_upstream_per_sample_transform_equivalence(equivalence):
    dev, _, _, roots, active, _, _, _ = equivalence
    first = SimpleCNN().to(dev)
    second = copy.deepcopy(first)
    for p, q in zip(first.parameters(), second.parameters()):
        p.grad_sample = torch.randn(3, *p.shape, device=dev)
        q.grad_sample = p.grad_sample.clone()
    # Exp3 accepts either a plain model (upstream supports it) or wrapper.
    apply(first, active)
    precondition_per_sample_gradients(second, *active[1])
    errors = {}
    for (n,p),q in zip(first.named_parameters(), second.parameters()):
        errors[n] = float((p.grad_sample-q.grad_sample).abs().max())
        assert torch.equal(p.grad_sample,q.grad_sample)
    print("TRANSFORM_EQUIVALENCE", str(dev), json.dumps(errors, sort_keys=True))


def test_independent_stale_budget_and_pairing():
    assert DEFAULT["M_stale"] == 512 and DEFAULT["M_syn"] == 2560
    c = read_config(ROOT/"configs/smoke.json")
    construction = RNGStream(45,"cpu")
    a,b = [RNGStream(42+c["stale_seed_offset"],"cpu") for _ in range(2)]
    before = construction.audit()
    for _ in range(3):
        _, x = synthetic_samples(c,torch.device("cpu"),a,budget=c["M_stale"])
        _, y = synthetic_samples(c,torch.device("cpu"),b,budget=c["M_stale"])
        assert x == y and x["count"] == 8 and x["rng_before"] != before
    assert construction.audit() == before


def test_step_zero_old_new_semantics():
    raw = dict(A=4.,S=6.,rank=4,eigen_tolerance=1e-12)
    new = dict(raw,A=2.,S=3.)
    row = oracle_compare(raw,new)
    assert row["R_diag_new"] == row["R_full_new"] == .5
    assert all(v is None for k,v in row.items() if k.endswith("_old") or k.startswith("private_delta"))
    later = oracle_compare(raw,new,raw)
    assert later["R_diag_old"] == later["R_full_old"] == 1
    assert later["private_delta_stale_diag"] == 2 and later["private_delta_stale_full"] == 3


def test_core_wall_time_accounting():
    assert core_wall_time(10.,3.) == 7.
    with pytest.raises(ValueError): core_wall_time(1.,2.)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_core_peak_excludes_diagnostic_allocations():
    dev = torch.device("cuda")
    tracker = CostTracker(dev)
    core = torch.ones(1024*1024, device=dev)
    with tracker.diagnostics():
        diagnostic = torch.ones(8*1024*1024,device=dev)
        del diagnostic
    result = tracker.finish()
    assert result["peak_cuda_memory_overall"] >= result["peak_cuda_memory_core"] + 8*1024*1024*4
    assert result["core_wall_time"] == result["wall_time"]-result["diagnostic_seconds"]
    del core


def test_provenance_and_dirty_behavior():
    info = checkout_info()
    p = provenance()
    assert p["upstream_git_commit"] == info["upstream_git_commit"]
    for name in ("models","data","optimizer","privacy","covariance","recorder","precondition","trainer","types"):
        assert len(p[f"upstream/{name}.py"]) == 64
    require_clean(dict(info,upstream_git_dirty=True),smoke=True)
    require_clean(dict(info,upstream_git_dirty=False),smoke=False)
    with pytest.raises(ValueError,match="clean upstream"):
        require_clean(dict(info,upstream_git_dirty=True),smoke=False)


def test_final_test_loss_and_old_new_aggregation(tmp_path):
    c = read_config(ROOT/"configs/smoke.json")
    runs = []
    for method, loss in zip(("dp_sgd","syn_diag","dp_kfc"), (3.,2.,1.)):
        summary = dict(method=method,seed=42,mean_refresh_time=1,total_refresh_time=2,wall_time=10,core_wall_time=8,diagnostic_seconds=2,number_of_refreshes=2,
                       peak_cuda_memory_core=10,peak_cuda_memory_overall=20,peak_cuda_memory_allocated=20,peak_cuda_memory_reserved=30,preconditioner_state_bytes=4,
                       final_test_loss=loss,final_accuracy=.1,best_accuracy=.2,late_mean_accuracy=.1)
        train = pd.DataFrame([dict(step=s,norm_cv=s,coefficient_cv=s,aggregate_cosine=s,clipping_shape_error=s,diagnostic_snr=s) for s in [1,2,3,4]])
        oracle = []
        for layer in LAYERS:
            for step in [3,4]:
                oracle.append(dict(layer=layer,step=step,**oracle_compare(dict(A=4.,S=6.,rank=4,eigen_tolerance=1e-12),dict(A=2.,S=3.,rank=4,eigen_tolerance=1e-12),dict(A=4.,S=6.,rank=4,eigen_tolerance=1e-12))))
        runs.append((tmp_path,dict(total_steps=4),summary,dict(train=train,oracle=pd.DataFrame(oracle),refresh=pd.DataFrame(columns=["step","layer"]))))
    with patch("exp3.summarize_exp3.load_runs",return_value=runs):
        result = summarize(tmp_path,c,tmp_path)
    assert result.final_test_loss.tolist() == [3.,2.,1.]
    assert np.allclose(result.G_diag_new,.5) and np.allclose(result.G_full_old,1)
    means = pd.read_csv(tmp_path/"summary_mean_std.csv")
    assert len(means[means.metric == "final_test_loss"]) == 3
    deltas = pd.read_csv(tmp_path/"paired_deltas.csv")
    assert deltas[(deltas.metric == "final_test_loss") & (deltas.pair == "syn_diag - dp_sgd") & (deltas.seed == "42")].delta.iloc[0] == -1


def test_aggregation_rejects_mixed_upstream_provenance(tmp_path):
    c = read_config(ROOT/"configs/smoke.json")
    from exp3.common import fingerprint
    # Test must remain valid when the real checkout is subsequently clean.
    p = dict(provenance(), upstream_git_dirty=True)
    for method in ("dp_sgd","syn_diag","dp_kfc"):
        root = tmp_path/"seed42"/method
        root.mkdir(parents=True)
        source = dict(p)
        if method == "dp_kfc":
            source["upstream/privacy.py"] = "0"*64
        meta = dict(seed=42,method=method,complete=True,fingerprint=fingerprint(c),provenance=source)
        meta.update({k:v for k,v in source.items() if k.startswith("upstream_")})
        for filename, value in (("config",c),("metadata",meta),("summary",dict(fingerprint=fingerprint(c)))):
            (root/f"{filename}.json").write_text(json.dumps(value))
        for name in ("train","oracle","refresh"):
            (root/f"{name}_metrics.csv").write_text("step,layer,seed,method,config_fingerprint\n")
    with pytest.raises(ValueError,match="Mixed implementation provenance"):
        load_runs(tmp_path,c)
    # Dirty provenance is checked before any metric aggregation in full mode.
    with pytest.raises(ValueError,match="clean upstream"):
        load_runs(tmp_path,dict(c,smoke=False))


def test_checkout_single_batch_privacy_matches_public_head():
    import subprocess
    import types
    from exp3.common import REPO
    from dp_kfac.privacy import clip_and_noise_gradients
    source = subprocess.check_output(["git","-C",str(REPO),"show","HEAD:src/dp_kfac/privacy.py"],text=True)
    public = types.ModuleType("dp_kfac._exp3_public_privacy_reference")
    public.__package__ = "dp_kfac"
    exec(compile(source,"public_HEAD/privacy.py","exec"),public.__dict__)
    first = torch.nn.Linear(3,2)
    second = copy.deepcopy(first)
    for p,q in zip(first.parameters(),second.parameters()):
        p.grad_sample = torch.randn(4,*p.shape)
        q.grad_sample = p.grad_sample.clone()
    for model,fn in ((first,clip_and_noise_gradients),(second,public.clip_and_noise_gradients)):
        with RNGStream(42,"cpu").use():
            fn(model,.7,1.,4,store_summed_grad=True)
    for p,q in zip(first.parameters(),second.parameters()):
        assert torch.equal(p.grad,q.grad) and torch.equal(p.summed_grad,q.summed_grad)
