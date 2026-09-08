from unittest.mock import patch
import inspect
import torch
from expv1.common import SimpleCNN, RNGStream, LAYERS, read_config, ROOT
from expv1 import fisher_wiener as fw


def test_pink_pairing(config):
    rng=RNGStream(45,'cpu')
    with patch.object(fw,'generate_pink_noise',wraps=fw.generate_pink_noise) as pink:
        (x,y),a=fw.synthetic_samples(config,torch.device('cpu'),rng)
        assert pink.call_count==2
        assert all(len(call.args)==3 and not call.kwargs for call in pink.call_args_list)
    (x2,y2),b=fw.synthetic_samples(config,torch.device('cpu'),RNGStream(45,'cpu'))
    assert torch.equal(x,x2) and torch.equal(y,y2) and a==b
    assert y.min()>=0 and y.max()<10


def test_covariance_conventions(config):
    dev=torch.device('cpu');model=SimpleCNN()
    samples,_=fw.synthetic_samples(config,dev,RNGStream(45,dev))
    with patch.object(fw,'compute_covariances',wraps=fw.compute_covariances) as compute, patch.object(fw.F,'cross_entropy',wraps=fw.F.cross_entropy) as ce, patch.object(fw,'accumulate_covariances',wraps=fw.accumulate_covariances) as acc:
        cov=fw.build_covariances(model.state_dict(),samples,config,dev)
        assert compute.call_count==ce.call_count==2
        assert all(not call.kwargs for call in compute.call_args_list)
        assert all(call.kwargs.get('reduction','mean')=='mean' for call in ce.call_args_list)
        assert len(acc.call_args.args[0])==2
    assert set(cov.A)==set(cov.G)==set(LAYERS)
    for n in LAYERS:
        l=getattr(model,n)
        assert cov.A[n].shape==(l.weight[0].numel()+1,)*2
        assert cov.G[n].shape==(l.weight.shape[0],)*2
        for m in (cov.A[n],cov.G[n]):
            torch.testing.assert_close(m,m.T)
            assert torch.linalg.eigvalsh(m).min()>-1e-5
    st=fw.build_wiener_state(cov,2,1,4)
    assert all(((v['H']>=0)&(v['H']<=1)).all() for v in st.values())
    assert all(t.dtype==torch.float32 for v in st.values() for t in v.values())


def test_fixed_definition(config):
    import pytest
    from expv1.common import check_config
    config['beta']=2
    with pytest.raises(ValueError):check_config(config)
    src=inspect.getsource(fw)
    for forbidden in ('summed_grad','grad_sample','compute_inverse_sqrt','precondition_per_sample_gradients','torch.kron'):
        assert forbidden not in src
    full=read_config(ROOT/'configs/full.json')
    assert full['M_syn']//full['batch_size']==10 and full['beta']==1


def test_recorder_matches_exp3_wrapper(config):
    from opacus import GradSampleModule
    dev=torch.device('cpu');model=SimpleCNN()
    samples,_=fw.synthetic_samples(config,dev,RNGStream(45,dev))
    expected=fw.build_covariances(model.state_dict(),samples,config,dev)
    wrapped=GradSampleModule(SimpleCNN(),loss_reduction='sum')
    wrapped._module.load_state_dict(model.state_dict())
    rec=fw.KFACRecorder(wrapped);rec.enable();factors=[]
    try:
        for x,y in zip(samples[0].split(4),samples[1].split(4)):
            wrapped.zero_grad(set_to_none=True)
            fw.F.cross_entropy(wrapped(x),y).backward()
            factors.append(fw.compute_covariances(wrapped,rec.activations,rec.backprops))
            rec.clear()
        actual=fw.accumulate_covariances(factors)
        for n in LAYERS:
            torch.testing.assert_close(actual.A[n],expected.A[n],rtol=0,atol=0)
            torch.testing.assert_close(actual.G[n],expected.G[n],rtol=0,atol=0)
    finally:
        rec.remove();wrapped.remove_hooks()
