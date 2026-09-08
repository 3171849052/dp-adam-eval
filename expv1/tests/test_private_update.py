from unittest.mock import patch
import torch
import pytest
from opacus import GradSampleModule
from expv1.common import SimpleCNN, LAYERS, METHODS, RNGStream
from expv1 import train_expv1 as tr
from expv1.fisher_wiener import apply_filter_to_copy, pack_layer_gradient
from dp_kfac.privacy import clip_and_noise_gradients, _compute_clip_factors, _compute_per_sample_norms_squared


def setup():
    torch.manual_seed(12)
    model=GradSampleModule(SimpleCNN(),loss_reduction='sum')
    x=torch.randn(4,1,28,28);y=torch.tensor([0,1,2,3])
    torch.nn.functional.cross_entropy(model(x),y,reduction='sum').backward()
    state={}
    for n in LAYERS:
        l=getattr(model._module,n);a=l.weight[0].numel()+1;g=l.weight.shape[0]
        state[n]=dict(Q_A=torch.eye(a),Q_G=torch.eye(g),H=torch.full((g,a),.3),scalar_h=torch.tensor(.4),
            lambda_A=torch.ones(a),lambda_G=torch.ones(g),trace_A=torch.tensor(float(a)),
            trace_G=torch.tensor(float(g)),trace_F=torch.tensor(float(a*g)))
    return model,state


def test_upstream_update_exact(config):
    model,st=setup();ref,_=setup()
    opt=tr.make_optimizer(model,config);ropt=tr.make_optimizer(ref,config)
    rng=RNGStream(99,'cpu'); rrng=RNGStream(99,'cpu')
    tr.private_update(model,st,opt,rng,2.,config,4,'dp_sgd')
    with rrng.use():clip_and_noise_gradients(ref,2.,1.,4,store_summed_grad=True)
    ropt.step()
    for a,b in zip(model.parameters(),ref.parameters()):assert torch.equal(a,b)
    assert rng.audit()==rrng.audit()
    model.remove_hooks();ref.remove_hooks()


@pytest.mark.parametrize('method',METHODS[1:])
def test_order_and_optimizer_input(config,method):
    model,st=setup();opt=tr.make_optimizer(model,config)
    events=[];clean=[];before=[p.detach().clone() for p in model.parameters()]
    raw=[p.grad_sample.clone() for p in model.parameters()]
    factors=_compute_clip_factors(_compute_per_sample_norms_squared(list(model.parameters()),4,torch.device('cpu')),1.)
    noisy={}
    def clip(*args,**kwargs):
        events.append('clip')
        for p,g in zip(model.parameters(),raw):assert torch.equal(p.grad_sample,g)
        clip_and_noise_gradients(*args,**kwargs)
        clean.extend(p.summed_grad.clone() for p in model.parameters())
        noisy.update({n:pack_layer_gradient(getattr(model._module,n)) for n in LAYERS})
    name='apply_fisher_wiener' if method=='dp_fisher_wiener' else 'apply_scalar_wiener'
    actual=getattr(tr,name)
    def filt(*args):
        assert events==['clip'];events.append('filter');actual(*args)
        for p,g in zip(model.parameters(),clean):assert torch.equal(p.summed_grad,g)
    step=opt.step
    def update():
        assert events==['clip','filter'];events.append('step')
        for n in LAYERS:
            torch.testing.assert_close(pack_layer_gradient(getattr(model._module,n)),apply_filter_to_copy(noisy[n],st[n],method),rtol=0,atol=0)
        step()
    with patch.object(tr,'clip_and_noise_gradients',side_effect=clip),patch.object(tr,name,side_effect=filt),patch.object(opt,'step',side_effect=update):
        tr.private_update(model,st,opt,RNGStream(99,'cpu'),2.,config,4,method)
    for p,old,g,raw_g in zip(model.parameters(),before,clean,raw):
        assert torch.equal(p.summed_grad,g)
        expected=(raw_g.flatten(1)*factors[:,None]).sum(0).reshape_as(p)/4
        torch.testing.assert_close(g,expected,rtol=0,atol=0)
        torch.testing.assert_close(p,old-.1*p.grad)
    assert events==['clip','filter','step']
    model.remove_hooks()


@pytest.mark.parametrize('method',METHODS)
def test_filter_never_mutates_other_attributes(method):
    model,st=setup()
    for p in model.parameters():p.summed_grad=torch.randn_like(p)
    samples=[p.grad_sample.clone() for p in model.parameters()]
    clean=[p.summed_grad.clone() for p in model.parameters()]
    if method!='dp_sgd':getattr(tr,'apply_fisher_wiener' if method=='dp_fisher_wiener' else 'apply_scalar_wiener')(model,st)
    for p,s,c in zip(model.parameters(),samples,clean):
        assert torch.equal(p.grad_sample,s) and torch.equal(p.summed_grad,c)
    model.remove_hooks()
