import pytest
import torch
from expv1.fisher_wiener import build_wiener_state, apply_fisher_matrix, apply_scalar_matrix, pack_layer_gradient, unpack_layer_gradient
from dp_kfac.types import CovariancePair


def case(sigma=2.):
    rng=torch.Generator().manual_seed(11)
    a=torch.randn(3,3,generator=rng);g=torch.randn(2,2,generator=rng)
    a=a@a.T+.3*torch.eye(3);g=g@g.T+.3*torch.eye(2)
    st=build_wiener_state(CovariancePair(A={'l':a},G={'l':g}),sigma,1,1)['l']
    return a,g,st


def test_kron_eigenvalues():
    a,g,st=case()
    torch.testing.assert_close(torch.linalg.eigvalsh(torch.kron(a,g)),
        (st['lambda_G'][:,None]*st['lambda_A']).flatten().sort().values,rtol=2e-5,atol=2e-5)


def test_column_major_explicit():
    a,g,st=case()
    y=torch.arange(6.).reshape(2,3)
    f=torch.kron(a,g); w=f@torch.linalg.inv(f+4*torch.eye(6))
    explicit=(w@y.T.contiguous().flatten()).reshape(3,2).T
    torch.testing.assert_close(apply_fisher_matrix(y,st),explicit,rtol=1e-5,atol=1e-5)


def test_zero_noise_identity():
    _,_,st=case(0)
    y=torch.randn(2,3)
    torch.testing.assert_close(apply_fisher_matrix(y,st),y,rtol=1e-5,atol=1e-5)


def test_huge_noise():
    _,_,st=case(1e10)
    assert st['H'].max()<1e-15


def test_scalar():
    a,g,st=case()
    mean=a.trace()*g.trace()/6
    torch.testing.assert_close(st['scalar_h'],mean/(mean+4))
    y=torch.randn(2,3)
    torch.testing.assert_close(apply_scalar_matrix(y,st),y*mean/(mean+4))


@pytest.mark.parametrize('layer',[torch.nn.Linear(3,2),torch.nn.Conv2d(2,3,3)])
def test_pack_inverse(layer):
    for p in layer.parameters():p.grad=torch.randn_like(p)
    old=[p.grad.clone() for p in layer.parameters()]
    m=pack_layer_gradient(layer)
    assert torch.equal(m[:,-1],old[1])
    unpack_layer_gradient(layer,m)
    for p,g in zip(layer.parameters(),old):assert torch.equal(p.grad,g)
