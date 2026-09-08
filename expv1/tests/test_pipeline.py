import json
from pathlib import Path
import torch
import pytest
from torch.utils.data import TensorDataset
from expv1.common import METHODS, provenance, require_pinned
from expv1.train_expv1 import train
from expv1.validate_expv1 import validate


def data():
    g=torch.Generator().manual_seed(123)
    ds=TensorDataset(torch.randn(32,1,28,28,generator=g),torch.randint(0,10,(32,),generator=g))
    return ds,ds


@pytest.mark.parametrize('method',METHODS)
def test_diagnostic_trajectory_isolation(config,tmp_path,method):
    results=[]
    for label,on,budget in [('on',True,None),('off',False,None),('budget',True,1)]:
        out=tmp_path/label
        meta=train(config,42,method,out,data(),diagnostics=on,eigen_budget=budget)
        root=out/'seed42'/method
        pair=json.loads((root/'pairing.json').read_text())
        results.append((meta['final_model_hash'],pair))
    assert results[0]==results[1]==results[2]


def test_pipeline_and_reject_tampering(config,tmp_path):
    for method in METHODS:train(config,42,method,tmp_path,data())
    assert validate(config,tmp_path,tmp_path,require_tests=False)['passed']
    p=tmp_path/'seed42/dp_fisher_wiener/pairing.json'
    pair=json.loads(p.read_text());pair['private'][0]['noise_rng_after']='tampered';p.write_text(json.dumps(pair))
    with pytest.raises(AssertionError):validate(config,tmp_path,tmp_path,require_tests=False)


def test_formal_pin_guard():
    p=provenance();p['upstream_git_dirty']=True
    with pytest.raises(ValueError):require_pinned(p,False)
    p['upstream_git_dirty']=False;p['upstream_git_commit']='wrong'
    with pytest.raises(ValueError):require_pinned(p,False)


def test_refresh_precedes_private_read(config,tmp_path):
    from unittest.mock import patch
    from expv1 import train_expv1 as tr
    events=[]
    original=tr.Indexed.__getitem__
    refresh=tr.build_covariances
    def get(self,i):
        events.append('read')
        return original(self,i)
    def build(*args,**kwargs):
        events.append('refresh')
        return refresh(*args,**kwargs)
    with patch.object(tr.Indexed,'__getitem__',get),patch.object(tr,'build_covariances',side_effect=build):
        train(config,42,'dp_fisher_wiener',tmp_path,data())
    assert events==['refresh']+['read']*8+['refresh']+['read']*8
