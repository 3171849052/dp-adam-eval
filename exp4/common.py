"""Fixed Exp4 protocol and shared, audited Exp3 adapters."""
import json
from pathlib import Path
import hashlib
import torch
from exp3.common import (SimpleCNN, RNGStream, digest, datasets, set_seed,
                         save_json, write_csv, fingerprint, provenance as base_provenance)

ROOT = Path(__file__).resolve().parent
TRAJECTORIES = ('adam', 'dp_adam')
DEFAULT = dict(dataset='MNIST', model='SimpleCNN', seeds=[42, 7, 91], batch_size=256,
               epochs=5, max_grad_norm=1., epsilon=1., delta=1e-5,
               adam_lr=.001, beta1=.9, beta2=.999, adam_eps=1e-8, weight_decay=0.,
               syn_lr=.1, M_syn=2560, K=50, **{'lambda': .001}, device='auto',
               threads=4, analysis_batch_size=32, eval_interval=100,
               diagnostic_interval=50, eps_num=1e-12, train_subset=None,
               test_subset=None, smoke=False)


def read_config(path):
    c = json.loads(Path(path).read_text())
    if set(c) != set(DEFAULT):
        raise ValueError('Config keys must match the documented protocol')
    flexible = {'device', 'threads', 'analysis_batch_size'}
    if c['smoke']:
        flexible |= {'smoke', 'seeds', 'batch_size', 'epochs', 'M_syn', 'K',
                     'eval_interval', 'diagnostic_interval', 'train_subset', 'test_subset'}
    for key in set(DEFAULT) - flexible:
        if c[key] != DEFAULT[key]:
            raise ValueError(f'Fixed protocol mismatch: {key}')
    for key in ('batch_size', 'epochs', 'M_syn', 'K', 'threads', 'analysis_batch_size',
                'eval_interval', 'diagnostic_interval'):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f'Invalid positive integer: {key}')
    if not c['seeds'] or len(set(c['seeds'])) != len(c['seeds']):
        raise ValueError('Seeds must be nonempty and unique')
    if c['M_syn'] % c['batch_size']:
        raise ValueError('M_syn must contain whole synthetic batches')
    return c


def provenance():
    value = base_provenance()
    value.update({f'exp4/{p.name}': hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in ROOT.glob('*.py')})
    return value


def adam(params, c, lr=None):
    return torch.optim.Adam(params, lr=c['adam_lr'] if lr is None else lr,
                            betas=(c['beta1'], c['beta2']), eps=c['adam_eps'],
                            weight_decay=c['weight_decay'])
