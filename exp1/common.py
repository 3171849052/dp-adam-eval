"""Shared configuration and direct adapters to the adjacent, read-only DP-KFC repo."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent / 'DP-KFC'
sys.path.insert(0, str(REPO / 'src'))
import torch
from dp_kfac.models import SimpleCNN
from dp_kfac.data import get_mnist_loaders, get_fashionmnist_loaders
from dp_kfac.optimizer import generate_pink_noise
from dp_kfac.trainer import set_seed

LAYERS = ('conv1', 'conv2', 'fc1', 'fc2')


def read_config(path=None):
    with open(path or ROOT / 'configs/default.json') as f:
        c = json.load(f)
    for key in ('batch_size', 'epochs', 'm_diag', 'm_full', 'analysis_batch_size', 'gram_chunk', 'threads'):
        if c[key] < 1:
            raise ValueError(f'{key} must be positive')
    if c['m_full'] > c['m_diag']:
        raise ValueError('m_full must be <= m_diag (fixed subset)')
    if len(set(c['pink_seeds'])) < 3:
        raise ValueError('At least three distinct pink_seeds are required')
    if c['m_diag'] % c['batch_size']:
        raise ValueError('m_diag must be divisible by batch_size for repo KFAC averaging')
    torch.set_num_threads(c['threads'])
    return c


def parse(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/default.json')
    return read_config(p.parse_args().config)


def run_root(c):
    path = (ROOT / c['output']).resolve()
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('output must be a subdirectory of exp1')
    path.mkdir(parents=True, exist_ok=True)
    return path


def device(c):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if c['device'] == 'auto' else torch.device(c['device'])


def fingerprint(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def provenance():
    files = ['models.py', 'data.py', 'optimizer.py', 'trainer.py', 'privacy.py', 'covariance.py', 'recorder.py']
    return {name: hashlib.sha256((REPO / 'src/dp_kfac' / name).read_bytes()).hexdigest() for name in files}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def save_torch(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temp)
    temp.replace(path)


@contextmanager
def in_exp1():
    old = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(old)


def loaders(c, public=False):
    # Upstream hardcodes ./data. Run the unmodified loader in exp1; copy any
    # available cache first so upstream download=True never writes to DP-KFC.
    name = 'FashionMNIST' if public else 'MNIST'
    cache = ROOT / 'data' / name
    upstream = REPO / 'data' / name
    if not cache.exists() and upstream.exists():
        shutil.copytree(upstream, cache)
    fn = get_fashionmnist_loaders if public else get_mnist_loaders
    with in_exp1():
        return fn(batch_size=c['batch_size'], num_workers=c['num_workers'])


def probes(c):
    path = run_root(c) / 'results/probes.pt'
    if path.exists():
        data = torch.load(path, weights_only=True)
        if data['fingerprint'] != fingerprint(c):
            raise ValueError('Probe config changed: select a fresh output directory')
        return data['sources']
    sources, indices = {}, {}
    for name, public, seed in [('private', False, c['seed'] + 100), ('public', True, c['seed'] + 101)]:
        dataset = loaders(c, public)[0].dataset
        if c['m_diag'] > len(dataset):
            raise ValueError('Diagnostic sample count exceeds dataset size')
        idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:c['m_diag']]
        samples = [dataset[int(i)] for i in idx]
        sources[name] = (torch.stack([s[0] for s in samples]), torch.tensor([s[1] for s in samples]))
        indices[name] = idx
    for seed in c['pink_seeds']:
        set_seed(seed)
        # Match upstream call size and uniform-label expression; no alpha override.
        batches = [(generate_pink_noise(c['batch_size'], (1, 28, 28), torch.device('cpu')),
                    torch.randint(0, 10, (c['batch_size'],), device='cpu'))
                   for _ in range(c['m_diag'] // c['batch_size'])]
        sources[f'pink_{seed}'] = tuple(torch.cat([b[j] for b in batches]) for j in (0, 1))
    save_torch(path, {'fingerprint': fingerprint(c), 'sources': sources, 'indices': indices})
    return sources
