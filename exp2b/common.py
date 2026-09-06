"""Read-only upstream adapters and isolated random streams."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from exp1.common import (LAYERS, SimpleCNN, REPO, generate_pink_noise, set_seed,
                         device, fingerprint, provenance, save_json, save_torch,
                         get_mnist_loaders)
# Exp1 modules use script-style imports; common exposes their required symbols.
sys.path.append(str(ROOT.parent / 'exp1'))
from fisher_utils import layer_gradient
from exp1.metrics import diagonal_metrics, spread

class RNGStream:
    """Swap only Torch RNG state; upstream randn_like has no generator argument."""
    def __init__(self, seed, dev):
        self.dev = torch.device(dev)
        self.cpu = torch.Generator().manual_seed(seed).get_state()
        self.cuda = (torch.Generator(device=self.dev).manual_seed(seed).get_state()
                     if self.dev.type == 'cuda' else None)

    @contextmanager
    def use(self):
        devices = [self.dev.index if self.dev.index is not None else torch.cuda.current_device()] if self.cuda is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu)
            if self.cuda is not None:
                torch.cuda.set_rng_state(self.cuda, self.dev)
            try:
                yield
            finally:
                self.cpu = torch.get_rng_state()
                if self.cuda is not None:
                    self.cuda = torch.cuda.get_rng_state(self.dev)


def read_config(path):
    c = json.loads(Path(path).read_text())
    for k in ('batch_size', 'epochs', 'K', 'M_syn', 'M_oracle', 'analysis_batch_size', 'eval_interval', 'threads'):
        if not isinstance(c[k], int) or c[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    if c['M_syn'] < 2 or any(c[k] <= 0 for k in ('lambda', 'learning_rate', 'epsilon', 'max_grad_norm', 'eps_num')):
        raise ValueError('Invalid positive scalar or M_syn < 2')
    if not 0 < c['delta'] < 1:
        raise ValueError('delta must be in (0,1)')
    torch.set_num_threads(c['threads'])
    return c


def parse(method=False):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/lambda1e4_K50.json'))
    if method:
        p.add_argument('--method', required=True, choices=['syn_diag'])
    args = p.parse_args()
    return (read_config(args.config), args.method) if method else read_config(args.config)


def run_root(c):
    root = (ROOT / c['output']).resolve()
    if not root.is_relative_to(ROOT / 'runs') or root == ROOT / 'runs':
        raise ValueError('output must be beneath exp2b/runs/')
    root.mkdir(parents=True, exist_ok=True)
    return root


def datasets(c):
    # Upstream hardcodes ./data: all new cache writes go below exp2b/runs/.
    cache_root = ROOT / 'runs/_cache'
    cache = cache_root / 'data/MNIST'
    if not cache.exists():
        for source in (ROOT.parent / 'exp1/data/MNIST', REPO / 'data/MNIST'):
            if source.exists():
                shutil.copytree(source, cache)
                break
    cache_root.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    try:
        os.chdir(cache_root)
        train, test, _ = get_mnist_loaders(c['batch_size'], num_workers=c['num_workers'])
    finally:
        os.chdir(old)
    return train.dataset, test.dataset


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
