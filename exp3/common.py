"""Read-only upstream adapters, fixed protocols, and isolated Torch streams."""
import csv
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import shutil
import sys
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent / "DP-KFC"
sys.path.insert(0, str(REPO / "src"))
from dp_kfac.models import SimpleCNN
from dp_kfac.data import get_mnist_loaders
from dp_kfac.optimizer import generate_pink_noise
from dp_kfac.trainer import set_seed

LAYERS = ("conv1", "conv2", "fc1", "fc2")
METHODS = ("dp_sgd", "syn_diag", "dp_kfc")
DEFAULT = dict(dataset="MNIST", model="SimpleCNN", seeds=[42, 7, 91], epochs=5,
               batch_size=256, epsilon=1.0, delta=1e-5, max_grad_norm=1.0,
               optimizer="SGD", learning_rate=0.1, momentum=0, M_oracle=512,
               K=50, M_syn=2560, damping=1e-3, **{"lambda": 1e-3},
               device="auto", threads=4, analysis_batch_size=32,
               eval_interval=100, eps_num=1e-12, gram_chunk=2048,
               oracle_seed=314159, oracle_enabled=True,
               train_subset=None, test_subset=None, smoke=False)


def read_config(path):
    c = json.loads(Path(path).read_text())
    if set(c) != set(DEFAULT):
        raise ValueError("Config must have exactly the documented keys")
    flexible = {"device", "threads", "analysis_batch_size", "eval_interval", "gram_chunk"}
    if c["smoke"]:
        flexible |= {"smoke", "seeds", "epochs", "batch_size", "K", "M_syn", "M_oracle", "train_subset", "test_subset"}
    for k in set(DEFAULT) - flexible:
        if c[k] != DEFAULT[k]:
            raise ValueError(f"Fixed protocol mismatch: {k}")
    for k in ("epochs", "batch_size", "K", "M_syn", "M_oracle", "threads", "analysis_batch_size", "eval_interval", "gram_chunk"):
        if type(c[k]) is not int or c[k] < 1:
            raise ValueError(f"Invalid positive integer: {k}")
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("Seeds must be unique and nonempty")
    if c["M_syn"] % c["batch_size"]:
        raise ValueError("Synthetic budget must contain whole equal microbatches")
    torch.set_num_threads(c["threads"])
    return c


def fingerprint(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def provenance():
    files = ["models", "data", "optimizer", "privacy", "covariance", "recorder", "precondition"]
    result = {f"upstream/{n}.py": hashlib.sha256((REPO / f"src/dp_kfac/{n}.py").read_bytes()).hexdigest() for n in files}
    result.update({f"exp3/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")})
    return result


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.with_suffix(".tmp").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)


class RNGStream:
    def __init__(self, seed, dev):
        self.dev = torch.device(dev)
        self.cpu = torch.Generator().manual_seed(seed).get_state()
        self.cuda = torch.Generator(device=self.dev).manual_seed(seed).get_state() if self.dev.type == "cuda" else None

    def audit(self):
        return digest([self.cpu] + ([] if self.cuda is None else [self.cuda]))

    @contextmanager
    def use(self):
        devices = [self.dev.index or 0] if self.cuda is not None else []
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


def layer_gradient(layer):
    return torch.cat([layer.weight.grad_sample.flatten(2), layer.bias.grad_sample.unsqueeze(-1)], -1).flatten(1)


def datasets(c):
    cache_root = ROOT / "runs/_cache"
    cache = cache_root / "data/MNIST"
    if not cache.exists():
        for source in (ROOT.parent / "exp1/data/MNIST", REPO / "data/MNIST", ROOT.parent / "exp2b/runs/_cache/data/MNIST"):
            if source.exists():
                shutil.copytree(source, cache)
                break
    cache_root.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    try:
        os.chdir(cache_root)
        train, test, _ = get_mnist_loaders(c["batch_size"], num_workers=0)
    finally:
        os.chdir(old)
    return train.dataset, test.dataset
