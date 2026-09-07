"""Shared Exp5b protocol, deterministic streams, and audit helpers."""
import csv
import hashlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from exp3.audit_upstream import checkout_info

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent / "DP-KFC"
sys.path.insert(0, str(REPO / "src"))
from dp_kfac.models import SimpleCNN  # noqa: E402
from dp_kfac.data import get_mnist_loaders  # noqa: E402
from dp_kfac.optimizer import generate_pink_noise  # noqa: E402
from dp_kfac.trainer import set_seed  # noqa: E402

METHODS = (
    "syn_diag", "syn_diag_beta1", "syn_diag_beta2", "syn_diag_beta12",
    "dp_sgd_momentum", "dp_kfc_momentum",
)
MOMENTUM_METHODS = (
    "syn_diag_beta1", "syn_diag_beta12", "dp_sgd_momentum", "dp_kfc_momentum",
)
SYNTHETIC_METHODS = (
    "syn_diag", "syn_diag_beta1", "syn_diag_beta2", "syn_diag_beta12", "dp_kfc_momentum",
)
LAYERS = ("conv1", "conv2", "fc1", "fc2")

DEFAULT = dict(
    dataset="MNIST", model="SimpleCNN", seeds=[42, 7, 91], learning_rate=.1,
    total_private_steps=1170, batch_size=256, epochs=5, max_grad_norm=1.,
    epsilon=1., delta=1e-5, beta1=.9, beta2=.999, adam_eps=1e-8,
    weight_decay=0., M_syn=2560, K=50, **{"lambda": 1e-3}, damping=1e-3,
    device="auto", threads=4, analysis_batch_size=32, eval_interval=100,
    diagnostic_interval=50, eps_num=1e-12, gram_chunk=2048, M_oracle=512,
    M_stale=512, stale_seed_offset=100003, oracle_seed=314159,
    oracle_enabled=True, train_subset=None, test_subset=None, smoke=False,
)


def read_config(path):
    c = json.loads(Path(path).read_text())
    if set(c) != set(DEFAULT):
        raise ValueError("Config must have exactly the documented keys")
    flexible = {"device", "threads", "analysis_batch_size", "eval_interval", "diagnostic_interval", "gram_chunk"}
    if c["smoke"]:
        flexible |= {
            "smoke", "seeds", "learning_rate", "total_private_steps", "batch_size", "epochs", "K", "M_syn",
            "M_oracle", "M_stale", "train_subset", "test_subset", "eval_interval", "diagnostic_interval",
            "oracle_enabled",
        }
    for key in set(DEFAULT) - flexible:
        if c[key] != DEFAULT[key]:
            raise ValueError(f"Fixed protocol mismatch: {key}")
    for key in (
        "batch_size", "epochs", "K", "M_syn", "M_oracle", "M_stale", "threads",
        "analysis_batch_size", "eval_interval", "diagnostic_interval", "gram_chunk", "total_private_steps",
    ):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"Invalid positive integer: {key}")
    if not c["seeds"] or len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("Seeds must be unique and nonempty")
    if c["M_syn"] % c["batch_size"]:
        raise ValueError("M_syn must contain whole synthetic batches")
    if c["learning_rate"] != .1:
        raise ValueError("Exp5b requires learning_rate exactly 0.1")
    torch.set_num_threads(c["threads"])
    return c


def fingerprint(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def digest(tensors):
    h = hashlib.sha256()
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(json.dumps(tensor, sort_keys=True).encode())
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix(".tmp").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)


def provenance():
    local_dependencies = [
        ROOT.parent / "exp3" / name for name in
        ("audit_upstream.py", "cost.py", "geometry.py", "metrics.py", "preconditioners.py", "common.py")
    ] + [ROOT.parent / "exp4" / name for name in ("common.py", "dynamics.py")]
    result = {f"upstream/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (REPO / "src/dp_kfac").glob("*.py")}
    result.update({f"{p.parent.name}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in local_dependencies})
    result.update(checkout_info())
    result.update({f"exp5b/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in ROOT.glob("*.py")})
    return result


class RNGStream:
    """A deterministic stream that does not touch the caller's Torch RNG."""
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


class Indexed(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = self.data[i]
        return x, y, i


def datasets(c):
    cache_root = ROOT / "runs/_cache"
    cache = cache_root / "data/MNIST"
    if not cache.exists():
        for source in (ROOT.parent / "exp1/data/MNIST", REPO / "data/MNIST", ROOT.parent / "exp2b/runs/_cache/data/MNIST"):
            if source.exists():
                cache_root.mkdir(parents=True, exist_ok=True)
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


def evaluate(model, loader, dev):
    model.eval()
    loss = correct = count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            output = model(x)
            loss += float(torch.nn.functional.cross_entropy(output, y, reduction="sum"))
            correct += int((output.argmax(1) == y).sum())
            count += len(y)
    model.train()
    return {"test_loss": loss / count, "test_accuracy": correct / count}
