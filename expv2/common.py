"""ExpV2 protocol, canonical run specifications, and isolated I/O helpers.

The Fisher-Wiener mathematics is deliberately imported from ExpV1.  This
module owns only ExpV2's protocol and experiment bookkeeping.
"""

import csv
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import sys

import torch

sys.dont_write_bytecode = True

from expv1.common import (  # noqa: E402
    LAYERS,
    REPO,
    RNGStream,
    SimpleCNN,
    digest,
    generate_pink_noise,
    set_seed,
)
from dp_kfac.data import get_mnist_loaders  # noqa: E402


ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "runs"
EXP1_REFERENCE_ROOT = ROOT.parent / "expv1" / "runs" / "formal_20260908_182603"
EXP1B_REFERENCE_ROOT = ROOT.parent / "expv1b" / "runs" / "formal_20260909_003544"
PINNED_COMMIT = "eb31b9aeb2280642684f4cedfa65cc02b76c76cd"

METHODS = ("dp_sgd", "dp_fisher_wiener")
BASE_LR = 0.1
STRESS_LR = 0.8
STRESS_SEED = 42
FORMAL_SEEDS = (42, 7, 91)
LAYERS = tuple(LAYERS)

CONFIG_KEYS = (
    "experiment", "seeds", "base_learning_rate", "stress_learning_rate",
    "stress_seed", "beta_train", "beta_primary_window",
    "beta_window_sensitivity", "epochs", "batch_size", "epsilon", "delta",
    "max_grad_norm", "optimizer", "momentum", "K", "M_syn", "eval_interval",
    "device", "threads", "train_subset", "test_subset", "smoke",
)

_FORMAL = dict(
    experiment="expv2", seeds=[42, 7, 91], base_learning_rate=0.1,
    stress_learning_rate=0.8, stress_seed=42, beta_train=1.0,
    beta_primary_window=50, beta_window_sensitivity=[10, 25, 50, 100],
    epochs=5, batch_size=256, epsilon=1.0, delta=1e-5, max_grad_norm=1.0,
    optimizer="SGD", momentum=0, K=50, M_syn=2560, eval_interval=100,
    device="auto", threads=4, train_subset=None, test_subset=None, smoke=False,
)

_SMOKE = dict(
    experiment="expv2", seeds=[42], base_learning_rate=0.1,
    stress_learning_rate=0.8, stress_seed=42, beta_train=1.0,
    beta_primary_window=2, beta_window_sensitivity=[1, 2, 4],
    epochs=1, batch_size=4, epsilon=1.0, delta=1e-5, max_grad_norm=1.0,
    optimizer="SGD", momentum=0, K=2, M_syn=8, eval_interval=1,
    device="auto", threads=4, train_subset=16, test_subset=32, smoke=True,
)


def lr_tag(lr):
    value = float(lr)
    if not (value > 0 and value < 10):
        raise ValueError("learning rate must be finite and positive")
    return f"{value:.2f}".replace(".", "p")


def run_id(method, learning_rate):
    if method == "dp_sgd":
        return f"dp_sgd_lr{lr_tag(learning_rate)}"
    if method == "dp_fisher_wiener":
        return f"dp_fisher_wiener_lr{lr_tag(learning_rate)}"
    raise ValueError(method)


def run_specs(config=None):
    """Return the three canonical run definitions; seed selection is separate."""
    if config is not None:
        check_config(config)
    return [
        dict(run_id=run_id("dp_sgd", BASE_LR), method="dp_sgd", learning_rate=BASE_LR),
        dict(run_id=run_id("dp_fisher_wiener", BASE_LR), method="dp_fisher_wiener", learning_rate=BASE_LR),
        dict(run_id=run_id("dp_fisher_wiener", STRESS_LR), method="dp_fisher_wiener", learning_rate=STRESS_LR),
    ]


def run_specs_for_seed(config, seed):
    check_config(config)
    if seed not in config["seeds"]:
        raise ValueError(f"seed {seed} is not in config")
    return [
        spec for spec in run_specs(config)
        if spec["method"] == "dp_sgd"
        or spec["learning_rate"] == config["base_learning_rate"]
        or seed == config["stress_seed"]
    ]


def run_spec(run_name, config=None):
    specs = run_specs(config)
    for spec in specs:
        if spec["run_id"] == run_name:
            return dict(spec)
    raise ValueError(f"Unknown canonical run id: {run_name}")


def check_config(config):
    """Reject protocol drift and undocumented adaptive controls."""
    if set(config) != set(CONFIG_KEYS):
        raise ValueError(
            f"Config keys mismatch; extra={sorted(set(config)-set(CONFIG_KEYS))}, "
            f"missing={sorted(set(CONFIG_KEYS)-set(config))}"
        )
    expected = _SMOKE if config["smoke"] else _FORMAL
    for key, value in expected.items():
        if key == "device":
            continue
        if config[key] != value:
            raise ValueError(f"Fixed ExpV2 protocol mismatch: {key}")
    if config["device"] != "auto" and not isinstance(config["device"], str):
        raise ValueError("device must be 'auto' or a device string")
    if type(config["beta_train"]) not in (int, float) or config["beta_train"] != 1:
        raise ValueError("beta_train must remain exactly 1")
    for key in ("epochs", "batch_size", "K", "M_syn", "eval_interval", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if type(config["stress_seed"]) is not int or config["stress_seed"] != 42:
        raise ValueError("stress_seed must be 42")
    if not config["seeds"] or any(type(seed) is not int for seed in config["seeds"]):
        raise ValueError("seeds must be nonempty integers")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be unique")
    if config["M_syn"] % config["batch_size"]:
        raise ValueError("M_syn must be an integer number of batches")
    if config["train_subset"] is not None and (type(config["train_subset"]) is not int or config["train_subset"] < 1):
        raise ValueError("train_subset must be null or a positive integer")
    if config["test_subset"] is not None and (type(config["test_subset"]) is not int or config["test_subset"] < 1):
        raise ValueError("test_subset must be null or a positive integer")
    if any(type(size) is not int or size < 1 for size in config["beta_window_sensitivity"]):
        raise ValueError("window sizes must be positive integers")
    torch.set_num_threads(config["threads"])
    return config


def read_config(path):
    return check_config(json.loads(Path(path).read_text()))


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def expected_total_steps(config, n=None):
    n = n or config["train_subset"] or 60000
    return (n // config["batch_size"]) * config["epochs"]


def synthetic_batches(config):
    if config["M_syn"] % config["batch_size"]:
        raise ValueError("M_syn must be an integer number of batches")
    return config["M_syn"] // config["batch_size"]


def output_path(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(RUNS_ROOT.resolve()):
        raise ValueError(f"All ExpV2 outputs must be inside {RUNS_ROOT}")
    return resolved


def _hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkout_info():
    def git(*args):
        return subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(REPO), *args], text=True
        ).strip()

    status = git("status", "--porcelain", "--untracked-files=all")
    return dict(
        upstream_git_commit=git("rev-parse", "HEAD"),
        upstream_git_dirty=bool(status),
        upstream_git_status=status,
        upstream_git_remote=git("remote", "-v"),
    )


def require_pinned(info, smoke):
    if not smoke and (info["upstream_git_dirty"] or info["upstream_git_commit"] != PINNED_COMMIT):
        raise ValueError("Formal mode requires clean pinned DP-KFC checkout")


def provenance():
    """Hash all code affecting ExpV2, including imported experiment helpers."""
    result = {}
    upstream = REPO / "src" / "dp_kfac"
    for path in sorted(upstream.rglob("*.py")):
        result[f"upstream/{path.relative_to(upstream)}"] = _hash_file(path)

    expv1_root = ROOT.parent / "expv1"
    for path in sorted(expv1_root.glob("*.py")):
        result[f"imported_expv1/{path.name}"] = _hash_file(path)
    expv1b_root = ROOT.parent / "expv1b"
    for name in ("common.py", "train_expv1b.py", "validate_expv1b.py"):
        path = expv1b_root / name
        if path.exists():
            result[f"imported_expv1b/{name}"] = _hash_file(path)
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" not in path.parts:
            result[f"expv2/{path.relative_to(ROOT)}"] = _hash_file(path)
    result["fixed_reference_expv1_root"] = str(EXP1_REFERENCE_ROOT)
    result["fixed_reference_expv1b_root"] = str(EXP1B_REFERENCE_ROOT)
    result.update(checkout_info())
    return result


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["step"]
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def datasets(config):
    """Load MNIST with a cache confined to ExpV2."""
    cache_root = RUNS_ROOT / "_cache"
    cache = cache_root / "data" / "MNIST"
    if not cache.exists():
        sources = (
            ROOT.parent / "exp1" / "data" / "MNIST",
            ROOT.parent / "expv1" / "runs" / "_cache" / "data" / "MNIST",
            REPO / "data" / "MNIST",
            ROOT.parent / "exp2b" / "runs" / "_cache" / "data" / "MNIST",
        )
        for source in sources:
            if source.exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, cache)
                break
    cache_root.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    try:
        os.chdir(cache_root)
        train, test, _ = get_mnist_loaders(config["batch_size"], num_workers=0)
    finally:
        os.chdir(old)
    return train.dataset, test.dataset


@contextmanager
def isolated_rng(seed, device):
    stream = RNGStream(seed, device)
    with stream.use():
        yield stream

