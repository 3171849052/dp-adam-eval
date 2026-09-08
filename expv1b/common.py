"""ExpV1b protocol, run specifications, provenance, and isolated I/O helpers.

The mathematical implementation is intentionally upstream of this package.  In
particular, Fisher-Wiener construction and application are imported from
``expv1`` and this module only owns the ExpV1b protocol and harness.
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

# The upstream checkout is read-only, including its bytecode cache.
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
REFERENCE_ROOT = ROOT.parent / "expv1" / "runs" / "formal_20260908_182603"
PINNED_COMMIT = "eb31b9aeb2280642684f4cedfa65cc02b76c76cd"

FISHER_LRS = (0.10, 0.15, 0.20, 0.30, 0.50, 0.80)
DP_SGD_LR = 0.10
METHODS = ("dp_sgd", "dp_fisher_wiener")

_PROTOCOL = dict(
    experiment="expv1b",
    seed=42,
    dp_sgd_learning_rate=0.1,
    fisher_learning_rates=list(FISHER_LRS),
    dataset="MNIST",
    model="SimpleCNN",
    epochs=5,
    batch_size=256,
    epsilon=1.0,
    delta=1e-5,
    max_grad_norm=1.0,
    optimizer="SGD",
    momentum=0,
    beta=1,
    K=50,
    M_syn=2560,
    eval_interval=100,
    threads=4,
    device="auto",
    train_subset=None,
    test_subset=None,
    smoke=False,
    eps_num=1e-12,
)

_SMOKE_OVERRIDES = dict(
    epochs=1,
    batch_size=4,
    K=2,
    M_syn=8,
    eval_interval=1,
    train_subset=16,
    test_subset=32,
    smoke=True,
)

CONFIG_KEYS = tuple(_PROTOCOL)


def lr_tag(lr):
    """Return a deterministic directory-safe tag, e.g. ``0p15``."""
    value = float(lr)
    if value < 0:
        raise ValueError("learning rate must be non-negative")
    return f"{value:.2f}".replace(".", "p")


def _run_id(method, lr):
    if method == "dp_sgd":
        return f"dp_sgd_lr{lr_tag(lr)}"
    if method == "dp_fisher_wiener":
        return f"dp_fisher_wiener_lr{lr_tag(lr)}"
    raise ValueError(method)


RUN_SPECS = tuple(
    ({"run_id": _run_id("dp_sgd", DP_SGD_LR), "method": "dp_sgd", "learning_rate": DP_SGD_LR},)
    + tuple(
        {"run_id": _run_id("dp_fisher_wiener", lr), "method": "dp_fisher_wiener", "learning_rate": lr}
        for lr in FISHER_LRS
    )
)
RUN_IDS = tuple(spec["run_id"] for spec in RUN_SPECS)
RUN_SPEC_BY_ID = {spec["run_id"]: spec for spec in RUN_SPECS}


def run_specs(config=None):
    """Return fresh canonical specs; learning rates never come from the CLI."""
    if config is not None:
        check_config(config)
    return [dict(spec) for spec in RUN_SPECS]


def run_spec(run_id, config=None):
    if config is not None:
        check_config(config)
    try:
        return dict(RUN_SPEC_BY_ID[run_id])
    except KeyError as exc:
        raise ValueError(f"Unknown canonical run id: {run_id}") from exc


def synthetic_batches(config):
    if config["M_syn"] % config["batch_size"]:
        raise ValueError("M_syn must be an integer number of batches")
    return config["M_syn"] // config["batch_size"]


def expected_total_steps(config, n=None):
    n = n or config["train_subset"] or 60000
    return (n // config["batch_size"]) * config["epochs"]


def check_config(config):
    """Reject protocol drift, including any unapproved LR grid changes."""
    if set(config) != set(CONFIG_KEYS):
        extra = sorted(set(config) - set(CONFIG_KEYS))
        missing = sorted(set(CONFIG_KEYS) - set(config))
        raise ValueError(f"Config keys mismatch; extra={extra}, missing={missing}")
    if config["experiment"] != "expv1b":
        raise ValueError("experiment must be expv1b")
    if type(config["seed"]) is not int or config["seed"] != 42:
        raise ValueError("ExpV1b uses seed=42 only")
    if config["dp_sgd_learning_rate"] != DP_SGD_LR:
        raise ValueError("DP-SGD anchor LR must be 0.1")
    if list(config["fisher_learning_rates"]) != list(FISHER_LRS):
        raise ValueError(f"Fisher LR grid must be exactly {list(FISHER_LRS)}")
    if config["optimizer"] != "SGD":
        raise ValueError("Only SGD is allowed")
    if config["momentum"] != 0:
        raise ValueError("momentum must be zero")
    if config["beta"] != 1:
        raise ValueError("beta must be one")
    fixed = dict(_PROTOCOL)
    if config["smoke"]:
        fixed.update(_SMOKE_OVERRIDES)
    for key, expected in fixed.items():
        if key == "device":
            continue
        if config[key] != expected:
            raise ValueError(f"Fixed protocol mismatch: {key}")
    for key in ("epochs", "batch_size", "K", "M_syn", "eval_interval", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if config["train_subset"] is not None and (type(config["train_subset"]) is not int or config["train_subset"] < 1):
        raise ValueError("train_subset must be null or positive integer")
    if config["test_subset"] is not None and (type(config["test_subset"]) is not int or config["test_subset"] < 1):
        raise ValueError("test_subset must be null or positive integer")
    synthetic_batches(config)
    torch.set_num_threads(config["threads"])
    return config


def read_config(path):
    return check_config(json.loads(Path(path).read_text()))


def output_path(path):
    """Resolve an output path and enforce the expv1b-only write boundary."""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(RUNS_ROOT.resolve()):
        raise ValueError(f"All ExpV1b outputs must be inside {RUNS_ROOT}")
    return resolved


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


def _hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    """Hash all code that can affect this experiment or its imported math."""
    result = {}
    upstream = REPO / "src" / "dp_kfac"
    for path in sorted(upstream.rglob("*.py")):
        result[f"upstream/{path.relative_to(upstream)}"] = _hash_file(path)
    expv1_root = ROOT.parent / "expv1"
    for name in ("common.py", "fisher_wiener.py", "metrics.py", "__init__.py"):
        path = expv1_root / name
        if path.exists():
            result[f"expv1/{name}"] = _hash_file(path)
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT)
        if "__pycache__" not in relative.parts:
            result[f"expv1b/{relative}"] = _hash_file(path)
    result.update(checkout_info())
    return result


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        fields = ["step"]
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def datasets(config):
    """Load MNIST while making any new cache only under expv1b/runs."""
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


def signal_amplitude_retention(signal_retention):
    return float(max(float(signal_retention), 0.0) ** 0.5)


def add_effective_signal_metrics(row, learning_rate):
    """Add diagnostics only; no value returned here is used by training."""
    amplitude = signal_amplitude_retention(row["signal_retention"])
    row.update(
        learning_rate=float(learning_rate),
        signal_amplitude_retention=amplitude,
        effective_signal_lr=float(learning_rate) * amplitude,
        optimizer_update_norm=float(learning_rate) * row["filtered_gradient_norm"],
        clean_reference_update_norm=0.1 * row["clean_clipped_norm"],
    )
    row["update_to_clean_reference_ratio"] = row["optimizer_update_norm"] / (
        row["clean_reference_update_norm"] + 1e-12
    )
    return row


def threshold_steps(evaluations, threshold):
    """Return the first completed private step meeting an accuracy threshold."""
    for row in evaluations:
        accuracy = row.get("test_accuracy")
        if accuracy is not None and float(accuracy) >= threshold:
            return int(row["step"]) + 1
    return None


# A descriptive alias useful to small, isolated tests.
steps_to_accuracy_threshold = threshold_steps


@contextmanager
def isolated_rng(seed, device):
    """Compatibility helper for tiny harness tests."""
    stream = RNGStream(seed, device)
    with stream.use():
        yield stream

