"""ExpV3 protocol, paired run specifications, and isolated I/O helpers.

The experiment deliberately imports the KFAC construction and beta algebra
from the earlier experiments.  This module owns only ExpV3 protocol and
bookkeeping; it contains no second implementation of those formulas.
"""

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

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
EXP2_REFERENCE_ROOT = ROOT.parent / "expv2" / "runs" / "formal_20260909_124540"
PINNED_COMMIT = "eb31b9aeb2280642684f4cedfa65cc02b76c76cd"

DP_METHOD = "dp_sgd"
FISHER_METHOD = "dp_fisher_wiener_adaptive_beta"
METHODS = (DP_METHOD, FISHER_METHOD)
DP_SGD_LR = 0.5
FISHER_LRS = (0.5, 1.0, 5.0)
FORMAL_SEEDS = (42, 7, 91)
LAYERS = tuple(LAYERS)

CONFIG_KEYS = (
    "experiment", "seeds", "dp_sgd_learning_rate", "fisher_learning_rates",
    "beta_initial", "beta_update_rule", "beta_window", "epochs", "batch_size",
    "epsilon", "delta", "max_grad_norm", "optimizer", "momentum", "K", "M_syn",
    "eval_interval", "device", "threads", "train_subset", "test_subset", "smoke",
)

FULL_CONFIG = {
    "experiment": "expv3", "seeds": [42, 7, 91],
    "dp_sgd_learning_rate": 0.5, "fisher_learning_rates": [0.5, 1.0, 5.0],
    "beta_initial": 1.0, "beta_update_rule": "one_interval_lag_hold_last_positive",
    "beta_window": 50, "epochs": 5, "batch_size": 256, "epsilon": 1.0,
    "delta": 1e-5, "max_grad_norm": 1.0, "optimizer": "SGD", "momentum": 0,
    "K": 50, "M_syn": 2560, "eval_interval": 100, "device": "auto", "threads": 4,
    "train_subset": None, "test_subset": None, "smoke": False,
}

SMOKE_CONFIG = {
    **FULL_CONFIG, "seeds": [42], "beta_window": 2, "epochs": 1, "batch_size": 4,
    "K": 2, "M_syn": 8, "eval_interval": 1, "train_subset": 16,
    "test_subset": 32, "smoke": True,
}


def lr_tag(lr):
    value = float(lr)
    if not math.isfinite(value) or value <= 0 or value >= 10:
        raise ValueError("learning rate must be finite, positive, and below 10")
    return f"{value:.2f}".replace(".", "p")


def run_id(method, learning_rate):
    tag = lr_tag(learning_rate)
    if method == DP_METHOD:
        return f"dp_sgd_lr{tag}"
    if method == FISHER_METHOD:
        return f"dp_fisher_wiener_adaptive_beta_lr{tag}"
    raise ValueError(method)


def run_specs(config=None):
    if config is not None:
        check_config(config)
    return [
        {"run_id": run_id(DP_METHOD, DP_SGD_LR), "method": DP_METHOD,
         "learning_rate": DP_SGD_LR},
        *[{"run_id": run_id(FISHER_METHOD, lr), "method": FISHER_METHOD,
           "learning_rate": lr} for lr in FISHER_LRS],
    ]


def run_specs_for_seed(config, seed):
    check_config(config)
    if seed not in config["seeds"]:
        raise ValueError(f"seed {seed} is not in config")
    return [dict(spec, seed=seed) for spec in run_specs(config)]


def run_spec(name, config=None):
    for spec in run_specs(config):
        if spec["run_id"] == name:
            return dict(spec)
    raise ValueError(f"Unknown canonical run id: {name}")


def expected_total_steps(config, n=None):
    n = n or config["train_subset"] or 60000
    return (n // config["batch_size"]) * config["epochs"]


def check_config(config):
    """Reject every protocol change, including undocumented controls."""
    if set(config) != set(CONFIG_KEYS):
        raise ValueError(
            f"Config keys mismatch; extra={sorted(set(config)-set(CONFIG_KEYS))}, "
            f"missing={sorted(set(CONFIG_KEYS)-set(config))}"
        )
    expected = SMOKE_CONFIG if config["smoke"] else FULL_CONFIG
    for key, value in expected.items():
        if key == "device":
            continue
        if config[key] != value:
            raise ValueError(f"Fixed ExpV3 protocol mismatch: {key}")
    if config["device"] != "auto" and not isinstance(config["device"], str):
        raise ValueError("device must be 'auto' or a device string")
    if list(config["fisher_learning_rates"]) != list(FISHER_LRS):
        raise ValueError("Fisher learning rates must be exactly [0.5, 1.0, 5.0]")
    if config["dp_sgd_learning_rate"] != DP_SGD_LR:
        raise ValueError("DP-SGD learning rate must be exactly 0.5")
    if config["beta_initial"] != 1.0:
        raise ValueError("beta_initial must be exactly 1.0")
    if config["beta_update_rule"] != "one_interval_lag_hold_last_positive":
        raise ValueError("unsupported beta update rule")
    if config["beta_window"] != config["K"]:
        raise ValueError("beta_window must equal K")
    if config["optimizer"] != "SGD" or config["momentum"] != 0:
        raise ValueError("ExpV3 requires SGD with momentum=0")
    for key in ("epochs", "batch_size", "beta_window", "K", "M_syn", "eval_interval", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if not config["seeds"] or any(type(seed) is not int for seed in config["seeds"]):
        raise ValueError("seeds must be nonempty integers")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be unique")
    if config["M_syn"] % config["batch_size"]:
        raise ValueError("M_syn must be an integer number of synthetic batches")
    for key in ("train_subset", "test_subset"):
        value = config[key]
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{key} must be null or a positive integer")
    torch.set_num_threads(config["threads"])
    return config


def read_config(path):
    return check_config(json.loads(Path(path).read_text()))


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def output_path(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(RUNS_ROOT.resolve()):
        raise ValueError(f"All ExpV3 outputs must be inside {RUNS_ROOT}")
    return resolved


def _hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkout_info():
    def git(*args):
        return subprocess.check_output(
            ["git", "--no-optional-locks", "-C", str(REPO), *args], text=True
        ).strip()
    status = git("status", "--porcelain", "--untracked-files=all")
    return {
        "upstream_git_commit": git("rev-parse", "HEAD"),
        "upstream_git_dirty": bool(status),
        "upstream_git_status": status,
        "upstream_git_remote": git("remote", "-v"),
    }


def require_pinned(info, smoke):
    if not smoke and (info["upstream_git_dirty"] or info["upstream_git_commit"] != PINNED_COMMIT):
        raise ValueError("Formal mode requires clean pinned DP-KFC checkout")


def provenance():
    """Hash all upstream/imported/local Python that can affect ExpV3."""
    result = {}
    upstream = REPO / "src" / "dp_kfac"
    for path in sorted(upstream.rglob("*.py")):
        result[f"upstream/{path.relative_to(upstream)}"] = _hash_file(path)
    for root, prefix in (
        (ROOT.parent / "expv1", "imported_expv1"),
        (ROOT.parent / "expv1b", "imported_expv1b"),
        (ROOT.parent / "expv2", "imported_expv2"),
    ):
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" not in path.parts:
                result[f"{prefix}/{path.relative_to(root)}"] = _hash_file(path)
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" not in path.parts:
            result[f"expv3/{path.relative_to(ROOT)}"] = _hash_file(path)
    result.update({
        "fixed_reference_expv1_root": str(EXP1_REFERENCE_ROOT),
        "fixed_reference_expv1b_root": str(EXP1B_REFERENCE_ROOT),
        "fixed_reference_expv2_formal_root": str(EXP2_REFERENCE_ROOT),
    })
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
    """Load MNIST while keeping any cache created by ExpV3 inside ExpV3."""
    cache_root = RUNS_ROOT / "_cache"
    cache = cache_root / "data" / "MNIST"
    if not cache.exists():
        for source in (
            ROOT.parent / "exp1" / "data" / "MNIST",
            ROOT.parent / "expv1" / "runs" / "_cache" / "data" / "MNIST",
            ROOT.parent / "expv2" / "runs" / "_cache" / "data" / "MNIST",
            REPO / "data" / "MNIST",
        ):
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
    return math.sqrt(max(float(signal_retention), 0.0))


def add_effective_step_metrics(row, learning_rate):
    """Add research-only effective-step fields; never feed them to training."""
    amplitude = signal_amplitude_retention(row["signal_retention"])
    row.update(
        learning_rate=float(learning_rate),
        signal_amplitude_retention=amplitude,
        effective_signal_lr=float(learning_rate) * amplitude,
        optimizer_update_norm=float(learning_rate) * float(row["filtered_gradient_norm"]),
        clean_reference_update_norm=0.5 * float(row["clean_clipped_norm"]),
        lr_compensation_to_dp_sgd_0p5=0.5 / (amplitude + 1e-12),
    )
    row["update_to_clean_reference_ratio"] = row["optimizer_update_norm"] / (
        row["clean_reference_update_norm"] + 1e-12
    )
    return row


def threshold_step(evaluations, threshold):
    for row in evaluations:
        value = row.get("test_accuracy")
        if value is not None and math.isfinite(float(value)) and float(value) >= threshold:
            return int(row["step"]) + 1
    return None


@contextmanager
def isolated_rng(seed, device):
    stream = RNGStream(seed, device)
    with stream.use():
        yield stream


def build_beta_interval_rows(rows, window):
    """Preserve deployable interval algebra when research oracle values are missing."""
    from expv2.beta_estimation import build_interval_rows
    normalized = [dict(row, clean_signal_energy=(float('nan')
                  if row.get('clean_signal_energy') is None else row['clean_signal_energy']))
                  for row in rows]
    return build_interval_rows(normalized, window)
