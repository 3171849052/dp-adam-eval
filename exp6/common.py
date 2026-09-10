"""Small shared pieces for the paired Exp6 experiment."""
import hashlib
import json
from pathlib import Path

import torch

from exp5.common import (Indexed, RNGStream, SimpleCNN, datasets, digest,
                         evaluate, generate_pink_noise, save_json, set_seed,
                         write_csv)

ROOT = Path(__file__).resolve().parent
METHODS = ("syn_adam", "syn_adam_ema", "dp_adam")
SYNTHETIC_METHODS = ("syn_adam", "syn_adam_ema")

DEFAULT = {
    "dataset": "MNIST",
    "model": "SimpleCNN",
    "seeds": [42, 7, 91],
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 5,
    "max_grad_norm": 1.0,
    "epsilon": 1.0,
    "delta": 1e-5,
    "beta1": 0.9,
    "beta2": 0.999,
    "adam_eps": 1e-8,
    "weight_decay": 0.0,
    "M_syn": 2560,
    "K": 50,
    "device": "auto",
    "threads": 4,
    "analysis_batch_size": 32,
    "eval_interval": 100,
    "diagnostic_interval": 50,
    "eps_num": 1e-12,
    "train_subset": None,
    "test_subset": None,
    "smoke": False,
}


def read_config(path):
    config = json.loads(Path(path).read_text())
    if set(config) != set(DEFAULT):
        raise ValueError("Config keys must match exp6/configs/full.json")
    if config["dataset"] != "MNIST" or config["model"] != "SimpleCNN":
        raise ValueError("Exp6 uses MNIST and SimpleCNN")
    for key in ("batch_size", "epochs", "M_syn", "K", "threads",
                "analysis_batch_size", "eval_interval", "diagnostic_interval"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"Invalid positive integer: {key}")
    if config["M_syn"] % config["batch_size"]:
        raise ValueError("M_syn must contain whole synthetic batches")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    if config["weight_decay"] != 0.0:
        raise ValueError("Exp6 explicit Adam protocol requires weight_decay=0")
    if not config["smoke"]:
        for key, expected in DEFAULT.items():
            if config[key] != expected:
                raise ValueError(f"Formal protocol mismatch: {key}")
    torch.set_num_threads(config["threads"])
    return config


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def stats(values, prefix=""):
    """Return q05/q50/q95 and norm for a collection of tensors."""
    flat = torch.cat([value.detach().reshape(-1).double() for value in values])
    quantiles = torch.quantile(
        flat, torch.tensor([.05, .50, .95], dtype=flat.dtype, device=flat.device))
    return {
        f"{prefix}q05": float(quantiles[0]),
        f"{prefix}q50": float(quantiles[1]),
        f"{prefix}q95": float(quantiles[2]),
        f"{prefix}norm": float(flat.norm()),
    }


def log_rms_distance(left, right, eps):
    values = []
    for name in left:
        a = left[name].detach().double()
        b = right[name].detach().double()
        values.append((torch.log(a + eps) - torch.log(b + eps)).reshape(-1))
    return float(torch.cat(values).square().mean().sqrt())


def parameter_names(model):
    return [name for name, _ in model._module.named_parameters()]


def target_device(config):
    return (torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")) if config["device"] == "auto" else torch.device(config["device"])
