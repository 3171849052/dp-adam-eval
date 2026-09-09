import torch
from torch.utils.data import TensorDataset

from expv4a.train_expv4a import SMOKE_CONFIG


def smoke_config():
    config = dict(SMOKE_CONFIG)
    config["device"] = "cpu"
    return config


def tiny_data():
    generator = torch.Generator().manual_seed(123)
    data = TensorDataset(
        torch.randn(32, 1, 28, 28, generator=generator),
        torch.randint(0, 10, (32,), generator=generator),
    )
    return data, data
