import pytest
import torch
from dp_wiener_mnist.config import load_config


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


@pytest.fixture
def smoke(tmp_path):
    c = load_config("config/mnist_dpsgd_smoke.yaml")
    c.runtime.device = "cpu"
    c.output.root = str(tmp_path)
    return c


@pytest.fixture
def tiny_data():
    from torch.utils.data import TensorDataset

    g = torch.Generator().manual_seed(123)
    return tuple(
        TensorDataset(torch.randn(n, 1, 28, 28, generator=g), torch.arange(n) % 10)
        for n in (16, 32)
    )
