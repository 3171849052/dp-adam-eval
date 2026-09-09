"""Original MNIST normalization and seeded fixed-shuffle/drop-last loaders."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from .config import Config


def build_data(
    c: Config, data_override: tuple | None = None
) -> tuple[DataLoader, DataLoader]:
    if data_override is None:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        source = datasets.MNIST(
            c.data.root, train=True, download=True, transform=transform
        )
        test = datasets.MNIST(
            c.data.root, train=False, download=True, transform=transform
        )
    else:
        source, test = data_override
    n = c.data.train_subset or len(source)
    m = c.data.test_subset or len(test)
    if not c.data.batch_size <= n <= len(source) or not 1 <= m <= len(test):
        raise ValueError("Invalid dataset subset sizes")
    train = DataLoader(
        Subset(source, range(n)),
        batch_size=c.data.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=c.data.num_workers,
        generator=torch.Generator().manual_seed(c.seed + 1),
    )
    test = DataLoader(
        Subset(test, range(m)),
        batch_size=c.data.eval_batch_size,
        num_workers=c.data.num_workers,
        generator=torch.Generator().manual_seed(c.seed + 2),
    )
    return train, test
