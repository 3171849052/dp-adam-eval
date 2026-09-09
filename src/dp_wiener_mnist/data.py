"""MNIST data loading and true Bernoulli Poisson private sampling."""

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, default_collate
from torchvision import datasets, transforms
from .config import Config


class PoissonBatchSampler(Sampler[list[int]]):
    """Yield independent Bernoulli-Poisson batches for a fixed step budget.

    A fresh mask is drawn for every planned step.  In particular, empty masks
    are yielded unchanged instead of being resampled or skipped.
    """

    def __init__(
        self,
        data_source: Dataset,
        sample_rate: float,
        steps_per_epoch: int,
        generator: torch.Generator,
    ) -> None:
        self.data_source = data_source
        self.sample_rate = float(sample_rate)
        self.steps_per_epoch = int(steps_per_epoch)
        self.generator = generator
        if not 0 < self.sample_rate <= 1:
            raise ValueError("sample_rate must satisfy 0 < q <= 1")
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")

    def __iter__(self):
        n = len(self.data_source)
        for _ in range(self.steps_per_epoch):
            included = torch.rand(n, generator=self.generator) < self.sample_rate
            yield included.nonzero(as_tuple=False).flatten().tolist()

    def __len__(self) -> int:
        return self.steps_per_epoch


def _poisson_collate(batch):
    """Make the legal empty Poisson batch representable to DataLoader."""
    return None if not batch else default_collate(batch)


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
    train_dataset = Subset(source, range(n))
    sample_rate = c.data.batch_size / n
    steps_per_epoch = n // c.data.batch_size
    if not 0 < sample_rate <= 1 or steps_per_epoch <= 0:
        raise ValueError("Invalid Poisson sampling parameters")
    train = DataLoader(
        train_dataset,
        batch_sampler=PoissonBatchSampler(
            train_dataset,
            sample_rate=sample_rate,
            steps_per_epoch=steps_per_epoch,
            generator=torch.Generator().manual_seed(c.seed + 1),
        ),
        num_workers=c.data.num_workers,
        collate_fn=_poisson_collate,
        # DataLoader itself draws a worker base seed on iterator creation.
        # Give that bookkeeping a private generator so it cannot consume the
        # global RNG or advance the sampler's Bernoulli generator.
        generator=torch.Generator().manual_seed(c.seed + 1),
    )
    test = DataLoader(
        Subset(test, range(m)),
        batch_size=c.data.eval_batch_size,
        num_workers=c.data.num_workers,
        generator=torch.Generator().manual_seed(c.seed + 2),
    )
    return train, test
