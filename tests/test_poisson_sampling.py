import torch
from torch.utils.data import TensorDataset

from dp_wiener_mnist.data import PoissonBatchSampler, build_data


def test_poisson_sampler_matches_independent_bernoulli_masks():
    dataset = TensorDataset(torch.arange(16), torch.arange(16))
    generator = torch.Generator().manual_seed(43)
    sampler = PoissonBatchSampler(dataset, sample_rate=0.25, steps_per_epoch=4, generator=generator)
    actual = list(sampler)

    reference_generator = torch.Generator().manual_seed(43)
    expected = []
    for _ in range(4):
        mask = torch.rand(16, generator=reference_generator) < 0.25
        expected.append(mask.nonzero(as_tuple=False).flatten().tolist())
    assert actual == expected
    assert [len(batch) for batch in actual] == [5, 3, 3, 5]
    assert len(sampler) == 4


def test_build_data_uses_poisson_and_fixed_planned_steps(smoke, tiny_data):
    loader, _ = build_data(smoke, tiny_data)
    sampler = loader.batch_sampler
    assert isinstance(sampler, PoissonBatchSampler)
    assert sampler.sample_rate == smoke.data.batch_size / 16
    assert len(loader) == 16 // smoke.data.batch_size
    assert loader.batch_size is None


def test_empty_poisson_batch_is_yielded_without_resampling():
    dataset = TensorDataset(torch.arange(4), torch.arange(4))
    sampler = PoissonBatchSampler(
        dataset,
        sample_rate=0.25,
        steps_per_epoch=1,
        generator=torch.Generator().manual_seed(1),
    )
    assert list(sampler) == [[]]


def test_dataloader_sampling_does_not_touch_global_rng(smoke, tiny_data):
    torch.manual_seed(901)
    before = torch.get_rng_state().clone()
    loader, test_loader = build_data(smoke, tiny_data)
    list(loader)
    list(test_loader)
    assert torch.equal(before, torch.get_rng_state())
