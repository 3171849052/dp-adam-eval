"""Bitwise multi-step comparison against hashes produced by actual ExpV1b."""

import json
from pathlib import Path
import pytest
import torch
from opacus import GradSampleModule
from dp_wiener_mnist import trainer as tr
from dp_wiener_mnist import fisher_wiener as fw
from dp_wiener_mnist.model import SimpleCNN
from dp_wiener_mnist.utils import RNGStream, digest, set_seed


def hashes(model, attr=None):
    return [
        digest([p if attr is None else getattr(p, attr)]) for p in model.parameters()
    ]


@pytest.mark.parametrize("algorithm", ["dp_sgd", "dp_fisher_wiener"])
def test_source_trajectory(algorithm, monkeypatch):
    fixture = json.loads(Path("tests/fixtures/expv1b_cpu_trajectory.json").read_text())
    expected = fixture["algorithms"][algorithm]
    x = torch.randn(12, 1, 28, 28, generator=torch.Generator().manual_seed(301))
    y = torch.arange(12) % 10
    set_seed(42)
    model = GradSampleModule(SimpleCNN(), loss_reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0)
    noise = RNGStream(46, "cpu")
    syn = RNGStream(45, "cpu")
    active = None
    assert hashes(model) == expected["initial"]
    c = {"batch_size": 4, "M_syn": 8}
    original = tr.clip_and_noise_gradients
    try:
        for step, row in enumerate(expected["steps"]):
            if algorithm == "dp_fisher_wiener" and step % 2 == 0:
                probes, audit = fw.synthetic_samples(c, torch.device("cpu"), syn)
                assert audit == row["synthetic"]
                cov = fw.build_covariances(
                    model._module.state_dict(), probes, c, torch.device("cpu")
                )
                active = fw.build_fisher_state(cov, 2.0, 1.0, 4)
            model.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(
                model(x[step * 4 : step * 4 + 4]),
                y[step * 4 : step * 4 + 4],
                reduction="sum",
            ).backward()

            def capture(*args, **kwargs):
                original(*args, **kwargs)
                assert (
                    hashes(model, "grad") == row["dp_noisy"]
                ), f"DP gradient differs at {step}"

            monkeypatch.setattr(tr, "clip_and_noise_gradients", capture)
            tr.private_update(model, active, optimizer, noise, 2.0, 1.0, 4, algorithm)
            assert (
                hashes(model, "grad") == row["filtered"]
            ), f"Filtered gradient differs at {step}"
            assert hashes(model) == row["parameters"], f"Parameters differ at {step}"
            assert noise.audit() == row["noise_rng"]
        assert digest(model.parameters()) == expected["final_hash"]
    finally:
        model.remove_hooks()
