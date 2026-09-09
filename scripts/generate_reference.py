#!/usr/bin/env python
"""Development-only: run the original ExpV1b to regenerate a portable hash fixture.

Invoke with PYTHONPATH pointing to the audited dp-adam-eval checkout. Training
and pytest never import that checkout. This script intentionally uses upstream
private_update, not the standalone implementation.
"""
import json
import subprocess
from pathlib import Path
import torch
from opacus import GradSampleModule
from expv1.common import SimpleCNN, RNGStream, digest, set_seed
from expv1 import fisher_wiener as fw
from expv1b import train_expv1b as source


def hashes(model, attr=None):
    return [
        digest([p if attr is None else getattr(p, attr)]) for p in model.parameters()
    ]


def generate():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 4,
        "seed": 42,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "batch_size": 4,
        "M_syn": 8,
        "K": 2,
        "sigma": 2.0,
        "steps": 3,
        "algorithms": {},
    }
    x = torch.randn(12, 1, 28, 28, generator=torch.Generator().manual_seed(301))
    y = torch.arange(12) % 10
    for algorithm in ("dp_sgd", "dp_fisher_wiener"):
        set_seed(42)
        model = GradSampleModule(SimpleCNN(), loss_reduction="sum")
        optimizer = source.make_optimizer(model, 0.1)
        noise = RNGStream(46, "cpu")
        syn = RNGStream(45, "cpu")
        record = {"initial": hashes(model), "steps": []}
        active = None
        c = {"batch_size": 4, "M_syn": 8, "smoke": True, "max_grad_norm": 1.0}
        for step in range(3):
            row = {}
            if algorithm == "dp_fisher_wiener" and step % 2 == 0:
                probes, audit = fw.synthetic_samples(c, torch.device("cpu"), syn)
                cov = fw.build_covariances(
                    model._module.state_dict(), probes, c, torch.device("cpu")
                )
                active = fw.build_fisher_state(cov, 2.0, 1.0, 4)
                row["synthetic"] = audit
            model.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(
                model(x[step * 4 : step * 4 + 4]),
                y[step * 4 : step * 4 + 4],
                reduction="sum",
            ).backward()
            original_clip = source.clip_and_noise_gradients
            original_step = optimizer.step

            def capture_clip(*a, **kw):
                original_clip(*a, **kw)
                row["dp_noisy"] = hashes(model, "grad")

            def capture_step(*a, **kw):
                row["filtered"] = hashes(model, "grad")
                return original_step(*a, **kw)

            source.clip_and_noise_gradients = capture_clip
            optimizer.step = capture_step
            try:
                source.private_update(
                    model,
                    active,
                    optimizer,
                    noise,
                    2.0,
                    c,
                    4,
                    algorithm,
                    0.1,
                    diagnostics=False,
                )
            finally:
                source.clip_and_noise_gradients = original_clip
                optimizer.step = original_step
            row["parameters"] = hashes(model)
            row["noise_rng"] = noise.audit()
            record["steps"].append(row)
        record["final_hash"] = digest(model.parameters())
        result["algorithms"][algorithm] = record
        model.remove_hooks()
    return result


if __name__ == "__main__":
    Path("tests/fixtures/expv1b_cpu_trajectory.json").write_text(
        json.dumps(generate(), indent=2) + "\n"
    )
