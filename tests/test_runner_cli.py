import os
from pathlib import Path
import subprocess
import sys
import json
import yaml
import pytest


@pytest.mark.parametrize("option", [[], ["--config"]])
def test_cli_bad_config(tmp_path, option):
    config = tmp_path / "bad.yaml"
    config.write_text("algorithm: dp_adam\n")
    p = subprocess.run(
        [sys.executable, "scripts/train.py", *option, str(config)],
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0 and "Unsupported algorithm" in p.stderr


def test_launcher_arguments():
    p = subprocess.run(
        ["bash", "run.sh", "--help"],
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0 and "--config" in p.stdout


@pytest.mark.parametrize("algorithm", ["dp_sgd", "dp_fisher_wiener"])
def test_real_mnist_cli(tmp_path, algorithm):
    # Real MNIST pipeline; uses the project's cache or torchvision download.
    tag = "dpsgd" if algorithm == "dp_sgd" else "fisher_wiener"
    c = yaml.safe_load(Path(f"config/mnist_{tag}_smoke.yaml").read_text())
    c["runtime"]["device"] = "cpu"
    c["output"]["root"] = str(tmp_path / "outputs")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(c))
    args = ["--config", str(config)] if algorithm == "dp_sgd" else [str(config)]
    p = subprocess.run(
        [sys.executable, "scripts/train.py", *args],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    runs = list((tmp_path / "outputs").iterdir())
    assert len(runs) == 1
    result = json.loads((runs[0] / "summary.json").read_text())
    assert result["status"] == "completed" and result["completed_steps"] == 4
