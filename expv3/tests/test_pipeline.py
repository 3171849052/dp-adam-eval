import json

import pandas as pd

from expv3.common import run_specs_for_seed
from expv3.plot_expv3 import plot
from expv3.summarize_expv3 import summarize
from expv3.train_expv3 import train
from expv3.validate_expv3 import validate


def test_pipeline(config, tiny_data, tmp_path):
    for spec in run_specs_for_seed(config, 42):
        train(config, 42, spec["run_id"], tmp_path, tiny_data)
    assert validate(config, tmp_path, tmp_path, require_tests=False)["passed"]
    summarize(config, tmp_path, tmp_path, require_tests=False)
    figures = plot(config, tmp_path, tmp_path, require_tests=False)
    assert figures and len(figures) >= 13
    assert validate(config, tmp_path, tmp_path, require_tests=False)["passed"]
    assert (tmp_path / "summary_paired_utility.csv").exists()
    for spec in run_specs_for_seed(config, 42):
        root = tmp_path / "seed42" / spec["run_id"]
        assert len(pd.read_csv(root / "beta_step_metrics.csv")) == 16
        assert len(pd.read_csv(root / "beta_interval_metrics.csv")) == 8
        assert len(pd.read_csv(root / "beta_controller_metrics.csv")) == 8
        assert json.loads((root / "metadata.json").read_text())["actual_noise_saved"] is False

