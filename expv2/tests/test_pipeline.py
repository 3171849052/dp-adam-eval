import json

import pandas as pd

from expv2.common import run_specs_for_seed
from expv2.plot_expv2 import plot
from expv2.summarize_expv2 import summarize
from expv2.train_expv2 import train
from expv2.validate_expv2 import validate


def test_pipeline(config, tiny_data, tmp_path):
    for spec in run_specs_for_seed(config, 42):
        train(config, 42, spec["run_id"], tmp_path, tiny_data)
    assert validate(config, tmp_path, tmp_path, require_tests=False)["passed"]
    summarize(config, tmp_path, tmp_path, require_tests=False)
    plot(config, tmp_path, tmp_path, require_tests=False)
    assert validate(config, tmp_path, tmp_path, require_tests=False)["passed"]
    for name in ("summary_runs.csv", "summary_beta_layers.csv", "summary_beta_trajectories.csv",
                 "summary_beta_lagged.csv", "summary_window_sensitivity.csv",
                 "beta_trajectory_dependence.csv", "summary.json", "validation.json"):
        assert (tmp_path / name).exists()
    for spec in run_specs_for_seed(config, 42):
        root = tmp_path / "seed42" / spec["run_id"]
        assert len(pd.read_csv(root / "beta_step_metrics.csv")) == 16
        assert len(pd.read_csv(root / "beta_interval_metrics.csv")) == 8
        assert len(pd.read_csv(root / "beta_lagged_metrics.csv")) == 4
    assert len(list((tmp_path / "figures").glob("*.png"))) == 7
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 7
