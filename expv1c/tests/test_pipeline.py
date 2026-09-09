import json

import pandas as pd
import pytest

from expv1c.common import RUN_IDS
from expv1c.plot_expv1c import plot
from expv1c.summarize_expv1c import summarize
from expv1b import train_expv1b as train_module
from expv1c.train_expv1c import train
from expv1c.validate_expv1c import validate


def test_pipeline(config, tiny_data, tmp_path):
    for run_id in RUN_IDS:
        train(config, run_id, tmp_path, tiny_data)
    assert validate(config, tmp_path, tmp_path, require_tests=False)["passed"]
    summarize(config, tmp_path, tmp_path, require_tests=False)
    plot(config, tmp_path, tmp_path, require_tests=False)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 7
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 7
