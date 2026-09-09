import json
import shutil
import tempfile
from pathlib import Path

from expv3 import common as v3_common
from expv3.train_expv3 import train as train_v3

from expv4a.summarize_expv4a import summarize
from expv4a.tests.conftest import smoke_config, tiny_data
from expv4a.train_expv4a import train
from expv4a.validate_expv4a import validate


def test_smoke_validate_summarize_and_v3_trajectory(tmp_path):
    config = smoke_config()
    data = tiny_data()
    run = "dp_fisher_wiener_adaptive_beta_lr0p50"
    v4a_root = tmp_path / "v4a"
    train(config, 42, run, v4a_root, data)
    validate(config, v4a_root, v4a_root, require_tests=False)
    result = summarize(config, v4a_root, v4a_root, require_tests=False)
    assert result["overall"]["gamma_model_median"] > 0
    assert result["overall"]["gamma_oracle_median"] > 0

    v3_root = Path(tempfile.mkdtemp(prefix="_expv4a_test_", dir=str(v3_common.RUNS_ROOT)))
    try:
        train_v3(v3_common_config(config), 42, run, v3_root, data_override=data)
        v4_summary = json.loads((v4a_root / "seed42" / run / "summary.json").read_text())
        v3_summary = json.loads((v3_root / "seed42" / run / "summary.json").read_text())
        assert v4_summary["final_model_hash"] == v3_summary["final_model_hash"]
        assert v4_summary["completed_steps"] == v3_summary["completed_steps"]
        assert v4_summary["privacy_steps"] == v3_summary["privacy_steps"]
        assert v4_summary["epsilon_spent"] == v3_summary["epsilon_spent"]
    finally:
        shutil.rmtree(v3_root, ignore_errors=True)


def v3_common_config(config):
    result = dict(config)
    result["experiment"] = "expv3"
    result["fisher_learning_rates"] = [0.5, 1.0, 5.0]
    return result
