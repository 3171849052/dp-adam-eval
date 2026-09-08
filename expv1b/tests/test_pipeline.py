import json

from expv1b.common import RUN_IDS
from expv1b.plot_expv1b import plot
from expv1b.summarize_expv1b import summarize
from expv1b.train_expv1b import train
from expv1b.validate_expv1b import validate


def test_diagnostic_isolation(config, tiny_data, tmp_path):
    train(config, "dp_fisher_wiener_lr0p30", tmp_path / "on", tiny_data, diagnostics=True)
    train(config, "dp_fisher_wiener_lr0p30", tmp_path / "off", tiny_data, diagnostics=False)
    on_root = tmp_path / "on" / "seed42" / "dp_fisher_wiener_lr0p30"
    off_root = tmp_path / "off" / "seed42" / "dp_fisher_wiener_lr0p30"
    on_meta = json.loads((on_root / "summary.json").read_text())
    off_meta = json.loads((off_root / "summary.json").read_text())
    assert on_meta["final_model_hash"] == off_meta["final_model_hash"]
    assert json.loads((on_root / "pairing.json").read_text()) == json.loads((off_root / "pairing.json").read_text())


def test_pipeline(config, tiny_data, tmp_path):
    for run_id in RUN_IDS:
        train(config, run_id, tmp_path, tiny_data)
    assert validate(config, tmp_path, tmp_path, require_tests=False)["passed"]
    summarize(config, tmp_path, tmp_path, require_tests=False)
    plot(config, tmp_path, tmp_path, require_tests=False)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 6
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 6

