import json
from pathlib import Path

from expv2.validate_expv2 import ANCHORS, _reference_root


def test_regression():
    for key, expected in ANCHORS.items():
        seed, run_name = key
        reference = _reference_root(seed, run_name)
        assert reference.exists()
        summary = json.loads((reference / "summary.json").read_text())
        assert summary["final_model_hash"] == expected
    for seed in (42, 7, 91):
        for run_name in ("dp_sgd_lr0p10", "dp_fisher_wiener_lr0p10"):
            root = _reference_root(seed, run_name)
            assert (root / "pairing.json").exists()
            assert len(json.loads((root / "pairing.json").read_text())["private"]) == 1170

