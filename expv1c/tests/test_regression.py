"""Compare complete tiny trajectories with the unchanged ExpV1b loop."""
from expv1b import train_expv1b as reference
from expv1b.common import read_config as read_reference_config
from expv1c.common import output_path, REFERENCE_ROOT
from expv1c.train_expv1c import train
from expv1c.validate_expv1c import load, ANCHORS


def test_regression(config, tiny_data, tmp_path, monkeypatch):
    ref_config = read_reference_config("expv1b/configs/smoke.json")
    ref_config["device"] = config["device"]
    # The only reference patch redirects all test writes into expv1c/runs.
    monkeypatch.setattr(reference, "output_path", output_path)
    for run_id in ANCHORS:
        reference.train(ref_config, run_id, tmp_path / "reference", tiny_data)
        train(config, run_id, tmp_path / "actual", tiny_data)
        ref, actual = [tmp_path / side / "seed42" / run_id for side in ("reference", "actual")]
        assert load(ref / "pairing.json") == load(actual / "pairing.json")
        assert load(ref / "summary.json")["final_model_hash"] == load(actual / "summary.json")["final_model_hash"]
        assert load(REFERENCE_ROOT / "seed42" / run_id / "summary.json")["final_model_hash"] == ANCHORS[run_id]
