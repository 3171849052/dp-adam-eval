import json

from expv1c.common import run_specs
from expv1c.train_expv1c import train


def test_paired_rng(config, tiny_data, tmp_path):
    selected = ["dp_sgd_lr0p10", "dp_fisher_wiener_lr0p80", "dp_fisher_wiener_lr1p00", "dp_fisher_wiener_lr1p50"]
    artifacts = {}
    for run_id in selected:
        train(config, run_id, tmp_path, tiny_data)
        root = tmp_path / "seed42" / run_id
        artifacts[run_id] = json.loads((root / "pairing.json").read_text())
    initial = [json.loads((tmp_path / "seed42" / run_id / "metadata.json").read_text())["initial_model_hash"] for run_id in selected]
    assert len(set(initial)) == 1
    base = artifacts[selected[0]]["private"]
    for run_id in selected[1:]:
        pair = artifacts[run_id]["private"]
        assert [x["batch_indices"] for x in pair] == [x["batch_indices"] for x in base]
        assert [x["noise_rng_before"] for x in pair] == [x["noise_rng_before"] for x in base]
        assert [x["noise_rng_after"] for x in pair] == [x["noise_rng_after"] for x in base]
    synthetic = [artifacts[run_id]["synthetic"] for run_id in selected[1:]]
    for pair in synthetic[1:]:
        assert pair == synthetic[0]

