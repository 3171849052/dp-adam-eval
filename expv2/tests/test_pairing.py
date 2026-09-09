import json

from expv2.train_expv2 import train


def test_paired_rng(config, tiny_data, tmp_path):
    names = ["dp_sgd_lr0p10", "dp_fisher_wiener_lr0p10", "dp_fisher_wiener_lr0p80"]
    pairs = {}
    for name in names:
        train(config, 42, name, tmp_path / name, tiny_data)
        pairs[name] = json.loads((tmp_path / name / "seed42" / name / "pairing.json").read_text())
    base = pairs[names[0]]
    for name in names[1:]:
        for key in ("batch_indices", "noise_rng_before", "noise_rng_after"):
            assert [row[key] for row in pairs[name]["private"]] == [row[key] for row in base["private"]]
        assert pairs[name]["synthetic"] == base["synthetic"]

