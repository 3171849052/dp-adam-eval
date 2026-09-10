import json

import torch
from torch.utils.data import TensorDataset

from exp6b.common import DEFAULT, METHODS
from exp6b.train_exp6b import train


def test_three_methods_share_private_and_synthetic_streams(tmp_path):
    config = dict(DEFAULT, smoke=True, seeds=[42], device="cpu", threads=1,
                  batch_size=4, epochs=1, M_syn=8, K=2,
                  analysis_batch_size=4, train_subset=8, test_subset=8,
                  eval_interval=2)
    torch.manual_seed(123)
    data = TensorDataset(torch.randn(8, 1, 28, 28), torch.arange(8) % 10)
    for method in METHODS:
        train(config, 42, method, tmp_path, (data, data))
    audits = {
        method: json.loads((tmp_path / "seed42" / method / "pairing.json").read_text())
        for method in METHODS
    }
    private = [(item["batch_hash"], item["noise_hash"])
               for item in audits["dp_adam"]["private"]]
    for method in METHODS:
        assert [(item["batch_hash"], item["noise_hash"])
                for item in audits[method]["private"]] == private
    synthetic = [(item["samples_hash"], item["labels_hash"])
                 for item in audits["syn_adam_floor"]["synthetic"]]
    assert [(item["samples_hash"], item["labels_hash"])
            for item in audits["syn_adam_ema_floor"]["synthetic"]] == synthetic
