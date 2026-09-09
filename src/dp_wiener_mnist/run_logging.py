"""Unique experiment directories and immediately flushed metrics/logs."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
import yaml

FIELDS = [
    "step",
    "epoch",
    "algorithm",
    "learning_rate",
    "train_loss",
    "test_loss",
    "test_accuracy",
    "epsilon_spent",
    "clip_rate",
    "gradient_norm_after_dp",
    "filter_refresh",
    "refresh_time",
    "filter_time",
    "filtered_gradient_norm",
    "active_state_bytes",
]


def write_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


class RunLog:
    def __init__(self, c, source: str):
        suffix = f"simplecnn_mnist_{c.algorithm}_eps{c.privacy.epsilon:g}_lr{c.training.learning_rate:g}"
        if c.algorithm == "dp_fisher_wiener":
            suffix += f"_K{c.wiener.refresh_interval}_M{c.wiener.synthetic_samples}"
        self.root = Path(c.output.root) / (
            datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "_" + suffix.replace(".", "p")
        )
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "config.yaml").write_text(source)
        write_yaml(self.root / "resolved_config.yaml", c.to_dict())
        self.handle = (self.root / "metrics.csv").open("w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=FIELDS)
        self.writer.writeheader()
        self.handle.flush()
        self.logger = logging.getLogger(str(self.root.resolve()))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = logging.FileHandler(self.root / "train.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)
        self.logger.info("Run initialized: %s", self.root)

    def append(self, row: dict) -> None:
        self.writer.writerow(row)
        self.handle.flush()
        self.logger.info("%s", row)

    def close(self) -> None:
        self.handle.close()
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
