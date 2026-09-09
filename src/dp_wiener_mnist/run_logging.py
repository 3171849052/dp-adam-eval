"""Per-run paths, complete protocol names, metadata, and durable metrics."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


METRICS_FIELDS = (
    "epoch",
    "global_step",
    "privacy_steps",
    "train_loss",
    "test_loss",
    "test_accuracy",
    "epsilon_spent",
    "noise_multiplier",
    "sample_rate",
    "expected_batch_size",
    "mean_sampled_batch_size",
    "total_sampled_examples",
    "clip_rate",
    "gradient_norm_after_dp",
    "filtered_gradient_norm",
    "refresh_count",
    "mean_refresh_time",
    "mean_filter_time",
    "active_state_bytes",
)
# Backwards-compatible import name; its schema is now epoch-level.
FIELDS = METRICS_FIELDS


@dataclass(frozen=True)
class RunPaths:
    """All files belonging to one prepared experiment directory."""

    directory: Path
    config: Path
    resolved_config: Path
    metrics: Path
    summary: Path
    train_log: Path


def _format_number(value: float | int) -> str:
    """Format a finite number as a stable, readable path-name token."""
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError("run name values must be finite")
    if number == number.to_integral_value():
        return str(number.quantize(Decimal("1")))
    if abs(number) < Decimal("0.001"):
        text = format(number.normalize(), "E").replace("E", "e")
        text = re.sub(r"e([+-])0+(\d+)$", r"e\1\2", text)
        return text.replace("e+", "e")
    return format(number.normalize(), "f")


def _format_name_component(value: object) -> str:
    """Convert an identifier to a safe, non-empty path-name component."""
    component = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    component = component.strip(".-_")
    if not component:
        raise ValueError("run name components must contain a path-safe character")
    return component


def _value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return _format_number(value)
    return _format_name_component(value)


def format_run_name(config, timestamp: datetime | None = None) -> str:
    """Encode the requested core experiment settings in a compact name."""
    stamp = timestamp or datetime.now()
    c = config
    tokens = [
        f"{stamp:%m%d-%H%M%S}",
        _format_name_component(c.model.name),
        _format_name_component(c.algorithm),
        f"s{_value(c.seed)}",
        f"ep{_value(c.training.epochs)}",
        f"lr{_value(c.training.learning_rate)}",
        f"eps{_value(c.privacy.epsilon)}",
        f"d{_value(c.privacy.delta)}",
        f"beta{_value(c.wiener.beta)}",
        f"K{_value(c.wiener.refresh_interval)}",
        f"M{_value(c.wiener.synthetic_samples)}",
    ]
    return "_".join(tokens)


def _run_paths(directory: Path) -> RunPaths:
    return RunPaths(
        directory=directory,
        config=directory / "config.yaml",
        resolved_config=directory / "resolved_config.yaml",
        metrics=directory / "metrics.csv",
        summary=directory / "summary.json",
        train_log=directory / "train.log",
    )


def run_paths_from_directory(directory: str | Path) -> RunPaths:
    """Resolve a directory previously created by ``--prepare-run``."""
    root = Path(directory).resolve()
    if not root.is_dir() or not (root / "config.yaml").is_file():
        raise ValueError("run directory must have been created by the runner")
    return _run_paths(root)


def format_tmux_session_name(directory: str | Path) -> str:
    """Build a deterministic tmux-safe name from a run directory."""
    run_name = Path(directory).name
    if not run_name:
        raise ValueError("run directory must have a name")
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"dp_wiener_mnist_{run_name}")


def create_run_directory(
    config,
    *,
    root: str | Path | None = None,
    now: datetime | None = None,
) -> RunPaths:
    """Create a collision-free second-precision output directory."""
    output_root = Path(root if root is not None else config.output.root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = now or datetime.now()
    for collision in range(10_000):
        candidate = output_root / format_run_name(
            config, timestamp + timedelta(seconds=collision)
        )
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        paths = _run_paths(candidate)
        paths.train_log.touch()
        return paths
    raise RuntimeError("could not allocate a unique run directory")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write JSON atomically so summaries survive interruptions."""
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(dict(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")


def write_run_metadata(
    paths: RunPaths,
    *,
    source_yaml: str | None = None,
    resolved_config: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> None:
    """Write source/resolved metadata and, when provided, the summary."""
    paths.directory.mkdir(parents=True, exist_ok=True)
    if source_yaml is not None:
        paths.config.write_text(source_yaml, encoding="utf-8")
    if resolved_config is not None:
        write_yaml(paths.resolved_config, resolved_config)
    paths.train_log.touch(exist_ok=True)
    if summary is not None:
        write_json(paths.summary, summary)


class MetricsCSVWriter:
    """Append one flushed and fsynced aggregate metric record per epoch."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or not self.path.stat().st_size:
            with self.path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=METRICS_FIELDS).writeheader()
                stream.flush()
                os.fsync(stream.fileno())

    def append(self, record: Mapping[str, Any]) -> None:
        if set(record) != set(METRICS_FIELDS):
            raise ValueError("metrics record must contain exactly the CSV fields")
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=METRICS_FIELDS).writerow(record)
            stream.flush()
            os.fsync(stream.fileno())


class RunLog:
    """Compatibility wrapper used by the trainer and direct Python callers."""

    def __init__(
        self, c=None, source: str | None = None, *, paths: RunPaths | None = None
    ):
        if paths is None:
            if c is None:
                raise ValueError("config is required when paths is omitted")
            paths = create_run_directory(c)
            write_run_metadata(
                paths,
                source_yaml=source
                if source is not None
                else yaml.safe_dump(c.to_dict(), sort_keys=False),
                resolved_config=c.to_dict(),
            )
        else:
            write_run_metadata(paths, source_yaml=source)
        self.paths = paths
        self.root = paths.directory
        self.metrics = MetricsCSVWriter(paths.metrics)
        self.logger = logging.getLogger(f"dp_wiener_mnist.{self.root.resolve()}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._handler = logging.FileHandler(paths.train_log, encoding="utf-8")
        self._handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        self.logger.addHandler(self._handler)
        self.logger.info("Run initialized: %s", self.root.resolve())

    def append(self, row: Mapping[str, Any]) -> None:
        self.metrics.append(row)
        self.logger.info("Epoch metrics: %s", dict(row))

    def close(self) -> None:
        self._handler.flush()
        self._handler.close()
        self.logger.removeHandler(self._handler)


__all__ = [
    "FIELDS",
    "METRICS_FIELDS",
    "MetricsCSVWriter",
    "RunLog",
    "RunPaths",
    "create_run_directory",
    "format_run_name",
    "format_tmux_session_name",
    "run_paths_from_directory",
    "write_json",
    "write_run_metadata",
    "write_yaml",
]
