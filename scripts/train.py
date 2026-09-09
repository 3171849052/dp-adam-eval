#!/usr/bin/env python
"""Prepare a run directory or train one configured MNIST experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dp_wiener_mnist.config import load_config  # noqa: E402


def _config_path(value: str | None) -> Path:
    if value is None:
        raise ValueError("a config path is required")
    path = Path(value)
    return (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()


def _set_configured_gpu(c) -> None:
    # This happens before importing torch in the training path.  A CPU run is
    # intentionally allowed to keep running even when no CUDA device exists.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(c.runtime.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _initial_resolved_config(c, paths, device) -> dict:
    """Create useful metadata before data loading or noise calibration."""
    resolved = c.to_dict()
    train_size = c.data.train_subset or (60_000 if c.data.dataset == "mnist" else None)
    test_size = c.data.test_subset or (10_000 if c.data.dataset == "mnist" else None)
    steps = train_size // c.data.batch_size if train_size is not None else None
    total = steps * c.training.epochs if steps is not None else None
    sample_rate = c.data.batch_size / train_size if train_size else None
    gpu_name = None
    if device.type == "cuda":
        import torch

        try:
            gpu_name = torch.cuda.get_device_name(0)
        except (AssertionError, IndexError, RuntimeError):
            gpu_name = None
    resolved["data"].update(train_size=train_size, test_size=test_size)
    resolved["privacy"].update(
        sample_rate=sample_rate,
        noise_multiplier=None,
        expected_batch_size=c.data.batch_size,
        steps_per_epoch=steps,
        total_steps=total,
        privacy_steps=0,
        epsilon_spent=0.0,
    )
    resolved["runtime"].update(
        actual_device=str(device),
        physical_gpu_index=c.runtime.gpu,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_name=gpu_name,
    )
    resolved["rng"] = dict(
        init_seed=c.seed,
        train_sampler_seed=c.seed + 1,
        test_seed=c.seed + 2,
        synthetic_seed=c.seed + 3,
        noise_seed=c.seed + 4,
    )
    resolved["run"] = {"directory": str(paths.directory.resolve())}
    resolved["wiener"]["enabled"] = c.algorithm == "dp_fisher_wiener"
    return resolved


def prepare_run(config_file: str | Path):
    """Create all initial run files without starting training."""
    config_path = Path(config_file)
    c = load_config(config_path)
    _set_configured_gpu(c)
    from dp_wiener_mnist.run_logging import (
        MetricsCSVWriter,
        create_run_directory,
        write_run_metadata,
    )
    from dp_wiener_mnist.utils import resolve_device

    device = resolve_device(c.runtime.device)
    paths = create_run_directory(c)
    write_run_metadata(
        paths,
        source_yaml=config_path.read_text(encoding="utf-8"),
        resolved_config=_initial_resolved_config(c, paths, device),
    )
    MetricsCSVWriter(paths.metrics)
    return paths


def _validate_gpu(c) -> None:
    """Validate only when the configured device mode needs CUDA."""
    if c.runtime.device == "cpu":
        return
    import torch

    if c.runtime.device == "auto" and not torch.cuda.is_available():
        return
    from dp_wiener_mnist.utils import validate_gpu_selection

    validate_gpu_selection(c.runtime.gpu)


def _validate_gpu_unmasked(c) -> None:
    """Check physical indices against all GPUs, like the shell launcher."""
    old = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        _validate_gpu(c)
    finally:
        if old is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def run_experiment(config_file: str | Path, run_dir: str | Path | None = None) -> int:
    config_path = Path(config_file)
    c = load_config(config_path)
    _set_configured_gpu(c)
    from dp_wiener_mnist.run_logging import RunLog, run_paths_from_directory
    from dp_wiener_mnist.trainer import train

    if run_dir is None:
        log = RunLog(c, config_path.read_text(encoding="utf-8"))
    else:
        paths = run_paths_from_directory(run_dir)
        log = RunLog(paths=paths)
    print(f"run_directory={log.root.resolve()}", flush=True)
    try:
        result = train(c, log)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["status"] == "completed" else 1
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="YAML configuration path")
    parser.add_argument("--config", dest="config_option", help="YAML configuration path")
    parser.add_argument(
        "--prepare-run", action="store_true", help="create metadata without training"
    )
    parser.add_argument("--run-dir", type=Path, help="use an existing prepared directory")
    parser.add_argument(
        "--tmux-session-name", type=Path, metavar="RUN_DIR", help="print a tmux-safe name"
    )
    parser.add_argument(
        "--print-gpu", action="store_true", help="print the configured physical GPU index"
    )
    parser.add_argument(
        "--validate-gpu", action="store_true", help="validate the configured GPU selection"
    )
    args = parser.parse_args()

    if args.config is not None and args.config_option is not None:
        parser.error("provide the config path either positionally or with --config")
    if args.prepare_run and args.run_dir is not None:
        parser.error("--prepare-run and --run-dir are mutually exclusive")
    if args.print_gpu and args.validate_gpu:
        parser.error("--print-gpu and --validate-gpu are mutually exclusive")
    if (args.prepare_run or args.run_dir is not None) and (
        args.print_gpu or args.validate_gpu
    ):
        parser.error("GPU inspection cannot be combined with a run mode")
    if args.tmux_session_name is not None:
        if args.prepare_run or args.run_dir is not None or args.print_gpu or args.validate_gpu:
            parser.error("--tmux-session-name cannot be combined with another mode")
        from dp_wiener_mnist.run_logging import format_tmux_session_name

        print(format_tmux_session_name(args.tmux_session_name))
        return 0

    try:
        config_path = _config_path(args.config_option or args.config)
        c = load_config(config_path)
        if args.print_gpu:
            # This mode intentionally performs no torch import or CUDA query.
            print(c.runtime.gpu)
            return 0
        if args.validate_gpu:
            _validate_gpu_unmasked(c)
            print(c.runtime.gpu)
            return 0
        if args.prepare_run:
            print(prepare_run(config_path).directory.resolve())
            return 0
        if args.run_dir is not None:
            return run_experiment(config_path, args.run_dir)
        return run_experiment(config_path)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
