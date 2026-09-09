#!/usr/bin/env python
"""Launch one installed dp_wiener_mnist experiment."""
import argparse
import os
from pathlib import Path
import traceback
from dp_wiener_mnist.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--config", dest="config_option", type=Path)
    args = parser.parse_args()
    if bool(args.config) == bool(args.config_option):
        parser.error("Provide exactly one positional config or --config")
    path = args.config_option or args.config
    c = load_config(path)
    # Must precede importing torch or querying CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(c.runtime.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from dp_wiener_mnist.run_logging import RunLog
    from dp_wiener_mnist.trainer import train

    log = RunLog(c, path.read_text())
    print(f"run_directory={log.root.resolve()}", flush=True)
    try:
        result = train(c, log)
        print(result, flush=True)
        return 0 if result["status"] == "completed" else 1
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
