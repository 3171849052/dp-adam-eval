#!/usr/bin/env python
"""Export only the standalone experiment into a new directory."""
import argparse
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.destination.resolve()
    if destination.exists():
        parser.error("Destination already exists; choose a new directory")
    destination.mkdir(parents=True)
    for name in ["src/dp_wiener_mnist", "config", "scripts", "tests", "docs"]:
        shutil.copytree(
            root / name,
            destination / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in [
        "README.md",
        "requirements.txt",
        "pyproject.toml",
        "run.sh",
        ".gitignore",
    ]:
        shutil.copy2(root / name, destination / name)
    (destination / "outputs").mkdir()
    (destination / "outputs/.gitkeep").touch()
    print(destination)


if __name__ == "__main__":
    main()
