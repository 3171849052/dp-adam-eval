"""ExpV1c: isolated upper-bound learning-rate sweep."""
import sys
import os
from pathlib import Path
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "runs" / "_cache" / "matplotlib"))
