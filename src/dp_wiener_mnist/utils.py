"""Source-compatible initialization and isolated CPU/CUDA RNG streams."""

import hashlib
from contextlib import contextmanager
import torch


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class RNGStream:
    def __init__(self, seed, dev):
        self.dev = torch.device(dev)
        self.cpu = torch.Generator().manual_seed(seed).get_state()
        self.cuda = (
            torch.Generator(device=self.dev).manual_seed(seed).get_state()
            if self.dev.type == "cuda"
            else None
        )

    def audit(self):
        return digest([self.cpu] + ([] if self.cuda is None else [self.cuda]))

    @contextmanager
    def use(self):
        devices = [self.dev.index or 0] if self.cuda is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self.cpu)
            if self.cuda is not None:
                torch.cuda.set_rng_state(self.cuda, self.dev)
            try:
                yield
            finally:
                self.cpu = torch.get_rng_state()
                if self.cuda is not None:
                    self.cuda = torch.cuda.get_rng_state(self.dev)


def set_seed(seed: int) -> None:
    import random
    import os
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_runtime(c) -> torch.device:
    """Configure deterministic FP32 execution after GPU visibility selection."""
    torch.set_num_threads(c.runtime.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    name = c.runtime.device
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        name = "cuda:0"
    return torch.device(name)


def require_finite(*tensors: torch.Tensor) -> None:
    """Fail at the first non-finite tensor without altering its values."""
    if any(not bool(torch.isfinite(t).all()) for t in tensors):
        raise FloatingPointError("Non-finite tensor")
