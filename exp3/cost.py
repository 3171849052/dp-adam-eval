"""Segment CUDA allocated peaks so diagnostic temporaries never enter core peak."""
from contextlib import contextmanager
import gc
import time
import torch


def core_wall_time(wall_time, diagnostic_seconds):
    if not 0 <= diagnostic_seconds <= wall_time:
        raise ValueError("Invalid diagnostic timing")
    return wall_time - diagnostic_seconds


class CostTracker:
    def __init__(self, dev):
        self.dev = dev
        self.core_peak = self.overall_peak = self.reserved_peak = 0
        self.diagnostic_seconds = 0.
        self.sync()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        self.started = time.perf_counter()

    def sync(self):
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    def capture(self, core):
        self.sync()
        if self.dev.type == "cuda":
            peak = torch.cuda.max_memory_allocated(self.dev)
            self.overall_peak = max(self.overall_peak, peak)
            self.reserved_peak = max(self.reserved_peak, torch.cuda.max_memory_reserved(self.dev))
            if core:
                self.core_peak = max(self.core_peak, peak)
            torch.cuda.reset_peak_memory_stats(self.dev)

    @contextmanager
    def diagnostics(self):
        self.capture(core=True)
        started = time.perf_counter()
        try:
            yield
        finally:
            # Separate model/hook cycles must be released before the next core
            # segment. Callers also delete probe tensors before leaving here.
            gc.collect()
            self.capture(core=False)
            self.diagnostic_seconds += time.perf_counter() - started

    def finish(self):
        self.capture(core=True)
        wall = time.perf_counter()-self.started
        return dict(wall_time=wall, diagnostic_seconds=self.diagnostic_seconds,
                    core_wall_time=core_wall_time(wall, self.diagnostic_seconds),
                    peak_cuda_memory_overall=self.overall_peak, peak_cuda_memory_core=self.core_peak,
                    peak_cuda_memory_allocated=self.overall_peak, peak_cuda_memory_reserved=self.reserved_peak)
