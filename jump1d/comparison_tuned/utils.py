"""
utils.py
========
Small shared helpers used across the comparison benchmark: device selection,
seeding, timing / GPU-memory instrumentation, an EMA weight tracker, and
checkpoint I/O.
"""

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch

from config import Config


# ---------------------------------------------------------------------------
# Device / seeding
# ---------------------------------------------------------------------------
def get_device(cfg: Config) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_enabled(cfg: Config, device: torch.device) -> bool:
    return bool(cfg.amp) and device.type == "cuda"


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
class Timer:
    """``with Timer() as t: ...`` -> ``t.elapsed`` in seconds."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed = None
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False


class GpuMemoryTracker:
    """``with GpuMemoryTracker(device) as m: ...`` -> ``m.peak_bytes``.

    Resets CUDA's peak-memory counter on entry and records the peak
    allocation observed during the block. Reports 0 on CPU.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.peak_bytes = 0

    def __enter__(self) -> "GpuMemoryTracker":
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *exc):
        if self.device.type == "cuda":
            self.peak_bytes = torch.cuda.max_memory_allocated(self.device)
        return False


@contextmanager
def timed_gpu_block(device: torch.device):
    """Convenience combinator: yields a dict filled in with 'time_s' and
    'peak_gpu_mem_bytes' once the `with` block exits."""
    result: Dict[str, float] = {}
    timer = Timer()
    mem = GpuMemoryTracker(device)
    with timer, mem:
        yield result
    result["time_s"] = timer.elapsed
    result["peak_gpu_mem_bytes"] = mem.peak_bytes


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    """Exponential moving average of a model's parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict:
        return self.shadow


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, **kwargs) -> None:
    torch.save(kwargs, path)


def load_checkpoint(path: str, map_location: Optional[torch.device] = None) -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
