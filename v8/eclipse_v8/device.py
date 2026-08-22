"""CUDA setup: match legacy eda notebooks (explicit device selection; refine later)."""

from __future__ import annotations

import os


def configure_cuda_visible_devices() -> None:
    """Set env before importing torch / running stages (same defaults as eda00.py)."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    # Which physical GPU: override with CUDA_VISIBLE_DEVICES in the shell if needed.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
    # seed_everything() needs this for reproducible GPU math, but it has to be set before
    # CUDA starts up, so it lives here instead of there.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this pipeline; no GPU is visible to PyTorch."
        )


def seed_everything(seed: int) -> None:
    """Make a run reproducible: same seed, same output every time.

    Fixes the random draws in stage0 and stage1 (moon-edge sampling, a debug-frame pick),
    and turns on PyTorch's deterministic mode so stage1's GPU optimizer gives identical
    numbers too. Call after configure_cuda_visible_devices(), before any stage runs.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
