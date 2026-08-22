"""CUDA setup: match legacy eda notebooks (explicit device selection; refine later)."""

from __future__ import annotations

import os


def configure_cuda_visible_devices() -> None:
    """Set env before importing torch / running stages (same defaults as eda00.py)."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    # Which physical GPU: override with CUDA_VISIBLE_DEVICES in the shell if needed.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
    # Required by torch.use_deterministic_algorithms(True) for deterministic cuBLAS; must be
    # set before CUDA initializes, so alongside CUDA_VISIBLE_DEVICES rather than in seed_everything.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this pipeline; no GPU is visible to PyTorch."
        )


def seed_everything(seed: int) -> None:
    """Seed every RNG the pipeline draws from and force deterministic GPU kernels.

    Covers stage0's moon-edge triplet sampling and stage1's debug-frame pick (both draw
    from the global `random` module, unseeded otherwise) and stage1's Adam pose fit, whose
    backward pass through indexed gather/scatter is only bitwise-reproducible under
    use_deterministic_algorithms. Call after configure_cuda_visible_devices() (which sets
    CUBLAS_WORKSPACE_CONFIG) and before any stage runs.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
