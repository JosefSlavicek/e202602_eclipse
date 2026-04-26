"""CUDA setup: match legacy eda notebooks (explicit device selection; refine later)."""

from __future__ import annotations

import os


def configure_cuda_visible_devices() -> None:
    """Set env before importing torch / running stages (same defaults as eda00.py)."""
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    # Which physical GPU: override with CUDA_VISIBLE_DEVICES in the shell if needed.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this pipeline; no GPU is visible to PyTorch."
        )
