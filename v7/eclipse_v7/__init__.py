"""Eclipse stacking pipeline (v7 library). Import submodules from notebooks or scripts."""

from eclipse_v7.device import configure_cuda_visible_devices, require_cuda

__all__ = ["configure_cuda_visible_devices", "require_cuda"]
