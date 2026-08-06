"""Eclipse stacking pipeline (v5 library). Import submodules from notebooks or scripts."""

from eclipse_v5.device import configure_cuda_visible_devices, require_cuda

__all__ = ["configure_cuda_visible_devices", "require_cuda"]
