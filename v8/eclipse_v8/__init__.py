"""Eclipse stacking pipeline (v8 library). Import submodules from notebooks or scripts."""

from eclipse_v8.device import configure_cuda_visible_devices, require_cuda

__all__ = ["configure_cuda_visible_devices", "require_cuda"]
