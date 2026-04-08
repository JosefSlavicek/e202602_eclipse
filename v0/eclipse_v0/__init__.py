"""Eclipse stacking pipeline (v0 library). Import submodules from notebooks or scripts."""

from eclipse_v0.device import configure_cuda_visible_devices, require_cuda

__all__ = ["configure_cuda_visible_devices", "require_cuda"]
