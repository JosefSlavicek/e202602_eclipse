"""Eclipse stacking pipeline (v1 library). Import submodules from notebooks or scripts."""

from eclipse_v1.device import configure_cuda_visible_devices, require_cuda

__all__ = ["configure_cuda_visible_devices", "require_cuda"]
