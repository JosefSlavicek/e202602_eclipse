"""Eclipse stacking pipeline (v6 library). Import submodules from notebooks or scripts."""

from eclipse_v6.device import configure_cuda_visible_devices, require_cuda

__all__ = ["configure_cuda_visible_devices", "require_cuda"]
