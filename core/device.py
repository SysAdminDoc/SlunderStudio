"""Resolve the configured local compute device for optional torch engines."""
from __future__ import annotations

from typing import Any


def configured_cuda_index(torch_module: Any) -> int:
    """Return a valid configured CUDA index, clamped to the installed devices."""
    try:
        count = int(torch_module.cuda.device_count())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return 0
    if count <= 0:
        return 0
    try:
        from core.settings import Settings

        requested = int(Settings().get("general.gpu_device", 0) or 0)
    except (TypeError, ValueError, RuntimeError):
        requested = 0
    return max(0, min(requested, count - 1))


def configured_torch_device(torch_module: Any | None = None) -> str:
    """Return ``cuda:<index>``, MPS, or CPU according to runtime and settings."""
    if torch_module is None:
        try:
            import torch as torch_module
        except (ImportError, OSError):
            torch_module = None
    if torch_module is not None:
        try:
            if torch_module.cuda.is_available():
                return f"cuda:{configured_cuda_index(torch_module)}"
        except (AttributeError, RuntimeError):
            pass
        try:
            if torch_module.backends.mps.is_available():
                return "mps"
        except (AttributeError, RuntimeError):
            pass
    return "cpu"
