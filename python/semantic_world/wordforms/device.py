"""Choosing the PyTorch device from the configuration's ``device`` setting."""

from __future__ import annotations


def resolve_device(name: str = "auto") -> str:
    """``cpu``, ``cuda``, or ``mps``. ``auto`` picks CUDA, then Apple's MPS, then the CPU."""
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
