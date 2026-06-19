"""Inference device helpers."""

from __future__ import annotations


def resolve_device(mode: str = "auto") -> str | int:
    """Map user device preference to an Ultralytics device argument."""

    text = mode.lower()
    if "cpu" in text:
        return "cpu"
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available() and ("auto" in text or "cuda" in text or "gpu" in text):
        return 0
    return "cpu"


def display_device(device: str | int) -> str:
    """Return a display-friendly device label."""

    return "CUDA:0" if device == 0 else "CPU"

