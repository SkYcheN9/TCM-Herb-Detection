"""GPU status helpers for desktop inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuStatus:
    """Display-ready GPU status."""

    available: bool
    device: str
    name: str
    memory: str

    @property
    def text(self) -> str:
        if not self.available:
            return "CPU 模式"
        return f"{self.device} · {self.name} · {self.memory}"


def query_gpu_status() -> GpuStatus:
    """Read CUDA status without requiring training code."""
    try:
        import torch
    except Exception:
        return GpuStatus(False, "CPU", "PyTorch 不可用", "-")

    if not torch.cuda.is_available():
        return GpuStatus(False, "CPU", "CUDA 不可用", "-")

    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    props = torch.cuda.get_device_properties(index)
    allocated = torch.cuda.memory_allocated(index) / 1024**3
    reserved = torch.cuda.memory_reserved(index) / 1024**3
    total = props.total_memory / 1024**3
    memory = f"{allocated:.1f}/{reserved:.1f}/{total:.1f} GB"
    return GpuStatus(True, f"CUDA:{index}", name, memory)


def resolve_inference_device(mode: str) -> str | int:
    """Map UI device text to an Ultralytics device argument."""
    text = mode.lower()
    if "cpu" in text:
        return "cpu"

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available() and ("cuda" in text or "自动" in mode or "auto" in text):
        return 0

    return "cpu"

