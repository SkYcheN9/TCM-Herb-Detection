"""Camera device discovery for desktop detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraDevice:
    """Camera option shown in the Detection view."""

    index: int
    label: str


def discover_camera_devices(max_index: int = 4) -> list[CameraDevice]:
    """Return common local camera entries without touching hardware at startup."""
    return [
        CameraDevice(0, "Laptop Camera / Camera 0"),
        CameraDevice(1, "USB Camera / Camera 1"),
        *[CameraDevice(index, f"Camera {index}") for index in range(2, max_index)],
    ]
