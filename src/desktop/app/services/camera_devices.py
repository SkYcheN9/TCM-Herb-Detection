"""Camera device discovery for desktop detection."""

from __future__ import annotations

from dataclasses import dataclass
import sys

import cv2


@dataclass(frozen=True)
class CameraDevice:
    """Camera option shown in the Detection view."""

    index: int
    label: str
    backend: int | None = None
    backend_label: str = "Auto"


WINDOWS_BACKENDS: tuple[tuple[int, str], ...] = (
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_MSMF, "Media Foundation"),
    (cv2.CAP_ANY, "Auto"),
)
DEFAULT_BACKENDS: tuple[tuple[int, str], ...] = ((cv2.CAP_ANY, "Auto"),)


def discover_camera_devices(max_index: int = 8) -> list[CameraDevice]:
    """Probe local camera entries and return devices that can provide frames."""

    devices: list[CameraDevice] = []
    device_names = _qt_video_input_names()
    for index in range(max_index):
        opened = open_camera_capture(index)
        if opened is None:
            continue
        capture, backend, backend_label = opened
        capture.release()
        devices.append(
            CameraDevice(
                index=index,
                label=_camera_label(index, backend_label, device_names[index] if index < len(device_names) else None),
                backend=backend,
                backend_label=backend_label,
            )
        )
    return devices


def open_camera_capture(index: int, preferred_backend: int | None = None) -> tuple[cv2.VideoCapture, int, str] | None:
    """Open a camera index using the preferred backend and compatible fallbacks."""

    backends = _camera_backends(preferred_backend)
    for backend, backend_label in backends:
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            continue

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, 30)
        ok, _ = capture.read()
        if ok:
            return capture, backend, backend_label
        capture.release()
    return None


def _camera_backends(preferred_backend: int | None = None) -> list[tuple[int, str]]:
    base = list(WINDOWS_BACKENDS if sys.platform.startswith("win") else DEFAULT_BACKENDS)
    if preferred_backend is None:
        return base

    preferred_label = _backend_label(preferred_backend)
    ordered = [(preferred_backend, preferred_label)]
    ordered.extend((backend, label) for backend, label in base if backend != preferred_backend)
    return ordered


def _backend_label(backend: int) -> str:
    for current_backend, label in WINDOWS_BACKENDS + DEFAULT_BACKENDS:
        if current_backend == backend:
            return label
    return "Auto"


def _qt_video_input_names() -> list[str]:
    try:
        from PySide6.QtMultimedia import QMediaDevices
    except Exception:
        return []

    try:
        return [device.description() for device in QMediaDevices.videoInputs() if device.description()]
    except Exception:
        return []


def _camera_label(index: int, backend_label: str, device_name: str | None = None) -> str:
    if device_name:
        return f"{device_name} / Camera {index} ({backend_label})"
    if index == 0:
        return f"Laptop Camera / Camera 0 ({backend_label})"
    if index == 1:
        return f"USB/Virtual Camera / Camera 1 ({backend_label})"
    return f"Camera {index} ({backend_label})"
