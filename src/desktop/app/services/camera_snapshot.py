"""Capture one still frame from a camera for desktop image detection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from .camera_devices import open_camera_capture
from .paths import SNAPSHOT_OUTPUT_DIR, ensure_desktop_dirs


def capture_camera_snapshot(camera_index: int, camera_backend: int | None = None) -> Path:
    """Capture one frame from the selected camera and save it as an image."""

    ensure_desktop_dirs()
    opened = open_camera_capture(camera_index, camera_backend)
    if opened is None:
        raise RuntimeError(
            f"无法打开摄像头 {camera_index}。请确认摄像头未被其他软件占用，"
            "或点击刷新摄像头后重试。"
        )

    capture, _, _ = opened
    try:
        frame = None
        for _ in range(5):
            ok, current_frame = capture.read()
            if ok and current_frame is not None:
                frame = current_frame
        if frame is None:
            raise RuntimeError("摄像头已打开，但没有读取到可用画面。")

        path = SNAPSHOT_OUTPUT_DIR / f"camera_{camera_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"无法保存摄像头快照：{path}")
        return path
    finally:
        capture.release()
