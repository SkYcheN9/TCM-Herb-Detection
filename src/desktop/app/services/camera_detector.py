"""Camera detection worker for the desktop client."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ultralytics import YOLO

from .gpu_status import query_gpu_status, resolve_inference_device
from .history_store import HistoryStore
from .paths import ensure_desktop_dirs


class CameraDetectorThread(QThread):
    """Run camera capture and YOLO inference away from the UI thread."""

    frame_ready = Signal(QImage)
    stats_ready = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        model_path: Path,
        camera_index: int,
        conf: float,
        iou: float,
        imgsz: int,
        device_mode: str,
        save_record: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.model_path = model_path
        self.camera_index = camera_index
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device_mode = device_mode
        self.save_record = save_record
        self._running = False

    def stop(self) -> None:
        """Request the camera loop to stop."""
        self._running = False

    def run(self) -> None:
        self._running = True
        capture = None

        try:
            ensure_desktop_dirs()
            self.status_changed.emit("正在加载模型")
            model = YOLO(str(self.model_path))
            device = resolve_inference_device(self.device_mode)

            self.status_changed.emit("正在打开摄像头")
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            capture = cv2.VideoCapture(self.camera_index, backend)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            capture.set(cv2.CAP_PROP_FPS, 30)

            if not capture.isOpened():
                self.error_occurred.emit(f"无法打开摄像头 {self.camera_index}")
                return

            self.status_changed.emit("实时检测中")
            last_frame_at = time.perf_counter()
            smooth_fps = 0.0
            frame_index = 0
            gpu_text = query_gpu_status().text
            merged_counts: dict[str, int] = {}
            last_fps = 0.0

            while self._running:
                ok, frame = capture.read()
                if not ok:
                    self.error_occurred.emit("摄像头画面读取失败")
                    break

                result = model.predict(
                    source=frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=device,
                    verbose=False,
                )[0]

                annotated = result.plot()
                now = time.perf_counter()
                instant_fps = 1.0 / max(now - last_frame_at, 1e-6)
                smooth_fps = instant_fps if smooth_fps <= 0 else smooth_fps * 0.85 + instant_fps * 0.15
                last_frame_at = now

                counts = _count_classes(result)
                merged_counts = counts
                if frame_index % 15 == 0:
                    gpu_text = query_gpu_status().text
                last_fps = smooth_fps

                self.frame_ready.emit(_bgr_to_qimage(annotated))
                self.stats_ready.emit(
                    {
                        "fps": smooth_fps,
                        "counts": counts,
                        "total": sum(counts.values()),
                        "gpu": gpu_text,
                        "device": "CUDA:0" if device == 0 else "CPU",
                        "model": self.model_path.name,
                        "camera": self.camera_index,
                    }
                )
                frame_index += 1

            if self.save_record and frame_index > 0:
                HistoryStore().add_record(
                    mode="摄像头检测",
                    source_path=f"Camera {self.camera_index}",
                    output_path="",
                    model_path=str(self.model_path),
                    device="CUDA:0" if device == 0 else "CPU",
                    fps=last_fps,
                    total_count=sum(merged_counts.values()),
                    class_counts=merged_counts,
                    status="已停止",
                )

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            if capture is not None:
                capture.release()
            self.status_changed.emit("摄像头已停止")
            self._running = False


def _count_classes(result: Any) -> dict[str, int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return {}

    class_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    names = getattr(result, "names", {}) or {}

    counts: dict[str, int] = {}
    for class_id in class_ids:
        name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
        counts[name] = counts.get(name, 0) + 1

    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return image.copy()
