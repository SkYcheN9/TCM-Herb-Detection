"""Image and video detection services."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
from ultralytics import YOLO

from .camera_detector import _bgr_to_qimage, _count_classes
from .gpu_status import query_gpu_status, resolve_inference_device
from .history_store import HistoryStore
from .paths import IMAGE_OUTPUT_DIR, VIDEO_OUTPUT_DIR, ensure_desktop_dirs


@dataclass(frozen=True)
class DetectionSummary:
    """Detection result shown in the UI and persisted to history."""

    mode: str
    source_path: Path
    output_path: Path
    model_path: Path
    device: str
    fps: float
    total_count: int
    class_counts: dict[str, int]
    frame: QImage | None = None


def detect_image(
    *,
    image_path: Path,
    model_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device_mode: str,
    save_record: bool,
) -> DetectionSummary:
    """Run image detection, save the annotated image, and optionally write history."""
    ensure_desktop_dirs()
    model = YOLO(str(model_path))
    device = resolve_inference_device(device_mode)

    started = time.perf_counter()
    result = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    elapsed = max(time.perf_counter() - started, 1e-6)

    annotated = result.plot()
    output_path = IMAGE_OUTPUT_DIR / f"{image_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(output_path), annotated)

    counts = _count_classes(result)
    summary = DetectionSummary(
        mode="图片检测",
        source_path=image_path,
        output_path=output_path,
        model_path=model_path,
        device="CUDA:0" if device == 0 else "CPU",
        fps=1.0 / elapsed,
        total_count=sum(counts.values()),
        class_counts=counts,
        frame=_bgr_to_qimage(annotated),
    )

    if save_record:
        _save_summary(summary)

    return summary


class VideoDetectorThread(QThread):
    """Run video detection in the background."""

    frame_ready = Signal(QImage)
    progress_ready = Signal(object)
    finished_summary = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        video_path: Path,
        model_path: Path,
        conf: float,
        iou: float,
        imgsz: int,
        device_mode: str,
        save_record: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device_mode = device_mode
        self.save_record = save_record
        self._running = False

    def stop(self) -> None:
        """Request video detection to stop."""
        self._running = False

    def run(self) -> None:
        self._running = True
        capture = None
        writer = None

        try:
            ensure_desktop_dirs()
            self.status_changed.emit("正在加载模型")
            model = YOLO(str(self.model_path))
            device = resolve_inference_device(self.device_mode)

            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                self.error_occurred.emit("无法打开视频文件")
                return

            fps_source = capture.get(cv2.CAP_PROP_FPS) or 25
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            output_path = VIDEO_OUTPUT_DIR / f"{self.video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps_source, (width, height))

            self.status_changed.emit("视频检测中")
            started = time.perf_counter()
            frame_count = 0
            merged_counts: dict[str, int] = {}
            last_gpu_text = query_gpu_status().text
            stopped_by_user = False

            while self._running:
                ok, frame = capture.read()
                if not ok:
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
                if annotated.shape[1] != width or annotated.shape[0] != height:
                    annotated = cv2.resize(annotated, (width, height))
                writer.write(annotated)

                counts = _count_classes(result)
                for name, count in counts.items():
                    merged_counts[name] = merged_counts.get(name, 0) + count

                frame_count += 1
                elapsed = max(time.perf_counter() - started, 1e-6)
                current_fps = frame_count / elapsed
                if frame_count % 15 == 0:
                    last_gpu_text = query_gpu_status().text

                self.frame_ready.emit(_bgr_to_qimage(annotated))
                self.progress_ready.emit(
                    {
                        "fps": current_fps,
                        "frame": frame_count,
                        "total_frames": total_frames,
                        "total": sum(merged_counts.values()),
                        "counts": dict(sorted(merged_counts.items(), key=lambda item: (-item[1], item[0]))),
                        "gpu": last_gpu_text,
                        "output_path": str(output_path),
                    }
                )

            stopped_by_user = not self._running
            elapsed = max(time.perf_counter() - started, 1e-6)
            summary = DetectionSummary(
                mode="视频检测",
                source_path=self.video_path,
                output_path=output_path,
                model_path=self.model_path,
                device="CUDA:0" if device == 0 else "CPU",
                fps=frame_count / elapsed if frame_count else 0.0,
                total_count=sum(merged_counts.values()),
                class_counts=dict(sorted(merged_counts.items(), key=lambda item: (-item[1], item[0]))),
                frame=None,
            )

            if self.save_record:
                _save_summary(summary, status="已停止" if stopped_by_user else "完成")

            self.finished_summary.emit(summary)

        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            if capture is not None:
                capture.release()
            if writer is not None:
                writer.release()
            self.status_changed.emit("视频检测已停止")
            self._running = False


def _save_summary(summary: DetectionSummary, status: str = "完成") -> None:
    HistoryStore().add_record(
        mode=summary.mode,
        source_path=str(summary.source_path),
        output_path=str(summary.output_path),
        model_path=str(summary.model_path),
        device=summary.device,
        fps=summary.fps,
        total_count=summary.total_count,
        class_counts=summary.class_counts,
        status=status,
    )
