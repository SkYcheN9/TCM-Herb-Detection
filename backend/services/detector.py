"""YOLO image detection service."""

from __future__ import annotations

import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import cv2
from fastapi import UploadFile
from sqlalchemy.orm import Session
from ultralytics import YOLO

from ..config import get_settings
from ..constants import CHINESE_CLASS_NAMES, CLASS_NAMES, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from ..schemas import BBox, DetectApiResponse, DetectImageResponse, DetectionApiItem, DetectionItem
from .device import display_device, resolve_device
from .history import create_detection_record, record_to_schema
from .model_locator import resolve_model_path


def validate_image_file(filename: str) -> None:
    """Reject unsupported image file extensions."""

    suffix = Path(filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image type '{suffix}'. Allowed: {allowed}")


def validate_video_file(filename: str) -> None:
    """Reject unsupported video file extensions."""

    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video type '{suffix}'. Allowed: {allowed}")


async def save_upload_file(upload: UploadFile) -> Path:
    """Save an uploaded image to backend storage."""

    validate_image_file(upload.filename or "")
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "image.jpg").suffix.lower()
    path = settings.upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}{suffix}"
    with path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return path


async def save_video_upload_file(upload: UploadFile) -> Path:
    """Save an uploaded video to backend storage."""

    validate_video_file(upload.filename or "")
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename or "video.mp4").suffix.lower()
    path = settings.upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}{suffix}"
    with path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return path


def detect_image_file(
    db: Session,
    *,
    image_path: Path,
    model_path: str | None,
    confidence: float,
    iou: float,
    image_size: int,
    device_mode: str,
    save_annotated: bool,
    persist: bool,
) -> DetectImageResponse:
    """Run YOLO detection for one image and optionally persist the result."""

    validate_image_file(str(image_path))
    resolved_model = resolve_model_path(model_path)
    device_arg = resolve_device(device_mode)
    device_label = display_device(device_arg)

    model = YOLO(str(resolved_model))
    started = time.perf_counter()
    result = model.predict(
        source=str(image_path),
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        device=device_arg,
        verbose=False,
    )[0]
    elapsed_ms = max((time.perf_counter() - started) * 1000.0, 0.001)

    detections = _extract_detections(result)
    class_counts = dict(Counter(item["class_name"] for item in detections))
    class_counts = dict(sorted(class_counts.items(), key=lambda item: (-item[1], item[0])))
    output_path = _save_annotated_image(result, image_path) if save_annotated else None

    if persist:
        record = create_detection_record(
            db,
            mode="image",
            source_type="upload",
            source_path=str(image_path),
            output_path=str(output_path) if output_path else None,
            model_path=str(resolved_model),
            device=device_label,
            elapsed_ms=elapsed_ms,
            fps=1000.0 / elapsed_ms,
            total_count=len(detections),
            class_counts=class_counts,
            detections=detections,
        )
        history = record_to_schema(record)
    else:
        history = _transient_response(
            image_path=image_path,
            output_path=output_path,
            model_path=resolved_model,
            device=device_label,
            elapsed_ms=elapsed_ms,
            class_counts=class_counts,
            detections=detections,
        )

    return DetectImageResponse(
        record_id=history.id,
        mode=history.mode,
        source_type=history.source_type,
        source_path=history.source_path,
        output_path=history.output_path,
        model_path=history.model_path,
        device=history.device,
        elapsed_ms=history.elapsed_ms,
        fps=history.fps,
        total_count=history.total_count,
        class_counts=history.class_counts,
        detections=history.detections,
        created_at=history.created_at,
        status=history.status,
    )


def detect_video_file(
    db: Session,
    *,
    video_path: Path,
    model_path: str | None,
    confidence: float,
    iou: float,
    image_size: int,
    device_mode: str,
    frame_stride: int,
    max_frames: int | None,
    save_annotated: bool,
) -> DetectApiResponse:
    """Run YOLO detection for one video and return bbox/class/confidence/count."""

    validate_video_file(str(video_path))
    if frame_stride < 1:
        raise ValueError("frame_stride must be greater than or equal to 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be greater than or equal to 1")

    resolved_model = resolve_model_path(model_path)
    device_arg = resolve_device(device_mode)
    device_label = display_device(device_arg)
    model = YOLO(str(resolved_model))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    writer = None
    output_path: Path | None = None
    started = time.perf_counter()
    processed_frames = 0
    all_detections: list[dict[str, object]] = []

    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if save_annotated:
            settings = get_settings()
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = settings.output_dir / (
                f"{video_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (width, height))

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            should_process = frame_index % frame_stride == 0
            can_process_more = max_frames is None or processed_frames < max_frames
            if should_process and can_process_more:
                result = model.predict(
                    source=frame,
                    conf=confidence,
                    iou=iou,
                    imgsz=image_size,
                    device=device_arg,
                    verbose=False,
                )[0]
                timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                frame_detections = _extract_detections(result)
                for item in frame_detections:
                    item["frame_index"] = frame_index
                    item["timestamp_ms"] = timestamp_ms
                all_detections.extend(frame_detections)
                processed_frames += 1

                if writer is not None:
                    annotated = result.plot()
                    if width and height and (annotated.shape[1] != width or annotated.shape[0] != height):
                        annotated = cv2.resize(annotated, (width, height))
                    writer.write(annotated)
            elif writer is not None:
                writer.write(frame)

            frame_index += 1
            if max_frames is not None and processed_frames >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed_ms = max((time.perf_counter() - started) * 1000.0, 0.001)
    class_counts = dict(Counter(str(item["class_name"]) for item in all_detections))
    class_counts = dict(sorted(class_counts.items(), key=lambda item: (-item[1], item[0])))

    create_detection_record(
        db,
        mode="video",
        source_type="upload",
        source_path=str(video_path),
        output_path=str(output_path) if output_path else None,
        model_path=str(resolved_model),
        device=device_label,
        elapsed_ms=elapsed_ms,
        fps=(processed_frames * 1000.0) / elapsed_ms if processed_frames else 0.0,
        total_count=len(all_detections),
        class_counts=class_counts,
        detections=all_detections,
    )

    return DetectApiResponse(
        count=len(all_detections),
        detections=[
            DetectionApiItem(
                bbox=BBox(**item["bbox"]),
                class_=str(item["class_name"]),
                confidence=float(item["confidence"]),
            )
            for item in all_detections
        ],
    )


def _extract_detections(result: object) -> list[dict[str, object]]:
    """Extract structured detections from an Ultralytics result."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    confidences = boxes.conf.detach().cpu().numpy().tolist()
    names = getattr(result, "names", {}) or {}

    detections: list[dict[str, object]] = []
    for bbox, class_id, confidence in zip(xyxy, class_ids, confidences, strict=False):
        class_name = _class_name(class_id, names)
        detections.append(
            {
                "bbox": {
                    "x1": float(bbox[0]),
                    "y1": float(bbox[1]),
                    "x2": float(bbox[2]),
                    "y2": float(bbox[3]),
                },
                "class_id": int(class_id),
                "class_name": class_name,
                "chinese_name": CHINESE_CLASS_NAMES.get(class_name, class_name),
                "confidence": float(confidence),
            }
        )
    return detections


def _class_name(class_id: int, names: object) -> str:
    """Resolve class name from model metadata or fixed project classes."""

    if isinstance(names, dict) and class_id in names:
        return str(names[class_id])
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return str(class_id)


def _save_annotated_image(result: object, image_path: Path) -> Path:
    """Save annotated detection image and return its path."""

    settings = get_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / (
        f"{image_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.jpg"
    )
    annotated = result.plot()
    cv2.imwrite(str(output_path), annotated)
    return output_path


def _transient_response(
    *,
    image_path: Path,
    output_path: Path | None,
    model_path: Path,
    device: str,
    elapsed_ms: float,
    class_counts: dict[str, int],
    detections: list[dict[str, object]],
) -> object:
    """Build a response-shaped object for non-persisted calls."""

    return SimpleNamespace(
        id=0,
        created_at=datetime.now(),
        mode="image",
        source_type="upload",
        source_path=str(image_path),
        output_path=str(output_path) if output_path else None,
        model_path=str(model_path),
        device=device,
        elapsed_ms=elapsed_ms,
        fps=1000.0 / elapsed_ms,
        total_count=len(detections),
        class_counts=class_counts,
        detections=[
            DetectionItem(
                bbox=BBox(**item["bbox"]),
                class_id=int(item["class_id"]),
                class_name=str(item["class_name"]),
                chinese_name=str(item["chinese_name"]),
                confidence=float(item["confidence"]),
            )
            for item in detections
        ],
        status="transient",
    )
