"""History serialization and persistence helpers."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from ..models import DetectionRecord
from ..schemas import BBox, DetectionItem, HistoryRecord


def dump_json(value: Any) -> str:
    """Serialize a JSON-compatible value with Chinese text preserved."""

    return json.dumps(value, ensure_ascii=False)


def load_json(raw: str, fallback: Any) -> Any:
    """Load JSON text, returning fallback on invalid content."""

    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def create_detection_record(
    db: Session,
    *,
    mode: str,
    source_type: str,
    source_path: str,
    output_path: str | None,
    model_path: str,
    device: str,
    elapsed_ms: float,
    fps: float,
    total_count: int,
    class_counts: dict[str, int],
    detections: list[dict[str, object]],
    success: bool = True,
    status: str = "completed",
    error_message: str | None = None,
) -> DetectionRecord:
    """Persist one detection record."""

    record = DetectionRecord(
        created_at=datetime.now(),
        mode=mode,
        source_type=source_type,
        source_path=source_path,
        output_path=output_path,
        model_path=model_path,
        device=device,
        elapsed_ms=elapsed_ms,
        fps=fps,
        total_count=total_count,
        success=success,
        status=status,
        error_message=error_message,
        class_counts_json=dump_json(class_counts),
        detections_json=dump_json(detections),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def record_to_schema(record: DetectionRecord) -> HistoryRecord:
    """Convert an ORM record to an API schema."""

    return HistoryRecord(
        id=record.id,
        created_at=record.created_at,
        mode=record.mode,
        source_type=record.source_type,
        source_path=record.source_path,
        output_path=record.output_path,
        model_path=record.model_path,
        device=record.device,
        elapsed_ms=record.elapsed_ms,
        fps=record.fps,
        total_count=record.total_count,
        success=bool(record.success),
        status=record.status,
        error_message=record.error_message,
        class_counts={str(k): int(v) for k, v in load_json(record.class_counts_json, {}).items()},
        detections=[
            DetectionItem(
                bbox=BBox(**item["bbox"]),
                class_id=int(item["class_id"]),
                class_name=str(item["class_name"]),
                chinese_name=str(item["chinese_name"]),
                confidence=float(item["confidence"]),
            )
            for item in load_json(record.detections_json, [])
        ],
    )
