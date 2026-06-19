"""Database models for detection history."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class DetectionRecord(Base):
    """Persisted inference result."""

    __tablename__ = "detection_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now,
        index=True,
        nullable=False,
    )
    mode: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_path: Mapped[str] = mapped_column(Text, nullable=False)
    device: Mapped[str] = mapped_column(String(32), nullable=False)
    elapsed_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fps: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="completed")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    class_counts_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    detections_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
