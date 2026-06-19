"""Pydantic schemas used by the API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    docs_url: str


class ClassInfo(BaseModel):
    """Single supported class label."""

    id: int
    name: str
    chinese_name: str


class BBox(BaseModel):
    """Detected bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionItem(BaseModel):
    """Single object detection result."""

    bbox: BBox
    class_id: int
    class_name: str
    chinese_name: str
    confidence: float


class DetectionApiItem(BaseModel):
    """Minimal detection item returned by detect APIs."""

    model_config = ConfigDict(populate_by_name=True)

    bbox: BBox
    class_: str = Field(alias="class")
    confidence: float


class DetectApiResponse(BaseModel):
    """Minimal image detection response."""

    count: int
    detections: list[DetectionApiItem]


class DetectImageResponse(BaseModel):
    """Image detection API response."""

    record_id: int
    mode: str
    source_type: str
    source_path: str
    output_path: str | None
    model_path: str
    device: str
    elapsed_ms: float
    fps: float
    total_count: int
    class_counts: dict[str, int]
    detections: list[DetectionItem]
    created_at: datetime
    status: str


class DetectBatchResponse(BaseModel):
    """Batch image detection response."""

    total_files: int
    success_count: int
    failed_count: int
    records: list[DetectImageResponse]
    errors: list[str] = Field(default_factory=list)


class HistoryRecord(BaseModel):
    """Detection history item."""

    id: int
    created_at: datetime
    mode: str
    source_type: str
    source_path: str
    output_path: str | None
    model_path: str
    device: str
    elapsed_ms: float
    fps: float
    total_count: int
    success: bool
    status: str
    error_message: str | None
    class_counts: dict[str, int]
    detections: list[DetectionItem]


class HistoryListResponse(BaseModel):
    """Paginated history list."""

    total: int
    limit: int
    offset: int
    page: int
    page_size: int
    pages: int
    items: list[HistoryRecord]


class DeleteResponse(BaseModel):
    """Deletion response."""

    deleted: int


class SummaryStatistics(BaseModel):
    """Global detection statistics."""

    total_records: int
    successful_records: int
    failed_records: int
    total_objects: int
    avg_fps: float
    avg_elapsed_ms: float
    top_class: str | None
    latest_detection_at: datetime | None


class ClassDistributionItem(BaseModel):
    """Aggregated object count for one class."""

    class_id: int
    class_name: str
    chinese_name: str
    count: int
    ratio: float


class TrendItem(BaseModel):
    """Time-bucketed detection trend item."""

    date: str
    record_count: int
    object_count: int


class StatisticsResponse(BaseModel):
    """Statistics API response."""

    summary: SummaryStatistics
    class_distribution: list[ClassDistributionItem]
    detection_trend: list[TrendItem]


class StatisticsQuery(BaseModel):
    """Statistics query options."""

    days: int = Field(default=30, ge=1, le=365)
    granularity: Literal["day"] = "day"
