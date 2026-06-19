"""Statistics aggregation service."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..constants import CHINESE_CLASS_NAMES, CLASS_NAMES
from ..models import DetectionRecord
from ..schemas import ClassDistributionItem, StatisticsResponse, SummaryStatistics, TrendItem
from .history import load_json


def build_statistics(db: Session, days: int = 30) -> StatisticsResponse:
    """Build dashboard statistics from persisted records."""

    since = datetime.now() - timedelta(days=days)
    records = db.scalars(
        select(DetectionRecord).where(DetectionRecord.created_at >= since)
    ).all()

    total_records = len(records)
    successful_records = sum(1 for record in records if bool(record.success))
    failed_records = total_records - successful_records
    total_objects = sum(record.total_count for record in records)
    avg_fps = _average(record.fps for record in records if record.fps > 0)
    avg_elapsed_ms = _average(record.elapsed_ms for record in records if record.elapsed_ms > 0)
    latest_detection_at = max((record.created_at for record in records), default=None)

    class_counter: Counter[str] = Counter()
    trend_records: defaultdict[str, int] = defaultdict(int)
    trend_objects: defaultdict[str, int] = defaultdict(int)
    for record in records:
        for class_name, count in load_json(record.class_counts_json, {}).items():
            class_counter[str(class_name)] += int(count)
        key = record.created_at.strftime("%Y-%m-%d")
        trend_records[key] += 1
        trend_objects[key] += record.total_count

    top_class = class_counter.most_common(1)[0][0] if class_counter else None
    class_distribution = _class_distribution(class_counter, total_objects)
    trends = [
        TrendItem(
            date=day,
            record_count=trend_records.get(day, 0),
            object_count=trend_objects.get(day, 0),
        )
        for day in _date_range(days)
    ]

    return StatisticsResponse(
        summary=SummaryStatistics(
            total_records=total_records,
            successful_records=successful_records,
            failed_records=failed_records,
            total_objects=total_objects,
            avg_fps=round(avg_fps, 2),
            avg_elapsed_ms=round(avg_elapsed_ms, 2),
            top_class=top_class,
            latest_detection_at=latest_detection_at,
        ),
        class_distribution=class_distribution,
        detection_trend=trends,
    )


def total_records(db: Session) -> int:
    """Return total number of stored detection records."""

    return int(db.scalar(select(func.count()).select_from(DetectionRecord)) or 0)


def _class_distribution(counter: Counter[str], total_objects: int) -> list[ClassDistributionItem]:
    items: list[ClassDistributionItem] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = int(counter.get(class_name, 0))
        ratio = (count / total_objects) if total_objects else 0.0
        items.append(
            ClassDistributionItem(
                class_id=class_id,
                class_name=class_name,
                chinese_name=CHINESE_CLASS_NAMES.get(class_name, class_name),
                count=count,
                ratio=round(ratio, 4),
            )
        )
    return sorted(items, key=lambda item: (-item.count, item.class_id))


def _date_range(days: int) -> list[str]:
    start = datetime.now().date() - timedelta(days=days - 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(days)]


def _average(values: object) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return float(sum(collected) / len(collected))
