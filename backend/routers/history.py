"""Detection history endpoints."""

from __future__ import annotations

from datetime import datetime, time

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import DetectionRecord
from ..schemas import DeleteResponse, HistoryListResponse, HistoryRecord
from ..services.history import record_to_schema
from ..services.history_exporter import export_history_to_excel


router = APIRouter(prefix="/history", tags=["history"])


@router.get(
    "",
    response_model=HistoryListResponse,
    summary="List detection history",
)
def list_history(
    page: int = Query(default=1, ge=1, description="Page number, starting from 1"),
    page_size: int = Query(default=20, ge=1, le=500, description="Rows per page"),
    keyword: str | None = Query(default=None, description="Search source path, model path, status or class counts"),
    mode: str | None = Query(default=None, description="Filter by mode, for example image or video"),
    source_type: str | None = Query(default=None, description="Filter by source type"),
    success: bool | None = Query(default=None, description="Filter by success status"),
    status_text: str | None = Query(default=None, alias="status", description="Filter by record status"),
    start_date: datetime | None = Query(default=None, description="Start datetime"),
    end_date: datetime | None = Query(default=None, description="End datetime"),
    db: Session = Depends(get_db),
) -> HistoryListResponse:
    """Return paginated and searchable detection history records."""

    filters = _build_filters(
        keyword=keyword,
        mode=mode,
        source_type=source_type,
        success=success,
        status_text=status_text,
        start_date=start_date,
        end_date=end_date,
    )
    statement = select(DetectionRecord).where(*filters)
    total = int(db.scalar(select(func.count()).select_from(DetectionRecord).where(*filters)) or 0)
    offset = (page - 1) * page_size
    rows = db.scalars(
        statement.order_by(DetectionRecord.created_at.desc(), DetectionRecord.id.desc())
        .offset(offset)
        .limit(page_size)
    ).all()

    return HistoryListResponse(
        total=total,
        limit=page_size,
        offset=offset,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size if total else 0,
        items=[record_to_schema(record) for record in rows],
    )


@router.get(
    "/export",
    response_class=FileResponse,
    summary="Export detection history to Excel",
)
def export_history(
    keyword: str | None = Query(default=None, description="Search source path, model path, status or class counts"),
    mode: str | None = Query(default=None, description="Filter by mode, for example image or video"),
    source_type: str | None = Query(default=None, description="Filter by source type"),
    success: bool | None = Query(default=None, description="Filter by success status"),
    status_text: str | None = Query(default=None, alias="status", description="Filter by record status"),
    start_date: datetime | None = Query(default=None, description="Start datetime"),
    end_date: datetime | None = Query(default=None, description="End datetime"),
    db: Session = Depends(get_db),
) -> FileResponse:
    """Export filtered detection history records as an Excel file."""

    filters = _build_filters(
        keyword=keyword,
        mode=mode,
        source_type=source_type,
        success=success,
        status_text=status_text,
        start_date=start_date,
        end_date=end_date,
    )
    records = db.scalars(
        select(DetectionRecord)
        .where(*filters)
        .order_by(DetectionRecord.created_at.desc(), DetectionRecord.id.desc())
    ).all()
    path = export_history_to_excel(list(records))
    return FileResponse(
        path=path,
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@router.get(
    "/{record_id}",
    response_model=HistoryRecord,
    summary="Get detection history detail",
)
def get_history(record_id: int, db: Session = Depends(get_db)) -> HistoryRecord:
    """Return one detection history item."""

    record = db.get(DetectionRecord, record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
    return record_to_schema(record)


@router.delete(
    "/{record_id}",
    response_model=DeleteResponse,
    summary="Delete one detection history record",
)
def delete_history(record_id: int, db: Session = Depends(get_db)) -> DeleteResponse:
    """Delete one detection history item."""

    record = db.get(DetectionRecord, record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found")
    db.delete(record)
    db.commit()
    return DeleteResponse(deleted=1)


@router.delete(
    "",
    response_model=DeleteResponse,
    summary="Clear detection history",
)
def clear_history(db: Session = Depends(get_db)) -> DeleteResponse:
    """Delete all detection history records."""

    result = db.execute(delete(DetectionRecord))
    db.commit()
    return DeleteResponse(deleted=int(result.rowcount or 0))


def _build_filters(
    *,
    keyword: str | None,
    mode: str | None,
    source_type: str | None,
    success: bool | None,
    status_text: str | None,
    start_date: datetime | None,
    end_date: datetime | None,
) -> list[object]:
    """Build SQLAlchemy filters shared by list and export."""

    filters: list[object] = []
    if keyword:
        pattern = f"%{keyword.strip()}%"
        filters.append(
            or_(
                DetectionRecord.source_path.like(pattern),
                DetectionRecord.output_path.like(pattern),
                DetectionRecord.model_path.like(pattern),
                DetectionRecord.device.like(pattern),
                DetectionRecord.status.like(pattern),
                DetectionRecord.error_message.like(pattern),
                DetectionRecord.class_counts_json.like(pattern),
            )
        )
    if mode:
        filters.append(DetectionRecord.mode == mode)
    if source_type:
        filters.append(DetectionRecord.source_type == source_type)
    if success is not None:
        filters.append(DetectionRecord.success == success)
    if status_text:
        filters.append(DetectionRecord.status == status_text)
    if start_date:
        filters.append(DetectionRecord.created_at >= start_date)
    if end_date:
        if end_date.time() == time.min:
            end_date = datetime.combine(end_date.date(), time.max)
        filters.append(DetectionRecord.created_at <= end_date)
    return filters
