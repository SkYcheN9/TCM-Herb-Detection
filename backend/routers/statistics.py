"""Statistics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import StatisticsResponse
from ..services.statistics import build_statistics


router = APIRouter(prefix="/statistics", tags=["statistics"])


@router.get(
    "",
    response_model=StatisticsResponse,
    summary="获取检测统计分析",
)
def get_statistics(
    days: int = Query(default=30, ge=1, le=365, description="统计最近 N 天"),
    db: Session = Depends(get_db),
) -> StatisticsResponse:
    """Return class distribution, detection counts and time trend."""

    return build_statistics(db, days=days)

