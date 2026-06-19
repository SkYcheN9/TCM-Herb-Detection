"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ensure_runtime_dirs, get_settings
from .constants import CHINESE_CLASS_NAMES, CLASS_NAMES
from .database import init_db
from .routers import detect, history, statistics
from .schemas import ClassInfo, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize runtime directories and database tables."""

    ensure_runtime_dirs()
    init_db()
    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "TCM-SliceAI 中医药饮片智能检测后端，提供 detect、history、statistics "
        "接口，并自动生成 Swagger 文档。"
    ),
    docs_url=settings.docs_url,
    redoc_url=settings.redoc_url,
    openapi_url=settings.openapi_url,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(history.router)
app.include_router(statistics.router)


@app.get("/", response_model=HealthResponse, tags=["system"], summary="服务健康检查")
def health() -> HealthResponse:
    """Return API health status."""

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        docs_url=settings.docs_url,
    )


@app.get("/classes", response_model=list[ClassInfo], tags=["system"], summary="获取类别列表")
def classes() -> list[ClassInfo]:
    """Return the fixed 15-class order."""

    return [
        ClassInfo(
            id=index,
            name=name,
            chinese_name=CHINESE_CLASS_NAMES.get(name, name),
        )
        for index, name in enumerate(CLASS_NAMES)
    ]

