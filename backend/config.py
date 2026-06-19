"""Application configuration and filesystem paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = Path(__file__).resolve().parent


class Settings(BaseModel):
    """Runtime settings used by the API server."""

    app_name: str = "TCM-SliceAI Backend"
    app_version: str = "0.1.0"
    database_url: str = f"sqlite:///{(BACKEND_ROOT / 'data' / 'backend.db').as_posix()}"
    upload_dir: Path = BACKEND_ROOT / "data" / "uploads"
    output_dir: Path = BACKEND_ROOT / "data" / "outputs"
    export_dir: Path = BACKEND_ROOT / "data" / "exports"
    default_confidence: float = 0.25
    default_iou: float = 0.45
    default_image_size: int = 640
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()


def ensure_runtime_dirs() -> None:
    """Create backend runtime directories."""

    settings = get_settings()
    for directory in (
        settings.upload_dir,
        settings.output_dir,
        settings.export_dir,
        BACKEND_ROOT / "data",
    ):
        directory.mkdir(parents=True, exist_ok=True)
