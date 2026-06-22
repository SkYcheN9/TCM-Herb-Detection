"""Filesystem paths used by the desktop client."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DESKTOP_DATA_DIR = PROJECT_ROOT / "src" / "desktop" / "data"
DESKTOP_REPORT_DIR = PROJECT_ROOT / "reports" / "desktop"
IMAGE_OUTPUT_DIR = DESKTOP_REPORT_DIR / "images"
SNAPSHOT_OUTPUT_DIR = DESKTOP_REPORT_DIR / "snapshots"
VIDEO_OUTPUT_DIR = DESKTOP_REPORT_DIR / "videos"
EXPORT_OUTPUT_DIR = DESKTOP_REPORT_DIR / "exports"
HISTORY_DB_PATH = DESKTOP_DATA_DIR / "history.db"


def ensure_desktop_dirs() -> None:
    """Create desktop runtime folders."""
    for directory in (
        DESKTOP_DATA_DIR,
        IMAGE_OUTPUT_DIR,
        SNAPSHOT_OUTPUT_DIR,
        VIDEO_OUTPUT_DIR,
        EXPORT_OUTPUT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

