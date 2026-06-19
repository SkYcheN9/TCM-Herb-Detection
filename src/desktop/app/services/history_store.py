"""SQLite history store for desktop detection records."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import HISTORY_DB_PATH, ensure_desktop_dirs


@dataclass(frozen=True)
class DetectionRecord:
    """Single detection history record."""

    id: int
    created_at: str
    mode: str
    source_path: str
    output_path: str
    model_path: str
    device: str
    fps: float
    total_count: int
    class_counts: dict[str, int]
    status: str


class HistoryStore:
    """Small SQLite wrapper for desktop detection history."""

    def __init__(self, db_path: Path = HISTORY_DB_PATH) -> None:
        ensure_desktop_dirs()
        self.db_path = db_path
        self._init_db()

    def add_record(
        self,
        *,
        mode: str,
        source_path: str,
        output_path: str,
        model_path: str,
        device: str,
        fps: float,
        total_count: int,
        class_counts: dict[str, int],
        status: str = "完成",
    ) -> int:
        """Persist a detection result and return its record id."""
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO detection_records (
                    created_at, mode, source_path, output_path, model_path,
                    device, fps, total_count, class_counts, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    mode,
                    source_path,
                    output_path,
                    model_path,
                    device,
                    fps,
                    total_count,
                    json.dumps(class_counts, ensure_ascii=False),
                    status,
                ),
            )
            return int(cursor.lastrowid)

    def list_records(self, limit: int = 500) -> list[DetectionRecord]:
        """Return recent detection records."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, mode, source_path, output_path, model_path,
                       device, fps, total_count, class_counts, status
                FROM detection_records
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        """Return compact statistics for the History view."""
        records = self.list_records()
        class_totals: dict[str, int] = {}
        for record in records:
            for name, count in record.class_counts.items():
                class_totals[name] = class_totals.get(name, 0) + count

        top_class = "-"
        if class_totals:
            top_class = max(class_totals.items(), key=lambda item: item[1])[0]

        today = datetime.now().strftime("%Y-%m-%d")
        today_count = sum(1 for record in records if record.created_at.startswith(today))

        return {
            "today": today_count,
            "total": len(records),
            "top_class": top_class,
            "exportable": len(records),
        }

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    device TEXT NOT NULL,
                    fps REAL NOT NULL,
                    total_count INTEGER NOT NULL,
                    class_counts TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DetectionRecord:
        counts = json.loads(row["class_counts"]) if row["class_counts"] else {}
        return DetectionRecord(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            mode=str(row["mode"]),
            source_path=str(row["source_path"]),
            output_path=str(row["output_path"]),
            model_path=str(row["model_path"]),
            device=str(row["device"]),
            fps=float(row["fps"]),
            total_count=int(row["total_count"]),
            class_counts={str(key): int(value) for key, value in counts.items()},
            status=str(row["status"]),
        )

