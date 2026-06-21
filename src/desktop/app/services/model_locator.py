"""Locate the model used by desktop inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ModelCandidate:
    """Resolved desktop model candidate."""

    path: Path
    source: str


PREFERRED_MODEL_PATHS = (
    (
        "final_results_full/reports/ablation/runs/baseline_cbam_bifpn/weights/best.pt",
        "Final deployment: CBAM+BiFPN balanced best",
    ),
    (
        "final_results_full/reports/ablation/runs/baseline_cbam/weights/best.pt",
        "Final accuracy best: CBAM",
    ),
    (
        "final_results_full/reports/ablation/runs/baseline_ghostconv/weights/best.pt",
        "Final lightweight best: GhostConv",
    ),
    ("runs/baseline/weights/best.pt", "Phase 1 Baseline best"),
    ("reports/ablation/runs/baseline/weights/best.pt", "Ablation baseline best"),
    ("runs/detect/runs/baseline/weights/best.pt", "Detection baseline best"),
    ("yolo26n.pt", "Local YOLO fallback"),
    ("yolov8n.pt", "Ultralytics YOLOv8n fallback"),
)


def find_best_model(project_root: Path | None = None) -> ModelCandidate | None:
    """Return the best available model for desktop inference."""
    root = project_root or PROJECT_ROOT

    for relative_path, source in PREFERRED_MODEL_PATHS:
        path = root / relative_path
        if path.is_file():
            return ModelCandidate(path=path, source=source)

    best_weights = sorted(
        root.glob("runs/**/weights/best.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in best_weights:
        if "smoke" not in str(path).lower():
            return ModelCandidate(path=path, source="Detected best.pt")

    if best_weights:
        return ModelCandidate(path=best_weights[0], source="Smoke best.pt fallback")

    return None

