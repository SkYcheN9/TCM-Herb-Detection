"""Locate inference weights without touching training code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import PROJECT_ROOT


@dataclass(frozen=True)
class ModelCandidate:
    """Resolved model path and display source."""

    path: Path
    source: str


PREFERRED_MODEL_PATHS: tuple[tuple[str, str], ...] = (
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


def find_best_model(project_root: Path = PROJECT_ROOT) -> ModelCandidate | None:
    """Return the preferred available model for inference."""

    for relative_path, source in PREFERRED_MODEL_PATHS:
        path = project_root / relative_path
        if path.is_file():
            return ModelCandidate(path=path, source=source)

    best_weights = sorted(
        project_root.glob("runs/**/weights/best.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in best_weights:
        if "smoke" not in str(path).lower():
            return ModelCandidate(path=path, source="Detected best.pt")
    if best_weights:
        return ModelCandidate(path=best_weights[0], source="Smoke best.pt fallback")
    return None


def resolve_model_path(model_path: str | None = None) -> Path:
    """Resolve a requested model path or return the best available candidate."""

    if model_path:
        path = Path(model_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            return path
        raise FileNotFoundError(f"Model file not found: {path}")

    candidate = find_best_model()
    if candidate is None:
        raise FileNotFoundError("No model weights found. Expected runs/**/weights/best.pt or yolov8n.pt.")
    return candidate.path

