"""Evaluate a trained YOLO model and export report-ready metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


PREFERRED_WEIGHTS: tuple[Path, ...] = (
    ROOT / "runs" / "full_model" / "weights" / "best.pt",
    ROOT / "runs" / "cbam_bifpn_focal" / "weights" / "best.pt",
    ROOT / "runs" / "cbam_bifpn" / "weights" / "best.pt",
    ROOT / "runs" / "baseline" / "weights" / "best.pt",
    ROOT / "reports" / "ablation" / "runs" / "baseline" / "weights" / "best.pt",
)


def parse_args() -> argparse.Namespace:
    """Parse evaluation options."""

    parser = argparse.ArgumentParser(description="Evaluate YOLO weights and export metrics.")
    parser.add_argument("--weights", default=None, help="Model weights. Defaults to best available model.")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", default="reports/evaluation")
    parser.add_argument("--name", default="default")
    parser.add_argument("--csv", default="reports/evaluation/metrics.csv")
    parser.add_argument("--json", default="reports/evaluation/metrics.json")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative path."""

    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_weights(value: str | None) -> Path:
    """Return requested weights or the best available checkpoint."""

    if value:
        path = resolve_path(value)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Weights not found: {path}")
    for path in PREFERRED_WEIGHTS:
        if path.is_file():
            return path
    candidates = sorted(ROOT.glob("runs/**/weights/best.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No best.pt found. Train a model first or pass --weights.")


def register_project_modules() -> None:
    """Register custom modules before loading project checkpoints."""

    try:
        from models.losses import register_focal_loss
        from models.modules import register_ultralytics_modules

        register_ultralytics_modules(enable_cbam=True, enable_bifpn=True)
        register_focal_loss()
    except Exception as exc:
        print(f"Warning: custom module registration skipped: {exc}")


def normalize_device(device: str) -> str | None:
    """Map auto to Ultralytics default device selection."""

    return None if device.lower() == "auto" else device


def summarize_metrics(weights: Path, metrics: object) -> dict[str, object]:
    """Convert Ultralytics validation metrics to report fields."""

    speed = getattr(metrics, "speed", {}) or {}
    preprocess = float(speed.get("preprocess", 0.0))
    inference = float(speed.get("inference", 0.0))
    postprocess = float(speed.get("postprocess", 0.0))
    total_ms = preprocess + inference + postprocess
    box = metrics.box
    return {
        "weights": str(weights),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "preprocess_ms": preprocess,
        "inference_ms": inference,
        "postprocess_ms": postprocess,
        "fps": 1000.0 / total_ms if total_ms > 0 else 0.0,
    }


def write_csv(path: Path, row: dict[str, object]) -> None:
    """Write one-row CSV metrics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    """Run validation and export metrics."""

    from ultralytics import YOLO

    args = parse_args()
    register_project_modules()
    weights = resolve_weights(args.weights)
    model = YOLO(str(weights))
    val_kwargs: dict[str, object] = {
        "data": str(resolve_path(args.data)),
        "imgsz": args.imgsz,
        "project": str(resolve_path(args.project)),
        "name": args.name,
        "plots": True,
        "exist_ok": True,
    }
    if args.batch is not None:
        val_kwargs["batch"] = args.batch
    device = normalize_device(args.device)
    if device is not None:
        val_kwargs["device"] = device

    metrics = model.val(**val_kwargs)
    summary = summarize_metrics(weights, metrics)
    json_path = resolve_path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(resolve_path(args.csv), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
