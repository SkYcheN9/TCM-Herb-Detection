"""Benchmark YOLO inference FPS on CPU and GPU environments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

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
    """Parse benchmark options."""

    parser = argparse.ArgumentParser(description="Benchmark YOLO inference latency and FPS.")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--source", default=None, help="Optional image or video source.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--devices", default="auto,cpu", help="Comma-separated devices: auto,cpu,0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--csv", default="reports/benchmark/benchmark.csv")
    parser.add_argument("--json", default="reports/benchmark/benchmark.json")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative path."""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_weights(value: str | None) -> Path:
    """Return requested weights or best available checkpoint."""

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
    """Register custom project modules before loading checkpoints."""

    try:
        from models.losses import register_focal_loss
        from models.modules import register_ultralytics_modules

        register_ultralytics_modules(enable_cbam=True, enable_bifpn=True)
        register_focal_loss()
    except Exception as exc:
        print(f"Warning: custom module registration skipped: {exc}")


def load_frame(source: str | None, imgsz: int) -> np.ndarray:
    """Load a benchmark frame or create a synthetic one."""

    if source:
        path = resolve_path(source)
        frame = cv2.imread(str(path))
        if frame is None:
            capture = cv2.VideoCapture(str(path))
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise FileNotFoundError(f"Unable to read source: {path}")
        return frame

    frame = np.full((imgsz, imgsz, 3), 240, dtype=np.uint8)
    cv2.rectangle(frame, (imgsz // 5, imgsz // 5), (imgsz * 4 // 5, imgsz * 4 // 5), (32, 132, 214), -1)
    cv2.putText(frame, "TCM", (imgsz // 3, imgsz // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)
    return frame


def normalize_device(device: str) -> str:
    """Map auto to CUDA when available, otherwise CPU."""

    text = device.strip().lower()
    if text == "auto":
        try:
            import torch

            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return text


def benchmark_device(
    *,
    model: object,
    frame: np.ndarray,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """Benchmark one device."""

    for _ in range(warmup):
        model.predict(source=frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)

    latencies: list[float] = []
    detections = 0
    for _ in range(iterations):
        started = time.perf_counter()
        result = model.predict(source=frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)
        boxes = getattr(result, "boxes", None)
        detections = len(boxes) if boxes is not None else 0

    mean_ms = statistics.fmean(latencies)
    return {
        "device": device,
        "iterations": iterations,
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(sorted(latencies)[max(int(iterations * 0.95) - 1, 0)], 3),
        "fps": round(1000.0 / mean_ms if mean_ms > 0 else 0.0, 2),
        "detections": detections,
    }


def write_outputs(csv_path: Path, json_path: Path, rows: list[dict[str, object]]) -> None:
    """Write benchmark CSV and JSON files."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Run benchmark."""

    from ultralytics import YOLO

    args = parse_args()
    register_project_modules()
    weights = resolve_weights(args.weights)
    model = YOLO(str(weights))
    frame = load_frame(args.source, args.imgsz)
    rows: list[dict[str, object]] = []

    for raw_device in [item.strip() for item in args.devices.split(",") if item.strip()]:
        device = normalize_device(raw_device)
        try:
            row = benchmark_device(
                model=model,
                frame=frame,
                device=device,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            row["weights"] = str(weights)
            rows.append(row)
            print(f"[ok] {device}: {row['fps']} FPS")
        except Exception as exc:
            print(f"[error] {device}: {exc}")

    if not rows:
        return 1
    write_outputs(resolve_path(args.csv), resolve_path(args.json), rows)
    print(f"Benchmark saved to {resolve_path(args.csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
