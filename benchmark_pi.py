"""Benchmark PyTorch, ONNX and OpenVINO exports on Raspberry Pi 5."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
PI_DIR = ROOT / "deployment" / "raspberry_pi"
sys.path.insert(0, str(PI_DIR))

from pi_runtime import CLASS_NAMES, YoloDetector  # noqa: E402


DEFAULT_MODELS = {
    "pytorch": ROOT / "best.pt",
    "onnx": ROOT / "best.onnx",
    "openvino": ROOT / "best_openvino",
}


def parse_args() -> argparse.Namespace:
    """Parse benchmark options."""

    parser = argparse.ArgumentParser(description="Benchmark Raspberry Pi deployment formats.")
    parser.add_argument("--pt", default=str(DEFAULT_MODELS["pytorch"]), help="PyTorch .pt model path.")
    parser.add_argument("--onnx", default=str(DEFAULT_MODELS["onnx"]), help="ONNX model path.")
    parser.add_argument("--openvino", default=str(DEFAULT_MODELS["openvino"]), help="OpenVINO model directory.")
    parser.add_argument("--image", default=None, help="Optional image used as benchmark input.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=100, help="Measured iterations.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "pi_benchmark.csv"),
        help="CSV output path.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    """Resolve a benchmark path relative to the project root."""

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return resolved


def load_frame(image_path: str | None, size: int) -> np.ndarray:
    """Load a benchmark frame or create a deterministic synthetic frame."""

    if image_path:
        path = resolve_path(image_path)
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Unable to read benchmark image: {path}")
        return frame

    frame = np.full((size, size, 3), 245, dtype=np.uint8)
    cv2.rectangle(frame, (size // 5, size // 5), (size * 4 // 5, size * 4 // 5), (42, 144, 214), 3)
    cv2.putText(
        frame,
        "TCM-SliceAI",
        (max(size // 12, 10), size // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(size / 640.0, 0.5),
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return frame


def benchmark_backend(
    *,
    backend: str,
    model_path: Path,
    frame: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """Benchmark one runtime backend."""

    detector = YoloDetector(
        model_path=model_path,
        backend=backend,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        class_names=CLASS_NAMES,
    )

    for _ in range(warmup):
        detector.detect(frame)

    latencies: list[float] = []
    detections = 0
    started = time.perf_counter()
    for _ in range(iterations):
        result = detector.detect(frame)
        latencies.append(result.elapsed_ms)
        detections = len(result.detections)
    wall_ms = (time.perf_counter() - started) * 1000.0

    mean_ms = statistics.fmean(latencies)
    p50_ms = statistics.median(latencies)
    p95_ms = sorted(latencies)[max(int(iterations * 0.95) - 1, 0)]
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    wall_fps = iterations * 1000.0 / wall_ms if wall_ms > 0 else 0.0
    return {
        "backend": backend,
        "model": str(model_path),
        "imgsz": imgsz,
        "iterations": iterations,
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(p50_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "fps": round(fps, 2),
        "wall_fps": round(wall_fps, 2),
        "detections": detections,
        "pass_10fps": fps >= 10.0,
    }


def main() -> int:
    """Run all available runtime benchmarks."""

    args = parse_args()
    frame = load_frame(args.image, args.imgsz)
    models = {
        "pytorch": resolve_path(args.pt),
        "onnx": resolve_path(args.onnx),
        "openvino": resolve_path(args.openvino),
    }

    rows: list[dict[str, object]] = []
    for backend, path in models.items():
        if not path.exists():
            print(f"[skip] {backend}: model not found at {path}")
            continue
        try:
            row = benchmark_backend(
                backend=backend,
                model_path=path,
                frame=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            rows.append(row)
            status = "PASS" if row["pass_10fps"] else "FAIL"
            print(f"[{status}] {backend}: {row['fps']} FPS, mean {row['mean_ms']} ms")
        except Exception as exc:
            print(f"[error] {backend}: {exc}")

    if not rows:
        print("No benchmark completed. Run python export.py first.")
        return 1

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Benchmark CSV saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
