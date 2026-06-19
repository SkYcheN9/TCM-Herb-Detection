"""Train the Ultralytics YOLOv8 baseline model."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_slice_ai.dataset import check_split_dataset


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""

    parser = argparse.ArgumentParser(description="Train YOLOv8 baseline.")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA id such as 0")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="baseline")
    parser.add_argument(
        "--skip-dataset-check",
        action="store_true",
        help="Skip strict dataset validation before training.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    """Prefer CUDA when available and fall back to CPU."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install Phase 1 dependencies first: "
            "python -m pip install -r requirements.txt"
        ) from exc

    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def default_batch_size(device: str, requested: int | None) -> int:
    """Choose a conservative batch size for GPU or CPU training."""

    if requested is not None:
        return requested
    if device == "cpu":
        return 4
    return 16


def main() -> int:
    """Run Ultralytics YOLOv8 training."""

    args = parse_args()
    data_path = Path(args.data)
    dataset_root = data_path.parent
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = ROOT / project_path

    if not args.skip_dataset_check:
        report = check_split_dataset(dataset_root)
        if not report.ok:
            print(
                "Dataset check failed. Run scripts/check_dataset.py "
                "--dataset-root dataset for details.",
                file=sys.stderr,
            )
            return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install Phase 1 dependencies first: "
            "python -m pip install -r requirements.txt"
        ) from exc

    device = resolve_device(args.device)
    batch = default_batch_size(device, args.batch)
    print(f"Training device: {device}")
    print(f"Batch size: {batch}")
    print(f"Output: {project_path / args.name}")

    try:
        model = YOLO(args.model)
    except (EOFError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            f"Model weights look incomplete or corrupted: {args.model}. "
            "Delete the local .pt file and run again so Ultralytics can "
            "download a fresh copy."
        ) from exc

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=args.workers,
        seed=args.seed,
        device=device,
        project=str(project_path),
        name=args.name,
        exist_ok=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
