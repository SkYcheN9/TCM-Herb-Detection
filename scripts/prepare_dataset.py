"""Normalize the raw LabelImg export into a YOLOv8 train/val dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_slice_ai.constants import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_RAW_IMAGE_DIR,
    DEFAULT_RAW_LABEL_DIR,
)
from tcm_slice_ai.dataset import normalize_dataset, write_json_report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Prepare Phase 1 YOLO dataset.")
    parser.add_argument("--images", default=DEFAULT_RAW_IMAGE_DIR)
    parser.add_argument("--labels", default=DEFAULT_RAW_LABEL_DIR)
    parser.add_argument("--output", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink"),
        default="copy",
        help="How files are placed into the normalized dataset.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clear existing train/val files before preparing.",
    )
    parser.add_argument(
        "--report-json",
        default="reports/phase1/normalization_report.json",
    )
    return parser.parse_args()


def main() -> int:
    """Create train/val folders and data.yaml."""

    args = parse_args()
    summary = normalize_dataset(
        image_dir=Path(args.images),
        label_dir=Path(args.labels),
        dataset_root=Path(args.output),
        train_ratio=args.train_ratio,
        seed=args.seed,
        mode=args.mode,
        clean=not args.no_clean,
    )
    write_json_report(Path(args.report_json), summary)

    print(f"data.yaml: {summary['data_yaml']}")
    print(f"train samples: {summary['train_count']}")
    print(f"val samples: {summary['val_count']}")
    print(f"excluded samples: {summary['excluded_count']}")
    print(f"report: {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
