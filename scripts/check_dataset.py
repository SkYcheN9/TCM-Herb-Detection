"""Command line dataset checker for Phase 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_slice_ai.constants import DEFAULT_RAW_IMAGE_DIR, DEFAULT_RAW_LABEL_DIR
from tcm_slice_ai.dataset import (
    check_flat_dataset,
    check_split_dataset,
    write_json_report,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Check YOLO dataset quality.")
    parser.add_argument("--images", default=DEFAULT_RAW_IMAGE_DIR)
    parser.add_argument("--labels", default=DEFAULT_RAW_LABEL_DIR)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument(
        "--report-json",
        default="reports/phase1/dataset_check.json",
    )
    parser.add_argument(
        "--report-md",
        default="reports/phase1/dataset_check.md",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status when blocking issues are found.",
    )
    return parser.parse_args()


def main() -> int:
    """Run dataset validation and write reports."""

    args = parse_args()
    if args.dataset_root:
        report = check_split_dataset(Path(args.dataset_root))
    else:
        report = check_flat_dataset(Path(args.images), Path(args.labels))

    write_json_report(Path(args.report_json), report.to_dict())
    write_markdown_report(Path(args.report_md), report)

    print(f"Images: {report.image_count}")
    print(f"Labels: {report.label_count}")
    print(f"Valid samples: {report.valid_sample_count}")
    print(f"Blocking issues: {report.blocking_issue_count}")
    print(f"Class order: {'OK' if report.class_order_ok else 'Mismatch'}")
    print(f"JSON report: {args.report_json}")
    print(f"Markdown report: {args.report_md}")

    if args.strict and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
