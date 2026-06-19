"""Clean raw YOLO datasets by detecting invalid, blurry and duplicate images."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_slice_ai.constants import CLASS_NAMES, DEFAULT_RAW_IMAGE_DIR, DEFAULT_RAW_LABEL_DIR, IMAGE_SUFFIXES
from tcm_slice_ai.dataset import find_images, find_labels, validate_label_file


@dataclass
class ImageQuality:
    """Quality metrics for one image."""

    image: str
    label: str | None
    readable: bool
    width: int = 0
    height: int = 0
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    exact_duplicate_of: str | None = None
    near_duplicate_of: str | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def keep(self) -> bool:
        """Return whether the sample should stay in the cleaned dataset."""

        return not self.issues


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Detect blurry, invalid and duplicate images.")
    parser.add_argument("--images", default=DEFAULT_RAW_IMAGE_DIR, help="Flat image directory.")
    parser.add_argument("--labels", default=DEFAULT_RAW_LABEL_DIR, help="Flat YOLO label directory.")
    parser.add_argument("--output", default=None, help="Optional cleaned flat dataset directory.")
    parser.add_argument("--blur-threshold", type=float, default=60.0)
    parser.add_argument("--near-duplicate-threshold", type=int, default=4)
    parser.add_argument("--min-width", type=int, default=640)
    parser.add_argument("--min-height", type=int, default=480)
    parser.add_argument("--report-json", default="reports/phase1/cleaning_report.json")
    parser.add_argument("--report-md", default="reports/phase1/cleaning_report.md")
    return parser.parse_args()


def exact_file_hash(path: Path) -> str:
    """Return a SHA256 hash for exact duplicate detection."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(gray_image, hash_size: int = 8) -> int:
    """Return a difference hash used for near-duplicate detection."""

    resized = cv2.resize(gray_image, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(left: int, right: int) -> int:
    """Return bit distance between two perceptual hashes."""

    return (left ^ right).bit_count()


def read_label_issues(label_path: Path | None) -> list[str]:
    """Return blocking label issues for a sample."""

    if label_path is None:
        return ["missing label"]
    if not label_path.exists():
        return ["missing label"]

    classes, problems = validate_label_file(label_path)
    issues = [problem.reason for problem in problems]
    if not classes:
        issues.append("empty label")
    return sorted(set(issues))


def analyze_dataset(
    image_dir: Path,
    label_dir: Path,
    *,
    blur_threshold: float,
    near_duplicate_threshold: int,
    min_width: int,
    min_height: int,
) -> list[ImageQuality]:
    """Analyze dataset quality and return one record per image."""

    images = find_images(image_dir)
    labels = find_labels(label_dir)
    records: list[ImageQuality] = []
    exact_seen: dict[str, str] = {}
    near_seen: list[tuple[str, int]] = []

    for stem, image_path in sorted(images.items()):
        label_path = labels.get(stem)
        record = ImageQuality(
            image=image_path.name,
            label=label_path.name if label_path else None,
            readable=False,
        )

        frame = cv2.imread(str(image_path))
        if frame is None:
            record.issues.append("unreadable image")
            record.issues.extend(read_label_issues(label_path))
            records.append(record)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        record.readable = True
        record.width = int(width)
        record.height = int(height)
        record.blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        record.brightness = float(gray.mean())
        record.contrast = float(gray.std())

        if width < min_width or height < min_height:
            record.issues.append(f"resolution below {min_width}x{min_height}")
        if record.blur_score < blur_threshold:
            record.issues.append(f"blur score below {blur_threshold:g}")

        current_hash = exact_file_hash(image_path)
        if current_hash in exact_seen:
            record.exact_duplicate_of = exact_seen[current_hash]
            record.issues.append("exact duplicate")
        else:
            exact_seen[current_hash] = image_path.name

        current_dhash = dhash(gray)
        if record.exact_duplicate_of is None:
            for previous_name, previous_dhash in near_seen:
                if hamming_distance(current_dhash, previous_dhash) <= near_duplicate_threshold:
                    record.near_duplicate_of = previous_name
                    record.issues.append("near duplicate")
                    break
        near_seen.append((image_path.name, current_dhash))

        record.issues.extend(read_label_issues(label_path))
        record.issues = sorted(set(record.issues))
        records.append(record)

    extra_labels = sorted(set(labels) - set(images))
    for stem in extra_labels:
        records.append(
            ImageQuality(
                image=f"{stem}<missing image>",
                label=labels[stem].name,
                readable=False,
                issues=["label without image"],
            )
        )

    return records


def write_clean_dataset(
    output_dir: Path,
    image_dir: Path,
    label_dir: Path,
    records: list[ImageQuality],
) -> dict[str, int]:
    """Copy kept samples to a clean flat dataset directory."""

    image_output = output_dir / "images"
    label_output = output_dir / "labels"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    kept = 0
    for path in list(image_output.iterdir()) + list(label_output.iterdir()):
        if path.is_file() and path.name != ".gitkeep":
            path.unlink()

    images = find_images(image_dir)
    labels = find_labels(label_dir)
    keep_names = {record.image for record in records if record.keep}
    for stem, image_path in sorted(images.items()):
        if image_path.name not in keep_names:
            continue
        label_path = labels.get(stem)
        if label_path is None:
            continue
        shutil.copy2(image_path, image_output / image_path.name)
        shutil.copy2(label_path, label_output / label_path.name)
        kept += 1

    (label_output / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    return {"kept_samples": kept, "excluded_samples": len([record for record in records if not record.keep])}


def summarize(records: list[ImageQuality]) -> dict[str, object]:
    """Return JSON-serializable cleaning summary."""

    issue_counts: dict[str, int] = {}
    for record in records:
        for issue in record.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    return {
        "total_records": len(records),
        "kept_samples": sum(1 for record in records if record.keep),
        "excluded_samples": sum(1 for record in records if not record.keep),
        "issue_counts": dict(sorted(issue_counts.items())),
        "records": [asdict(record) for record in records],
    }


def write_reports(json_path: Path, md_path: Path, summary: dict[str, object]) -> None:
    """Write JSON and Markdown cleaning reports."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Dataset Cleaning Report",
        "",
        f"- Total records: {summary['total_records']}",
        f"- Kept samples: {summary['kept_samples']}",
        f"- Excluded samples: {summary['excluded_samples']}",
        "",
        "## Issue Counts",
        "",
    ]
    issue_counts = summary["issue_counts"]
    if isinstance(issue_counts, dict) and issue_counts:
        lines.extend(f"- {issue}: {count}" for issue, count in issue_counts.items())
    else:
        lines.append("- No issues found")

    lines.extend(["", "## Excluded Samples", ""])
    records = summary["records"]
    excluded = [record for record in records if isinstance(record, dict) and record.get("issues")]
    if excluded:
        for record in excluded[:200]:
            issues = ", ".join(str(issue) for issue in record["issues"])
            lines.append(f"- `{record['image']}`: {issues}")
        if len(excluded) > 200:
            lines.append(f"- ... {len(excluded) - 200} more")
    else:
        lines.append("- None")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run dataset cleaning analysis."""

    args = parse_args()
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    records = analyze_dataset(
        image_dir,
        label_dir,
        blur_threshold=args.blur_threshold,
        near_duplicate_threshold=args.near_duplicate_threshold,
        min_width=args.min_width,
        min_height=args.min_height,
    )
    summary = summarize(records)

    if args.output:
        summary["clean_output"] = str(Path(args.output))
        summary["copy_summary"] = write_clean_dataset(Path(args.output), image_dir, label_dir, records)

    write_reports(Path(args.report_json), Path(args.report_md), summary)
    print(f"Kept samples: {summary['kept_samples']}")
    print(f"Excluded samples: {summary['excluded_samples']}")
    print(f"JSON report: {args.report_json}")
    print(f"Markdown report: {args.report_md}")
    if args.output:
        print(f"Clean dataset: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
