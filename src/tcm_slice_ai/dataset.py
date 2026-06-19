"""Dataset checks and normalization helpers for YOLOv8 training."""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .constants import CLASS_NAMES, IMAGE_SUFFIXES


@dataclass(frozen=True)
class LabelProblem:
    """A single label-file problem found during validation."""

    file: str
    line: int
    reason: str
    content: str = ""


@dataclass
class DatasetReport:
    """Structured result returned by dataset validation."""

    image_count: int = 0
    label_count: int = 0
    valid_sample_count: int = 0
    missing_label_files: list[str] = field(default_factory=list)
    extra_label_files: list[str] = field(default_factory=list)
    empty_label_files: list[str] = field(default_factory=list)
    class_order_ok: bool = True
    class_order_source: str | None = None
    expected_classes: list[str] = field(default_factory=lambda: CLASS_NAMES.copy())
    actual_classes: list[str] | None = None
    label_problems: list[LabelProblem] = field(default_factory=list)
    object_counts: dict[int, int] = field(default_factory=dict)
    file_counts_by_class: dict[int, int] = field(default_factory=dict)

    @property
    def blocking_issue_count(self) -> int:
        """Return the number of issues that should block strict training."""

        return (
            len(self.missing_label_files)
            + len(self.extra_label_files)
            + len(self.empty_label_files)
            + len(self.label_problems)
            + (0 if self.class_order_ok else 1)
        )

    @property
    def ok(self) -> bool:
        """Whether the dataset passed all strict checks."""

        return self.blocking_issue_count == 0

    def to_dict(self) -> dict[str, object]:
        """Serialize the report to plain JSON-compatible values."""

        data = asdict(self)
        data["blocking_issue_count"] = self.blocking_issue_count
        data["ok"] = self.ok
        return data


@dataclass(frozen=True)
class Sample:
    """A valid image/label pair ready for dataset split."""

    stem: str
    image: Path
    label: Path
    primary_class: int


def find_images(image_dir: Path) -> dict[str, Path]:
    """Return image files keyed by stem."""

    return {
        path.stem: path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def find_labels(label_dir: Path) -> dict[str, Path]:
    """Return YOLO label files keyed by stem."""

    return {
        path.stem: path
        for path in sorted(label_dir.glob("*.txt"))
        if path.name != "classes.txt"
    }


def read_classes_file(path: Path | None) -> tuple[list[str] | None, str | None]:
    """Read a LabelImg classes file if one is available."""

    if path is None or not path.exists():
        return None, None
    classes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return classes, str(path)


def read_data_yaml_classes(path: Path) -> tuple[list[str] | None, str | None]:
    """Read the names mapping from a generated Ultralytics data.yaml file."""

    if not path.exists():
        return None, None

    names: dict[int, str] = {}
    in_names_block = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.strip() == "names:":
            in_names_block = True
            continue
        if not in_names_block:
            continue
        if not line.startswith("  "):
            break
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        try:
            index = int(key)
        except ValueError:
            return None, str(path)
        names[index] = value.strip().strip("'\"")

    if not names:
        return None, str(path)
    ordered = [names[index] for index in sorted(names)]
    return ordered, str(path)


def validate_label_file(path: Path) -> tuple[list[int], list[LabelProblem]]:
    """Validate a YOLO txt label file and return class ids plus problems."""

    classes: list[int] = []
    problems: list[LabelProblem] = []
    lines = path.read_text(encoding="utf-8").splitlines()

    if not any(line.strip() for line in lines):
        problems.append(
            LabelProblem(
                file=path.name,
                line=0,
                reason="empty label file",
            )
        )
        return classes, problems

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(
                LabelProblem(
                    file=path.name,
                    line=line_number,
                    reason="YOLO labels must contain 5 fields",
                    content=line,
                )
            )
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            problems.append(
                LabelProblem(
                    file=path.name,
                    line=line_number,
                    reason="class id is not an integer",
                    content=line,
                )
            )
            continue

        classes.append(class_id)
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            problems.append(
                LabelProblem(
                    file=path.name,
                    line=line_number,
                    reason=(
                        f"class id {class_id} is outside expected range "
                        f"0-{len(CLASS_NAMES) - 1}"
                    ),
                    content=line,
                )
            )

        try:
            x_center, y_center, width, height = (float(v) for v in parts[1:])
        except ValueError:
            problems.append(
                LabelProblem(
                    file=path.name,
                    line=line_number,
                    reason="bbox values must be numeric",
                    content=line,
                )
            )
            continue

        coords = (x_center, y_center, width, height)
        if any(value < 0.0 or value > 1.0 for value in coords):
            problems.append(
                LabelProblem(
                    file=path.name,
                    line=line_number,
                    reason="bbox values must be normalized to 0-1",
                    content=line,
                )
            )
        if width <= 0.0 or height <= 0.0:
            problems.append(
                LabelProblem(
                    file=path.name,
                    line=line_number,
                    reason="bbox width and height must be positive",
                    content=line,
                )
            )

    return classes, problems


def check_flat_dataset(
    image_dir: Path,
    label_dir: Path,
    classes_file: Path | None = None,
) -> DatasetReport:
    """Validate a flat image/label dataset exported by LabelImg."""

    images = find_images(image_dir)
    labels = find_labels(label_dir)
    report = DatasetReport(
        image_count=len(images),
        label_count=len(labels),
    )

    actual_classes, source = read_classes_file(classes_file or label_dir / "classes.txt")
    report.actual_classes = actual_classes
    report.class_order_source = source
    if actual_classes != CLASS_NAMES:
        report.class_order_ok = False

    image_stems = set(images)
    label_stems = set(labels)
    report.missing_label_files = sorted(
        f"{stem}{images[stem].suffix}" for stem in image_stems - label_stems
    )
    report.extra_label_files = sorted(f"{stem}.txt" for stem in label_stems - image_stems)

    object_counts: Counter[int] = Counter()
    file_counts: Counter[int] = Counter()
    valid_samples = 0

    for stem in sorted(image_stems & label_stems):
        classes, problems = validate_label_file(labels[stem])
        report.label_problems.extend(problems)
        if any(problem.reason == "empty label file" for problem in problems):
            report.empty_label_files.append(labels[stem].name)

        has_blocking_problem = bool(problems)
        if not has_blocking_problem:
            valid_samples += 1

        unique_classes = set(classes)
        object_counts.update(classes)
        file_counts.update(unique_classes)

    report.valid_sample_count = valid_samples
    report.object_counts = dict(sorted(object_counts.items()))
    report.file_counts_by_class = dict(sorted(file_counts.items()))
    return report


def check_split_dataset(dataset_root: Path) -> DatasetReport:
    """Validate train and val folders under a normalized YOLO dataset."""

    data_yaml_classes, data_yaml_source = read_data_yaml_classes(
        dataset_root / "data.yaml"
    )
    merged = DatasetReport(
        actual_classes=data_yaml_classes,
        class_order_source=data_yaml_source,
    )
    if data_yaml_classes != CLASS_NAMES:
        merged.class_order_ok = False

    object_counts: Counter[int] = Counter()
    file_counts: Counter[int] = Counter()

    for split in ("train", "val"):
        split_report = check_flat_dataset(
            dataset_root / "images" / split,
            dataset_root / "labels" / split,
            classes_file=dataset_root / "classes.txt",
        )
        merged.image_count += split_report.image_count
        merged.label_count += split_report.label_count
        merged.valid_sample_count += split_report.valid_sample_count
        merged.missing_label_files.extend(
            f"{split}/{name}" for name in split_report.missing_label_files
        )
        merged.extra_label_files.extend(
            f"{split}/{name}" for name in split_report.extra_label_files
        )
        merged.empty_label_files.extend(
            f"{split}/{name}" for name in split_report.empty_label_files
        )
        merged.label_problems.extend(
            LabelProblem(
                file=f"{split}/{problem.file}",
                line=problem.line,
                reason=problem.reason,
                content=problem.content,
            )
            for problem in split_report.label_problems
        )
        if not split_report.class_order_ok:
            merged.class_order_ok = False
            merged.actual_classes = split_report.actual_classes
            merged.class_order_source = split_report.class_order_source
        if split_report.actual_classes and split_report.actual_classes != CLASS_NAMES:
            merged.class_order_ok = False
            merged.actual_classes = split_report.actual_classes
            merged.class_order_source = split_report.class_order_source
        object_counts.update(split_report.object_counts)
        file_counts.update(split_report.file_counts_by_class)

    merged.object_counts = dict(sorted(object_counts.items()))
    merged.file_counts_by_class = dict(sorted(file_counts.items()))
    return merged


def collect_valid_samples(
    image_dir: Path,
    label_dir: Path,
    report: DatasetReport,
) -> list[Sample]:
    """Build sample records that are safe to include in training."""

    images = find_images(image_dir)
    labels = find_labels(label_dir)
    bad_label_files = {problem.file for problem in report.label_problems}
    missing_stems = {Path(name).stem for name in report.missing_label_files}
    samples: list[Sample] = []

    for stem in sorted(set(images) & set(labels)):
        if labels[stem].name in bad_label_files or stem in missing_stems:
            continue
        classes, _ = validate_label_file(labels[stem])
        if not classes:
            continue
        class_counter = Counter(classes)
        primary_class = class_counter.most_common(1)[0][0]
        samples.append(
            Sample(
                stem=stem,
                image=images[stem],
                label=labels[stem],
                primary_class=primary_class,
            )
        )

    return samples


def split_samples(
    samples: Iterable[Sample],
    train_ratio: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    """Split samples into class-balanced train and validation sets."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    rng = random.Random(seed)
    grouped: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.primary_class].append(sample)

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    for class_id in sorted(grouped):
        group = sorted(grouped[class_id], key=lambda item: item.stem)
        rng.shuffle(group)
        if len(group) == 1:
            split_index = 1
        else:
            split_index = max(1, min(len(group) - 1, round(len(group) * train_ratio)))
        train_samples.extend(group[:split_index])
        val_samples.extend(group[split_index:])

    return (
        sorted(train_samples, key=lambda item: item.stem),
        sorted(val_samples, key=lambda item: item.stem),
    )


def ensure_dataset_dirs(dataset_root: Path) -> None:
    """Create the normalized YOLO directory structure."""

    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        (dataset_root / relative).mkdir(parents=True, exist_ok=True)


def clean_split_dirs(dataset_root: Path) -> None:
    """Remove generated train/val files while keeping the directory layout."""

    resolved_root = dataset_root.resolve()
    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        directory = (dataset_root / relative).resolve()
        if resolved_root not in directory.parents:
            raise RuntimeError(f"Refusing to clean unexpected path: {directory}")
        for path in directory.iterdir():
            if path.is_file() and path.name != ".gitkeep":
                path.unlink()


def place_file(source: Path, destination: Path, mode: str) -> None:
    """Copy, hardlink, or symlink a source file into the normalized dataset."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return
    if mode == "symlink":
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)
        return

    raise ValueError("mode must be copy, hardlink, or symlink")


def write_data_yaml(dataset_root: Path) -> Path:
    """Write an Ultralytics-compatible data.yaml with fixed class order."""

    data_yaml = dataset_root / "data.yaml"
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dataset_root / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n",
        encoding="utf-8",
    )
    return data_yaml


def normalize_dataset(
    image_dir: Path,
    label_dir: Path,
    dataset_root: Path,
    train_ratio: float,
    seed: int,
    mode: str,
    clean: bool = True,
) -> dict[str, object]:
    """Normalize a flat dataset into YOLO train/val folders."""

    report = check_flat_dataset(image_dir, label_dir)
    samples = collect_valid_samples(image_dir, label_dir, report)
    train_samples, val_samples = split_samples(samples, train_ratio, seed)

    ensure_dataset_dirs(dataset_root)
    if clean:
        clean_split_dirs(dataset_root)

    for split, current_samples in (("train", train_samples), ("val", val_samples)):
        for sample in current_samples:
            place_file(
                sample.image,
                dataset_root / "images" / split / sample.image.name,
                mode,
            )
            place_file(
                sample.label,
                dataset_root / "labels" / split / sample.label.name,
                mode,
            )

    data_yaml = write_data_yaml(dataset_root)
    prepared_report = check_split_dataset(dataset_root)

    return {
        "source_report": report.to_dict(),
        "prepared_report": prepared_report.to_dict(),
        "data_yaml": str(data_yaml),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "excluded_count": report.image_count - len(samples),
        "mode": mode,
        "train_ratio": train_ratio,
        "seed": seed,
    }


def write_json_report(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON report with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_markdown_report(path: Path, report: DatasetReport) -> None:
    """Write a short human-readable dataset validation report."""

    lines = [
        "# Dataset Check Report",
        "",
        f"- Images: {report.image_count}",
        f"- Labels: {report.label_count}",
        f"- Valid samples: {report.valid_sample_count}",
        f"- Blocking issues: {report.blocking_issue_count}",
        f"- Class order: {'OK' if report.class_order_ok else 'Mismatch'}",
        "",
        "## Fixed Class Order",
        "",
    ]
    lines.extend(f"{index}. `{name}`" for index, name in enumerate(CLASS_NAMES))

    if report.missing_label_files:
        lines.extend(["", "## Missing Labels", ""])
        lines.extend(f"- `{name}`" for name in report.missing_label_files[:200])
        if len(report.missing_label_files) > 200:
            lines.append(f"- ... {len(report.missing_label_files) - 200} more")

    if report.extra_label_files:
        lines.extend(["", "## Extra Labels", ""])
        lines.extend(f"- `{name}`" for name in report.extra_label_files[:200])
        if len(report.extra_label_files) > 200:
            lines.append(f"- ... {len(report.extra_label_files) - 200} more")

    if report.label_problems:
        lines.extend(["", "## Label Problems", ""])
        for problem in report.label_problems[:200]:
            lines.append(
                f"- `{problem.file}:{problem.line}` {problem.reason}"
                + (f" `{problem.content}`" if problem.content else "")
            )
        if len(report.label_problems) > 200:
            lines.append(f"- ... {len(report.label_problems) - 200} more")

    lines.extend(["", "## Object Counts", ""])
    for index, name in enumerate(CLASS_NAMES):
        count = report.object_counts.get(index, 0)
        lines.append(f"- `{index} {name}`: {count}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
