"""Offline YOLO dataset augmentation with Albumentations, Mosaic and MixUp."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_slice_ai.constants import CLASS_NAMES
from tcm_slice_ai.dataset import find_images, find_labels, write_data_yaml


@dataclass(frozen=True)
class YoloBox:
    """One YOLO-format bounding box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class AugmentStats:
    """Offline augmentation summary."""

    copied_train: int
    copied_val: int
    albumentations_images: int
    mosaic_images: int
    mixup_images: int
    output: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Create an offline augmented YOLO dataset.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output", default="dataset_augmented")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copies", type=int, default=1, help="Albumentations copies per training image.")
    parser.add_argument("--mosaic-count", type=int, default=200)
    parser.add_argument("--mixup-count", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--disable-albumentations", action="store_true")
    parser.add_argument("--disable-mosaic", action="store_true")
    parser.add_argument("--disable-mixup", action="store_true")
    parser.add_argument("--report-json", default="reports/phase1/augmentation_report.json")
    return parser.parse_args()


def import_albumentations():
    """Import Albumentations lazily with a clear setup hint."""

    try:
        import albumentations as album
    except ImportError as exc:
        raise RuntimeError(
            "Albumentations is required for offline augmentation. "
            "Install it with: .\\.venv\\Scripts\\python.exe -m pip install albumentations"
        ) from exc
    return album


def read_boxes(path: Path) -> list[YoloBox]:
    """Read YOLO labels."""

    boxes: list[YoloBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x_center, y_center, width, height = line.split()
        boxes.append(
            YoloBox(
                class_id=int(class_id),
                x_center=float(x_center),
                y_center=float(y_center),
                width=float(width),
                height=float(height),
            )
        )
    return boxes


def write_boxes(path: Path, boxes: list[YoloBox]) -> None:
    """Write YOLO labels."""

    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
        if box.width > 0.0 and box.height > 0.0
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def yolo_to_albu(boxes: list[YoloBox]) -> tuple[list[list[float]], list[int]]:
    """Convert boxes to Albumentations YOLO format."""

    return (
        [[box.x_center, box.y_center, box.width, box.height] for box in boxes],
        [box.class_id for box in boxes],
    )


def albu_to_yolo(boxes: list[list[float]], class_ids: list[int]) -> list[YoloBox]:
    """Convert Albumentations boxes to project label records."""

    result: list[YoloBox] = []
    for class_id, box in zip(class_ids, boxes, strict=False):
        x_center, y_center, width, height = (float(value) for value in box)
        if width <= 0.001 or height <= 0.001:
            continue
        result.append(
            YoloBox(
                class_id=int(class_id),
                x_center=min(max(x_center, 0.0), 1.0),
                y_center=min(max(y_center, 0.0), 1.0),
                width=min(max(width, 0.0), 1.0),
                height=min(max(height, 0.0), 1.0),
            )
        )
    return result


def copy_split(dataset_root: Path, output_root: Path, split: str) -> int:
    """Copy original images and labels for a split."""

    image_output = output_root / "images" / split
    label_output = output_root / "labels" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    images = find_images(dataset_root / "images" / split)
    labels = find_labels(dataset_root / "labels" / split)

    copied = 0
    for stem, image_path in sorted(images.items()):
        label_path = labels.get(stem)
        if label_path is None:
            continue
        shutil.copy2(image_path, image_output / image_path.name)
        shutil.copy2(label_path, label_output / label_path.name)
        copied += 1
    return copied


def clean_output(output_root: Path) -> None:
    """Clear generated augmented dataset contents."""

    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        directory = output_root / relative
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.is_file() and path.name != ".gitkeep":
                path.unlink()

    labels_root = output_root / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    for cache_path in labels_root.glob("*.cache"):
        cache_path.unlink()


def build_albumentations_pipeline(imgsz: int):
    """Build an explicit offline augmentation pipeline for train copies.

    HSV jitter and bbox-safe random crop are kept as first-class transforms
    instead of being hidden inside a generic Albumentations branch, so the
    generated dataset visibly uses every strategy required by the project.
    """

    album = import_albumentations()
    return album.Compose(
        [
            album.HueSaturationValue(
                hue_shift_limit=12,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0,
            ),
            album.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.65,
            ),
            album.CLAHE(p=0.35),
            album.HorizontalFlip(p=0.5),
            album.RandomSizedBBoxSafeCrop(
                height=imgsz,
                width=imgsz,
                erosion_rate=0.1,
                p=1.0,
            ),
        ],
        bbox_params=album.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=0.2,
        ),
    )


def augment_with_albumentations(
    dataset_root: Path,
    output_root: Path,
    copies: int,
    rng: random.Random,
    imgsz: int,
) -> int:
    """Create Albumentations image copies for the training split."""

    if copies <= 0:
        return 0

    pipeline = build_albumentations_pipeline(imgsz)
    images = find_images(dataset_root / "images" / "train")
    labels = find_labels(dataset_root / "labels" / "train")
    created = 0
    for stem, image_path in sorted(images.items()):
        label_path = labels.get(stem)
        if label_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        boxes = read_boxes(label_path)
        albu_boxes, class_ids = yolo_to_albu(boxes)
        for index in range(copies):
            seed = rng.randint(0, 2**32 - 1)
            random.seed(seed)
            np.random.seed(seed % (2**32 - 1))
            transformed = pipeline(image=image, bboxes=albu_boxes, class_ids=class_ids)
            next_boxes = albu_to_yolo(list(transformed["bboxes"]), list(transformed["class_ids"]))
            if not next_boxes:
                continue
            name = f"{stem}_alb_{index + 1}"
            cv2.imwrite(str(output_root / "images" / "train" / f"{name}.jpg"), transformed["image"])
            write_boxes(output_root / "labels" / "train" / f"{name}.txt", next_boxes)
            created += 1
    return created


def xyxy_from_yolo(box: YoloBox, width: int, height: int) -> tuple[float, float, float, float]:
    """Convert YOLO normalized box to pixel xyxy."""

    box_width = box.width * width
    box_height = box.height * height
    x1 = box.x_center * width - box_width / 2
    y1 = box.y_center * height - box_height / 2
    x2 = x1 + box_width
    y2 = y1 + box_height
    return x1, y1, x2, y2


def yolo_from_xyxy(class_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> YoloBox | None:
    """Convert clipped pixel xyxy to YOLO format."""

    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 1.0 or box_height <= 1.0:
        return None
    return YoloBox(
        class_id=class_id,
        x_center=((x1 + x2) / 2) / width,
        y_center=((y1 + y2) / 2) / height,
        width=box_width / width,
        height=box_height / height,
    )


def resize_with_boxes(image_path: Path, label_path: Path, size: int) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Resize an image to a square and return pixel boxes."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read {image_path}")
    source_h, source_w = image.shape[:2]
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    pixel_boxes = []
    for box in read_boxes(label_path):
        x1, y1, x2, y2 = xyxy_from_yolo(box, source_w, source_h)
        pixel_boxes.append(
            (
                box.class_id,
                x1 * size / source_w,
                y1 * size / source_h,
                x2 * size / source_w,
                y2 * size / source_h,
            )
        )
    return resized, pixel_boxes


def augment_mosaic(dataset_root: Path, output_root: Path, count: int, size: int, rng: random.Random) -> int:
    """Create Mosaic training samples."""

    if count <= 0:
        return 0
    images = find_images(dataset_root / "images" / "train")
    labels = find_labels(dataset_root / "labels" / "train")
    samples = [(stem, images[stem], labels[stem]) for stem in sorted(set(images) & set(labels))]
    if len(samples) < 4:
        return 0

    created = 0
    tile = size // 2
    for index in range(count):
        chosen = rng.sample(samples, 4)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        boxes: list[YoloBox] = []
        offsets = [(0, 0), (tile, 0), (0, tile), (tile, tile)]
        for (_, image_path, label_path), (offset_x, offset_y) in zip(chosen, offsets, strict=True):
            image, pixel_boxes = resize_with_boxes(image_path, label_path, tile)
            canvas[offset_y : offset_y + tile, offset_x : offset_x + tile] = image
            for class_id, x1, y1, x2, y2 in pixel_boxes:
                next_box = yolo_from_xyxy(class_id, x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y, size, size)
                if next_box:
                    boxes.append(next_box)
        if not boxes:
            continue
        name = f"mosaic_{index + 1:05d}"
        cv2.imwrite(str(output_root / "images" / "train" / f"{name}.jpg"), canvas)
        write_boxes(output_root / "labels" / "train" / f"{name}.txt", boxes)
        created += 1
    return created


def augment_mixup(dataset_root: Path, output_root: Path, count: int, size: int, rng: random.Random) -> int:
    """Create MixUp training samples."""

    if count <= 0:
        return 0
    images = find_images(dataset_root / "images" / "train")
    labels = find_labels(dataset_root / "labels" / "train")
    samples = [(stem, images[stem], labels[stem]) for stem in sorted(set(images) & set(labels))]
    if len(samples) < 2:
        return 0

    created = 0
    for index in range(count):
        first, second = rng.sample(samples, 2)
        first_image, first_boxes = resize_with_boxes(first[1], first[2], size)
        second_image, second_boxes = resize_with_boxes(second[1], second[2], size)
        mixed = cv2.addWeighted(first_image, 0.55, second_image, 0.45, 0.0)
        boxes: list[YoloBox] = []
        for class_id, x1, y1, x2, y2 in first_boxes + second_boxes:
            next_box = yolo_from_xyxy(class_id, x1, y1, x2, y2, size, size)
            if next_box:
                boxes.append(next_box)
        if not boxes:
            continue
        name = f"mixup_{index + 1:05d}"
        cv2.imwrite(str(output_root / "images" / "train" / f"{name}.jpg"), mixed)
        write_boxes(output_root / "labels" / "train" / f"{name}.txt", boxes)
        created += 1
    return created


def write_report(path: Path, stats: AugmentStats) -> None:
    """Write augmentation JSON report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    """Create an offline augmented dataset."""

    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output)
    rng = random.Random(args.seed)

    clean_output(output_root)
    copied_train = copy_split(dataset_root, output_root, "train")
    copied_val = copy_split(dataset_root, output_root, "val")
    albumentations_count = (
        0
        if args.disable_albumentations
        else augment_with_albumentations(dataset_root, output_root, args.copies, rng, args.imgsz)
    )
    mosaic_count = 0 if args.disable_mosaic else augment_mosaic(dataset_root, output_root, args.mosaic_count, args.imgsz, rng)
    mixup_count = 0 if args.disable_mixup else augment_mixup(dataset_root, output_root, args.mixup_count, args.imgsz, rng)
    write_data_yaml(output_root)

    stats = AugmentStats(
        copied_train=copied_train,
        copied_val=copied_val,
        albumentations_images=albumentations_count,
        mosaic_images=mosaic_count,
        mixup_images=mixup_count,
        output=str(output_root),
    )
    write_report(Path(args.report_json), stats)
    print(f"Augmented dataset: {output_root}")
    print(f"Train originals: {copied_train}")
    print(f"Val originals: {copied_val}")
    print(f"Albumentations: {albumentations_count}")
    print(f"Mosaic: {mosaic_count}")
    print(f"MixUp: {mixup_count}")
    print(f"Report: {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
