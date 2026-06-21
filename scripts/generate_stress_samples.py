"""Generate complex interference samples for acceptance testing.

The generated images are meant for robustness checks before the live
acceptance demo. They are synthetic test cases, not automatically trusted
training labels. If these samples are later used for training, review the
generated labels first.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcm_slice_ai.constants import CLASS_NAMES
from tcm_slice_ai.dataset import find_images, find_labels


GANCao_CLASS_ID = CLASS_NAMES.index("gancao")
JIEGENG_CLASS_ID = CLASS_NAMES.index("jiegeng")


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class ObjectCrop:
    class_id: int
    image: np.ndarray
    alpha: np.ndarray
    source: str


@dataclass(frozen=True)
class GeneratedSample:
    image: str
    label: str
    scenario: str
    object_count: int
    class_ids: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate complex TCM slice stress-test samples.")
    parser.add_argument("--images", default="data/images", help="Raw image directory.")
    parser.add_argument("--labels", default="data/labels", help="Raw YOLO label directory.")
    parser.add_argument("--output", default="reports/stress_test_samples", help="Output directory.")
    parser.add_argument("--count", type=int, default=60, help="Number of synthetic test images.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-crops-per-class", type=int, default=140)
    return parser.parse_args()


def read_boxes(path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            continue
        if 0 <= class_id < len(CLASS_NAMES) and width > 0 and height > 0:
            boxes.append(YoloBox(class_id, x_center, y_center, width, height))
    return boxes


def write_boxes(path: Path, boxes: list[YoloBox]) -> None:
    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
        if box.width > 0 and box.height > 0
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def xyxy_from_yolo(box: YoloBox, width: int, height: int) -> tuple[int, int, int, int]:
    box_width = box.width * width
    box_height = box.height * height
    x1 = round(box.x_center * width - box_width / 2)
    y1 = round(box.y_center * height - box_height / 2)
    x2 = round(x1 + box_width)
    y2 = round(y1 + box_height)
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


def yolo_from_xyxy(class_id: int, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> YoloBox | None:
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < 6 or box_height < 6:
        return None
    return YoloBox(
        class_id=class_id,
        x_center=((x1 + x2) / 2) / width,
        y_center=((y1 + y2) / 2) / height,
        width=box_width / width,
        height=box_height / height,
    )


def clean_output(output_root: Path) -> tuple[Path, Path]:
    image_dir = output_root / "images"
    label_dir = output_root / "labels"
    for directory in (image_dir, label_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()
    return image_dir, label_dir


def collect_crop_metadata(image_dir: Path, label_dir: Path, rng: random.Random, max_per_class: int) -> list[tuple[int, Path, YoloBox]]:
    images = find_images(image_dir)
    labels = find_labels(label_dir)
    grouped: dict[int, list[tuple[int, Path, YoloBox]]] = defaultdict(list)
    for stem in sorted(set(images) & set(labels)):
        for box in read_boxes(labels[stem]):
            grouped[box.class_id].append((box.class_id, images[stem], box))

    selected: list[tuple[int, Path, YoloBox]] = []
    for class_id, records in grouped.items():
        rng.shuffle(records)
        selected.extend(records[:max_per_class])
    rng.shuffle(selected)
    return selected


def load_object_crops(image_dir: Path, label_dir: Path, rng: random.Random, max_per_class: int) -> dict[int, list[ObjectCrop]]:
    records = collect_crop_metadata(image_dir, label_dir, rng, max_per_class)
    crops_by_class: dict[int, list[ObjectCrop]] = defaultdict(list)
    image_cache: dict[Path, np.ndarray] = {}

    for class_id, image_path, box in records:
        image = image_cache.get(image_path)
        if image is None:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image_cache[image_path] = image

        height, width = image.shape[:2]
        x1, y1, x2, y2 = xyxy_from_yolo(box, width, height)
        pad = max(6, round(max(x2 - x1, y2 - y1) * 0.08))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        if x2 - x1 < 18 or y2 - y1 < 18:
            continue

        crop = image[y1:y2, x1:x2].copy()
        alpha = build_foreground_alpha(crop)
        crop, alpha = trim_to_alpha(crop, alpha)
        if crop.shape[0] < 14 or crop.shape[1] < 14:
            continue
        crops_by_class[class_id].append(ObjectCrop(class_id, crop, alpha, image_path.name))

    return dict(crops_by_class)


def build_foreground_alpha(crop: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    border_width = max(2, min(height, width) // 12)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate(
        [
            lab[:border_width, :, :].reshape(-1, 3),
            lab[-border_width:, :, :].reshape(-1, 3),
            lab[:, :border_width, :].reshape(-1, 3),
            lab[:, -border_width:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    delta = np.linalg.norm(lab - background, axis=2)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((delta > 20) | ((hsv[:, :, 1] > 25) & (hsv[:, :, 2] < 248))).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground_ratio = float(np.count_nonzero(mask)) / max(mask.size, 1)
    if foreground_ratio < 0.015 or foreground_ratio > 0.92:
        mask = np.full((height, width), 255, dtype=np.uint8)
    else:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def trim_to_alpha(image: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = cv2.findNonZero((alpha > 8).astype(np.uint8))
    if coords is None:
        return image, alpha
    x, y, width, height = cv2.boundingRect(coords)
    margin = 3
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + width + margin)
    y2 = min(image.shape[0], y + height + margin)
    return image[y1:y2, x1:x2].copy(), alpha[y1:y2, x1:x2].copy()


def make_background(width: int, height: int, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    style = rng.choice(["paper", "cloth", "table", "clutter"])
    base_colors = {
        "paper": np.array([214, 218, 216], dtype=np.float32),
        "cloth": np.array([180, 190, 184], dtype=np.float32),
        "table": np.array([168, 150, 126], dtype=np.float32),
        "clutter": np.array([190, 196, 188], dtype=np.float32),
    }
    base = np.tile(base_colors[style], (height, width, 1))
    noise = np_rng.normal(0, 10 if style != "clutter" else 16, (height, width, 1))
    background = np.clip(base + noise, 0, 255).astype(np.uint8)

    yy, xx = np.indices((height, width))
    if style == "cloth":
        pattern = (np.sin(xx / 13.0) + np.sin((xx + yy) / 23.0)) * 5
        background = np.clip(background.astype(np.float32) + pattern[..., None], 0, 255).astype(np.uint8)
    elif style == "table":
        grain = np.sin(xx / 19.0 + np_rng.normal(0, 0.1, (height, width))) * 9
        background = np.clip(background.astype(np.float32) + grain[..., None], 0, 255).astype(np.uint8)
    elif style == "clutter":
        for _ in range(rng.randint(5, 10)):
            color = tuple(int(value) for value in np_rng.integers(90, 230, size=3))
            pt1 = (rng.randint(0, width - 1), rng.randint(0, height - 1))
            pt2 = (rng.randint(0, width - 1), rng.randint(0, height - 1))
            cv2.line(background, pt1, pt2, color, rng.randint(2, 8), cv2.LINE_AA)
        for _ in range(rng.randint(2, 5)):
            color = tuple(int(value) for value in np_rng.integers(120, 230, size=3))
            center = (rng.randint(0, width - 1), rng.randint(0, height - 1))
            radius = rng.randint(24, 90)
            cv2.circle(background, center, radius, color, -1, cv2.LINE_AA)

    vignette = 1.0 - 0.22 * (((xx - width / 2) / width) ** 2 + ((yy - height / 2) / height) ** 2)
    return np.clip(background.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)


def choose_objects(
    crops_by_class: dict[int, list[ObjectCrop]],
    scenario_index: int,
    rng: random.Random,
) -> list[ObjectCrop]:
    available_classes = [class_id for class_id, crops in crops_by_class.items() if crops]
    if not available_classes:
        return []

    objects: list[ObjectCrop] = []
    if scenario_index % 4 == 0:
        for class_id in (GANCao_CLASS_ID, JIEGENG_CLASS_ID):
            if crops_by_class.get(class_id):
                objects.append(rng.choice(crops_by_class[class_id]))

    target_count = rng.randint(4, 8)
    while len(objects) < target_count:
        class_id = rng.choice(available_classes)
        objects.append(rng.choice(crops_by_class[class_id]))
    rng.shuffle(objects)
    return objects


def transform_object(crop: ObjectCrop, scenario: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    scale_range = (0.48, 0.82) if scenario == "stacking" else (0.55, 1.1)
    scale = rng.uniform(*scale_range)
    angle = rng.uniform(-42, 42)
    image = cv2.resize(crop.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(crop.alpha, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    next_width = max(1, int((height * sin_value) + (width * cos_value)))
    next_height = max(1, int((height * cos_value) + (width * sin_value)))
    matrix[0, 2] += (next_width / 2) - center[0]
    matrix[1, 2] += (next_height / 2) - center[1]
    rotated = cv2.warpAffine(image, matrix, (next_width, next_height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rotated_alpha = cv2.warpAffine(alpha, matrix, (next_width, next_height), flags=cv2.INTER_LINEAR, borderValue=0)
    return trim_to_alpha(rotated, rotated_alpha)


def paste_object(canvas: np.ndarray, image: np.ndarray, alpha: np.ndarray, x: int, y: int) -> tuple[int, int, int, int] | None:
    height, width = canvas.shape[:2]
    obj_height, obj_width = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + obj_width)
    y2 = min(height, y + obj_height)
    if x2 <= x1 or y2 <= y1:
        return None

    obj_x1 = x1 - x
    obj_y1 = y1 - y
    obj_x2 = obj_x1 + (x2 - x1)
    obj_y2 = obj_y1 + (y2 - y1)
    alpha_crop = alpha[obj_y1:obj_y2, obj_x1:obj_x2].astype(np.float32) / 255.0
    if np.count_nonzero(alpha_crop > 0.08) < 20:
        return None

    alpha_3 = alpha_crop[..., None]
    canvas[y1:y2, x1:x2] = (
        image[obj_y1:obj_y2, obj_x1:obj_x2].astype(np.float32) * alpha_3
        + canvas[y1:y2, x1:x2].astype(np.float32) * (1.0 - alpha_3)
    ).astype(np.uint8)

    coords = cv2.findNonZero((alpha_crop > 0.08).astype(np.uint8))
    if coords is None:
        return None
    bx, by, bw, bh = cv2.boundingRect(coords)
    return x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh


def place_objects(
    canvas: np.ndarray,
    objects: list[ObjectCrop],
    scenario: str,
    rng: random.Random,
) -> list[YoloBox]:
    height, width = canvas.shape[:2]
    boxes: list[YoloBox] = []
    center_x = rng.randint(width // 3, width * 2 // 3)
    center_y = rng.randint(height // 3, height * 2 // 3)

    for index, crop in enumerate(objects):
        image, alpha = transform_object(crop, scenario, rng)
        obj_height, obj_width = image.shape[:2]
        if obj_width >= width or obj_height >= height:
            continue

        if scenario == "stacking":
            x = center_x + rng.randint(-150, 150) - obj_width // 2
            y = center_y + rng.randint(-110, 110) - obj_height // 2
        elif scenario == "mixed_gancao_jiegeng" and crop.class_id in (GANCao_CLASS_ID, JIEGENG_CLASS_ID):
            x = width // 5 + index * rng.randint(20, 45) - obj_width // 2
            y = height * 3 // 4 + rng.randint(-40, 35) - obj_height // 2
        else:
            x = rng.randint(16, max(17, width - obj_width - 16))
            y = rng.randint(16, max(17, height - obj_height - 16))

        pasted_box = paste_object(canvas, image, alpha, x, y)
        if pasted_box is None:
            continue
        yolo_box = yolo_from_xyxy(crop.class_id, *pasted_box, width, height)
        if yolo_box is not None:
            boxes.append(yolo_box)
    return boxes


def add_occlusion(canvas: np.ndarray, boxes: list[YoloBox], rng: random.Random) -> None:
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    for box in rng.sample(boxes, k=min(len(boxes), rng.randint(1, 3))):
        x1, y1, x2, y2 = xyxy_from_yolo(box, width, height)
        if rng.random() < 0.55:
            strip_width = rng.randint(18, 56)
            pt1 = (max(0, x1 - strip_width), rng.randint(y1, max(y1, y2 - 1)))
            pt2 = (min(width - 1, x2 + strip_width), rng.randint(y1, max(y1, y2 - 1)))
            color = rng.choice([(232, 226, 213), (86, 78, 66), (194, 178, 150)])
            cv2.line(overlay, pt1, pt2, color, strip_width, cv2.LINE_AA)
        else:
            center = (rng.randint(x1, max(x1, x2 - 1)), rng.randint(y1, max(y1, y2 - 1)))
            axes = (rng.randint(18, 70), rng.randint(12, 50))
            color = rng.choice([(210, 206, 194), (80, 74, 65), (186, 150, 111)])
            cv2.ellipse(overlay, center, axes, rng.randint(0, 180), 0, 360, color, -1, cv2.LINE_AA)
    alpha = rng.uniform(0.62, 0.86)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)


def add_stains_and_dust(canvas: np.ndarray, rng: random.Random, np_rng: np.random.Generator) -> None:
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    for _ in range(rng.randint(8, 18)):
        center = (rng.randint(0, width - 1), rng.randint(0, height - 1))
        axes = (rng.randint(6, 42), rng.randint(4, 28))
        color = rng.choice([(96, 67, 38), (138, 96, 54), (78, 74, 66), (162, 135, 92)])
        cv2.ellipse(overlay, center, axes, rng.randint(0, 180), 0, 360, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, dst=canvas)

    dust_count = rng.randint(80, 180)
    xs = np_rng.integers(0, width, size=dust_count)
    ys = np_rng.integers(0, height, size=dust_count)
    for x, y in zip(xs, ys, strict=False):
        radius = rng.randint(1, 2)
        color = rng.choice([(95, 85, 73), (215, 205, 186), (64, 60, 54)])
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def apply_camera_artifacts(canvas: np.ndarray, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.randint(-7, 7)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + rng.randint(-18, 22), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + rng.randint(-18, 18), 0, 255)
    canvas = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < 0.45:
        kernel_size = rng.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (kernel_size, kernel_size), 0)
    if rng.random() < 0.35:
        kernel_size = rng.choice([5, 7, 9])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        canvas = cv2.filter2D(canvas, -1, kernel)

    noise = np_rng.normal(0, rng.uniform(2.0, 6.0), canvas.shape)
    return np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def write_data_yaml(output_root: Path) -> None:
    lines = [
        "path: .",
        "train: images",
        "val: images",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")


def write_contact_sheet(image_dir: Path, output_root: Path, sample_names: list[str]) -> None:
    thumbs: list[np.ndarray] = []
    for name in sample_names[:16]:
        image = cv2.imread(str(image_dir / name))
        if image is None:
            continue
        thumb = cv2.resize(image, (240, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(thumb, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    if not thumbs:
        return
    while len(thumbs) % 4 != 0:
        thumbs.append(np.full_like(thumbs[0], 235))
    rows = [np.hstack(thumbs[index : index + 4]) for index in range(0, len(thumbs), 4)]
    cv2.imwrite(str(output_root / "preview_contact_sheet.jpg"), np.vstack(rows))


def generate_samples(args: argparse.Namespace) -> dict[str, object]:
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    output_root = Path(args.output)
    image_output, label_output = clean_output(output_root)
    crops_by_class = load_object_crops(Path(args.images), Path(args.labels), rng, args.max_crops_per_class)
    if not any(crops_by_class.values()):
        raise RuntimeError("No valid object crops were found. Check data/images and data/labels.")

    scenarios = ["occlusion", "stacking", "stained", "complex_background", "mixed_gancao_jiegeng"]
    generated: list[GeneratedSample] = []
    class_counter: Counter[int] = Counter()

    for index in range(args.count):
        scenario = scenarios[index % len(scenarios)]
        canvas = make_background(args.width, args.height, rng, np_rng)
        objects = choose_objects(crops_by_class, index, rng)
        boxes = place_objects(canvas, objects, scenario, rng)
        if not boxes:
            continue

        if scenario in {"occlusion", "mixed_gancao_jiegeng"} or rng.random() < 0.35:
            add_occlusion(canvas, boxes, rng)
        if scenario in {"stained", "complex_background"} or rng.random() < 0.45:
            add_stains_and_dust(canvas, rng, np_rng)
        canvas = apply_camera_artifacts(canvas, rng, np_rng)

        name = f"stress_{index + 1:04d}_{scenario}.jpg"
        label_name = f"{Path(name).stem}.txt"
        cv2.imwrite(str(image_output / name), canvas)
        write_boxes(label_output / label_name, boxes)
        class_ids = [box.class_id for box in boxes]
        class_counter.update(class_ids)
        generated.append(
            GeneratedSample(
                image=f"images/{name}",
                label=f"labels/{label_name}",
                scenario=scenario,
                object_count=len(boxes),
                class_ids=class_ids,
            )
        )

    write_data_yaml(output_root)
    write_contact_sheet(image_output, output_root, [Path(sample.image).name for sample in generated])
    report = {
        "output": str(output_root),
        "count": len(generated),
        "image_size": [args.width, args.height],
        "scenarios": scenarios,
        "class_counts": {CLASS_NAMES[class_id]: count for class_id, count in sorted(class_counter.items())},
        "notes": [
            "Generated samples are for acceptance stress testing.",
            "Review labels manually before using them as training data.",
            "mixed_gancao_jiegeng cases intentionally place gancao and jiegeng close together.",
        ],
        "samples": [asdict(sample) for sample in generated],
    }
    (output_root / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    report = generate_samples(args)
    print(f"Generated {report['count']} stress-test images")
    print(f"Output: {report['output']}")
    print("Preview: preview_contact_sheet.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
