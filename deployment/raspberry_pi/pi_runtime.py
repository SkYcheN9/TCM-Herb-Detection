"""Lightweight YOLO runtime used by Raspberry Pi benchmark and web deployment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_NAMES: list[str] = [
    "zexie",
    "niuxi",
    "gaoliangjiang",
    "mudanpi",
    "yuzhu",
    "baizhi",
    "baishao",
    "dazao",
    "danshen",
    "gancao",
    "baixianpi",
    "baihe",
    "sangzhi",
    "jiegeng",
    "banlangen",
]


@dataclass(frozen=True)
class Detection:
    """Single detection in original-frame coordinates."""

    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class DetectionResult:
    """Detection result and timing."""

    detections: list[Detection]
    elapsed_ms: float

    @property
    def fps(self) -> float:
        """Return frames per second for this result."""

        return 1000.0 / self.elapsed_ms if self.elapsed_ms > 0 else 0.0


@dataclass(frozen=True)
class LetterboxMeta:
    """Preprocessing metadata needed to map boxes back to the source image."""

    ratio: float
    pad_x: float
    pad_y: float
    source_width: int
    source_height: int


class YoloDetector:
    """YOLO detector with PyTorch, ONNX Runtime and OpenVINO backends."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        backend: str,
        imgsz: int = 416,
        conf: float = 0.25,
        iou: float = 0.45,
        class_names: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.backend = self._resolve_backend(backend, self.model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.class_names = class_names or CLASS_NAMES
        self._session: Any = None
        self._input_name: str | None = None
        self._compiled_model: Any = None
        self._openvino_output: Any = None
        self._model: Any = None
        self._load()

    @staticmethod
    def _resolve_backend(backend: str, model_path: Path) -> str:
        if backend != "auto":
            return backend.lower()
        if model_path.suffix == ".pt":
            return "pytorch"
        if model_path.suffix == ".onnx":
            return "onnx"
        if model_path.is_dir():
            return "openvino"
        raise ValueError(f"Unable to infer backend from {model_path}")

    def _load(self) -> None:
        if self.backend == "onnx":
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            return

        if self.backend == "openvino":
            try:
                from openvino import Core
            except ImportError:  # pragma: no cover - older OpenVINO releases.
                from openvino.runtime import Core

            model_xml = self.model_path
            if self.model_path.is_dir():
                xml_files = sorted(self.model_path.glob("*.xml"))
                if not xml_files:
                    raise FileNotFoundError(f"No OpenVINO .xml found in {self.model_path}")
                model_xml = xml_files[0]
            core = Core()
            model = core.read_model(str(model_xml))
            self._compiled_model = core.compile_model(model, "CPU")
            self._openvino_output = self._compiled_model.outputs[0]
            return

        if self.backend == "pytorch":
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
            return

        raise ValueError(f"Unsupported backend: {self.backend}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a BGR frame."""

        started = time.perf_counter()
        if self.backend == "pytorch":
            detections = self._detect_pytorch(frame)
        else:
            tensor, meta = preprocess(frame, self.imgsz)
            output = self._infer_tensor(tensor)
            detections = postprocess(
                output=output,
                meta=meta,
                conf_threshold=self.conf,
                iou_threshold=self.iou,
                class_names=self.class_names,
            )
        elapsed_ms = max((time.perf_counter() - started) * 1000.0, 0.001)
        return DetectionResult(detections=detections, elapsed_ms=elapsed_ms)

    def _infer_tensor(self, tensor: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            outputs = self._session.run(None, {self._input_name: tensor})
            return np.asarray(outputs[0])

        if self.backend == "openvino":
            outputs = self._compiled_model([tensor])
            return np.asarray(outputs[self._openvino_output])

        raise ValueError(f"Unsupported direct inference backend: {self.backend}")

    def _detect_pytorch(self, frame: np.ndarray) -> list[Detection]:
        result = self._model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device="cpu",
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.cls is None:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
        confidences = boxes.conf.detach().cpu().numpy().tolist()
        names = getattr(result, "names", {}) or {}

        detections: list[Detection] = []
        for bbox, class_id, confidence in zip(xyxy, class_ids, confidences, strict=False):
            name = str(names.get(class_id, class_id)) if isinstance(names, dict) else class_name(class_id)
            detections.append(
                Detection(
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    class_id=int(class_id),
                    class_name=name,
                    confidence=float(confidence),
                )
            )
        return detections


def preprocess(frame: np.ndarray, imgsz: int) -> tuple[np.ndarray, LetterboxMeta]:
    """Letterbox and normalize a BGR image for YOLO inference."""

    source_height, source_width = frame.shape[:2]
    ratio = min(imgsz / source_width, imgsz / source_height)
    resized_width = int(round(source_width * ratio))
    resized_height = int(round(source_height * ratio))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_x = (imgsz - resized_width) // 2
    pad_y = (imgsz - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    meta = LetterboxMeta(
        ratio=ratio,
        pad_x=float(pad_x),
        pad_y=float(pad_y),
        source_width=source_width,
        source_height=source_height,
    )
    return np.ascontiguousarray(tensor), meta


def postprocess(
    *,
    output: np.ndarray,
    meta: LetterboxMeta,
    conf_threshold: float,
    iou_threshold: float,
    class_names: list[str],
) -> list[Detection]:
    """Convert raw YOLO output to filtered detections."""

    pred = np.asarray(output)
    while pred.ndim > 2:
        pred = pred[0]
    if pred.ndim != 2:
        return []

    if pred.shape[0] < pred.shape[1] and pred.shape[0] <= len(class_names) + 5:
        pred = pred.T

    if pred.shape[1] == 6:
        boxes_xyxy = pred[:, :4]
        scores = pred[:, 4]
        class_ids = pred[:, 5].astype(int)
    elif pred.shape[1] >= len(class_names) + 5:
        boxes_xywh = pred[:, :4]
        objectness = pred[:, 4:5]
        class_scores = pred[:, 5 : 5 + len(class_names)] * objectness
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    elif pred.shape[1] >= len(class_names) + 4:
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4 : 4 + len(class_names)]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    else:
        return []

    keep = scores >= conf_threshold
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    if len(scores) == 0:
        return []

    boxes_xyxy = scale_boxes(boxes_xyxy, meta)
    keep_indices = nms(boxes_xyxy, scores, class_ids, iou_threshold)

    detections: list[Detection] = []
    for index in keep_indices:
        class_id = int(class_ids[index])
        detections.append(
            Detection(
                bbox=tuple(float(value) for value in boxes_xyxy[index]),
                class_id=class_id,
                class_name=class_name(class_id, class_names),
                confidence=float(scores[index]),
            )
        )
    return detections


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert center x/y/width/height boxes to x1/y1/x2/y2."""

    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def scale_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Map letterboxed boxes back to the original frame."""

    scaled = boxes.copy()
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - meta.pad_x) / meta.ratio
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - meta.pad_y) / meta.ratio
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, meta.source_width - 1)
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, meta.source_height - 1)
    return scaled


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Class-aware non-maximum suppression."""

    selected: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size > 0:
            current = int(order[0])
            selected.append(current)
            if order.size == 1:
                break
            overlaps = box_iou(boxes[current], boxes[order[1:]])
            order = order[1:][overlaps <= iou_threshold]
    selected.sort(key=lambda item: float(scores[item]), reverse=True)
    return selected


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and many boxes."""

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max((box[2] - box[0]) * (box[3] - box[1]), 0)
    area_b = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 0)
    return intersection / np.maximum(area_a + area_b - intersection, 1e-6)


def class_name(class_id: int, class_names: list[str] | None = None) -> str:
    """Return a class label for the fixed project class order."""

    names = class_names or CLASS_NAMES
    if 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)


def annotate_frame(frame: np.ndarray, detections: list[Detection], fps: float | None = None) -> np.ndarray:
    """Draw detections and FPS on a BGR frame."""

    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
        color = color_for_class(detection.class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        draw_label(annotated, label, x1, max(y1 - 8, 0), color)

    if fps is not None:
        draw_label(annotated, f"FPS {fps:.1f}", 12, 28, (32, 156, 238))
    return annotated


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Draw a readable text label."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    y1 = max(y - height - baseline, 0)
    cv2.rectangle(frame, (x, y1), (x + width + 8, y + baseline), color, -1)
    cv2.putText(frame, text, (x + 4, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color for a class id."""

    palette = [
        (20, 184, 166),
        (16, 185, 129),
        (14, 165, 233),
        (245, 158, 11),
        (239, 68, 68),
        (132, 204, 22),
        (99, 102, 241),
        (217, 119, 6),
    ]
    return palette[class_id % len(palette)]
