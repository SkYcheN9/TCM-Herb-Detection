"""Detection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..config import get_settings
from ..constants import CHINESE_CLASS_NAMES, CLASS_NAMES
from ..database import get_db
from ..schemas import (
    ClassInfo,
    DetectApiResponse,
    DetectBatchResponse,
    DetectImageResponse,
    DetectionApiItem,
)
from ..services.detector import (
    detect_image_file,
    detect_video_file,
    save_upload_file,
    save_video_upload_file,
)


router = APIRouter(prefix="/detect", tags=["detect"])


@router.get(
    "/classes",
    response_model=list[ClassInfo],
    summary="List supported classes",
)
def list_classes() -> list[ClassInfo]:
    """Return the fixed YOLO class order used by the project."""

    return [
        ClassInfo(
            id=index,
            name=name,
            chinese_name=CHINESE_CLASS_NAMES.get(name, name),
        )
        for index, name in enumerate(CLASS_NAMES)
    ]


@router.post(
    "/image",
    response_model=DetectApiResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
    summary="Detect objects in an uploaded image",
)
async def detect_image(
    file: UploadFile = File(..., description="Image file to detect"),
    model_path: str | None = Form(default=None, description="Optional model path"),
    confidence: float | None = Form(default=None, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float | None = Form(default=None, ge=0.0, le=1.0, description="NMS IoU threshold"),
    image_size: int | None = Form(default=None, ge=32, le=2048, description="Inference image size"),
    device: str = Form(default="auto", description="auto, cpu or cuda"),
    save_annotated: bool = Form(default=True, description="Save annotated result image"),
    db: Session = Depends(get_db),
) -> DetectApiResponse:
    """Return bbox, class, confidence and count for one image."""

    settings = get_settings()
    try:
        image_path = await save_upload_file(file)
        result = detect_image_file(
            db,
            image_path=image_path,
            model_path=model_path,
            confidence=confidence if confidence is not None else settings.default_confidence,
            iou=iou if iou is not None else settings.default_iou,
            image_size=image_size if image_size is not None else settings.default_image_size,
            device_mode=device,
            save_annotated=save_annotated,
            persist=True,
        )
        return _to_detect_api_response(result)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post(
    "/video",
    response_model=DetectApiResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
    summary="Detect objects in an uploaded video",
)
async def detect_video(
    file: UploadFile = File(..., description="Video file to detect"),
    model_path: str | None = Form(default=None, description="Optional model path"),
    confidence: float | None = Form(default=None, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float | None = Form(default=None, ge=0.0, le=1.0, description="NMS IoU threshold"),
    image_size: int | None = Form(default=None, ge=32, le=2048, description="Inference image size"),
    device: str = Form(default="auto", description="auto, cpu or cuda"),
    frame_stride: int = Form(default=1, ge=1, description="Process every Nth frame"),
    max_frames: int | None = Form(default=None, ge=1, description="Maximum processed frames"),
    save_annotated: bool = Form(default=True, description="Save annotated result video"),
    db: Session = Depends(get_db),
) -> DetectApiResponse:
    """Return bbox, class, confidence and count for one video."""

    settings = get_settings()
    try:
        video_path = await save_video_upload_file(file)
        return detect_video_file(
            db,
            video_path=video_path,
            model_path=model_path,
            confidence=confidence if confidence is not None else settings.default_confidence,
            iou=iou if iou is not None else settings.default_iou,
            image_size=image_size if image_size is not None else settings.default_image_size,
            device_mode=device,
            frame_stride=frame_stride,
            max_frames=max_frames,
            save_annotated=save_annotated,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post(
    "/batch",
    response_model=DetectBatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Detect objects in uploaded images",
)
async def detect_batch(
    files: list[UploadFile] = File(..., description="Image files to detect"),
    model_path: str | None = Form(default=None, description="Optional model path"),
    confidence: float | None = Form(default=None, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float | None = Form(default=None, ge=0.0, le=1.0, description="NMS IoU threshold"),
    image_size: int | None = Form(default=None, ge=32, le=2048, description="Inference image size"),
    device: str = Form(default="auto", description="auto, cpu or cuda"),
    save_annotated: bool = Form(default=True, description="Save annotated result image"),
    db: Session = Depends(get_db),
) -> DetectBatchResponse:
    """Detect multiple uploaded images and persist every successful result."""

    settings = get_settings()
    records: list[DetectImageResponse] = []
    errors: list[str] = []
    for file in files:
        try:
            image_path = await save_upload_file(file)
            records.append(
                detect_image_file(
                    db,
                    image_path=image_path,
                    model_path=model_path,
                    confidence=confidence if confidence is not None else settings.default_confidence,
                    iou=iou if iou is not None else settings.default_iou,
                    image_size=image_size if image_size is not None else settings.default_image_size,
                    device_mode=device,
                    save_annotated=save_annotated,
                    persist=True,
                )
            )
        except Exception as exc:
            errors.append(f"{file.filename}: {exc}")

    return DetectBatchResponse(
        total_files=len(files),
        success_count=len(records),
        failed_count=len(errors),
        records=records,
        errors=errors,
    )


def _to_detect_api_response(result: DetectImageResponse) -> DetectApiResponse:
    """Return the minimal detection API shape."""

    chinese_counts = {
        CHINESE_CLASS_NAMES.get(class_name, class_name): count
        for class_name, count in result.class_counts.items()
    }
    return DetectApiResponse(
        count=result.total_count,
        class_counts=result.class_counts,
        chinese_class_counts=chinese_counts,
        image_width=result.image_width,
        image_height=result.image_height,
        detections=[
            DetectionApiItem(
                bbox=item.bbox,
                class_=item.class_name,
                confidence=item.confidence,
            )
            for item in result.detections
        ],
    )
