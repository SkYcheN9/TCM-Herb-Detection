# Phase 1 Baseline Summary

## Environment

- Python environment: `.venv`
- PyTorch: `2.12.1+cu126`
- Training device: `CUDA:0`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Framework: `Ultralytics YOLOv8`

## Dataset

- Raw images: 1049
- Raw labels: 1048
- Missing labels: 1
- Label files with out-of-range classes: 84
- Prepared valid samples: 964
- Train samples: 771
- Val samples: 193
- Class order: fixed 15-class order from project spec

## Training

- Model: `yolov8n.pt`
- Epochs: 100
- Image size: 640
- Batch size: 16
- Output directory: `runs/baseline`

## Best Validation Result

- Best epoch by mAP50-95: 51
- Precision: 0.94926
- Recall: 0.91324
- mAP50: 0.94042
- mAP50-95: 0.73898

## Final Epoch Result

- Epoch: 100
- Precision: 0.94537
- Recall: 0.90156
- mAP50: 0.91438
- mAP50-95: 0.71305
