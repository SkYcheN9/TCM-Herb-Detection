"""Export the finished TCM-SliceAI model for Raspberry Pi deployment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


PREFERRED_WEIGHTS: tuple[Path, ...] = (
    ROOT / "final_results_full" / "reports" / "ablation" / "runs" / "baseline_ghostconv" / "weights" / "best.pt",
    ROOT / "reports" / "ablation" / "runs" / "baseline_ghostconv" / "weights" / "best.pt",
    ROOT / "runs" / "ghostconv" / "weights" / "best.pt",
    ROOT / "final_results_full" / "reports" / "ablation" / "runs" / "baseline_cbam_bifpn" / "weights" / "best.pt",
    ROOT / "runs" / "cbam_bifpn" / "weights" / "best.pt",
    ROOT / "runs" / "baseline" / "weights" / "best.pt",
    ROOT / "reports" / "ablation" / "runs" / "baseline" / "weights" / "best.pt",
    ROOT / "runs" / "detect" / "runs" / "baseline" / "weights" / "best.pt",
)


def parse_args() -> argparse.Namespace:
    """Parse export options."""

    parser = argparse.ArgumentParser(
        description="Export best.pt, best.onnx, best.torchscript, best_openvino/ and best_ncnn_model/."
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Source .pt weights. Defaults to the best available trained model.",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory. Default creates best.pt, best.onnx and best_openvino/ in project root.",
    )
    parser.add_argument("--imgsz", type=int, default=416, help="Export image size for Pi 5 real-time inference.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--dynamic", action="store_true", help="Export dynamic input shapes.")
    parser.add_argument("--half", action="store_true", help="Export FP16 where the backend supports it.")
    parser.add_argument("--skip-openvino", action="store_true", help="Only export best.pt and best.onnx.")
    parser.add_argument("--skip-onnx", action="store_true", help="Only export best.pt and best_openvino/.")
    parser.add_argument("--skip-torchscript", action="store_true", help="Skip TorchScript export.")
    parser.add_argument("--skip-ncnn", action="store_true", help="Skip NCNN export.")
    return parser.parse_args()


def resolve_weights(requested: str | None) -> Path:
    """Resolve the finished training weights."""

    if requested:
        path = Path(requested)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return path
        raise FileNotFoundError(f"Model weights not found: {path}")

    for path in PREFERRED_WEIGHTS:
        if path.is_file():
            return path

    candidates = sorted(
        ROOT.glob("runs/**/weights/best.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if "smoke" not in str(path).lower():
            return path
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No best.pt found. Train the model first or pass --weights.")


def register_project_modules() -> None:
    """Register custom YOLO modules before loading custom checkpoints."""

    try:
        from models.losses import register_focal_loss
        from models.modules import register_ultralytics_modules

        register_ultralytics_modules(enable_cbam=True, enable_bifpn=True, enable_decoupled_head=True)
        register_focal_loss()
    except Exception as exc:  # pragma: no cover - only needed for custom checkpoints.
        print(f"Warning: project module registration skipped: {exc}")


def clean_path(path: Path) -> None:
    """Remove an existing generated artifact."""

    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def move_exported_path(exported: str | Path, target: Path) -> Path:
    """Move an Ultralytics export result to the requested deterministic path."""

    exported_path = Path(exported)
    if not exported_path.is_absolute():
        exported_path = ROOT / exported_path
    if exported_path.resolve() == target.resolve():
        return target
    clean_path(target)
    shutil.move(str(exported_path), str(target))
    return target


def export_model(args: argparse.Namespace) -> dict[str, Any]:
    """Export the selected weights to Raspberry Pi deployment formats."""

    from ultralytics import YOLO

    register_project_modules()
    weights = resolve_weights(args.weights)
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    best_pt = output_dir / "best.pt"
    best_onnx = output_dir / "best.onnx"
    best_torchscript = output_dir / "best.torchscript"
    best_openvino = output_dir / "best_openvino"
    best_ncnn = output_dir / "best_ncnn_model"

    if weights.resolve() != best_pt.resolve():
        shutil.copy2(weights, best_pt)

    model = YOLO(str(best_pt))
    artifacts: dict[str, Any] = {
        "source_weights": str(weights),
        "best_pt": str(best_pt),
        "imgsz": args.imgsz,
        "dynamic": bool(args.dynamic),
        "half": bool(args.half),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
    }

    if not args.skip_onnx:
        exported_onnx = model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=args.dynamic,
            half=args.half,
            simplify=True,
            nms=False,
        )
        artifacts["best_onnx"] = str(move_exported_path(exported_onnx, best_onnx))

    if not args.skip_torchscript:
        exported_torchscript = model.export(
            format="torchscript",
            imgsz=args.imgsz,
            optimize=False,
            half=args.half,
        )
        artifacts["best_torchscript"] = str(move_exported_path(exported_torchscript, best_torchscript))

    if not args.skip_openvino:
        exported_openvino = model.export(
            format="openvino",
            imgsz=args.imgsz,
            dynamic=args.dynamic,
            half=args.half,
            int8=False,
        )
        artifacts["best_openvino"] = str(move_exported_path(exported_openvino, best_openvino))

    if not args.skip_ncnn:
        exported_ncnn = model.export(
            format="ncnn",
            imgsz=args.imgsz,
            half=args.half,
        )
        artifacts["best_ncnn"] = str(move_exported_path(exported_ncnn, best_ncnn))

    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifacts["manifest"] = str(manifest_path)
    return artifacts


def main() -> int:
    """Run model export."""

    args = parse_args()
    artifacts = export_model(args)
    print("Raspberry Pi export completed:")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
