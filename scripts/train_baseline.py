"""Train the Ultralytics YOLOv8 baseline model."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tcm_slice_ai.dataset import check_split_dataset


def load_yaml_config(path: str | None) -> dict[str, Any]:
    """Load an optional training config YAML file."""

    if path is None:
        return {}
    import yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def config_value(
    args: argparse.Namespace,
    config: dict[str, Any],
    name: str,
    default: Any,
) -> Any:
    """Return CLI value unless it is unset, otherwise fall back to config."""

    value = getattr(args, name)
    if value is not None:
        return value
    return config.get(name, default)


def str_to_bool(value: str | bool | None) -> bool:
    """Parse common CLI/config boolean values."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""

    parser = argparse.ArgumentParser(description="Train YOLOv8 baseline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, or CUDA id such as 0")
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--enable-cbam",
        default=None,
        help="Enable the CBAM model path. Accepts true/false.",
    )
    parser.add_argument(
        "--enable-bifpn",
        default=None,
        help="Enable the BiFPN neck path. Accepts true/false.",
    )
    parser.add_argument(
        "--enable-focal-loss",
        default=None,
        help="Replace YOLOv8 classification BCE with Focal Loss. Accepts true/false.",
    )
    parser.add_argument(
        "--enable-ghostconv",
        default=None,
        help="Use the GhostConv lightweight backbone path. Accepts true/false.",
    )
    parser.add_argument(
        "--enable-decoupled-head",
        default=None,
        help="Use DecoupledDetect instead of the default YOLOv8 Detect head. Accepts true/false.",
    )
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument(
        "--focal-alpha",
        default=None,
        help="Focal Loss alpha in [0, 1], or none to disable alpha weighting.",
    )
    parser.add_argument(
        "--skip-dataset-check",
        action="store_true",
        help="Skip strict dataset validation before training.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    """Prefer CUDA when available and fall back to CPU."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install Phase 1 dependencies first: "
            "python -m pip install -r requirements.txt"
        ) from exc

    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def default_batch_size(
    device: str,
    requested: int | None,
    config: dict[str, Any],
) -> int:
    """Choose a conservative batch size for GPU or CPU training."""

    if requested is not None:
        return requested
    if device == "cpu":
        return int(config.get("batch_cpu", 4))
    return int(config.get("batch_gpu", 16))


def optional_float(value: Any) -> float | None:
    """Parse a float config value, accepting none/null to disable it."""

    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return float(value)


def resolve_model_and_name(
    config: dict[str, Any],
    args: argparse.Namespace,
    feature_flags: tuple[bool, bool, bool, bool],
    feature_cli_supplied: bool,
) -> tuple[str, str]:
    """Resolve the model YAML/weights and output run name for selected features."""

    enable_cbam, enable_bifpn, enable_ghostconv, enable_decoupled_head = feature_flags
    model_by_feature = {
        (False, False, False, False): "yolov8n.pt",
        (True, False, False, False): "models/yolov8n_cbam.yaml",
        (False, True, False, False): "models/yolov8n_bifpn.yaml",
        (True, True, False, False): "models/yolov8n_cbam_bifpn.yaml",
        (False, False, True, False): "models/yolov8n_ghost.yaml",
        (False, False, False, True): "models/yolov8n_decoupled.yaml",
        (True, True, True, True): "models/yolov8n_full.yaml",
    }
    name_by_feature = {
        (False, False, False, False): "baseline",
        (True, False, False, False): "cbam",
        (False, True, False, False): "bifpn",
        (True, True, False, False): "cbam_bifpn",
        (False, False, True, False): "ghostconv",
        (False, False, False, True): "decoupled_head",
        (True, True, True, True): "full_model",
    }
    if feature_flags not in model_by_feature and args.model is None and "model" not in config:
        enabled = [
            name
            for name, enabled_flag in zip(
                ("CBAM", "BiFPN", "GhostConv", "DecoupledHead"),
                feature_flags,
            )
            if enabled_flag
        ]
        raise ValueError(
            "No default model YAML for feature combination: "
            f"{', '.join(enabled) or 'Baseline'}. Provide --model explicitly."
        )

    default_model = model_by_feature.get(feature_flags, config.get("model", "yolov8n.pt"))
    default_name = name_by_feature.get(feature_flags, config.get("name", "custom_model"))

    if args.model is not None:
        model_path = args.model
    elif feature_cli_supplied:
        model_path = default_model
    else:
        model_path = config.get("model", default_model)

    if args.name is not None:
        name = args.name
    elif feature_cli_supplied:
        name = default_name
    else:
        name = config.get("name", default_name)

    return model_path, name


def main() -> int:
    """Run Ultralytics YOLOv8 training."""

    args = parse_args()
    config = load_yaml_config(args.config)
    enable_cbam_cli_supplied = args.enable_cbam is not None
    enable_bifpn_cli_supplied = args.enable_bifpn is not None
    enable_ghostconv_cli_supplied = args.enable_ghostconv is not None
    enable_decoupled_head_cli_supplied = args.enable_decoupled_head is not None
    enable_cbam = str_to_bool(
        args.enable_cbam if enable_cbam_cli_supplied else config.get("enable_cbam")
    )
    enable_bifpn = str_to_bool(
        args.enable_bifpn if enable_bifpn_cli_supplied else config.get("enable_bifpn")
    )
    enable_focal_loss = str_to_bool(
        args.enable_focal_loss
        if args.enable_focal_loss is not None
        else config.get("enable_focal_loss")
    )
    enable_ghostconv = str_to_bool(
        args.enable_ghostconv
        if enable_ghostconv_cli_supplied
        else config.get("enable_ghostconv")
    )
    enable_decoupled_head = str_to_bool(
        args.enable_decoupled_head
        if enable_decoupled_head_cli_supplied
        else config.get("enable_decoupled_head")
    )
    focal_gamma = float(config_value(args, config, "focal_gamma", 2.0))
    focal_alpha = optional_float(
        args.focal_alpha if args.focal_alpha is not None else config.get("focal_alpha", 0.25)
    )

    data_path = Path(config_value(args, config, "data", "dataset/data.yaml"))
    model_path, name = resolve_model_and_name(
        config=config,
        args=args,
        feature_flags=(enable_cbam, enable_bifpn, enable_ghostconv, enable_decoupled_head),
        feature_cli_supplied=(
            enable_cbam_cli_supplied
            or enable_bifpn_cli_supplied
            or enable_ghostconv_cli_supplied
            or enable_decoupled_head_cli_supplied
        ),
    )
    epochs = int(config_value(args, config, "epochs", 100))
    imgsz = int(config_value(args, config, "imgsz", 640))
    workers = int(config_value(args, config, "workers", 4))
    seed = int(config_value(args, config, "seed", 42))
    device_arg = config_value(args, config, "device", "auto")
    project_arg = config_value(args, config, "project", "runs")

    if enable_cbam or enable_bifpn or enable_decoupled_head:
        from models.modules import register_ultralytics_modules

        register_ultralytics_modules(
            enable_cbam=enable_cbam,
            enable_bifpn=enable_bifpn,
            enable_decoupled_head=enable_decoupled_head,
        )
    if enable_focal_loss:
        from models.losses import register_focal_loss

        register_focal_loss()

    dataset_root = data_path.parent
    project_path = Path(project_arg)
    if not project_path.is_absolute():
        project_path = ROOT / project_path

    if not args.skip_dataset_check:
        report = check_split_dataset(dataset_root)
        if not report.ok:
            print(
                "Dataset check failed. Run scripts/check_dataset.py "
                "--dataset-root dataset for details.",
                file=sys.stderr,
            )
            return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install Phase 1 dependencies first: "
            "python -m pip install -r requirements.txt"
        ) from exc

    device = resolve_device(device_arg)
    batch = default_batch_size(device, args.batch, config)
    print(f"Training device: {device}")
    print(f"Batch size: {batch}")
    print(f"CBAM enabled: {enable_cbam}")
    print(f"BiFPN enabled: {enable_bifpn}")
    print(f"GhostConv enabled: {enable_ghostconv}")
    print(f"Decoupled Head enabled: {enable_decoupled_head}")
    print(f"Focal Loss enabled: {enable_focal_loss}")
    if enable_focal_loss:
        print(f"Focal gamma: {focal_gamma}")
        print(f"Focal alpha: {focal_alpha}")
    print(f"Output: {project_path / name}")

    try:
        model = YOLO(model_path)
    except (EOFError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            f"Model weights look incomplete or corrupted: {model_path}. "
            "Delete the local .pt file and run again so Ultralytics can "
            "download a fresh copy."
        ) from exc

    trainer = None
    if enable_focal_loss:
        from scripts.trainers import build_focal_trainer

        trainer = build_focal_trainer(
            enable_focal_loss=enable_focal_loss,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )

    model.train(
        trainer=trainer,
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        seed=seed,
        device=device,
        project=str(project_path),
        name=name,
        exist_ok=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
