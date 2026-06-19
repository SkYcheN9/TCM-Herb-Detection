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


def main() -> int:
    """Run Ultralytics YOLOv8 training."""

    args = parse_args()
    config = load_yaml_config(args.config)
    enable_cbam_cli_supplied = args.enable_cbam is not None
    enable_bifpn_cli_supplied = args.enable_bifpn is not None
    enable_cbam = str_to_bool(
        args.enable_cbam if enable_cbam_cli_supplied else config.get("enable_cbam")
    )
    enable_bifpn = str_to_bool(
        args.enable_bifpn if enable_bifpn_cli_supplied else config.get("enable_bifpn")
    )
    model_by_feature = {
        (False, False): "yolov8n.pt",
        (True, False): "models/yolov8n_cbam.yaml",
        (False, True): "models/yolov8n_bifpn.yaml",
        (True, True): "models/yolov8n_cbam_bifpn.yaml",
    }
    name_by_feature = {
        (False, False): "baseline",
        (True, False): "cbam",
        (False, True): "bifpn",
        (True, True): "cbam_bifpn",
    }
    default_model = model_by_feature[(enable_cbam, enable_bifpn)]
    default_name = name_by_feature[(enable_cbam, enable_bifpn)]

    data_path = Path(config_value(args, config, "data", "dataset/data.yaml"))
    if args.model is not None:
        model_path = args.model
    elif enable_cbam_cli_supplied or enable_bifpn_cli_supplied:
        model_path = default_model
    else:
        model_path = config.get("model", default_model)
    epochs = int(config_value(args, config, "epochs", 100))
    imgsz = int(config_value(args, config, "imgsz", 640))
    workers = int(config_value(args, config, "workers", 4))
    seed = int(config_value(args, config, "seed", 42))
    device_arg = config_value(args, config, "device", "auto")
    project_arg = config_value(args, config, "project", "runs")
    if args.name is not None:
        name = args.name
    elif enable_cbam_cli_supplied or enable_bifpn_cli_supplied:
        name = default_name
    else:
        name = config.get("name", default_name)

    if enable_cbam or enable_bifpn:
        from models.modules import register_ultralytics_modules

        register_ultralytics_modules(enable_cbam=enable_cbam, enable_bifpn=enable_bifpn)

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
    print(f"Output: {project_path / name}")

    try:
        model = YOLO(model_path)
    except (EOFError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            f"Model weights look incomplete or corrupted: {model_path}. "
            "Delete the local .pt file and run again so Ultralytics can "
            "download a fresh copy."
        ) from exc

    model.train(
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
