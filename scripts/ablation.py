"""Run YOLOv8 ablation experiments and export reports."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import zipfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_workspace_path(value: str | Path) -> Path:
    """Resolve a project-relative path against the repository root."""

    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


@dataclass(frozen=True)
class Experiment:
    """A single ablation experiment definition."""

    key: str
    display_name: str
    config: str
    run_name: str
    enable_cbam: bool
    enable_bifpn: bool
    enable_focal_loss: bool
    enable_ghostconv: bool
    enable_decoupled_head: bool
    suite: str = "default"


EXPERIMENTS = [
    Experiment(
        key="baseline",
        display_name="Baseline",
        config="configs/baseline.yaml",
        run_name="baseline",
        enable_cbam=False,
        enable_bifpn=False,
        enable_focal_loss=False,
        enable_ghostconv=False,
        enable_decoupled_head=False,
    ),
    Experiment(
        key="cbam",
        display_name="Baseline+CBAM",
        config="configs/cbam.yaml",
        run_name="baseline_cbam",
        enable_cbam=True,
        enable_bifpn=False,
        enable_focal_loss=False,
        enable_ghostconv=False,
        enable_decoupled_head=False,
    ),
    Experiment(
        key="bifpn",
        display_name="Baseline+BiFPN",
        config="configs/bifpn.yaml",
        run_name="baseline_bifpn",
        enable_cbam=False,
        enable_bifpn=True,
        enable_focal_loss=False,
        enable_ghostconv=False,
        enable_decoupled_head=False,
        suite="extended",
    ),
    Experiment(
        key="focal",
        display_name="Baseline+Focal",
        config="configs/focal.yaml",
        run_name="baseline_focal",
        enable_cbam=False,
        enable_bifpn=False,
        enable_focal_loss=True,
        enable_ghostconv=False,
        enable_decoupled_head=False,
        suite="extended",
    ),
    Experiment(
        key="ghostconv",
        display_name="Baseline+GhostConv",
        config="configs/ghostconv.yaml",
        run_name="baseline_ghostconv",
        enable_cbam=False,
        enable_bifpn=False,
        enable_focal_loss=False,
        enable_ghostconv=True,
        enable_decoupled_head=False,
        suite="extended",
    ),
    Experiment(
        key="decoupled_head",
        display_name="Baseline+DecoupledHead",
        config="configs/decoupled_head.yaml",
        run_name="baseline_decoupled_head",
        enable_cbam=False,
        enable_bifpn=False,
        enable_focal_loss=False,
        enable_ghostconv=False,
        enable_decoupled_head=True,
        suite="extended",
    ),
    Experiment(
        key="cbam_bifpn",
        display_name="Baseline+CBAM+BiFPN",
        config="configs/cbam_bifpn.yaml",
        run_name="baseline_cbam_bifpn",
        enable_cbam=True,
        enable_bifpn=True,
        enable_focal_loss=False,
        enable_ghostconv=False,
        enable_decoupled_head=False,
    ),
    Experiment(
        key="cbam_bifpn_focal",
        display_name="Baseline+CBAM+BiFPN+Focal",
        config="configs/cbam_bifpn_focal.yaml",
        run_name="baseline_cbam_bifpn_focal",
        enable_cbam=True,
        enable_bifpn=True,
        enable_focal_loss=True,
        enable_ghostconv=False,
        enable_decoupled_head=False,
    ),
    Experiment(
        key="full_model",
        display_name="FullModel",
        config="configs/full_model.yaml",
        run_name="full_model",
        enable_cbam=True,
        enable_bifpn=True,
        enable_focal_loss=True,
        enable_ghostconv=True,
        enable_decoupled_head=True,
    ),
]


SUMMARY_FIELDS = [
    "Experiment",
    "Config",
    "Suite",
    "CBAM",
    "BiFPN",
    "FocalLoss",
    "GhostConv",
    "DecoupledHead",
    "Init",
    "PretrainedWeights",
    "FocalLossType",
    "FocalGamma",
    "FocalAlpha",
    "Data",
    "Epochs",
    "ImageSize",
    "Batch",
    "Workers",
    "Device",
    "Weights",
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95",
    "FPS",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run ablation training and validation.")
    parser.add_argument("--output", default="reports/ablation")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--experiments",
        default="default",
        help="default, extended, all, or comma-separated experiment keys.",
    )
    parser.add_argument(
        "--init",
        default=None,
        choices=["default", "scratch", "pretrained"],
        help="Forwarded to train.py; use pretrained for fair partial weight transfer.",
    )
    parser.add_argument(
        "--pretrained-weights",
        default=None,
        help="Weights used when --init pretrained is selected.",
    )
    parser.add_argument(
        "--focal-loss-type",
        default=None,
        choices=["soft_focal", "legacy_focal", "varifocal"],
        help="Override Focal variants for Focal-enabled experiments.",
    )
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--focal-alpha", default=None)
    parser.add_argument("--skip-train", action="store_true", help="Reuse existing weights.")
    parser.add_argument("--skip-val", action="store_true", help="Only export training curves already available.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing them.")
    return parser.parse_args()


def selected_experiments(selection: str) -> list[Experiment]:
    """Return selected experiments in canonical order."""

    normalized = selection.lower()
    if normalized == "all":
        return list(EXPERIMENTS)
    if normalized == "default":
        return [experiment for experiment in EXPERIMENTS if experiment.suite == "default"]
    if normalized == "extended":
        return list(EXPERIMENTS)
    keys = {item.strip() for item in selection.split(",") if item.strip()}
    known = {experiment.key for experiment in EXPERIMENTS}
    unknown = sorted(keys - known)
    if unknown:
        raise ValueError(f"Unknown experiments: {', '.join(unknown)}")
    return [experiment for experiment in EXPERIMENTS if experiment.key in keys]


def train_command(args: argparse.Namespace, experiment: Experiment, runs_dir: Path) -> list[str]:
    """Build the training command for an experiment."""

    data_path = resolve_workspace_path(args.data)
    command = [
        sys.executable,
        "train.py",
        "--config",
        experiment.config,
        "--project",
        str(runs_dir),
        "--name",
        experiment.run_name,
        "--data",
        str(data_path),
    ]
    optional_args = {
        "--epochs": getattr(args, "epochs", None),
        "--imgsz": getattr(args, "imgsz", None),
        "--batch": getattr(args, "batch", None),
        "--workers": getattr(args, "workers", None),
        "--device": getattr(args, "device", None),
        "--init": getattr(args, "init", None),
        "--pretrained-weights": getattr(args, "pretrained_weights", None),
    }
    for flag, value in optional_args.items():
        if value is not None:
            command.extend([flag, str(value)])
    if experiment.enable_focal_loss:
        focal_args = {
            "--focal-loss-type": getattr(args, "focal_loss_type", None),
            "--focal-gamma": getattr(args, "focal_gamma", None),
            "--focal-alpha": getattr(args, "focal_alpha", None),
        }
        for flag, value in focal_args.items():
            if value is not None:
                command.extend([flag, str(value)])
    return command


def experiment_metadata(args: argparse.Namespace, experiment: Experiment) -> dict[str, object]:
    """Return reproducibility metadata for a summary row."""

    return {
        "Experiment": experiment.display_name,
        "Config": experiment.config,
        "Suite": experiment.suite,
        "CBAM": experiment.enable_cbam,
        "BiFPN": experiment.enable_bifpn,
        "FocalLoss": experiment.enable_focal_loss,
        "GhostConv": experiment.enable_ghostconv,
        "DecoupledHead": experiment.enable_decoupled_head,
        "Init": args.init or "config/default",
        "PretrainedWeights": args.pretrained_weights or "",
        "FocalLossType": args.focal_loss_type if experiment.enable_focal_loss else "",
        "FocalGamma": args.focal_gamma if experiment.enable_focal_loss else "",
        "FocalAlpha": args.focal_alpha if experiment.enable_focal_loss else "",
        "Data": str(resolve_workspace_path(args.data)),
        "Epochs": args.epochs if args.epochs is not None else "",
        "ImageSize": args.imgsz if args.imgsz is not None else "",
        "Batch": args.batch if args.batch is not None else "",
        "Workers": args.workers if args.workers is not None else "",
        "Device": args.device or "",
    }


def run_training(args: argparse.Namespace, experiment: Experiment, runs_dir: Path) -> None:
    """Run one training job."""

    command = train_command(args, experiment, runs_dir)
    print(f"[train] {experiment.display_name}: {' '.join(command)}", flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def register_for_experiment(experiment: Experiment) -> None:
    """Register custom modules needed to load or validate an experiment."""

    if experiment.enable_cbam or experiment.enable_bifpn or experiment.enable_decoupled_head:
        from models.modules import register_ultralytics_modules

        register_ultralytics_modules(
            enable_cbam=experiment.enable_cbam,
            enable_bifpn=experiment.enable_bifpn,
            enable_decoupled_head=experiment.enable_decoupled_head,
        )
    if experiment.enable_focal_loss:
        from models.losses import register_focal_loss

        register_focal_loss()


def best_weights(run_dir: Path) -> Path:
    """Return best weights if available, otherwise last weights."""

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        return best
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No weights found in {run_dir / 'weights'}")


def validate_experiment(
    args: argparse.Namespace,
    experiment: Experiment,
    weights: Path,
    val_dir: Path,
) -> tuple[dict[str, object], object | None]:
    """Validate one trained model and return summary metrics."""

    if args.skip_val:
        return empty_summary(args, experiment, weights), None

    register_for_experiment(experiment)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    val_kwargs = {
        "data": str(resolve_workspace_path(args.data)),
        "project": str(val_dir),
        "name": experiment.run_name,
        "plots": True,
        "exist_ok": True,
    }
    if args.imgsz is not None:
        val_kwargs["imgsz"] = args.imgsz
    if args.batch is not None:
        val_kwargs["batch"] = args.batch
    if args.device is not None and args.device.lower() != "auto":
        val_kwargs["device"] = args.device

    print(f"[val] {experiment.display_name}: {weights}", flush=True)
    metrics = model.val(**val_kwargs)
    return summarize_metrics(args, experiment, weights, metrics), metrics


def empty_summary(args: argparse.Namespace, experiment: Experiment, weights: Path) -> dict[str, object]:
    """Return a placeholder summary row when validation is skipped."""

    return {
        **experiment_metadata(args, experiment),
        "Weights": str(weights),
        "Precision": "",
        "Recall": "",
        "mAP50": "",
        "mAP50-95": "",
        "FPS": "",
        "preprocess_ms": "",
        "inference_ms": "",
        "postprocess_ms": "",
    }


def summarize_metrics(
    args: argparse.Namespace,
    experiment: Experiment,
    weights: Path,
    metrics: object,
) -> dict[str, object]:
    """Convert Ultralytics validation metrics into a summary row."""

    speed = getattr(metrics, "speed", {}) or {}
    preprocess = float(speed.get("preprocess", 0.0))
    inference = float(speed.get("inference", 0.0))
    postprocess = float(speed.get("postprocess", 0.0))
    fps_denominator = preprocess + inference + postprocess
    fps = 1000.0 / fps_denominator if fps_denominator > 0 else 0.0
    box = metrics.box
    return {
        **experiment_metadata(args, experiment),
        "Weights": str(weights),
        "Precision": float(box.mp),
        "Recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "FPS": fps,
        "preprocess_ms": preprocess,
        "inference_ms": inference,
        "postprocess_ms": postprocess,
    }


def read_history(results_csv: Path, experiment: Experiment) -> list[dict[str, object]]:
    """Read Ultralytics per-epoch results.csv."""

    if not results_csv.exists():
        return []
    rows: list[dict[str, object]] = []
    with results_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for raw in reader:
            row = {key.strip(): value for key, value in raw.items()}
            row["experiment"] = experiment.display_name
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    """Write rows to a CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fields = keys
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def column_name(index: int) -> str:
    """Convert a one-based column index to an Excel column name."""

    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def write_xlsx(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    """Write a minimal XLSX workbook without external dependencies."""

    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [fields] + [[row.get(field, "") for field in fields] for row in rows]
    sheet_rows = []
    for row_index, row in enumerate(all_rows, start=1):
        cells = []
        for col_index, value in enumerate(row, start=1):
            ref = f"{column_name(col_index)}{row_index}"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        "</worksheet>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="summary" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            "</Relationships>",
        )
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def metric_float(row: dict[str, object], key: str) -> float | None:
    """Parse a numeric value from a result row."""

    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_loss_curve(histories: dict[str, list[dict[str, object]]], path: Path) -> None:
    """Plot total training loss curves."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plotted = False
    for name, rows in histories.items():
        epochs, losses = [], []
        for index, row in enumerate(rows, start=1):
            box = metric_float(row, "train/box_loss") or 0.0
            cls = metric_float(row, "train/cls_loss") or 0.0
            dfl = metric_float(row, "train/dfl_loss") or 0.0
            epochs.append(metric_float(row, "epoch") or index)
            losses.append(box + cls + dfl)
        if epochs:
            plt.plot(epochs, losses, marker="o", linewidth=1.5, label=name)
            plotted = True
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Ablation Loss Curve")
    plt.grid(True, alpha=0.3)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_map_curve(histories: dict[str, list[dict[str, object]]], path: Path) -> None:
    """Plot mAP50 and mAP50-95 curves."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plotted = False
    for name, rows in histories.items():
        epochs, map50, map95 = [], [], []
        for index, row in enumerate(rows, start=1):
            m50 = metric_float(row, "metrics/mAP50(B)")
            m95 = metric_float(row, "metrics/mAP50-95(B)")
            if m50 is None or m95 is None:
                continue
            epochs.append(metric_float(row, "epoch") or index)
            map50.append(m50)
            map95.append(m95)
        if epochs:
            plt.plot(epochs, map50, marker="o", linewidth=1.5, label=f"{name} mAP50")
            plt.plot(epochs, map95, linestyle="--", linewidth=1.5, label=f"{name} mAP50-95")
            plotted = True
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Ablation mAP Curve")
    plt.grid(True, alpha=0.3)
    if plotted:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def precision_recall_curve(metrics: object):
    """Return a mean precision-recall curve from Ultralytics metrics."""

    import numpy as np

    curves = getattr(metrics, "curves_results", None)
    if not curves:
        return None
    pr_curve = None
    for curve in curves:
        if len(curve) >= 4 and curve[2] == "Recall" and curve[3] == "Precision":
            pr_curve = curve
            break
    if pr_curve is None:
        return None

    recall = np.asarray(pr_curve[0], dtype=float).reshape(-1)
    precision = np.asarray(pr_curve[1], dtype=float)
    if precision.ndim == 0 or recall.size == 0:
        return None
    if precision.ndim > 1:
        if precision.shape[-1] == recall.size:
            precision = np.nanmean(precision.reshape(-1, precision.shape[-1]), axis=0)
        elif precision.shape[0] == recall.size:
            precision = np.nanmean(precision.reshape(precision.shape[0], -1), axis=1)
        else:
            precision = np.nanmean(precision, axis=0).reshape(-1)
    precision = precision.reshape(-1)
    points = min(recall.size, precision.size)
    if points == 0:
        return None
    return recall[:points], precision[:points]


def plot_pr_curve(metrics_by_name: dict[str, object], path: Path) -> None:
    """Plot mean precision-recall curves from validation metrics."""

    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plotted = False
    for name, metrics in metrics_by_name.items():
        curve = precision_recall_curve(metrics)
        if curve is None:
            continue
        recall, precision = curve
        plt.plot(recall, precision, linewidth=1.5, label=name)
        plotted = True
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Ablation PR Curve")
    plt.grid(True, alpha=0.3)
    if plotted:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> int:
    """Run all selected ablation experiments."""

    args = parse_args()
    output_dir = ROOT / args.output
    runs_dir = output_dir / "runs"
    val_dir = output_dir / "val"
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = selected_experiments(args.experiments)

    if args.dry_run:
        for experiment in experiments:
            print(" ".join(train_command(args, experiment, runs_dir)))
        return 0

    summaries: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, object]]] = {}
    all_history_rows: list[dict[str, object]] = []
    metrics_by_name: dict[str, object] = {}

    for experiment in experiments:
        run_dir = runs_dir / experiment.run_name
        if not args.skip_train:
            run_training(args, experiment, runs_dir)
        weights = best_weights(run_dir)
        summary, metrics = validate_experiment(args, experiment, weights, val_dir)
        summaries.append(summary)
        if metrics is not None:
            metrics_by_name[experiment.display_name] = metrics
        history = read_history(run_dir / "results.csv", experiment)
        histories[experiment.display_name] = history
        all_history_rows.extend(history)

    write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    write_xlsx(output_dir / "summary.xlsx", summaries, SUMMARY_FIELDS)
    write_csv(output_dir / "history.csv", all_history_rows)
    plot_loss_curve(histories, plots_dir / "loss_curve.png")
    plot_map_curve(histories, plots_dir / "map_curve.png")
    if metrics_by_name:
        plot_pr_curve(metrics_by_name, plots_dir / "pr_curve.png")

    print(f"Ablation reports saved to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
