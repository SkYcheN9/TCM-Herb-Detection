"""Run Focal Loss parameter search and merge experiment summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ablation import SUMMARY_FIELDS, write_csv, write_xlsx


SEARCH_FIELDS = [
    "SearchKey",
    "LossType",
    "Gamma",
    "Alpha",
    *SUMMARY_FIELDS,
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run Focal Loss parameter search.")
    parser.add_argument("--output", default="reports/focal_search")
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--experiments", default="cbam_bifpn_focal")
    parser.add_argument("--init", default="pretrained", choices=["default", "scratch", "pretrained"])
    parser.add_argument("--pretrained-weights", default="yolov8n.pt")
    parser.add_argument("--loss-types", default="soft_focal,varifocal")
    parser.add_argument("--gammas", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--alphas", default="none,0.25,0.5,0.75")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def split_values(value: str) -> list[str]:
    """Split comma-separated CLI values while ignoring blanks."""

    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_token(value: str) -> str:
    """Return a filesystem-friendly token."""

    return value.lower().replace(".", "p").replace("/", "_").replace("\\", "_")


def search_key(loss_type: str, gamma: str, alpha: str) -> str:
    """Return a stable key for one Focal parameter combination."""

    return f"{normalize_token(loss_type)}_g{normalize_token(gamma)}_a{normalize_token(alpha)}"


def build_ablation_command(args: argparse.Namespace, key: str, loss_type: str, gamma: str, alpha: str) -> list[str]:
    """Build the delegated ablation command for one parameter combination."""

    command = [
        sys.executable,
        "scripts/ablation.py",
        "--output",
        str(Path(args.output) / key),
        "--experiments",
        args.experiments,
        "--data",
        args.data,
        "--init",
        args.init,
        "--pretrained-weights",
        args.pretrained_weights,
        "--focal-loss-type",
        loss_type,
        "--focal-gamma",
        gamma,
        "--focal-alpha",
        alpha,
    ]
    optional_args = {
        "--epochs": args.epochs,
        "--imgsz": args.imgsz,
        "--batch": args.batch,
        "--workers": args.workers,
        "--device": args.device,
    }
    for flag, value in optional_args.items():
        if value is not None:
            command.extend([flag, str(value)])
    if args.skip_train:
        command.append("--skip-train")
    if args.skip_val:
        command.append("--skip-val")
    return command


def read_summary(path: Path) -> list[dict[str, object]]:
    """Read one ablation summary CSV."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def run_search(args: argparse.Namespace) -> list[dict[str, object]]:
    """Run every Focal combination and return merged summary rows."""

    rows: list[dict[str, object]] = []
    output_dir = ROOT / args.output
    loss_types = split_values(args.loss_types)
    gammas = split_values(args.gammas)
    alphas = split_values(args.alphas)

    for loss_type in loss_types:
        for gamma in gammas:
            for alpha in alphas:
                key = search_key(loss_type, gamma, alpha)
                command = build_ablation_command(args, key, loss_type, gamma, alpha)
                print(f"[focal-search] {key}: {' '.join(command)}", flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, check=True)
                for row in read_summary(output_dir / key / "summary.csv"):
                    rows.append(
                        {
                            "SearchKey": key,
                            "LossType": loss_type,
                            "Gamma": gamma,
                            "Alpha": alpha,
                            **row,
                        }
                    )
    return rows


def metric_value(row: dict[str, object], key: str) -> float:
    """Return a sortable metric value."""

    try:
        return float(row.get(key, "") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    """Run the Focal search and write merged reports."""

    args = parse_args()
    rows = run_search(args)
    if args.dry_run:
        return 0

    rows.sort(
        key=lambda row: (
            metric_value(row, "mAP50-95"),
            metric_value(row, "mAP50"),
            metric_value(row, "Recall"),
        ),
        reverse=True,
    )
    output_dir = ROOT / args.output
    write_csv(output_dir / "summary.csv", rows, SEARCH_FIELDS)
    write_xlsx(output_dir / "summary.xlsx", rows, SEARCH_FIELDS)
    print(f"Focal search reports saved to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
