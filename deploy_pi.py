"""Package Raspberry Pi deployment artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse deployment packaging options."""

    parser = argparse.ArgumentParser(description="Prepare Raspberry Pi deployment package.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional source best.pt. Defaults to the final GhostConv lightweight model when available.",
    )
    parser.add_argument("--output", default="dist/raspberry_pi")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--skip-export", action="store_true", help="Reuse existing best.* artifacts.")
    parser.add_argument("--zip", action="store_true", help="Create a zip archive.")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    """Resolve a project-relative path."""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_export(output_dir: Path, weights: str | None, imgsz: int) -> None:
    """Run export.py for deployment formats."""

    command = [
        sys.executable,
        str(ROOT / "export.py"),
        "--output",
        str(output_dir),
        "--imgsz",
        str(imgsz),
    ]
    if weights:
        command.extend(["--weights", weights])
    subprocess.run(command, cwd=ROOT, check=True)


def copy_tree(source: Path, target: Path) -> None:
    """Copy a directory tree, replacing any previous generated version."""

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def write_manifest(output_dir: Path, imgsz: int) -> Path:
    """Write deployment manifest."""

    manifest = {
        "name": "TCM-Herb-Detection Raspberry Pi Package",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "imgsz": imgsz,
        "recommended_model": "Baseline+GhostConv",
        "deployment_note": "Raspberry Pi 5 without accelerator prioritizes GhostConv for speed; web/desktop use CBAM+BiFPN.",
        "artifacts": sorted(path.name for path in output_dir.iterdir()),
        "entrypoints": {
            "camera_web": "python pi_camera_web.py",
            "benchmark": "python ../../benchmark_pi.py --pt best.pt --onnx best.onnx --openvino best_openvino",
        },
    }
    path = output_dir / "deploy_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def zip_package(output_dir: Path) -> Path:
    """Create a zip archive for transfer to Raspberry Pi."""

    archive_path = output_dir.with_suffix(".zip")
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir))
    return archive_path


def main() -> int:
    """Prepare Raspberry Pi package."""

    args = parse_args()
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_export:
        run_export(output_dir, args.weights, args.imgsz)

    copy_tree(ROOT / "deployment" / "raspberry_pi", output_dir / "runtime")
    docs_target = output_dir / "RASPBERRY_PI_DEPLOYMENT.md"
    shutil.copy2(ROOT / "docs" / "RASPBERRY_PI_DEPLOYMENT.md", docs_target)
    manifest = write_manifest(output_dir, args.imgsz)
    print(f"Deployment package: {output_dir}")
    print(f"Manifest: {manifest}")
    if args.zip:
        archive = zip_package(output_dir)
        print(f"Archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
