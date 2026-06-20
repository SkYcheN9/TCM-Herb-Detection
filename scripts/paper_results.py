"""Generate paper-ready experimental result tables and analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ablation import write_csv


PAPER_FIELDS = [
    "Experiment",
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95",
    "FPS",
    "Delta_mAP50",
    "Delta_mAP50-95",
    "Delta_FPS",
    "Init",
    "FocalLossType",
    "FocalGamma",
    "FocalAlpha",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Generate final paper experiment results.")
    parser.add_argument("--ablation-summary", default="reports/ablation/summary.csv")
    parser.add_argument("--focal-summary", default=None)
    parser.add_argument("--output", default="reports/final_experiments")
    parser.add_argument("--title", default="论文实验结果汇总")
    return parser.parse_args()


def read_rows(path: str | Path) -> list[dict[str, object]]:
    """Read CSV rows if the file exists."""

    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def numeric(row: dict[str, object], key: str) -> float | None:
    """Parse a float from a row."""

    try:
        value = row.get(key)
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: object, digits: int = 4) -> str:
    """Format a table value."""

    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def find_baseline(rows: list[dict[str, object]]) -> dict[str, object] | None:
    """Return the baseline row."""

    for row in rows:
        if str(row.get("Experiment", "")).lower() == "baseline":
            return row
    return rows[0] if rows else None


def add_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Add baseline-relative metric differences."""

    baseline = find_baseline(rows)
    if baseline is None:
        return rows
    base_map50 = numeric(baseline, "mAP50") or 0.0
    base_map95 = numeric(baseline, "mAP50-95") or 0.0
    base_fps = numeric(baseline, "FPS") or 0.0
    enriched = []
    for row in rows:
        item = dict(row)
        item["Delta_mAP50"] = (numeric(row, "mAP50") or 0.0) - base_map50
        item["Delta_mAP50-95"] = (numeric(row, "mAP50-95") or 0.0) - base_map95
        item["Delta_FPS"] = (numeric(row, "FPS") or 0.0) - base_fps
        enriched.append(item)
    return enriched


def best_row(rows: list[dict[str, object]], metric: str) -> dict[str, object] | None:
    """Return the best row by metric."""

    valid = [row for row in rows if numeric(row, metric) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: numeric(row, metric) or 0.0)


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> str:
    """Render rows as a Markdown table."""

    headers = fields
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [fmt(row.get(field)) for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def focal_section(rows: list[dict[str, object]]) -> str:
    """Return the optional Focal search section."""

    if not rows:
        return "## Focal 参数搜索\n\n尚未提供 Focal 参数搜索结果。"
    top_rows = rows[: min(8, len(rows))]
    best = best_row(rows, "mAP50-95")
    fields = ["SearchKey", "LossType", "Gamma", "Alpha", "mAP50", "mAP50-95", "Precision", "Recall", "FPS"]
    text = ["## Focal 参数搜索", "", markdown_table(top_rows, fields)]
    if best is not None:
        text.append("")
        text.append(
            "最优 Focal 组合为 "
            f"`{best.get('LossType')}`，gamma={best.get('Gamma')}，alpha={best.get('Alpha')}，"
            f"mAP50={fmt(best.get('mAP50'))}，mAP50-95={fmt(best.get('mAP50-95'))}。"
        )
    return "\n".join(text)


def diagnosis(rows: list[dict[str, object]]) -> str:
    """Generate a concise result diagnosis."""

    baseline = find_baseline(rows)
    best_accuracy = best_row(rows, "mAP50-95")
    best_speed = best_row(rows, "FPS")
    if baseline is None:
        return "未读取到 Baseline 结果，无法进行相对分析。"

    baseline_name = baseline.get("Experiment", "Baseline")
    baseline_map95 = fmt(baseline.get("mAP50-95"))
    lines = [
        f"以 `{baseline_name}` 为参照，其 mAP50-95 为 {baseline_map95}。",
    ]
    if best_accuracy is not None:
        lines.append(
            "精度最优模型为 "
            f"`{best_accuracy.get('Experiment')}`，mAP50-95={fmt(best_accuracy.get('mAP50-95'))}。"
        )
    if best_speed is not None:
        lines.append(
            "速度最优模型为 "
            f"`{best_speed.get('Experiment')}`，FPS={fmt(best_speed.get('FPS'), digits=2)}。"
        )
    lines.append(
        "改进模型低于 Baseline 的主要原因是：早期实验中 Baseline 直接使用 `yolov8n.pt` "
        "完整预训练权重，而改进结构从 YAML 随机初始化或只能部分迁移权重，初始化条件不完全对等；"
        "其次，CBAM、BiFPN、GhostConv 与 Decoupled Head 会改变特征分布和优化路径，"
        "在当前小样本数据规模下更容易出现欠拟合或收敛速度不足；"
        "最后，旧 Focal 设置对 YOLOv8 的软标签质量分数不够友好，"
        "gamma=2.0、alpha=0.25 会过度压低易样本梯度，使分类分支收敛明显变慢。"
    )
    lines.append(
        "因此，论文中应将旧结果表述为阶段性消融结果；正式公平消融需要统一数据集、epoch、batch、seed、"
        "验证集和初始化策略，并优先采用 `--init pretrained` 的部分预训练迁移。"
    )
    return "\n\n".join(lines)


def generate_report(
    title: str,
    ablation_rows: list[dict[str, object]],
    focal_rows: list[dict[str, object]],
) -> str:
    """Generate the final Markdown report."""

    enriched = add_deltas(ablation_rows)
    fields = [
        "Experiment",
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
        "FPS",
        "Delta_mAP50",
        "Delta_mAP50-95",
        "Delta_FPS",
    ]
    parts = [
        f"# {title}",
        "",
        "## 公平消融结果",
        "",
        markdown_table(enriched, fields) if enriched else "尚未读取到消融结果。",
        "",
        "## 结果分析",
        "",
        diagnosis(enriched),
        "",
        focal_section(focal_rows),
        "",
        "## 论文结论写法",
        "",
        "当前结果说明，简单叠加注意力、特征融合、轻量化卷积、解耦检测头和 Focal Loss "
        "并不必然带来精度提升。对于本项目的小规模 15 类中药饮片数据集，官方 YOLOv8n 预训练 Baseline "
        "具有更强的先验优势；改进结构应以公平预训练迁移和 Focal 参数搜索后的最优组合为最终报告依据。"
    ]
    return "\n".join(parts) + "\n"


def project_fields(rows: list[dict[str, object]], fields: list[str]) -> list[dict[str, object]]:
    """Return rows containing only the requested fields."""

    return [{field: row.get(field, "") for field in fields} for row in rows]


def main() -> int:
    """Write paper-ready result files."""

    args = parse_args()
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_rows = read_rows(args.ablation_summary)
    focal_rows = read_rows(args.focal_summary) if args.focal_summary else []
    enriched = add_deltas(ablation_rows)
    write_csv(output_dir / "paper_table.csv", project_fields(enriched, PAPER_FIELDS), PAPER_FIELDS)
    report = generate_report(args.title, ablation_rows, focal_rows)
    (output_dir / "paper_results.md").write_text(report, encoding="utf-8")
    print(f"Paper results saved to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
