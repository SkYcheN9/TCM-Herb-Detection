"""Generate the practice-project Word report required by the task book."""

from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from report_generator import DocxBuilder, read_simple_yaml  # noqa: E402


OUT_DIR = ROOT / "reports" / "generated_experiment_materials"
PNG_DIR = OUT_DIR / "png"
WORD_DIR = OUT_DIR / "word"
CSV_DIR = OUT_DIR / "csv"
REPORT_PATH = WORD_DIR / "experiment_report.docx"
BACKUP_PATH = WORD_DIR / "experiment_report_auto_summary_backup.docx"


CHINESE_NAMES = {
    "zexie": "泽泻",
    "niuxi": "牛膝",
    "gaoliangjiang": "高良姜",
    "mudanpi": "牡丹皮",
    "yuzhu": "玉竹",
    "baizhi": "白芷",
    "baishao": "白芍",
    "dazao": "大枣",
    "danshen": "丹参",
    "gancao": "甘草片",
    "baixianpi": "白鲜皮",
    "baihe": "百合",
    "sangzhi": "桑枝",
    "jiegeng": "桔梗",
    "banlangen": "板蓝根",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file with UTF-8 BOM tolerance."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def fnum(value: Any, digits: int = 4) -> str:
    """Format a number for report tables."""

    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value) if value not in (None, "") else "-"


def count_dataset_boxes() -> tuple[list[str], dict[str, dict[str, int]]]:
    """Count images and boxes for train/val splits."""

    data = read_simple_yaml(ROOT / "dataset_augmented" / "data.yaml")
    raw_names = data.get("names", {})
    if isinstance(raw_names, dict):
        names = [raw_names[index] for index in sorted(raw_names)]
    else:
        names = list(raw_names)

    result: dict[str, dict[str, int]] = {}
    for split in ["train", "val"]:
        labels_dir = ROOT / "dataset_augmented" / "labels" / split
        images_dir = ROOT / "dataset_augmented" / "images" / split
        counter: Counter[str] = Counter()
        boxes = 0
        for label_path in labels_dir.glob("*.txt"):
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if not parts:
                    continue
                class_id = int(float(parts[0]))
                if 0 <= class_id < len(names):
                    counter[names[class_id]] += 1
                    boxes += 1
        result[split] = {
            "images": len(list(images_dir.glob("*.*"))),
            "labels": len(list(labels_dir.glob("*.txt"))),
            "boxes": boxes,
            **{name: counter[name] for name in names},
        }
    return names, result


def create_model_structure_figure(output: Path) -> Path:
    """Create a before/after YOLOv8 structure comparison figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=170)
    ax.axis("off")

    def box(x: float, y: float, text: str, color: str) -> None:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            color="#1f2933",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=color, edgecolor="#264653", linewidth=1.2),
        )

    baseline = ["Input", "YOLOv8n\nBackbone", "SPPF", "PAN-FPN\nNeck", "Detect\nHead", "bbox/class/conf"]
    improved = [
        "Input",
        "GhostConv\nBackbone",
        "CBAM\nAttention",
        "SPPF",
        "BiFPN\nNeck",
        "Decoupled\nHead",
        "bbox/class/conf\ncount",
    ]
    xs1 = [0.08, 0.24, 0.38, 0.52, 0.67, 0.83]
    xs2 = [0.06, 0.20, 0.34, 0.47, 0.60, 0.74, 0.89]
    ax.text(0.02, 0.82, "Baseline YOLOv8n", fontsize=13, fontweight="bold", color="#0f172a")
    ax.text(0.02, 0.42, "Improved Project Model", fontsize=13, fontweight="bold", color="#0f172a")
    for index, item in enumerate(baseline):
        box(xs1[index], 0.72, item, "#edf2f7")
        if index < len(baseline) - 1:
            ax.annotate("", xy=(xs1[index + 1] - 0.06, 0.72), xytext=(xs1[index] + 0.06, 0.72), arrowprops=dict(arrowstyle="->", lw=1.6))
    for index, item in enumerate(improved):
        color = "#e0f2fe" if index not in {1, 2, 4, 5} else "#ccfbf1"
        box(xs2[index], 0.32, item, color)
        if index < len(improved) - 1:
            ax.annotate("", xy=(xs2[index + 1] - 0.052, 0.32), xytext=(xs2[index] + 0.052, 0.32), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(
        0.50,
        0.08,
        "Focal Loss is enabled during training to replace the classification term when configured.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def create_annotation_sample(output: Path) -> Path:
    """Render one real YOLO label file as a LabelImg-style annotation sample."""

    from PIL import Image, ImageDraw, ImageFont

    image_path = ROOT / "dataset_augmented" / "images" / "train" / "1-5-001.jpg"
    label_path = ROOT / "dataset_augmented" / "labels" / "train" / "1-5-001.txt"
    names, _ = count_dataset_boxes()
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((820, 610))
    width, height = image.size
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    colors = ["#e63946", "#2a9d8f", "#457b9d", "#f4a261", "#7b2cbf"]
    raw_image = Image.open(image_path)
    raw_width, raw_height = raw_image.size
    labels: list[str] = []
    for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, xc, yc, bw, bh = [float(part) for part in parts]
        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh / 2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh / 2) * height)
        color = colors[index % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        label = names[int(class_id)]
        labels.append(label)
        draw.rectangle([x1, max(0, y1 - 28), x1 + 150, y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 26)), label, fill="white", font=font)

    canvas_width = 1240
    canvas_height = 760
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#edf2f7")
    side = ImageDraw.Draw(canvas)
    side.rectangle([0, 0, canvas_width, 42], fill="#dbeafe")
    side.text((18, 11), "LabelImg - 1-5-001.jpg", fill="#0f172a", font=small_font)
    side.rectangle([0, 42, 72, canvas_height], fill="#f8fafc", outline="#cbd5e1")
    for i, icon in enumerate(["Open", "Save", "Box", "Prev", "Next"]):
        y = 72 + i * 58
        side.rounded_rectangle([10, y, 62, y + 38], radius=4, fill="#ffffff", outline="#cbd5e1")
        side.text((18, y + 11), icon[:4], fill="#334155", font=small_font)
    image_x, image_y = 92, 78
    side.rectangle([image_x - 8, image_y - 8, image_x + width + 8, image_y + height + 8], fill="#ffffff", outline="#cbd5e1")
    canvas.paste(image, (image_x, image_y))
    panel_x = 940
    side.rectangle([panel_x, 58, 1220, 735], fill="#ffffff", outline="#cbd5e1")
    side.text((panel_x + 18, 82), "Label List", fill="#0f172a", font=font)
    for row, label in enumerate(labels[:8]):
        y = 126 + row * 34
        side.rectangle([panel_x + 18, y, panel_x + 250, y + 24], fill="#ecfeff", outline="#99f6e4")
        side.text((panel_x + 28, y + 4), f"{row + 1}. {label}", fill="#0f766e", font=small_font)
    side.text((panel_x + 18, 430), "YOLO format", fill="#0f172a", font=font)
    for row, text in enumerate(["class_id", "x_center", "y_center", "width", "height"]):
        side.text((panel_x + 34, 468 + row * 28), text, fill="#475569", font=small_font)
    side.text((panel_x + 18, 635), f"Raw size: {raw_width} x {raw_height}", fill="#475569", font=small_font)
    side.text((panel_x + 18, 664), "Fixed 15-class order", fill="#475569", font=small_font)
    canvas.save(output, quality=92)
    return output


def create_augmentation_comparison(output: Path) -> Path:
    """Create an augmentation-before/after comparison board."""

    from PIL import Image, ImageDraw, ImageFont

    candidates = [
        (ROOT / "dataset_augmented" / "images" / "train" / "1-5-001.jpg", "Original"),
        (ROOT / "dataset_augmented" / "images" / "train" / "1-5-001_alb_1.jpg", "Albumentations"),
        (ROOT / "reports" / "desktop" / "images" / "mosaic_00220_20260623_091936.jpg", "Mosaic"),
        (ROOT / "reports" / "stress_test_samples" / "images" / "stress_0004_complex_background.jpg", "Hard Scene"),
    ]
    thumb_size = (420, 300)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    canvas = Image.new("RGB", (thumb_size[0] * 2 + 48, thumb_size[1] * 2 + 108), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    for index, (path, title) in enumerate(candidates):
        image = Image.open(path).convert("RGB")
        image.thumbnail(thumb_size)
        row, col = divmod(index, 2)
        x = 24 + col * (thumb_size[0] + 24)
        y = 54 + row * (thumb_size[1] + 48)
        canvas.paste(image, (x, y))
        draw.rectangle([x, y, x + thumb_size[0], y + thumb_size[1]], outline="#cbd5e1", width=2)
        draw.text((x, y - 32), title, fill="#0f172a", font=font)
    draw.text((24, 14), "Data Augmentation and Stress Scene Comparison", fill="#0f172a", font=font)
    canvas.save(output, quality=92)
    return output


def add_code_block(doc: DocxBuilder, code: str, title: str) -> None:
    """Append a compact monospace code block to a DOCX builder."""

    doc.add_paragraph(title)
    for raw_line in code.strip("\n").splitlines():
        line = raw_line[:120]
        doc.body.append(
            "<w:p>"
            "<w:pPr><w:spacing w:before=\"0\" w:after=\"0\"/></w:pPr>"
            "<w:r><w:rPr><w:rFonts w:ascii=\"Consolas\" w:eastAsia=\"Microsoft YaHei\"/>"
            "<w:sz w:val=\"17\"/><w:color w:val=\"334155\"/></w:rPr>"
            f"<w:t xml:space=\"preserve\">{escape(line)}</w:t></w:r></w:p>"
        )


def p(doc: DocxBuilder, text: str) -> None:
    """Add a normal paragraph."""

    doc.add_paragraph(text)


def b(doc: DocxBuilder, text: str) -> None:
    """Add a bullet paragraph."""

    doc.add_bullet(text)


def table_from_rows(headers: list[str], rows: list[list[Any]]) -> list[list[Any]]:
    """Build a table matrix."""

    return [headers] + rows


def add_paragraphs(doc: DocxBuilder, paragraphs: list[str]) -> None:
    """Append a list of report paragraphs."""

    for text in paragraphs:
        p(doc, text)


def build_report() -> None:
    """Build and save the current practice-project report."""

    WORD_DIR.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(REPORT_PATH, BACKUP_PATH)

    structure_png = create_model_structure_figure(PNG_DIR / "model_structure_comparison.png")
    annotation_png = create_annotation_sample(PNG_DIR / "annotation_quality_sample.png")
    aug_png = create_augmentation_comparison(PNG_DIR / "augmentation_comparison.png")
    stress_png = ROOT / "reports" / "stress_test_samples" / "preview_contact_sheet.jpg"
    preview_png = ROOT / "src" / "frontend" / "public" / "images" / "tcm-detection-preview.png"

    names, counts = count_dataset_boxes()
    ablation_rows = read_csv(CSV_DIR / "ablation_summary.csv")
    training_rows = read_csv(CSV_DIR / "training_metrics.csv")
    best_training = max(training_rows, key=lambda row: float(row.get("metrics/mAP50-95(B)", 0) or 0)) if training_rows else {}
    train_images = counts.get("train", {}).get("images", 0)
    val_images = counts.get("val", {}).get("images", 0)
    train_boxes = counts.get("train", {}).get("boxes", 0)
    val_boxes = counts.get("val", {}).get("boxes", 0)

    doc = DocxBuilder("基于 CBAM 与 BiFPN 改进 YOLOv8 的中医药饮片智能检测与识别系统实践项目报告")
    doc.add_heading("基于 CBAM 与 BiFPN 改进 YOLOv8 的中医药饮片智能检测与识别系统实践项目报告", level=0)
    p(doc, "项目：TCM-SliceAI 中医药饮片智能检测与识别系统")
    p(doc, "报告性质：实践项目设计、实现、评估与验收材料。本文不是单一实验记录，而是围绕任务书要求，对数据构建、算法改进、系统开发、自动化材料生成和量化验证进行综合说明。")
    p(doc, "交付范围说明：本版系统部署范围为本地 CPU/GPU 推理、PC 桌面端和 Web 端。由于硬件支持条件已经调整，专用硬件方向不再纳入本项目交付、测试和验收范围。")

    doc.add_heading("目录", level=1)
    for item in [
        "一、绪论",
        "二、任务书要求与总体方案",
        "三、数据集构建与数据治理",
        "四、改进 YOLOv8 模型设计",
        "五、系统功能实现与工程化部署",
        "六、训练评估与消融验证",
        "七、项目验收覆盖与问题讨论",
        "八、总结与展望",
    ]:
        p(doc, item)

    doc.add_heading("一、绪论", level=1)
    doc.add_heading("1.1 项目背景与业务意义", level=2)
    add_paragraphs(
        doc,
        [
            "中医药饮片是中药处方调配、药房盘点和仓储流转中的基础对象，具有种类多、形态差异细、包装和摆放方式不统一等特点。传统管理流程主要依赖药师经验完成识别、复核和数量估计，在样本量较大或工作节奏较快时容易受到疲劳、光照、遮挡和经验差异影响。对于教学和实践项目而言，饮片识别也不是简单的分类题目，因为系统必须同时判断图像中是否存在多个目标、每个目标所在位置、所属类别以及总数和分项数量。任务书要求开发基于改进 YOLOv8 的中医药饮片智能检测与识别系统，正是希望把目标检测算法、数据治理方法和可运行软件系统结合起来，形成能够被现场样本验证的完整实践成果。",
            "本项目面向的场景具有明显的细粒度识别难度。桔梗、甘草片、白芍、白芷等切片类饮片在颜色、纹理、横截面形态和边缘破损程度上存在相似性；桑枝、牛膝、泽泻等类别在不同拍摄角度、不同堆叠状态下可能呈现接近的长条或块状轮廓；丹参、牡丹皮等类别又会受到色泽深浅和表面纹理复杂度的影响。如果只在干净背景、单个样本、规整摆放的图像上训练和展示，模型很容易学习到过于理想化的特征，无法支撑任务书中“复杂真实样本现场验收”的要求。因此，项目在设计之初就把复杂背景、遮挡、堆叠、污损和光照变化作为数据和算法共同面对的问题。",
            "从工程应用角度看，饮片检测系统的价值不仅在于给出一个类别名称，更在于把检测结果转化为可以被药房盘点和复核流程使用的信息。系统输出需要包含检测框、类别、置信度、总数量、按类别统计和可视化结果，并且要能够在桌面端和 Web 端稳定查看、保存、检索和导出。这样的要求决定了本文不能沿用单纯实验报告的写法，而应当以实践项目报告的形式组织内容：先解释任务书约束和整体路线，再说明数据集、模型和系统的实现，最后使用训练曲线、混淆矩阵和消融表验证方案是否可靠。",
            "本项目还承担了课程实践中的可复核要求。所有实验材料不能只停留在截图或口头描述，而要由脚本从训练日志和评估文件中自动生成 CSV、Excel、PNG 和 Word 材料。这样做一方面减少手工整理时的误填和漏填，另一方面也让报告中的数据、图表和模型选型结论能够追溯到具体文件。对于一个需要答辩和验收的项目来说，可追溯性本身就是工程质量的一部分。本文后续所有量化结论均来自当前工程目录中的训练日志、消融汇总表和生成图像，而不是脱离项目的主观描述。",
        ],
    )
    doc.add_heading("1.2 任务定义与应用边界", level=2)
    add_paragraphs(
        doc,
        [
            "本项目输入为药房、教学或实验环境中采集的中医药饮片图片、视频片段或摄像头画面，输出为结构化检测结果和带框可视化图像。结构化结果至少包括目标坐标、英文类别名、中文类别名、置信度、总目标数量以及按类别统计的数量。系统功能覆盖单张图片检测、多图批量检测、视频检测、摄像头实时检测、历史记录查询、统计分析和结果导出。算法侧以 YOLOv8n 为基线模型，在结构和损失层面实现实质性改进，避免只通过调整 conf、iou、imgsz 或 epochs 等基础参数来声称创新。",
            "根据当前项目条件，部署边界已明确调整为 PC 桌面端、Web 端和本地 CPU/GPU 推理。桌面端主要用于本机摄像头、图片和视频的交互式检测，适合离线演示、课堂验收和本地批处理；Web 端主要用于浏览器上传、摄像头采集、历史记录和统计展示，适合局域网或单机服务场景；后端统一提供 API，使两个客户端共享同一套模型加载、预处理、推理和结果解析逻辑。由于硬件支持问题，专用硬件方向已经从交付范围中移除，本文不再把该方向作为验收项、部署方案或模型选择依据。",
            "在模型选型上，本文将 Baseline+CBAM+BiFPN 作为默认部署模型，将 Baseline+CBAM 作为最高精度参考模型，将 GhostConv、Decoupled Head 和 Focal Loss 相关组合作为消融对照。这样的划分既保留任务书要求的多种改进尝试，也避免把轻量化模块直接等同于最终部署目标。最终模型选择不只看单一 mAP 指标，而是在 mAP50-95、FPS、功能稳定性、接口一致性和报告可复核性之间进行综合判断。",
        ],
    )
    doc.add_heading("1.3 本文工作概述", level=2)
    add_paragraphs(
        doc,
        [
            "本文围绕实践项目交付物组织，共包括八个部分。第一部分说明项目背景、业务意义和任务边界；第二部分从任务书出发拆解关键要求，并给出总体技术路线；第三部分介绍 15 类饮片数据集、YOLO 标注规范、数据清洗、类别分布和增强策略；第四部分说明 YOLOv8 基线模型与 CBAM、BiFPN、GhostConv、Decoupled Head、Focal Loss 等改进模块；第五部分介绍自动化训练评估、FastAPI 后端、桌面端、Web 端和 CPU/GPU 兼容实现；第六部分通过 Loss 曲线、mAP 曲线、混淆矩阵和 11 组消融结果完成量化验证；第七部分对照验收要求分析完成情况和风险；第八部分总结项目成果并提出后续优化方向。",
            "与传统深度学习实验报告相比，本文减少了单纯罗列实验参数的篇幅，增加了数据治理、软件接口、客户端功能和验收材料生成的说明。这种调整符合任务性质：项目不是为了比较两个现成模型的单次训练结果，而是要完成一个能够训练、评估、推理、展示、保存、导出和复核的智能检测系统。因此，训练结果只是实践链路中的一个证据，系统完整性、材料自动化和部署边界同样是报告的重要内容。",
        ],
    )

    doc.add_heading("二、任务书要求与总体方案", level=1)
    doc.add_heading("2.1 任务书关键约束", level=2)
    doc.add_table(
        table_from_rows(
            ["要求类别", "任务书要求", "本项目落实情况"],
            [
                ["数据集", "15 类常见中医药饮片，覆盖遮挡、堆叠、污损、光照变化和复杂背景", f"固定 15 类顺序，增强后训练集 {train_images} 张、验证集 {val_images} 张，并保留压力测试样本"],
                ["标注规范", "LabelImg 与 YOLO 标准格式：class_id, x_center, y_center, width, height", "提供数据检查与类别顺序锁定，阻断缺失标注、空标注、越界类别和 bbox 归一化异常"],
                ["算法改进", "必须完成代码落地的实质性结构或损失改进", "完成 CBAM、BiFPN、GhostConv、Decoupled Head、Focal Loss 五类改进并进行消融"],
                ["系统功能", "多目标检测、实时推理、CPU/GPU 兼容、可视化结果和统计分析", "FastAPI 后端、PySide6 桌面端和 Next.js Web 端均输出 bbox、class、confidence、count"],
                ["验收材料", "提供结构对比图、训练曲线、混淆矩阵、消融表和核心代码展示", "自动生成 Excel、CSV、PNG 与 Word，图表说明均超过 30 个中文字符"],
                ["部署范围", "根据项目实际硬件条件完成可运行系统", "本版聚焦本地 CPU/GPU、PC 桌面端和 Web 端，专用硬件方向不纳入验收"],
            ],
        )
    )
    add_paragraphs(
        doc,
        [
            "任务书对本项目提出了两个层面的要求：一是算法层面的实质改进，二是系统层面的完整交付。算法层面要求不能仅靠调参完成，因此本文把结构模块和损失函数改动作为核心实现对象，并通过消融表证明每个方向的实际效果。系统层面要求能够在真实复杂样本中完成检测和计数，因此项目不仅需要训练权重，还需要有统一推理接口、结果可视化、历史记录、统计分析和材料生成能力。",
            "任务书还特别强调报告配图必须有 30 字以上说明，且要包含模型结构对比、数据可视化、训练曲线、混淆矩阵和消融结果。这意味着报告不是训练日志的简单截图集合，而需要把每张图放回项目问题中解释：图像说明应当回答该图用于验证什么、结果反映什么风险、后续可以怎样改进。本文在插图下方均加入针对性分析，避免只写“如图所示”而没有工程含义。",
        ],
    )
    doc.add_heading("2.2 总体技术路线", level=2)
    add_paragraphs(
        doc,
        [
            "系统整体分为数据层、算法层、服务层、应用层和材料层。数据层负责原始图像清洗、YOLO 标注检查、类别顺序锁定、增强数据生成和压力样本管理；算法层负责基线训练、改进模块注册、损失函数替换、模型验证、速度测试和消融调度；服务层基于 FastAPI 暴露检测、历史记录和统计接口；应用层包括 PySide6 桌面端和 Next.js Web 端；材料层由 report_generator.py 与本报告脚本自动汇总训练日志、消融表、曲线、混淆矩阵和 Word 文档。",
            "这种路线把任务书中的“算法改进”和“工程系统”放在同一条链路中实现。训练脚本与推理服务共享固定类别顺序，避免训练、后端和前端之间 class_id 映射错位；模型评估结果直接进入消融汇总表，避免手工复制造成数值错误；报告材料从 CSV 和 PNG 自动生成，保证最终文档与项目目录中的证据一致。项目最终不是只给出一个 best.pt 文件，而是给出数据、模型、接口、客户端和验收材料之间能够相互印证的工程闭环。",
            "在功能流程上，用户可通过桌面端或 Web 端提交图片、视频或摄像头画面，客户端将数据发送给后端检测服务，后端完成图像读取、尺寸适配、模型推理、结果过滤和可视化保存，再把统一 JSON 返回给客户端。客户端负责展示检测框、类别、置信度、总数量、分类统计和历史记录。该流程将模型内部输出转化为业务可读信息，使系统不仅能“识别是什么”，还能回答“有多少、在哪里、是否需要复核”。",
            "在模型流程上，项目先训练 YOLOv8n 基线模型，再分别引入 CBAM、BiFPN、GhostConv、Decoupled Head 和 Focal Loss，随后进行 11 组消融对比。消融结果用于回答三个问题：哪些模块确实提高 mAP50-95，哪些模块提升速度但不一定提升精度，哪些复杂组合在当前数据规模下收益不足。最终默认模型选择 Baseline+CBAM+BiFPN，因为它在精度和速度之间取得较好平衡，也与桌面端和 Web 端的实际运行条件相匹配。",
        ],
    )
    doc.add_image(
        structure_png,
        "图 1  改进前后模型结构对比图。图中展示 Baseline YOLOv8n 与项目改进模型的主要差异，新增模块包括 GhostConv、CBAM、BiFPN、Decoupled Head，并在训练阶段引入 Focal Loss 作为可选分类损失。该图说明项目改进发生在骨干、特征融合、检测头和损失函数多个位置，而不是简单修改训练轮数或置信度阈值。",
    )
    doc.add_heading("2.3 交付边界与验收口径", level=2)
    add_paragraphs(
        doc,
        [
            "结合当前硬件条件，本项目的验收口径明确为本地 CPU/GPU、PC 桌面端和 Web 端。训练和评估环节在可用 GPU 上完成，推理服务支持自动选择 cuda 或 cpu；桌面端面向本机演示和批量检测；Web 端面向浏览器交互和结果统计。模型导出仍保留 pt、ONNX 等通用格式能力，但这些格式在本文中主要用于兼容不同本地推理环境和后续扩展，不再作为专用硬件方向部署的证据。",
            "这种边界调整不会削弱项目的核心目标。任务书真正关注的是 15 类饮片的可靠检测、实质性算法改进、复杂样本鲁棒性、系统可运行性和材料可复核性。PC 桌面端与 Web 端已经覆盖图片、视频、摄像头、历史记录、统计分析和导出功能，能够完整展示模型从训练到应用的链路。报告中因此不再设置专用硬件方向的验收表项，也不把 GhostConv 的轻量化结果解释为某一特定硬件平台的默认方案。",
            "为了避免交付范围混乱，本文在模型选择中区分“部署默认模型”和“消融对照模型”。Baseline+CBAM+BiFPN 是默认部署模型，Baseline+CBAM 是最高精度参考，Baseline+GhostConv 是轻量化结构对照，Focal Loss 和 FullModel 组合则作为负向或候选结论保留。这样的表述更符合当前项目实际，也能让教师在验收时清楚看到每类模型的作用边界。",
        ],
    )

    doc.add_heading("三、数据集构建与数据治理", level=1)
    doc.add_heading("3.1 类别体系与标注规范", level=2)
    class_rows = []
    for index, name in enumerate(names):
        class_rows.append([index, name, CHINESE_NAMES.get(name, name), counts["train"].get(name, 0), counts["val"].get(name, 0)])
    doc.add_table(table_from_rows(["ID", "英文类别", "中文名称", "训练框数", "验证框数"], class_rows))
    add_paragraphs(
        doc,
        [
            f"数据集严格使用任务书指定的 15 类中医药饮片，并固定 YOLO 类别顺序。增强后训练集包含 {train_images} 张图像、{train_boxes} 个标注框，验证集包含 {val_images} 张图像、{val_boxes} 个标注框。固定类别顺序的意义在于保证训练、验证、后端解析、前端展示、历史统计和报告表格使用同一套 class_id 映射，避免因为自动排序或中文名称转换导致模型输出错位。",
            "标注格式采用 YOLO 标准五元组，即 class_id、x_center、y_center、width、height，后四项均为相对图像宽高归一化后的数值。该格式简洁、训练兼容性好，但对标注质量要求较高：如果框体过大，模型会把背景纹理学习为饮片特征；如果框体过小，模型会忽略边缘形态和局部破损；如果类别编号错位，后端和界面将显示错误类别。项目因此在训练前对标签文件进行完整性检查，把数据质量控制前置到模型训练之前。",
            "对于中医药饮片场景，标注不仅要框住可见主体，还要处理堆叠、遮挡和边缘缺损。若多个饮片互相接触但仍可辨认，应分别标注；若目标被遮挡但主体类别仍明显，应以可见轮廓为依据给出紧贴框；若饮片碎片过小且无法判断类别，则不宜强行标注。这样的标注策略有助于模型学习真实业务中的局部可见目标，同时避免把模糊碎屑当作稳定类别特征。",
        ],
    )
    doc.add_image(
        annotation_png,
        "图 2  YOLO 标注质量示例图。图中基于真实标签文件反绘检测框，展示 class_id 与归一化坐标转换后的目标范围。框体紧贴饮片主体，能够减少桌面纹理、包装材料和相邻饮片进入正样本区域，从而降低模型学习错误背景特征的风险。",
    )
    doc.add_heading("3.2 数据规模、类别分布与业务难度", level=2)
    add_paragraphs(
        doc,
        [
            "从当前统计看，训练集和验证集均包含多实例图片，标注框数量明显大于图片数量，说明数据并非简单的一图一物分类集合，而是更接近真实盘点场景。多实例数据能够训练模型同时处理不同位置、不同尺度和不同遮挡状态的目标，也能在推理阶段直接输出总数与分项数量。对于药房盘点和处方复核而言，这一点比单标签分类更有实用价值。",
            "类别分布对检测模型有直接影响。样本数量较多的类别通常更容易获得稳定特征，而样本数量较少或形态变化较大的类别更容易在验证集中出现波动。中医药饮片还存在类别内差异较大的问题，同一类别可能因切制厚度、干燥程度、拍摄角度和光照条件不同而呈现明显外观变化。项目通过增强数据和压力样本缓解这一问题，但后续仍需要把现场失败样本纳入难例库，持续完善类别覆盖。",
            "难点不只来自样本数量，还来自类别之间的视觉相似性。甘草片与桔梗都可能呈浅色切片形态，白芍与白芷在局部纹理和颜色上容易接近，桑枝与牛膝在某些角度下都可能呈长条结构。面对这类细粒度差异，模型需要利用纹理、边缘、尺度和上下文信息共同判断，而不能只依赖单一颜色特征。CBAM 和 BiFPN 的设计正是围绕这一问题展开：前者帮助模型突出关键通道和空间位置，后者增强不同尺度特征的融合能力。",
        ],
    )
    doc.add_heading("3.3 数据清洗与工程验证", level=2)
    add_paragraphs(
        doc,
        [
            "数据清洗环节主要处理四类问题。第一类是文件级问题，包括图片无法读取、标签缺失、图片与标签命名不一致；第二类是格式级问题，包括标签列数不是五列、坐标无法解析、类别编号不是整数；第三类是范围级问题，包括归一化坐标小于 0 或大于 1、宽高为 0 或接近 0；第四类是语义级问题，包括框体明显偏移、类别误标和重复标注。前三类可由脚本自动发现，第四类需要结合可视化抽检和人工复核。",
            "项目将 dataset/data.yaml 与 dataset_augmented/data.yaml 作为训练入口配置文件，并通过脚本维护路径和类别顺序。这样可以减少手动修改 YAML 造成的路径错误，尤其是在原始数据、增强数据和消融训练之间切换时，统一配置能保证模型读取同一套类别定义。数据检查脚本作为训练前置门禁，只有当图片、标签、类别顺序和 bbox 均通过检查时，才进入训练与消融阶段。",
            "从验收角度看，数据治理需要同时满足“训练可用”和“验收可信”。训练可用强调格式正确、图像可读、类别覆盖和标签完整；验收可信强调样本能够代表真实业务干扰，例如非均匀光照、饮片堆叠、纸张遮挡、表面污损、相机角度变化和桌面背景干扰。项目保留压力测试样本目录，目的就是在正式验收前提前观察模型是否只适应干净背景，避免出现训练指标较好但现场样本失效的问题。",
            "数据治理还为系统端提供了稳定基础。后端返回中文类别名、前端展示类别统计、历史记录保存检测结果，都依赖训练时固定的类别顺序和名称映射。如果数据集在后续迭代中增加或删除类别，必须同步更新训练配置、后端类别表、前端展示文案和报告表格，否则会出现模型输出与界面显示不一致的问题。本文当前报告基于 15 类固定体系完成，后续扩展应以版本号管理类别映射。",
        ],
    )
    doc.add_heading("3.4 数据增强与难例构造", level=2)
    add_paragraphs(
        doc,
        [
            "任务书要求数据集覆盖复杂桌面背景、纸张或包装袋遮挡、光照强弱变化、饮片堆叠和表面污损等真实场景。项目采用 Albumentations 完成离线增强，包含 Mosaic、MixUp、HSV 扰动、随机裁剪、亮度对比度变化、水平翻转和 CLAHE 等策略。增强后的数据集扩大了目标尺度、颜色、亮度和背景变化，有助于模型在面对非规整摆放饮片时保持稳定。",
            "增强策略需要控制强度，不能为了追求数量而生成不符合实际业务的样本。过强的颜色扰动可能使饮片失去真实色泽，过度裁剪可能破坏类别判别信息，过高比例的 MixUp 可能让目标边界变得不自然。因此，本项目把增强作为提升鲁棒性的手段，而不是替代真实采集。真正决定模型上限的仍然是真实样本质量，增强数据主要用于覆盖合理范围内的视觉变化。",
            "压力测试样本包括遮挡、堆叠、污损、复杂背景和甘草片与桔梗混合等类型。这些样本并不直接替代教师现场验收样本，但可以作为项目自检集合。若模型在压力样本中出现大量漏检，说明训练集中类似场景不足；若出现大量误检，说明模型对背景或相似类别过于敏感；若检测框明显偏移，则需要检查标注框质量和定位损失收敛情况。压力样本的价值在于把失败模式提前暴露出来，使后续数据补充更有方向。",
        ],
    )
    doc.add_image(
        aug_png,
        "图 3  数据增强与复杂场景对比图。图中同时展示原始样本、Albumentations 增强、Mosaic 样本和复杂背景压力样本，说明增强策略覆盖尺度变化、颜色扰动、局部遮挡和背景干扰。该设计用于提升模型对现场非规整摆放饮片的泛化能力。",
    )
    if stress_png.exists():
        doc.add_image(
            stress_png,
            "图 4  压力测试样本总览图。该图汇总遮挡、堆叠、污损、复杂背景和易混类别混合等难例场景，可用于验收前快速检查模型是否只适应干净背景。若模型在这些样本中大面积漏检或误检，需要继续补充同类训练数据。",
        )

    doc.add_heading("四、改进 YOLOv8 模型设计", level=1)
    doc.add_heading("4.1 YOLOv8 基线模型与改进原则", level=2)
    add_paragraphs(
        doc,
        [
            "Baseline 使用 Ultralytics YOLOv8n，输入尺寸为 640，检测头输出 P3、P4、P5 三个尺度特征上的边界框、类别和置信度。YOLOv8n 具有速度快、生态成熟、训练和导出流程稳定的优点，适合作为实践项目的基准模型。与二阶段检测器相比，YOLO 系列更适合实时识别和多端应用；与更大的 YOLOv8 模型相比，YOLOv8n 在当前数据规模和系统运行条件下更容易达到速度与精度平衡。",
            "本项目的改进原则有三点。第一，改进必须有代码落地，能够在模型配置或训练流程中真实启用；第二，改进必须可消融，能够单独或组合验证对指标的影响；第三，改进必须服务于饮片检测的实际难点，而不是为了堆叠模块。基于这三个原则，项目实现 CBAM 注意力、BiFPN 特征融合、GhostConv 轻量卷积、Decoupled Head 解耦检测头和 Focal Loss 分类损失，并统一纳入消融脚本比较。",
            "饮片检测的核心挑战是细粒度纹理、多尺度目标和复杂背景共存。细粒度纹理要求模型能够关注局部通道和空间区域，多尺度目标要求模型融合低层细节与高层语义，复杂背景要求模型抑制桌面、包装、阴影和碎屑干扰。CBAM 对应注意力增强，BiFPN 对应多尺度融合，GhostConv 对应计算量优化，Decoupled Head 对应分类与回归任务解耦，Focal Loss 对应难例分类关注。各模块的作用不同，因此需要通过消融结果判断是否适合当前数据。",
        ],
    )
    doc.add_heading("4.2 CBAM 注意力模块", level=2)
    add_paragraphs(
        doc,
        [
            "CBAM 包含通道注意力和空间注意力两部分。通道注意力通过平均池化和最大池化提取全局统计信息，使模型突出对饮片颜色、纹理和局部形态更敏感的特征通道；空间注意力在二维位置上重新加权，帮助模型从复杂桌面背景中聚焦饮片主体。对切片类饮片而言，很多判别信息并不来自整体轮廓，而来自局部纹理、边缘破损、内部纤维和颜色分布，因此注意力机制具有明确的任务动机。",
            "项目将 CBAM 插入 Backbone 的多个 C2f 后方，保持特征图尺寸不变，便于与 Ultralytics YAML 解析器兼容。这样的插入方式不会破坏检测头输出尺度，也便于在消融中与 Baseline 进行公平比较。消融结果显示 Baseline+CBAM 取得最高 mAP50-95，说明注意力机制确实增强了模型对饮片细节特征的表达能力。",
        ],
    )
    add_code_block(
        doc,
        """
class ChannelAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        attention = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        return x * self.sigmoid(attention)

class CBAM(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        return self.spatial_attention(x)
""",
        "核心代码片段 1：CBAM 通道与空间注意力的主要 PyTorch 实现，完整文件位于 models/modules/cbam.py。",
    )
    doc.add_heading("4.3 BiFPN 特征融合模块", level=2)
    add_paragraphs(
        doc,
        [
            "YOLOv8 默认 Neck 能够完成自顶向下和自底向上的特征传递，但不同尺度特征的融合权重相对固定。BiFPN 引入可学习权重，并在融合前进行归一化，使模型自动判断低层纹理、中层结构和高层语义在当前任务中的相对重要性。对于堆叠小目标，P3 细节特征非常关键；对于较大目标和背景抑制，P4、P5 语义信息更有价值。加权融合能够让模型在不同尺度之间更灵活地分配注意力。",
            "在本项目中，BiFPN 的作用不是单独追求复杂网络，而是补足饮片多尺度检测中的特征传递问题。验证结果显示，单独加入 BiFPN 并未超过 Baseline，但与 CBAM 组合后取得接近最高精度且保持较高 FPS 的结果。这说明特征融合模块需要与前面的注意力增强相互配合：CBAM 先提高关键特征质量，BiFPN 再对不同尺度特征进行加权整合，二者组合更适合作为默认部署模型。",
        ],
    )
    add_code_block(
        doc,
        """
class BiFPNFusion(nn.Module):
    def normalized_weights(self) -> torch.Tensor:
        weights = F.relu(self.weights)
        return weights / (weights.sum() + self.epsilon)

    def forward(self, inputs):
        weights = self.normalized_weights().to(inputs[0].device, inputs[0].dtype)
        fused = 0
        for weight, feature in zip(weights, inputs):
            fused = fused + weight * feature
        return self.conv(fused)
""",
        "核心代码片段 2：BiFPN 加权融合逻辑，完整文件位于 models/modules/bifpn.py。",
    )
    doc.add_heading("4.4 GhostConv、Decoupled Head 与 Focal Loss", level=2)
    add_paragraphs(
        doc,
        [
            "GhostConv 通过较少普通卷积生成主特征，再以轻量变换生成冗余特征，用于降低部分卷积层的计算开销。消融结果显示 Baseline+GhostConv 在 FPS 上表现较好，同时 mAP50-95 也高于 Baseline，说明轻量卷积并非只能牺牲精度换速度。在当前报告中，GhostConv 被定位为轻量化结构对照，用来说明模型复杂度和检测精度之间的关系，而不是特定硬件平台的部署依据。",
            "Decoupled Head 将分类塔与回归塔拆分，试图减少分类任务和定位任务在同一卷积分支中的梯度干扰。对于饮片检测而言，分类关注纹理和形态，回归关注目标边界和尺度，两者确实存在不同优化目标。但当前消融结果中 Baseline+DecoupledHead 未带来明显提升，说明在数据规模、模型容量和训练配置固定时，解耦结构的收益并不一定稳定，需要更多训练轮数、学习率或结构细节配合。",
            "Focal Loss 用于提高难例和易混类别在分类损失中的权重，理论上适合处理类别不均衡和困难样本。然而本项目的 Focal 相关组合没有取得最优结果，可能原因包括 gamma 参数不完全适配、预训练迁移后的分类分布已经较稳定、当前数据增强方式改变了正负样本比例等。这个结果并不说明 Focal Loss 无价值，而是提醒我们：复杂损失函数必须通过消融验证，不能只凭理论预期作为最终方案。",
        ],
    )
    add_code_block(
        doc,
        """
class FocalClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        hard_targets = targets.gt(0).to(dtype=targets.dtype)
        p_t = hard_targets * prob + (1 - hard_targets) * (1 - prob)
        focal_factor = (1 - p_t).clamp(min=0).pow(self.gamma)
        return bce_loss * focal_factor

class DecoupledDetect(Detect):
    def __init__(self, nc: int = 80, ch: tuple[int, ...] = ()) -> None:
        super().__init__(nc=nc, end2end=False, ch=ch)
        self.cv2 = nn.ModuleList([...])  # bbox regression tower
        self.cv3 = nn.ModuleList([...])  # classification tower
""",
        "核心代码片段 3：Focal 分类损失与解耦检测头的关键实现，完整文件位于 models/losses/focal_loss.py 和 models/modules/decoupled_head.py。",
    )
    doc.add_heading("4.5 模型选型策略", level=2)
    add_paragraphs(
        doc,
        [
            "模型选型遵循“最高精度参考、默认部署平衡、消融对照保留”的原则。Baseline+CBAM 是最高 mAP50-95 参考，说明注意力机制对当前任务最有直接帮助；Baseline+CBAM+BiFPN 与最高精度仅有极小差距，同时 FPS 更高，因此作为桌面端和 Web 端默认模型更合理；Baseline+GhostConv 作为轻量化对照，说明减少计算量并不一定导致明显精度下降；Focal Loss 和 FullModel 的结果则用于提醒后续工作不要盲目叠加模块。",
            "最终推荐 Baseline+CBAM+BiFPN，并不是因为它在每个单项指标上都绝对第一，而是因为实践项目更关注系统可用性。一个用于检测和计数的系统需要稳定响应、结果可解释、部署维护简单，并且能够在普通本地设备上运行。mAP50-95 的极小差异在实际展示中未必可感知，但 FPS、接口一致性和模型结构可解释性会影响验收体验。因此本文把 CBAM+BiFPN 作为默认模型，把 CBAM 单模块作为精度上限参考。",
        ],
    )

    doc.add_heading("五、系统功能实现与工程化部署", level=1)
    doc.add_heading("5.1 自动化训练、评估与报告材料生成", level=2)
    add_paragraphs(
        doc,
        [
            "项目提供 train.py、evaluate.py、benchmark.py、export.py、scripts/ablation.py 和 report_generator.py 等自动化入口。训练脚本自动检测 torch.cuda.is_available，在 GPU 可用时优先使用 CUDA，否则回退 CPU；评估脚本读取验证集并输出 Precision、Recall、mAP50、mAP50-95 等指标；基准测试脚本记录 FPS；消融脚本统一输出 summary.csv、summary.xlsx、history.csv、Loss/mAP 曲线和各实验运行目录。",
            "report_generator.py 的作用是把分散在 runs、final_results_full 和 reports 中的训练日志、消融表、混淆矩阵和曲线统一整理成 CSV、Excel、PNG 与 Word 材料。这样的自动化材料链路对于实践项目非常重要，因为报告中的每一个数值都应当能追溯到工程文件。如果后续重新训练模型，只需要重新运行材料生成脚本，就能得到更新后的曲线、表格和文档，避免人工替换图表造成前后不一致。",
            "本次 Word 报告在自动摘要的基础上进行了项目化重写：一方面保留任务书要求的 Loss 曲线、mAP 曲线、混淆矩阵和消融实验表，另一方面增加数据治理、系统架构、客户端功能、部署边界和验收口径说明。这样生成的文档既能满足训练评估材料要求，也能体现本项目不是单次实验，而是完整的智能检测与识别系统实践。",
        ],
    )
    doc.add_heading("5.2 后端服务与统一推理接口", level=2)
    add_paragraphs(
        doc,
        [
            "后端基于 FastAPI 实现，核心接口包括 /detect/classes、/detect/image、/detect/video 和 /detect/batch。检测服务加载最终模型后返回 bbox、class、confidence、count 和 class_counts，并可将带框结果保存到 backend/data/outputs，同时把检测记录写入 SQLite。统一后端的价值在于让桌面端和 Web 端共享同一套推理逻辑，避免不同客户端各自解析模型输出导致字段不一致。",
            "统一推理接口的关键是把 Ultralytics 原始结果标准化。后端在结果解析阶段将 xyxy 坐标、类别编号、英文类别名、中文类别名和置信度整理为统一结构，并在响应层进一步汇总总数量和按类别统计。这样客户端无需关心模型内部张量格式，只需要展示和保存标准 JSON。对于后续维护来说，如果更换权重或调整阈值，只要后端输出结构不变，前端和桌面端就不需要大规模修改。",
            "历史记录和统计接口使系统从“即时检测工具”扩展为“可追溯识别系统”。每次检测的来源、输出图像、类别统计和时间信息都可以写入数据库，后续可按日期、类别或来源查看。对于药房盘点场景，这一功能能够支持复核和追责；对于课程验收场景，它能够展示系统确实完成了从检测到记录再到统计的闭环。",
        ],
    )
    doc.add_table(
        table_from_rows(
            ["模块", "实现路径", "主要功能"],
            [
                ["检测服务", "backend/services/detector.py", "图片、视频推理，输出检测框、类别、置信度、计数和 FPS"],
                ["接口路由", "backend/routers/detect.py", "提供 classes、image、video、batch API"],
                ["历史记录", "backend/routers/history.py", "保存检测来源、结果路径、类别统计和导出记录"],
                ["统计分析", "backend/routers/statistics.py", "汇总类别频次、检测次数和趋势数据"],
                ["模型定位", "backend/services/model_locator.py", "按默认模型和可用权重路径寻找 best.pt"],
            ],
        )
    )
    doc.add_heading("5.3 桌面端与 Web 端实现", level=2)
    add_paragraphs(
        doc,
        [
            "桌面端采用 PySide6 和 qfluentwidgets，实现 Dashboard、Detection、History、Settings 和 About 等模块。Detection 页面支持摄像头、图片和视频检测，能够显示 FPS、类别数量、总数和 GPU 状态，并将检测结果保存到 reports/desktop。桌面端适合课堂答辩和本机演示，因为它不依赖浏览器上传流程，能够直接调用本机摄像头或选择本地文件。",
            "Web 端采用 Next.js、TypeScript 和 TailwindCSS，实现上传检测、摄像头采集、历史记录、统计分析和模型状态展示。Web 端的优势在于界面访问方便、结构展示清晰，也便于后续扩展为局域网演示或集中管理页面。前端不直接运行模型，而是通过 API 调用后端服务，这样可以保证桌面端和 Web 端使用同一套权重、阈值和类别映射。",
            "两个客户端的功能重点略有不同，但数据结构保持一致。桌面端强调本机交互、摄像头线程和结果导出，Web 端强调浏览器体验、响应式布局和统计展示。由于后端统一返回 total_count、class_counts 和 detections 列表，客户端可以围绕同一套字段实现不同展示方式。这种设计降低了维护成本，也提高了项目在答辩时的稳定性。",
        ],
    )
    if preview_png.exists():
        doc.add_image(
            preview_png,
            "图 5  Web 检测界面预览图。该界面用于展示上传或摄像头检测结果，并将模型输出转化为可读的类别、置信度和计数信息。界面层不是单纯展示图片，而是承接后端结构化输出，服务于药房盘点、复核和统计分析场景。",
        )
    doc.add_heading("5.4 CPU/GPU 兼容与本地部署流程", level=2)
    add_paragraphs(
        doc,
        [
            "本项目部署流程聚焦本地 CPU/GPU 环境。训练和推理入口均提供 auto、cpu、cuda 等设备选择方式，在 CUDA 可用时使用 GPU 加速，在无 GPU 环境下回退 CPU。对课程项目而言，这种兼容性比绑定某一专用设备更重要，因为验收环境可能与开发环境不同，系统需要能够在普通电脑上启动并完成演示。",
            "模型文件以 pt 权重作为默认加载格式，并保留 ONNX 等通用导出能力。导出能力主要用于提高本地推理环境的兼容性和后续扩展空间，不再承担专用硬件方向交付任务。默认模型由后端模型定位服务统一寻找，避免桌面端、Web 端和脚本各自写死权重路径。若后续替换为新的 best.pt，只需要按目录规范放置权重并更新配置即可。",
            "本地部署的核心要求包括：后端服务能够启动并加载模型，桌面端能够连接后端或使用本地配置完成检测，Web 端能够访问检测接口并展示结果，历史记录数据库能够正常写入和读取，报告材料能够重新生成。本文最终验收也围绕这些可运行能力展开，而不是把尚未纳入交付范围的硬件环境作为证明材料。",
        ],
    )

    doc.add_heading("六、训练评估与消融验证", level=1)
    doc.add_heading("6.1 训练设置与评价指标", level=2)
    add_paragraphs(
        doc,
        [
            "虽然本文不是单一实验报告，但任务书明确要求展示训练日志、mAP 曲线、混淆矩阵和消融量化对比，因此本章作为项目验证部分保留。主训练日志读取 final_results_full/reports/ablation/runs/baseline_cbam_bifpn/results.csv，对应最终默认模型 Baseline+CBAM+BiFPN 的 150 epoch 训练结果；正式消融结果读取 final_results_full/reports/ablation/summary.csv，并结合候选补充结果形成最终消融表。评价指标包括 Precision、Recall、mAP50、mAP50-95 和 FPS。mAP50 用于观察宽松 IoU 阈值下的检出能力，mAP50-95 更能体现定位质量，FPS 则用于判断桌面端和 Web 端推理体验。",
            f"Baseline+CBAM+BiFPN 最终 150 epoch 训练日志中最佳 mAP50-95 出现在第 {best_training.get('epoch', '-')} 轮，对应 mAP50={fnum(best_training.get('metrics/mAP50(B)'))}，mAP50-95={fnum(best_training.get('metrics/mAP50-95(B)'))}。正式 11 组消融统一使用增强数据集、固定输入尺寸和相近训练配置，因此更适合作为最终模型选型依据。Precision 表示预测为某类饮片的结果中有多少是真正目标，Recall 表示真实目标中有多少被模型检出，二者分别对应误报和漏检风险。",
            "对于药房盘点场景，漏检会导致数量偏低，误检会导致类别或库存偏差，因此 Precision 与 Recall 都需要关注。mAP50-95 综合多个 IoU 阈值，能够避免模型只在较宽松定位标准下表现良好。FPS 不是唯一目标，但会影响系统交互体验：如果检测延迟过高，摄像头实时检测和批量处理都会变得不顺畅。本文最终选型因此同时考虑精度和速度。",
        ],
    )
    doc.add_heading("6.2 Loss 与 mAP 曲线分析", level=2)
    for key, caption in [
        ("box_loss_curve.png", "图 6  Baseline+CBAM+BiFPN（150 epoch）box_loss 收敛曲线。该曲线反映边界框回归误差变化，后期趋于平稳说明模型对饮片位置和外接框尺度已有稳定学习；若验证曲线明显反弹，则需要检查遮挡、堆叠或小目标样本是否不足。"),
        ("cls_loss_curve.png", "图 7  Baseline+CBAM+BiFPN（150 epoch）cls_loss 收敛曲线。分类损失下降说明模型逐步学习不同饮片的颜色、纹理和形态差异；对易混类别而言，cls_loss 是否持续波动可作为判断类别边界是否清晰的重要依据。"),
        ("dfl_loss_curve.png", "图 8  Baseline+CBAM+BiFPN（150 epoch）dfl_loss 收敛曲线。DFL 与边界框分布建模相关，稳定下降表示定位分布更集中，有助于减少饮片边缘被桌面纹理或相邻饮片干扰时的框体偏移。"),
        ("map50_curve.png", "图 9  Baseline+CBAM+BiFPN（150 epoch）mAP50 曲线。mAP50 体现模型在 IoU=0.50 阈值下的整体检出能力，曲线快速上升并进入平台期，说明模型能较快捕捉大部分饮片目标和主要类别特征。"),
        ("map50_95_curve.png", "图 10  Baseline+CBAM+BiFPN（150 epoch）mAP50-95 曲线。该指标比 mAP50 更严格，能够反映定位精细度和多阈值稳定性，因而本文将其作为消融排序和部署模型选择的核心依据。"),
    ]:
        path = PNG_DIR / key
        if path.exists():
            doc.add_image(path, caption)
    add_paragraphs(
        doc,
        [
            "从 Loss 曲线看，box_loss、cls_loss 和 dfl_loss 分别对应定位、分类和边界框分布学习。三者都应当在训练前中期明显下降，并在后期逐步进入平台。如果 box_loss 收敛较慢，通常说明目标边界不清晰、标注框质量不稳定或小目标比例较高；如果 cls_loss 波动较大，通常说明类别间差异不足或易混类别样本不平衡；如果 dfl_loss 难以下降，则需要关注目标边缘、堆叠遮挡和输入尺寸设置。",
            "mAP50 与 mAP50-95 的差异也能提供诊断信息。若 mAP50 较高但 mAP50-95 明显偏低，说明模型能找到目标大致位置，但框体精度仍不足；若两者同步提升并进入稳定区间，说明模型在检出和定位方面都具备较好表现。本项目最终以 mAP50-95 作为消融排序核心指标，是因为饮片计数和可视化展示都需要较准确的边界框，不能只满足宽松定位标准。",
        ],
    )
    doc.add_heading("6.3 混淆矩阵分析", level=2)
    add_paragraphs(
        doc,
        [
            "混淆矩阵用于分析高频误检与漏检关系。中医药饮片类别之间存在显著的细粒度相似性，例如甘草片与桔梗均可能呈现浅色切片形态，白芍与白芷在局部纹理和颜色上也可能接近；在堆叠和遮挡场景中，目标边界不完整会进一步放大混淆。与单纯给出整体 mAP 相比，混淆矩阵能帮助定位具体类别对，为后续数据补充提供方向。",
            "观察混淆矩阵时需要同时看对角线和非对角线。对角线越集中，说明真实类别与预测类别匹配越稳定；非对角线数值偏高，说明某些类别存在系统性误判。归一化混淆矩阵能够消除类别样本数量差异的影响，更适合观察少样本类别是否被频繁错分。对于验收场景，如果某个类别在归一化矩阵中表现较弱，应优先补充该类别在复杂背景和不同光照下的样本。",
        ],
    )
    for key, caption in [
        ("confusion_matrix.png", "图 11  Baseline+CBAM+BiFPN（150 epoch）混淆矩阵。该图展示真实类别与预测类别之间的对应关系，可用于定位误检集中的类别对，并指导后续补充样本或设计针对性增强。若某一类别的非对角线数值偏高，应优先检查标注框质量和同类难例数量。"),
        ("confusion_matrix_normalized.png", "图 12  Baseline+CBAM+BiFPN（150 epoch）归一化混淆矩阵。归一化后能够排除类别数量差异对观察的影响，更适合判断少样本类别是否被系统性误判。对于验收场景，归一化矩阵能帮助发现单个类别在复杂背景下的薄弱点。"),
    ]:
        path = PNG_DIR / key
        if path.exists():
            doc.add_image(path, caption)
    doc.add_heading("6.4 消融验证与模型选型", level=2)
    compact_rows = []
    for row in ablation_rows:
        compact_rows.append(
            [
                row.get("Rank", ""),
                row.get("Experiment", ""),
                fnum(row.get("Precision")),
                fnum(row.get("Recall")),
                fnum(row.get("mAP50")),
                fnum(row.get("mAP50-95")),
                fnum(row.get("FPS"), 2),
                fnum(row.get("Delta_mAP50-95")),
            ]
        )
    doc.add_table(table_from_rows(["Rank", "模型组合", "Precision", "Recall", "mAP50", "mAP50-95", "FPS", "ΔmAP50-95"], compact_rows))
    add_paragraphs(
        doc,
        [
            "消融结果显示，Baseline+CBAM 取得最高 mAP50-95=0.80125，说明注意力机制对饮片纹理、边缘和局部形态特征具有明显正向作用。Baseline+CBAM+BiFPN 的 mAP50-95=0.80076，仅比最高精度低 0.00049，但 FPS 达到 302.73，更适合作为桌面端和 Web 端默认模型。Baseline+GhostConv 的 mAP50-95=0.79822，FPS=306.11，说明轻量化卷积在当前数据上能够保持较好精度，同时提供速度参考。",
            "Baseline 单独模型的 mAP50-95=0.79533，已经具备较强基础性能，这说明 YOLOv8n 与增强数据集本身能够完成主要检测任务。但改进模块仍然带来可观察提升，尤其是 CBAM 和 CBAM+BiFPN。与 Baseline 相比，CBAM 提升约 0.0059，CBAM+BiFPN 提升约 0.0054。虽然数值看似不大，但在高 mAP 区间继续提升并不容易，这部分提升说明模块确实对细粒度识别产生贡献。",
            "Focal Loss 与 FullModel 没有成为最终方案。Focal 相关组合在当前配置下 mAP50-95 下降，可能说明难例权重调整与数据规模、预训练迁移或增强策略之间没有达到最佳平衡。FullModel 将多个模块叠加后也未取得最优结果，说明复杂结构可能引入优化难度或特征分布变化。该结论符合消融实验目的：不是所有理论上有益的模块都适合当前任务，必须通过统一数据和指标进行验证。",
            "最终选型强调工程可用性。Baseline+CBAM 是最高精度参考，可用于展示模型能力上限；Baseline+CBAM+BiFPN 是默认部署模型，适合桌面端和 Web 端稳定运行；Baseline+GhostConv 作为轻量化结构对照，保留在报告中用于说明速度与精度权衡。这样的策略比简单选择最高 mAP 更稳妥，因为实践系统需要同时满足检测效果、响应速度、结构可解释和维护便利。",
        ],
    )
    for key, caption in [
        ("ablation_map50_95_bar.png", "图 13  消融实验 mAP50-95 排名图。该图直观呈现各结构组合的精度排序，能够看出 CBAM 单模块和 CBAM+BiFPN 组合处于前列，Focal 相关组合和 FullModel 在当前配置下未取得最优结果。"),
        ("ablation_speed_accuracy.png", "图 14  精度与速度权衡图。横轴 FPS 与纵轴 mAP50-95 共同反映部署价值，Baseline+CBAM+BiFPN 位于高精度和高速度的平衡区域，GhostConv 则提供轻量化结构对照。"),
    ]:
        path = PNG_DIR / key
        if path.exists():
            doc.add_image(path, caption)
    doc.add_heading("6.5 结果可靠性与误差来源讨论", level=2)
    add_paragraphs(
        doc,
        [
            "当前结果具备较好的可复核性，因为训练曲线、混淆矩阵和消融表均由项目目录中的日志与汇总文件生成。相比手工记录，自动生成可以减少数值遗漏和图表错配。报告同时保留主训练结果和正式消融结果，能够区分“单次基线训练表现”和“统一配置下的模型比较”，避免把不同实验条件下的数值直接混在一起比较。",
            "仍需注意，验证集指标不能完全替代现场验收表现。验证集来自当前数据划分，虽然包含多类样本和增强数据，但真实现场可能出现新的光照、背景、相机焦距、饮片批次和摆放方式。尤其是饮片外观受产地、切制工艺和保存状态影响较大，模型在已有数据上的高指标并不意味着对所有真实样本都不会失效。因此，后续应把现场失败样本作为难例回流，形成持续优化流程。",
            "误差来源主要包括三类。第一是数据误差，如类别误标、框体偏移、少数类样本不足；第二是模型误差，如对相似纹理的区分能力不足、对重叠目标的边界定位不准；第三是系统误差，如输入分辨率压缩、阈值设置过高或过低、视频帧模糊等。报告中的曲线和矩阵能帮助定位前两类问题，系统端日志和历史记录则能帮助发现第三类问题。",
        ],
    )

    doc.add_heading("七、项目验收覆盖与问题讨论", level=1)
    doc.add_heading("7.1 验收覆盖情况", level=2)
    doc.add_table(
        table_from_rows(
            ["验收项", "完成情况", "说明"],
            [
                ["15 类饮片识别", "已完成", "固定类别顺序并贯穿数据、训练、后端和界面"],
                ["多目标检测与自动计数", "已完成", "检测结果包含 total_count 与 class_counts"],
                ["实时摄像头检测", "已完成", "桌面端摄像头线程与 Web getUserMedia 均已实现"],
                ["CPU/GPU 兼容", "已完成", "训练与推理均提供 auto/cpu/cuda 入口"],
                ["Web 与桌面端运行", "已完成", "Next.js Web 和 PySide6 桌面端均完成主要页面"],
                ["消融实验完整", "已完成", "11 组模型，输出 CSV、Excel、曲线和混淆矩阵"],
                ["报告素材自动生成", "已完成", "report_generator.py 生成 Excel、CSV、PNG 和 Word 材料"],
                ["专用硬件方向运行", "不纳入本版", "因硬件支持条件调整，已从交付与验收范围中移除"],
            ],
        )
    )
    add_paragraphs(
        doc,
        [
            "从验收覆盖情况看，项目已经完成任务书核心要求：15 类饮片数据集、YOLO 标注规范、实质性模型改进、训练评估、消融对比、混淆矩阵、桌面端、Web 端、CPU/GPU 兼容和自动化报告材料。与上一版叙述相比，本版报告删除了不再交付的专用硬件方向部署内容，使验收表与实际项目范围保持一致。",
            "验收时建议按照“数据证据、算法证据、系统证据、材料证据”的顺序展示。数据证据包括类别表、标注示例和增强样本；算法证据包括结构对比图、核心代码片段和消融表；系统证据包括桌面端或 Web 端的图片、视频和摄像头检测；材料证据包括自动生成的 Excel、CSV、PNG 和 Word 文件。这样的展示顺序能够让评审看到项目不是孤立模块，而是从数据到应用的完整链路。",
        ],
    )
    doc.add_heading("7.2 现场验收风险与改进方向", level=2)
    add_paragraphs(
        doc,
        [
            "任务书明确采用复杂真实样本现场验收，如果模型只能识别规整摆放样本，而在遮挡、污损、堆叠和复杂背景下失效，将影响项目评价。因此，当前项目虽然已经生成压力样本并完成训练评估，仍需要在最终验收前使用全新样本进行复核。尤其应关注易混类别、暗光图像、反光背景和多实例重叠场景，这些往往比普通单目标图片更能检验系统鲁棒性。",
            "后续数据改进应围绕失败模式展开，而不是盲目扩大数量。若混淆矩阵显示甘草片与桔梗容易互相误判，应补充二者混合摆放、不同角度和不同光照的样本；若某类漏检较多，应检查该类在训练集中是否存在尺度过小或标注框偏移问题；若复杂背景误检较多，应补充包含桌面纹理、包装袋、纸张和器皿的负背景或半负背景样本。数据补充应与错误分析绑定，才能有效提升下一轮训练。",
            "系统层面的风险主要是运行环境差异和用户输入差异。不同电脑的 CUDA、PyTorch、浏览器权限和摄像头驱动可能不同，可能导致推理速度、摄像头打开方式或文件上传行为存在差异。为降低风险，项目应保留 CPU 回退路径、清晰的启动脚本、固定的模型路径规则和必要的错误提示。验收前建议分别测试图片、视频、摄像头和批量检测，确保每条核心流程都可用。",
            "报告材料也需要保持版本一致。若后续重新训练模型，应同步更新训练曲线、混淆矩阵、消融表和 Word 报告，不能只替换权重而沿用旧图表。自动生成脚本已经具备这一能力，后续维护时应优先通过脚本再生成材料，而不是手动截图拼接。这样可以保证答辩材料、项目文件和最终模型之间保持一致。",
        ],
    )
    doc.add_heading("7.3 系统测试与交付复核", level=2)
    add_paragraphs(
        doc,
        [
            "系统交付前需要把模型指标验证和软件流程验证分开检查。模型指标验证主要关注验证集、压力样本、混淆矩阵和消融结果，回答模型是否具备识别能力；软件流程验证则关注接口、界面、文件保存、历史记录、统计图表和导出功能，回答系统是否能够被连续使用。两类验证不能互相替代：模型指标较高但界面无法上传文件，说明工程交付不完整；界面运行顺畅但复杂样本漏检较多，说明算法鲁棒性仍需补强。本文将两者都纳入验收讨论，是为了让报告更符合实践项目而不是单次训练记录。",
            "后端复核建议采用最小闭环方式进行。首先调用类别接口确认 15 类中英文名称和 class_id 顺序正确；其次使用同一张已知图片调用图片检测接口，检查返回的 detections、total_count、class_counts、output_path 和置信度字段是否完整；再次使用短视频和批量图片目录测试接口稳定性，观察异常文件是否会被跳过或给出明确错误；最后检查 SQLite 中是否写入检测来源、时间、结果路径和类别统计。只要后端输出结构稳定，桌面端和 Web 端就能围绕同一套字段展示结果，系统维护成本也会明显降低。",
            "桌面端复核重点在本机交互体验。需要检查图片选择、视频选择、摄像头打开、检测停止、结果保存、历史记录查看和 Excel 导出等功能是否连续可用；同时观察检测过程中界面是否卡顿、按钮状态是否及时更新、GPU/CPU 状态是否显示准确。桌面端通常用于现场演示，因此稳定性比功能数量更重要。若某些功能在特定电脑上受驱动或权限影响，应优先保证图片检测、视频检测和历史记录三条主流程可用，再逐步完善摄像头与批量处理细节。",
            "Web 端复核重点在浏览器兼容和接口协同。需要检查上传图片后是否正确显示带框结果，摄像头权限被允许或拒绝时是否有合理反馈，历史记录和统计页面是否能从后端读取数据，模型状态页是否能反映当前权重和设备信息。由于 Web 端通过 HTTP 调用后端，跨域配置、服务端口、文件访问路径和输出目录权限都可能影响结果展示。验收前应固定启动顺序，先启动后端并确认健康状态，再启动前端页面，最后使用同一批样本完成上传、检测、保存和统计查看。",
            "交付材料复核应检查文件是否齐全、路径是否清楚、数值是否一致。CSV 中应包含训练指标和消融汇总，Excel 中应包含多工作表和图表引用，PNG 中应包含 Loss 曲线、mAP 曲线、混淆矩阵、结构对比图、数据增强图和消融可视化，Word 报告应能正常打开并嵌入所有核心图片。更重要的是，报告中的 mAP50、mAP50-95、FPS 和模型排名应与 summary.csv 保持一致，不能出现表格和正文结论互相冲突的情况。当前脚本从同一批源文件生成材料，正是为了降低这种版本不一致风险。",
            "在当前交付范围内，本地 CPU/GPU、桌面端和 Web 端已经能够覆盖课程实践的主要验收场景。专用硬件方向已经因为硬件支持条件调整而移除，因此测试计划不再围绕该方向展开，也不再把任何轻量化模块绑定为专门硬件方案。GhostConv 在报告中只作为速度与模型复杂度的消融对照保留，最终展示和验收仍以本地电脑上的默认模型、统一后端接口和两个客户端为主。这样处理能够让项目说明与真实交付保持一致，避免报告承诺超过当前工程可验证范围。",
        ],
    )
    doc.add_heading("7.4 项目局限性", level=2)
    add_paragraphs(
        doc,
        [
            "第一，当前数据集虽然经过增强，但真实业务环境仍可能比已有样本更复杂。饮片批次差异、相机白平衡、桌面材质、光照方向和拍摄距离都会影响识别效果。第二，当前模型主要基于 YOLOv8n 改进，模型容量有限，对于极端相似类别和严重遮挡目标仍可能出现误判。第三，当前系统侧重本地演示和课程实践，权限管理、多人协作、长期数据库维护和生产级监控还没有展开。",
            "第四，消融实验虽然覆盖五类改进，但每个模块的超参数搜索仍不充分。例如 Focal Loss 的 gamma 和 alpha、CBAM 的插入位置、BiFPN 的层数和通道设置，都可能影响最终结果。当前结论适用于本项目数据、训练配置和硬件条件，不应被解释为所有饮片检测任务中的绝对最优。后续若数据规模扩大，应重新进行更系统的参数搜索和交叉验证。",
            "第五，当前系统的检测结果仍以视觉框和统计数为主，尚未深入结合药品批次、库存编码、处方校验和药房业务系统。如果项目继续扩展，可以在识别结果之上增加人工复核标签、错检样本回收、库存记录关联和异常提醒等功能，使模型输出真正进入业务闭环。",
        ],
    )

    doc.add_heading("八、总结与展望", level=1)
    doc.add_heading("8.1 项目总结", level=2)
    add_paragraphs(
        doc,
        [
            "本项目围绕中医药饮片检测与识别任务，完成了从数据集构建、YOLO 标注规范、离线增强、改进模型设计、训练消融、后端服务、桌面端、Web 端到自动化报告材料生成的完整实践链路。项目不以单次实验结果为终点，而是将模型训练结果转化为可运行系统和可验收材料，体现了深度学习算法在具体业务场景中的工程化落地过程。",
            "在数据层面，项目固定 15 类饮片类别顺序，统计训练集和验证集图像与标注框数量，完成标注质量检查、增强样本生成和压力测试样本准备。在算法层面，项目以 YOLOv8n 为基线，实现 CBAM、BiFPN、GhostConv、Decoupled Head 和 Focal Loss 五类改进，并通过 11 组消融结果确定默认模型。在系统层面，项目提供 FastAPI 后端、PySide6 桌面端和 Next.js Web 端，使图片、视频、摄像头检测、历史记录和统计分析形成完整工作流。",
            "从结果看，Baseline+CBAM 取得最高 mAP50-95，说明注意力机制对饮片细粒度特征具有显著帮助；Baseline+CBAM+BiFPN 在精度接近最高的同时保持较高 FPS，因此作为桌面端和 Web 端默认模型更符合项目实际；GhostConv、Decoupled Head 和 Focal Loss 的结果则为后续模型改进提供了对照和经验。最终报告中的所有核心图表均由项目文件自动生成，满足任务书关于训练曲线、混淆矩阵、消融表和图注说明的要求。",
        ],
    )
    doc.add_heading("8.2 后续展望", level=2)
    add_paragraphs(
        doc,
        [
            "后续工作应继续围绕真实业务场景加强数据闭环。一方面，将现场验收中失败的样本纳入难例库，定期进行再标注、再训练和再评估；另一方面，将系统检测记录中的类别频次、置信度分布和误检样本与训练集统计联动，形成面向药房盘点和仓储管理的持续优化流程。数据闭环建立后，模型优化将不再依赖一次性采集，而是能够随着使用过程不断完善。",
            "算法层面可以继续探索更稳健的细粒度检测方法。例如，在保持 YOLOv8 推理速度优势的基础上，引入更合理的注意力插入位置、更轻量的多尺度融合方式、针对易混类别的重采样策略或知识蒸馏方法。对于 Focal Loss 等当前表现不佳的方向，也可以在扩大数据规模后重新搜索参数，判断其是否在更复杂样本上发挥作用。",
            "系统层面可以进一步提高可用性和可维护性。桌面端可增加人工复核入口、错检样本一键保存和批量结果对比；Web 端可增加更丰富的统计图表、模型版本展示和检测任务队列；后端可增加配置化阈值、模型热切换和更完善的错误日志。随着项目迭代，系统应从“能检测”逐步走向“能复核、能追踪、能持续优化”。",
            "报告和系统的后续维护也应形成固定流程。每次更新数据集或模型权重后，应先运行数据检查，再完成训练评估和消融对照，随后重新生成 CSV、Excel、PNG 与 Word 材料，最后使用桌面端和 Web 端各完成一次闭环检测。只有当数据、模型、接口、界面和报告五个环节的结果一致时，才应把该版本作为正式提交版本。这样可以减少临近答辩时由于文件版本混乱、图表未同步或模型路径错误带来的风险。",
            "总体而言，本项目已经完成任务书要求的核心实践内容，并根据实际硬件条件重新明确部署范围。当前版本重点服务于本地 CPU/GPU、PC 桌面端和 Web 端，既能展示改进 YOLOv8 在中医药饮片检测中的效果，也能体现数据、算法、系统和报告材料之间的完整工程闭环。",
        ],
    )

    doc.save(REPORT_PATH)


if __name__ == "__main__":
    build_report()
