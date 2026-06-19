"""Dashboard view."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, PrimaryPushButton, PushButton, SubtitleLabel
from qfluentwidgets import FluentIcon as FIF

from ..widgets.layout import HeroPanel, Page, SectionCard, TagRow, add_metric_grid


class DashboardView(Page):
    """System overview for model and detection operations."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "dashboardView",
            "Dashboard",
            "实时掌握检测工作站状态、模型能力和近期任务概览。",
            parent,
        )

        hero = HeroPanel(self)
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(26, 24, 26, 24)
        hero_layout.setSpacing(24)

        left = QVBoxLayout()
        left.setSpacing(12)

        eyebrow = CaptionLabel("INDUSTRIAL VISION WORKSTATION", hero)
        eyebrow.setObjectName("eyebrowLabel")
        title = SubtitleLabel("TCM-SliceAI Detection Console", hero)
        detail = BodyLabel(
            "面向 15 类中医药饮片的检测、计数与结果追踪。当前阶段已完成训练侧 Phase 1，桌面端正在进入客户端开发。",
            hero,
        )
        detail.setObjectName("mutedLabel")
        detail.setWordWrap(True)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(PrimaryPushButton(FIF.CAMERA, "进入检测", hero))
        actions.addWidget(PushButton(FIF.HISTORY, "查看历史", hero))
        actions.addStretch(1)

        left.addWidget(eyebrow)
        left.addWidget(title)
        left.addWidget(detail)
        left.addStretch(1)
        left.addLayout(actions)

        right = QGridLayout()
        right.setHorizontalSpacing(10)
        right.setVerticalSpacing(10)
        right.addWidget(_StatusPill("模型状态", "Baseline 可用", "statusPill", hero), 0, 0)
        right.addWidget(_StatusPill("运行设备", "GPU 优先", "statusPill", hero), 0, 1)
        right.addWidget(_StatusPill("Detector", "待接入", "warningPill", hero), 1, 0)
        right.addWidget(_StatusPill("数据集", "15 类固定", "statusPill", hero), 1, 1)

        hero_layout.addLayout(left, 3)
        hero_layout.addLayout(right, 2)
        self.root_layout.addWidget(hero)

        add_metric_grid(
            self.root_layout,
            [
                ("类别数量", "15", "饮片类别顺序固定"),
                ("Baseline mAP50", "94.04%", "Phase 1 最优验证结果"),
                ("验证样本", "193", "规范化验证集数量"),
                ("Smoke FPS", "169.4", "消融流程快速验证记录"),
            ],
        )

        lower = QHBoxLayout()
        lower.setSpacing(14)

        pipeline = SectionCard("工作流", self)
        pipeline.layout.addWidget(TagRow(["Image", "Video", "Camera", "RTSP", "Batch"], pipeline))
        for step, text in [
            ("01", "选择输入源并加载待检测素材"),
            ("02", "调用统一 Detector 接口完成识别"),
            ("03", "展示 bbox、类别、置信度与计数"),
            ("04", "保存记录并支持后续导出"),
        ]:
            pipeline.layout.addWidget(_TimelineItem(step, text, pipeline))

        model = SectionCard("模型能力", self)
        model.layout.addWidget(BodyLabel("当前已实现模块", model))
        model.layout.addWidget(TagRow(["YOLOv8", "CBAM", "BiFPN", "Focal Loss"], model))
        note = BodyLabel("GhostConv 与 Decoupled Head 属于后续模型阶段，本桌面端不会修改训练或模型代码。", model)
        note.setObjectName("mutedLabel")
        note.setWordWrap(True)
        model.layout.addWidget(note)
        model.layout.addStretch(1)

        lower.addWidget(pipeline, 3)
        lower.addWidget(model, 2)
        self.root_layout.addLayout(lower)
        self.root_layout.addStretch(1)


class _StatusPill(QFrame):
    def __init__(self, title: str, value: str, style_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        title_label = CaptionLabel(title, self)
        title_label.setObjectName("mutedLabel")
        value_label = QLabel(value, self)
        value_label.setObjectName(style_name)
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addWidget(value_label)


class _TimelineItem(QWidget):
    def __init__(self, step: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        number = QLabel(step, self)
        number.setObjectName("statusPill")
        number.setFixedWidth(46)
        number.setAlignment(Qt.AlignCenter)

        body = BodyLabel(text, self)
        body.setWordWrap(True)

        layout.addWidget(number)
        layout.addWidget(body, 1)

