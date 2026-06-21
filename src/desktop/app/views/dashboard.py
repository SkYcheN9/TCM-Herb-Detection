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
            "最终实验、模型部署与检测统计总览。",
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
        eyebrow.setStyleSheet("color: #7CE7F4; font-weight: 700;")
        title = SubtitleLabel("TCM-SliceAI Detection Console", hero)
        title.setObjectName("heroTitle")
        title.setStyleSheet("color: #F8FAFC; font-size: 22px; font-weight: 700;")
        detail = BodyLabel(
            "面向 15 类中医药饮片的检测、识别、计数与结果追踪。当前 11 组消融实验、网页端、桌面端和树莓派端部署链路均已完成。",
            hero,
        )
        detail.setObjectName("heroBody")
        detail.setStyleSheet("color: rgba(248, 250, 252, 0.88); font-size: 15px;")
        detail.setWordWrap(True)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.detect_button = PrimaryPushButton(FIF.CAMERA, "进入检测", hero)
        self.history_button = PushButton(FIF.HISTORY, "查看历史", hero)
        actions.addWidget(self.detect_button)
        actions.addWidget(self.history_button)
        actions.addStretch(1)

        left.addWidget(eyebrow)
        left.addWidget(title)
        left.addWidget(detail)
        left.addStretch(1)
        left.addLayout(actions)

        right = QGridLayout()
        right.setHorizontalSpacing(10)
        right.setVerticalSpacing(10)
        right.addWidget(_StatusPill("默认模型", "CBAM+BiFPN", "statusPill", hero), 0, 0)
        right.addWidget(_StatusPill("最高精度", "CBAM 80.12%", "statusPill", hero), 0, 1)
        right.addWidget(_StatusPill("树莓派端", "GhostConv", "statusPill", hero), 1, 0)
        right.addWidget(_StatusPill("数据集", "15 类 / 3225 样本", "statusPill", hero), 1, 1)

        hero_layout.addLayout(left, 3)
        hero_layout.addLayout(right, 2)
        self.root_layout.addWidget(hero)

        add_metric_grid(
            self.root_layout,
            [
                ("消融模型", "11", "公平预训练迁移与候选组合均已完成"),
                ("部署 mAP50-95", "80.08%", "CBAM+BiFPN 平衡精度与速度"),
                ("最高 mAP50-95", "80.12%", "CBAM 单模块精度最高"),
                ("树莓派模型", "GhostConv", "轻量化部署优先"),
            ],
        )

        lower = QHBoxLayout()
        lower.setSpacing(14)

        pipeline = SectionCard("工作流", self)
        pipeline.layout.addWidget(TagRow(["Image", "Video", "Camera", "RTSP", "Batch"], pipeline))
        for step, text in [
            ("01", "选择摄像头、图片或视频输入源"),
            ("02", "调用统一 Detector 接口完成识别"),
            ("03", "展示 bbox、类别、置信度与药材计数"),
            ("04", "保存记录并支持历史查询与导出"),
        ]:
            pipeline.layout.addWidget(_TimelineItem(step, text, pipeline))

        model = SectionCard("模型部署", self)
        model.layout.addWidget(BodyLabel("最终选型", model))
        model.layout.addWidget(TagRow(["YOLOv8n", "CBAM", "BiFPN", "GhostConv"], model))
        note = BodyLabel(
            "网页端和桌面端默认使用 CBAM+BiFPN；最高精度结果为 CBAM；树莓派 5 无算力棒部署使用 GhostConv 轻量模型。Focal Loss 与 FullModel 已保留为负向消融结论。",
            model,
        )
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
        title_label.setObjectName("heroMeta")
        title_label.setStyleSheet("color: rgba(248, 250, 252, 0.82);")
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

