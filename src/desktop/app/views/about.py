"""About view."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, FluentIcon as FIF, PrimaryPushButton, PushButton, SubtitleLabel

from ..widgets.layout import Page, SectionCard, TagRow, add_metric_grid


class AboutView(Page):
    """Project information view."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "aboutView",
            "About",
            "TCM-SliceAI 中医药饮片智能检测与识别系统桌面端。",
            parent,
        )

        overview = SectionCard("项目概览", self)
        title = SubtitleLabel("Traditional Chinese Medicine Decoction Pieces Detection System", overview)
        title.setWordWrap(True)
        overview.layout.addWidget(title)

        body = BodyLabel(
            "本客户端面向 PC 端检测工作流，使用 PySide6 与 qfluentwidgets 构建。当前第一步完成目录结构与主界面，不修改训练代码和模型代码。",
            overview,
        )
        body.setObjectName("mutedLabel")
        body.setWordWrap(True)
        overview.layout.addWidget(body)
        overview.layout.addWidget(TagRow(["PySide6", "qfluentwidgets", "YOLOv8", "15 classes"], overview))

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        buttons.addWidget(PrimaryPushButton(FIF.DOCUMENT, "项目文档", overview))
        buttons.addWidget(PushButton(FIF.GITHUB, "代码仓库", overview))
        buttons.addStretch(1)
        overview.layout.addLayout(buttons)
        self.root_layout.addWidget(overview)

        add_metric_grid(
            self.root_layout,
            [
                ("Phase", "1", "训练与消融基础已完成"),
                ("Desktop", "P1", "当前开发重点"),
                ("Web", "P2", "后续阶段"),
                ("Raspberry Pi", "P3", "边缘部署阶段"),
            ],
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        classes = SectionCard("固定类别", self)
        classes.layout.addWidget(
            TagRow(
                [
                    "zexie",
                    "niuxi",
                    "gaoliangjiang",
                    "mudanpi",
                    "yuzhu",
                    "baizhi",
                    "baishao",
                    "dazao",
                ],
                classes,
            )
        )
        classes.layout.addWidget(
            TagRow(["danshen", "gancao", "baixianpi", "baihe", "sangzhi", "jiegeng", "banlangen"], classes)
        )

        scope = SectionCard("当前范围", self)
        for item in ["完成 src/desktop 客户端目录", "完成主窗口与导航", "完成五个模块页面", "预留后续 Detector 接入位置"]:
            label = CaptionLabel(item, scope)
            label.setObjectName("metaLabel")
            scope.layout.addWidget(label)

        grid.addWidget(classes, 0, 0)
        grid.addWidget(scope, 0, 1)
        self.root_layout.addLayout(grid)
        self.root_layout.addStretch(1)

