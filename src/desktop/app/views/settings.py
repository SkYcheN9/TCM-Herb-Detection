"""Settings view."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QWidget
from qfluentwidgets import BodyLabel, ComboBox, LineEdit, PushButton, Slider, SpinBox, SwitchButton
from qfluentwidgets import FluentIcon as FIF

from ..widgets.layout import Page, SectionCard


class SettingsView(Page):
    """Application settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "settingsView",
            "Settings",
            "配置运行设备、模型路径、保存位置和界面偏好。",
            parent,
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        runtime = SectionCard("运行配置", self)
        runtime_grid = QGridLayout()
        runtime_grid.setHorizontalSpacing(12)
        runtime_grid.setVerticalSpacing(12)

        device = ComboBox(runtime)
        device.addItems(["自动选择", "CUDA:0", "CPU"])

        workers = SpinBox(runtime)
        workers.setRange(0, 16)
        workers.setValue(4)

        cache = SwitchButton(runtime)
        cache.setText("缓存最近使用的模型")
        cache.setChecked(True)

        runtime_grid.addWidget(BodyLabel("默认设备", runtime), 0, 0)
        runtime_grid.addWidget(device, 0, 1)
        runtime_grid.addWidget(BodyLabel("后台任务数", runtime), 1, 0)
        runtime_grid.addWidget(workers, 1, 1)
        runtime_grid.addWidget(cache, 2, 0, 1, 2)
        runtime.layout.addLayout(runtime_grid)

        paths = SectionCard("路径", self)
        path_grid = QGridLayout()
        path_grid.setHorizontalSpacing(12)
        path_grid.setVerticalSpacing(12)
        for row, (label, placeholder) in enumerate(
            [
                (
                    "模型权重",
                    "final_results_full/reports/ablation/runs/baseline_cbam_bifpn/weights/best.pt",
                ),
                ("导出目录", "reports/desktop_exports"),
                ("历史数据库", "src/desktop/data/history.db"),
            ]
        ):
            line = LineEdit(paths)
            line.setPlaceholderText(placeholder)
            path_grid.addWidget(BodyLabel(label, paths), row, 0)
            path_grid.addWidget(line, row, 1)
            path_grid.addWidget(PushButton(FIF.FOLDER, "", paths), row, 2)
        paths.layout.addLayout(path_grid)

        interface = SectionCard("界面", self)
        interface_grid = QGridLayout()
        interface_grid.setHorizontalSpacing(12)
        interface_grid.setVerticalSpacing(12)

        theme = ComboBox(interface)
        theme.addItems(["深色模式", "浅色模式", "跟随系统"])

        scale = Slider(Qt.Horizontal, interface)
        scale.setRange(80, 140)
        scale.setValue(100)

        interface_grid.addWidget(BodyLabel("主题", interface), 0, 0)
        interface_grid.addWidget(theme, 0, 1)
        interface_grid.addWidget(BodyLabel("界面缩放", interface), 1, 0)
        interface_grid.addWidget(scale, 1, 1)
        interface.layout.addLayout(interface_grid)

        safety = SectionCard("检测行为", self)
        for text in ["低置信度结果保留为待复核", "检测完成后写入历史", "导出时包含类别统计", "启动时检查模型路径"]:
            switch = SwitchButton(safety)
            switch.setText(text)
            switch.setChecked(True)
            safety.layout.addWidget(switch)

        grid.addWidget(runtime, 0, 0)
        grid.addWidget(paths, 0, 1)
        grid.addWidget(interface, 1, 0)
        grid.addWidget(safety, 1, 1)
        self.root_layout.addLayout(grid)

        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(PushButton(FIF.SYNC, "恢复默认", self))
        actions.addWidget(PushButton(FIF.SAVE, "保存设置", self))
        self.root_layout.addLayout(actions)
        self.root_layout.addStretch(1)

