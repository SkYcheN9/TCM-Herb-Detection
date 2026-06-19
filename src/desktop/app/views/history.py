"""History view."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QTableWidgetItem, QWidget
from qfluentwidgets import BodyLabel, ComboBox, FluentIcon as FIF, InfoBar, InfoBarPosition, LineEdit, PushButton, TableWidget

from ..services.excel_exporter import export_records_to_excel
from ..services.history_store import DetectionRecord, HistoryStore
from ..widgets.layout import Page, SectionCard


class HistoryView(Page):
    """Detection records and export center."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "historyView",
            "History",
            "查看图片、视频、摄像头检测记录，并导出 Excel。",
            parent,
        )

        self.store = HistoryStore()
        self.records: list[DetectionRecord] = []

        self.metrics_holder = QGridLayout()
        self.metrics_holder.setContentsMargins(0, 0, 0, 0)
        self.metrics_holder.setHorizontalSpacing(14)
        self.metrics_holder.setVerticalSpacing(14)
        self.root_layout.addLayout(self.metrics_holder)

        card = SectionCard("检测记录", self)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        self.search = LineEdit(card)
        self.search.setPlaceholderText("搜索文件名、类别或日期")

        self.source = ComboBox(card)
        self.source.addItems(["全部来源", "图片检测", "视频检测", "摄像头检测"])

        refresh_button = PushButton(FIF.SYNC, "刷新", card)
        export_button = PushButton(FIF.DOWNLOAD, "导出 Excel", card)
        open_button = PushButton(FIF.FOLDER, "打开导出目录", card)

        toolbar.addWidget(self.search, 1)
        toolbar.addWidget(self.source)
        toolbar.addWidget(refresh_button)
        toolbar.addWidget(export_button)
        toolbar.addWidget(open_button)
        card.layout.addLayout(toolbar)

        self.table = TableWidget(card)
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["ID", "时间", "模式", "输入源", "目标数", "类别统计", "性能", "状态"])
        self.table.verticalHeader().hide()
        self.table.setMinimumHeight(380)
        card.layout.addWidget(self.table)

        note = BodyLabel("结果图片和视频保存在 reports/desktop，历史记录保存在桌面端 SQLite 数据库。", card)
        note.setObjectName("mutedLabel")
        note.setWordWrap(True)
        card.layout.addWidget(note)

        self.root_layout.addWidget(card, 1)

        refresh_button.clicked.connect(self.refresh)
        export_button.clicked.connect(self.export_excel)
        open_button.clicked.connect(self.open_export_folder)
        self.search.textChanged.connect(self.render_table)
        self.source.currentIndexChanged.connect(self.render_table)
        self.refresh()

    def refresh(self) -> None:
        """Reload records from SQLite."""
        self.records = self.store.list_records()
        self.render_metrics()
        self.render_table()

    def render_metrics(self) -> None:
        """Render summary KPI cards."""
        while self.metrics_holder.count():
            item = self.metrics_holder.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        summary = self.store.summary()
        metrics = [
            ("今日检测", str(summary["today"]), "当天保存的检测记录"),
            ("累计记录", str(summary["total"]), "图片 / 视频 / 摄像头"),
            ("高频类别", str(summary["top_class"]), "按历史记录累计统计"),
            ("可导出", str(summary["exportable"]), "Excel 导出记录数"),
        ]
        from ..widgets.layout import MetricCard

        for index, (title, value, detail) in enumerate(metrics):
            self.metrics_holder.addWidget(MetricCard(title, value, detail), index // 4, index % 4)

    def render_table(self) -> None:
        """Render filtered records."""
        records = self._filtered_records()
        self.table.setRowCount(len(records))

        for row_index, record in enumerate(records):
            classes = ", ".join(f"{name}: {count}" for name, count in record.class_counts.items()) or "-"
            row = [
                record.id,
                record.created_at,
                record.mode,
                record.source_path,
                record.total_count,
                classes,
                self._format_performance(record),
                record.status,
            ]
            for col_index, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                if col_index in {0, 4, 6}:
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_index, col_index, item)

        self.table.resizeColumnsToContents()

    def export_excel(self) -> None:
        """Export filtered records to Excel."""
        records = self._filtered_records()
        if not records:
            self._show_error("没有可导出的记录", "请先完成一次检测，或调整筛选条件。")
            return

        path = export_records_to_excel(records)
        self._show_success("Excel 导出完成", str(path))
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    def open_export_folder(self) -> None:
        from ..services.paths import EXPORT_OUTPUT_DIR, ensure_desktop_dirs

        ensure_desktop_dirs()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(EXPORT_OUTPUT_DIR)))

    def _filtered_records(self) -> list[DetectionRecord]:
        keyword = self.search.text().strip().lower()
        mode = self.source.currentText()
        records = self.records
        if mode != "全部来源":
            records = [record for record in records if record.mode == mode]
        if keyword:
            records = [
                record
                for record in records
                if keyword in record.created_at.lower()
                or keyword in record.mode.lower()
                or keyword in record.source_path.lower()
                or keyword in ", ".join(record.class_counts).lower()
            ]
        return records

    @staticmethod
    def _format_performance(record: DetectionRecord) -> str:
        if record.performance_unit.lower() == "ms":
            return f"{record.fps:.0f} ms"
        return f"{record.fps:.1f} FPS"

    def _show_error(self, title: str, content: str) -> None:
        InfoBar.error(
            title=title,
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=3500,
            parent=self.window(),
        )

    def _show_success(self, title: str, content: str) -> None:
        InfoBar.success(
            title=title,
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=3500,
            parent=self.window(),
        )
