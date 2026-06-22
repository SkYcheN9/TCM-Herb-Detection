"""Detection workspace view."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QBrush, QCloseEvent, QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    ComboBox,
    FluentIcon as FIF,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    Slider,
    SmoothScrollArea,
    SpinBox,
    SwitchButton,
    TableWidget,
)

from ..services.camera_detector import CameraDetectorThread
from ..services.camera_devices import discover_camera_devices
from ..services.camera_snapshot import capture_camera_snapshot
from ..services.gpu_status import query_gpu_status
from ..services.media_detector import DetectionCandidate, DetectionSummary, VideoDetectorThread, detect_image
from ..services.model_locator import find_best_model
from ..widgets.layout import Page, SectionCard, VideoPanel


CHINESE_CLASS_NAMES = {
    "zexie": "泽泻",
    "niuxi": "牛膝",
    "gaoliangjiang": "高良姜",
    "mudanpi": "牡丹皮",
    "yuzhu": "玉竹",
    "baizhi": "白芷",
    "baishao": "白芍",
    "dazao": "大枣",
    "danshen": "丹参",
    "gancao": "甘草",
    "baixianpi": "白鲜皮",
    "baihe": "百合",
    "sangzhi": "桑枝",
    "jiegeng": "桔梗",
    "banlangen": "板蓝根",
}


class DetectionView(Page):
    """Main detection workspace."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "detectionView",
            "Detection",
            "支持摄像头、图片和视频检测，并自动保存结果与历史记录。",
            parent,
        )

        self.model_candidate = find_best_model()
        self.camera_devices = discover_camera_devices()
        self.camera_thread: CameraDetectorThread | None = None
        self.video_thread: VideoDetectorThread | None = None
        self.selected_image_path: Path | None = None
        self.selected_video_path: Path | None = None
        self.last_output_path: Path | None = None

        self.workspace_splitter = QSplitter(Qt.Horizontal, self)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(8)

        self.video_panel = VideoPanel(
            "等待输入源",
            "选择摄像头、图片或视频后点击开始检测，结果会保存到 reports/desktop。",
            self,
        )
        self.video_panel.setMinimumWidth(640)
        self.workspace_splitter.addWidget(self.video_panel)

        side_container = QWidget(self)
        side = QVBoxLayout(side_container)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(14)

        self.source_card = SectionCard("输入源", self)
        self.source_card.setMinimumHeight(250)
        source_grid = QGridLayout()
        source_grid.setHorizontalSpacing(10)
        source_grid.setVerticalSpacing(10)

        self.mode_selector = ComboBox(self.source_card)
        self.mode_selector.addItems(["摄像头检测", "图片检测", "视频检测"])

        self.camera_selector = ComboBox(self.source_card)
        self.camera_selector.addItems([device.label for device in self.camera_devices])
        self.refresh_camera_button = PushButton(FIF.SYNC, "刷新", self.source_card)

        self.file_path = LineEdit(self.source_card)
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText("选择图片或视频文件")
        self.file_path.setMinimumWidth(260)

        self.device_selector = ComboBox(self.source_card)
        self.device_selector.addItems(["自动选择", "CUDA:0", "CPU"])

        self.model_path = LineEdit(self.source_card)
        self.model_path.setReadOnly(True)
        self.model_path.setText(str(self.model_candidate.path) if self.model_candidate else "未找到可用模型")
        self.model_path.setMinimumWidth(260)

        self.model_source = BodyLabel(
            self.model_candidate.source if self.model_candidate else "请先准备 best.pt 权重文件",
            self.source_card,
        )
        self.model_source.setObjectName("mutedLabel")
        self.model_source.setWordWrap(True)

        self.pick_file_button = PushButton(FIF.FOLDER, "选择文件", self.source_card)

        source_grid.addWidget(BodyLabel("模式", self.source_card), 0, 0)
        source_grid.addWidget(self.mode_selector, 0, 1)
        source_grid.addWidget(BodyLabel("摄像头", self.source_card), 1, 0)
        camera_row = QHBoxLayout()
        camera_row.setContentsMargins(0, 0, 0, 0)
        camera_row.setSpacing(8)
        camera_row.addWidget(self.camera_selector, 1)
        camera_row.addWidget(self.refresh_camera_button)

        source_grid.addLayout(camera_row, 1, 1)
        source_grid.addWidget(BodyLabel("文件", self.source_card), 2, 0)
        source_grid.addWidget(self.file_path, 2, 1)
        source_grid.addWidget(BodyLabel("运行设备", self.source_card), 3, 0)
        source_grid.addWidget(self.device_selector, 3, 1)
        source_grid.addWidget(BodyLabel("当前模型", self.source_card), 4, 0)
        source_grid.addWidget(self.model_path, 4, 1)
        self.source_card.layout.addLayout(source_grid)
        self.source_card.layout.addWidget(self.pick_file_button)
        self.source_card.layout.addWidget(self.model_source)

        self.params_card = SectionCard("推理参数", self)
        params_grid = QGridLayout()
        params_grid.setHorizontalSpacing(10)
        params_grid.setVerticalSpacing(12)

        self.conf_slider = Slider(Qt.Horizontal, self.params_card)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(85)
        self.conf_value = QLabel("0.85", self.params_card)
        self.conf_value.setObjectName("valueBadge")
        self.conf_value.setAlignment(Qt.AlignCenter)
        self.conf_value.setMinimumWidth(54)

        self.iou_slider = Slider(Qt.Horizontal, self.params_card)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)
        self.iou_value = QLabel("0.45", self.params_card)
        self.iou_value.setObjectName("valueBadge")
        self.iou_value.setAlignment(Qt.AlignCenter)
        self.iou_value.setMinimumWidth(54)

        self.size_spin = SpinBox(self.params_card)
        self.size_spin.setRange(320, 1280)
        self.size_spin.setSingleStep(32)
        self.size_spin.setValue(640)

        params_grid.addWidget(BodyLabel("置信度", self.params_card), 0, 0)
        params_grid.addWidget(self.conf_slider, 0, 1)
        params_grid.addWidget(self.conf_value, 0, 2)
        params_grid.addWidget(BodyLabel("IoU", self.params_card), 1, 0)
        params_grid.addWidget(self.iou_slider, 1, 1)
        params_grid.addWidget(self.iou_value, 1, 2)
        params_grid.addWidget(BodyLabel("图像尺寸", self.params_card), 2, 0)
        params_grid.addWidget(self.size_spin, 2, 1, 1, 2)
        self.params_card.layout.addLayout(params_grid)

        self.options_card = SectionCard("保存与显示", self)
        for text in ["显示 bbox", "显示类别名称", "显示置信度", "自动计数"]:
            checkbox = CheckBox(text, self.options_card)
            checkbox.setChecked(True)
            self.options_card.layout.addWidget(checkbox)

        self.record_switch = SwitchButton("保存检测记录", self.options_card)
        self.record_switch.setChecked(True)
        self.options_card.layout.addWidget(self.record_switch)

        self.status_card = SectionCard("实时状态", self)
        self.status_card.setMinimumHeight(210)
        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(10)
        status_grid.setVerticalSpacing(10)
        status_grid.setColumnStretch(0, 0)
        status_grid.setColumnStretch(1, 1)

        self.status_value = QLabel("待机", self.status_card)
        self.status_value.setObjectName("statusPill")
        self.status_value.setAlignment(Qt.AlignCenter)
        self.status_value.setMinimumWidth(120)

        self.fps_value = _MetricValue("0.0 FPS", self.status_card)
        self.total_value = _MetricValue("0", self.status_card)
        self.progress_value = BodyLabel("-", self.status_card)
        self.progress_value.setObjectName("mutedLabel")
        self.progress_value.setWordWrap(True)
        self.progress_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.gpu_value = BodyLabel(query_gpu_status().text, self.status_card)
        self.gpu_value.setObjectName("mutedLabel")
        self.gpu_value.setWordWrap(True)
        self.gpu_value.setTextInteractionFlags(Qt.TextSelectableByMouse)

        status_grid.addWidget(BodyLabel("状态", self.status_card), 0, 0)
        status_grid.addWidget(self.status_value, 0, 1)
        status_grid.addWidget(BodyLabel("性能", self.status_card), 1, 0)
        status_grid.addWidget(self.fps_value, 1, 1)
        status_grid.addWidget(BodyLabel("总数量", self.status_card), 2, 0)
        status_grid.addWidget(self.total_value, 2, 1)
        status_grid.addWidget(BodyLabel("进度", self.status_card), 3, 0)
        status_grid.addWidget(self.progress_value, 3, 1)
        status_grid.addWidget(BodyLabel("GPU", self.status_card), 4, 0)
        status_grid.addWidget(self.gpu_value, 4, 1)
        self.status_card.layout.addLayout(status_grid)

        self.classes_card = SectionCard("药材计数", self)
        self.class_counts = QVBoxLayout()
        self.class_counts.setSpacing(8)
        self.classes_card.layout.addLayout(self.class_counts)
        self._render_class_counts({})

        self.details_card = SectionCard("检测明细", self)
        self.details_table = TableWidget(self.details_card)
        self.details_table.setColumnCount(3)
        self.details_table.setHorizontalHeaderLabels(["类别", "置信度", "定位框"])
        self.details_table.verticalHeader().hide()
        self.details_table.setMinimumHeight(180)
        self.details_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.details_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.details_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.details_card.layout.addWidget(self.details_table)
        details_note = BodyLabel("低置信候选仅用于复核混合/重叠饮片，不参与正式计数。", self.details_card)
        details_note.setObjectName("mutedLabel")
        details_note.setWordWrap(True)
        self.details_card.layout.addWidget(details_note)
        self._render_detection_details([])

        self.start_button = PrimaryPushButton(FIF.PLAY, "开始检测", self)
        self.snapshot_button = PushButton(FIF.PHOTO, "拍照识别", self)
        self.stop_button = PushButton(FIF.PAUSE, "停止", self)
        self.open_output_button = PushButton(FIF.FOLDER, "打开结果", self)
        self.clear_button = PushButton(FIF.SYNC, "清空画面", self)
        self.stop_button.setEnabled(False)
        self.open_output_button.setEnabled(False)

        run_buttons = QGridLayout()
        run_buttons.setHorizontalSpacing(10)
        run_buttons.setVerticalSpacing(10)
        run_buttons.addWidget(self.start_button, 0, 0)
        run_buttons.addWidget(self.snapshot_button, 0, 1)
        run_buttons.addWidget(self.stop_button, 1, 0)
        run_buttons.addWidget(self.open_output_button, 1, 1)
        run_buttons.addWidget(self.clear_button, 2, 0, 1, 2)

        side.addWidget(self.source_card)
        side.addWidget(self.params_card)
        side.addWidget(self.options_card)
        side.addWidget(self.status_card)
        side.addWidget(self.classes_card)
        side.addWidget(self.details_card)
        side.addLayout(run_buttons)
        side.addStretch(1)

        side_scroll = SmoothScrollArea(self)
        side_scroll.setWidget(side_container)
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side_scroll.setMinimumWidth(460)
        side_scroll.setMaximumWidth(660)
        self.workspace_splitter.addWidget(side_scroll)
        self.workspace_splitter.setStretchFactor(0, 1)
        self.workspace_splitter.setStretchFactor(1, 0)
        self.workspace_splitter.setSizes([1220, 520])
        self.root_layout.addWidget(self.workspace_splitter, 1)

        self.mode_selector.currentIndexChanged.connect(self._update_mode_controls)
        self.conf_slider.valueChanged.connect(self._update_threshold_labels)
        self.iou_slider.valueChanged.connect(self._update_threshold_labels)
        self.pick_file_button.clicked.connect(self.pick_source_file)
        self.refresh_camera_button.clicked.connect(self.refresh_camera_devices)
        self.start_button.clicked.connect(self.start_detection)
        self.snapshot_button.clicked.connect(self.capture_snapshot_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.open_output_button.clicked.connect(self.open_last_output)
        self.clear_button.clicked.connect(self.clear_current_view)
        self._update_threshold_labels()
        self._update_mode_controls()

    def start_detection(self) -> None:
        """Start detection for the selected source type."""
        mode = self.mode_selector.currentText()
        if mode == "摄像头检测":
            self.start_camera_detection()
        elif mode == "图片检测":
            self.start_image_detection()
        else:
            self.start_video_detection()

    def start_camera_detection(self) -> None:
        """Start live camera detection."""
        if self.camera_thread and self.camera_thread.isRunning():
            return
        if not self._ensure_model():
            return
        if not self.camera_devices:
            self._show_error("未发现摄像头", "请先打开 Camo/摄像头软件，然后点击“刷新”重新检测摄像头。")
            return

        camera_device = self.camera_devices[self.camera_selector.currentIndex()]
        self.camera_thread = CameraDetectorThread(
            model_path=self.model_candidate.path,
            camera_index=camera_device.index,
            camera_backend=camera_device.backend,
            conf=self.conf_slider.value() / 100,
            iou=self.iou_slider.value() / 100,
            imgsz=self.size_spin.value(),
            device_mode=self.device_selector.currentText(),
            save_record=self.record_switch.isChecked(),
            parent=self,
        )
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.stats_ready.connect(self.update_stats)
        self.camera_thread.status_changed.connect(self.update_status)
        self.camera_thread.error_occurred.connect(self.handle_error)
        self.camera_thread.finished.connect(self._on_thread_finished)

        self._set_running(True)
        self.update_status("启动中")
        self.camera_thread.start()

    def capture_snapshot_detection(self) -> None:
        """Capture one camera frame and run still-image detection on it."""
        if not self._ensure_model():
            return
        if self.camera_thread and self.camera_thread.isRunning():
            self._show_error("请先停止实时检测", "拍照识别会临时打开摄像头，请先停止当前实时检测。")
            return
        if not self.camera_devices:
            self._show_error("未发现摄像头", "请先打开摄像头或 Camo，然后点击“刷新”重新检测摄像头。")
            return

        camera_device = self.camera_devices[self.camera_selector.currentIndex()]
        self._set_running(True, can_stop=False)
        self.update_status("拍照中")
        try:
            snapshot_path = capture_camera_snapshot(camera_device.index, camera_device.backend)
            self.selected_image_path = snapshot_path
            self.file_path.setText(str(snapshot_path))
            self.update_status("快照检测中")
            summary = detect_image(
                image_path=snapshot_path,
                model_path=self.model_candidate.path,
                conf=self.conf_slider.value() / 100,
                iou=self.iou_slider.value() / 100,
                imgsz=self.size_spin.value(),
                device_mode=self.device_selector.currentText(),
                save_record=self.record_switch.isChecked(),
            )
            self.apply_summary(summary)
            self.update_status("拍照识别完成")
            self._show_success("拍照识别完成", f"快照：{snapshot_path.name}；结果已保存。")
        except Exception as exc:
            self.handle_error(str(exc))
        finally:
            self._set_running(False)

    def refresh_camera_devices(self) -> None:
        """Probe camera devices again after USB or virtual camera changes."""
        if self.camera_thread and self.camera_thread.isRunning():
            self._show_error("无法刷新", "请先停止当前摄像头检测。")
            return

        current_index = self.camera_devices[self.camera_selector.currentIndex()].index if self.camera_devices else 0
        self.camera_devices = discover_camera_devices()
        self.camera_selector.clear()
        if not self.camera_devices:
            self.camera_devices = []
            self.snapshot_button.setEnabled(False)
            self._show_error("未发现摄像头", "没有检测到可打开的摄像头，请确认权限、Camo 虚拟摄像头和占用情况。")
            return

        self.camera_selector.addItems([device.label for device in self.camera_devices])
        for option_index, device in enumerate(self.camera_devices):
            if device.index == current_index:
                self.camera_selector.setCurrentIndex(option_index)
                break
        self.snapshot_button.setEnabled(self.mode_selector.currentText() == "摄像头检测")
        self._show_success("摄像头已刷新", f"发现 {len(self.camera_devices)} 个可用摄像头。")

    def start_image_detection(self) -> None:
        """Run still image detection."""
        if not self._ensure_model() or not self._ensure_selected_file(self.selected_image_path, "请先选择图片文件。"):
            return

        self._set_running(True, can_stop=False)
        self.update_status("图片检测中")
        try:
            summary = detect_image(
                image_path=self.selected_image_path,
                model_path=self.model_candidate.path,
                conf=self.conf_slider.value() / 100,
                iou=self.iou_slider.value() / 100,
                imgsz=self.size_spin.value(),
                device_mode=self.device_selector.currentText(),
                save_record=self.record_switch.isChecked(),
            )
            self.apply_summary(summary)
            self.update_status("图片检测完成")
            self._show_success("图片检测完成", f"结果已保存：{summary.output_path}")
        except Exception as exc:
            self.handle_error(str(exc))
        finally:
            self._set_running(False)

    def start_video_detection(self) -> None:
        """Run video file detection."""
        if self.video_thread and self.video_thread.isRunning():
            return
        if not self._ensure_model() or not self._ensure_selected_file(self.selected_video_path, "请先选择视频文件。"):
            return

        self.video_thread = VideoDetectorThread(
            video_path=self.selected_video_path,
            model_path=self.model_candidate.path,
            conf=self.conf_slider.value() / 100,
            iou=self.iou_slider.value() / 100,
            imgsz=self.size_spin.value(),
            device_mode=self.device_selector.currentText(),
            save_record=self.record_switch.isChecked(),
            parent=self,
        )
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.progress_ready.connect(self.update_stats)
        self.video_thread.finished_summary.connect(self.apply_summary)
        self.video_thread.status_changed.connect(self.update_status)
        self.video_thread.error_occurred.connect(self.handle_error)
        self.video_thread.finished.connect(self._on_thread_finished)

        self._set_running(True)
        self.video_thread.start()

    def stop_detection(self) -> None:
        """Stop running camera or video detection."""
        self.stop_camera_detection()
        if self.video_thread and self.video_thread.isRunning():
            self.update_status("正在停止视频")
            self.video_thread.stop()
            self.video_thread.wait(3000)

    def stop_camera_detection(self) -> None:
        """Stop live camera detection."""
        if self.camera_thread and self.camera_thread.isRunning():
            self.update_status("正在停止摄像头")
            self.camera_thread.stop()
            self.camera_thread.wait(3000)

    def pick_source_file(self) -> None:
        """Select an image or video according to current mode."""
        mode = self.mode_selector.currentText()
        if mode == "图片检测":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择图片",
                "",
                "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
            )
            if path:
                self.selected_image_path = Path(path)
                self.file_path.setText(path)
        elif mode == "视频检测":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择视频",
                "",
                "Videos (*.mp4 *.avi *.mov *.mkv *.wmv)",
            )
            if path:
                self.selected_video_path = Path(path)
                self.file_path.setText(path)

    def update_frame(self, image) -> None:
        pixmap = QPixmap.fromImage(image)
        self.video_panel.set_frame(pixmap)

    def update_stats(self, stats: object) -> None:
        data = dict(stats)
        self.fps_value.setText(f"{data.get('fps', 0):.1f} FPS")
        self.total_value.setText(str(data.get("total", 0)))
        self.gpu_value.setText(str(data.get("gpu", "CPU 模式")))
        if data.get("total_frames"):
            self.progress_value.setText(f"{data.get('frame', 0)} / {data.get('total_frames', 0)} 帧")
        elif data.get("backend"):
            self.progress_value.setText(f"摄像头后端：{data.get('backend')}")
        self._render_class_counts(data.get("counts", {}))
        self._render_detection_details([])

    def apply_summary(self, summary: object) -> None:
        """Show final image or video summary."""
        result = summary if isinstance(summary, DetectionSummary) else None
        if not result:
            return
        if result.frame is not None:
            self.update_frame(result.frame)
        self.last_output_path = result.output_path
        self.open_output_button.setEnabled(True)
        self.fps_value.setText(self._format_performance(result.mode, result.fps))
        self.total_value.setText(str(result.total_count))
        self.progress_value.setText(str(result.output_path))
        if result.review_count:
            self.progress_value.setText(
                f"{result.output_path}\n"
                f"提示：{result.review_count} 个目标置信度低于 0.85，混合或重叠饮片建议人工复核。"
            )
        self.gpu_value.setText(query_gpu_status().text)
        self._render_class_counts(result.class_counts)
        self._render_detection_details(result.candidates or [])

    def update_status(self, text: str) -> None:
        self.status_value.setText(text)

    def handle_error(self, message: str) -> None:
        self._show_error("检测失败", message)
        self.update_status("错误")
        self._set_running(False)

    def open_last_output(self) -> None:
        """Open the last output folder."""
        if not self.last_output_path:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output_path.parent)))

    def clear_current_view(self) -> None:
        """Clear the current preview, metrics, and count display."""
        self.stop_detection()
        self._reset_visual_state()
        self.update_status("已清空")

    def _on_thread_finished(self) -> None:
        self._set_running(False)

    def _render_class_counts(self, counts: dict[str, int]) -> None:
        while self.class_counts.count():
            item = self.class_counts.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if not counts:
            empty = BodyLabel("暂无目标", self.classes_card)
            empty.setObjectName("mutedLabel")
            self.class_counts.addWidget(empty)
            return

        for name, count in counts.items():
            self.class_counts.addWidget(_ClassCountRow(name, count, self.classes_card))

    def _render_detection_details(self, candidates: list[DetectionCandidate]) -> None:
        self.details_table.setRowCount(len(candidates))
        current_conf = self.conf_slider.value() / 100
        muted_brush = QBrush(QColor("#8A94A3"))
        for row_index, candidate in enumerate(candidates):
            x1, y1, x2, y2 = candidate.bbox
            values = [
                CHINESE_CLASS_NAMES.get(candidate.class_name, candidate.class_name),
                f"{candidate.confidence * 100:.1f}%",
                f"{x1}, {y1}, {x2}, {y2}",
            ]
            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_index == 1:
                    item.setTextAlignment(Qt.AlignCenter)
                if candidate.confidence < current_conf:
                    item.setForeground(muted_brush)
                self.details_table.setItem(row_index, col_index, item)

    def _set_running(self, running: bool, can_stop: bool = True) -> None:
        self.start_button.setEnabled(not running)
        self.snapshot_button.setEnabled(
            not running and self.mode_selector.currentText() == "摄像头检测" and bool(self.camera_devices)
        )
        self.stop_button.setEnabled(running and can_stop)
        self.mode_selector.setEnabled(not running)
        self.camera_selector.setEnabled(not running)
        self.refresh_camera_button.setEnabled(not running and self.mode_selector.currentText() == "摄像头检测")
        self.device_selector.setEnabled(not running)
        self.pick_file_button.setEnabled(not running)

    def _update_mode_controls(self) -> None:
        mode = self.mode_selector.currentText()
        is_camera = mode == "摄像头检测"
        self.camera_selector.setEnabled(is_camera)
        self.refresh_camera_button.setEnabled(is_camera)
        self.snapshot_button.setEnabled(is_camera and bool(self.camera_devices))
        self.pick_file_button.setEnabled(not is_camera)
        self.file_path.setEnabled(not is_camera)
        if is_camera:
            self.file_path.clear()
        else:
            selected = self.selected_image_path if mode == "图片检测" else self.selected_video_path
            self.file_path.setText(str(selected) if selected else "")
        self.fps_value.setText("耗时 -" if mode == "图片检测" else "0.0 FPS")
        if mode == "摄像头检测":
            self.progress_value.setText("请将药材样本置于摄像头画面内，非药材画面可能产生误检。")
        else:
            self.progress_value.setText("-")
        self._reset_visual_state(update_progress=False, update_status=False)

    def _reset_visual_state(self, update_progress: bool = True, update_status: bool = True) -> None:
        mode = self.mode_selector.currentText()
        self.video_panel.clear_frame()
        self.last_output_path = None
        self.open_output_button.setEnabled(False)
        self.fps_value.setText("耗时 -" if mode == "图片检测" else "0.0 FPS")
        self.total_value.setText("0")
        self.gpu_value.setText(query_gpu_status().text)
        self._render_class_counts({})
        self._render_detection_details([])
        if update_status:
            self.update_status("待机")
        if update_progress:
            if mode == "摄像头检测":
                self.progress_value.setText("请将药材样本置于摄像头画面内，非药材画面可能产生误检。")
            else:
                self.progress_value.setText("-")

    def _update_threshold_labels(self) -> None:
        self.conf_value.setText(f"{self.conf_slider.value() / 100:.2f}")
        self.iou_value.setText(f"{self.iou_slider.value() / 100:.2f}")

    @staticmethod
    def _format_performance(mode: str, value: float) -> str:
        if mode == "图片检测":
            return f"{value:.0f} ms"
        return f"{value:.1f} FPS"

    def _ensure_model(self) -> bool:
        if self.model_candidate:
            return True
        self._show_error("未找到模型", "没有找到可用于检测的 best.pt 或备用模型。")
        return False

    def _ensure_selected_file(self, path: Path | None, message: str) -> bool:
        if path and path.is_file():
            return True
        self._show_error("未选择文件", message)
        return False

    def _show_error(self, title: str, content: str) -> None:
        InfoBar.error(
            title=title,
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP_RIGHT,
            duration=4500,
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

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_detection()
        super().closeEvent(event)


class _MetricValue(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("metricValue")


class _ClassCountRow(QWidget):
    def __init__(self, name: str, count: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        name_label = BodyLabel(CHINESE_CLASS_NAMES.get(name, name), self)
        value_label = QLabel(str(count), self)
        value_label.setObjectName("statusPill")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setFixedWidth(52)

        layout.addWidget(name_label, 1)
        layout.addWidget(value_label)
