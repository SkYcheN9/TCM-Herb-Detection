"""Detection workspace view."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices, QPixmap
from PySide6.QtWidgets import QFileDialog, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget
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
    SpinBox,
    SwitchButton,
)

from ..services.camera_detector import CameraDetectorThread
from ..services.camera_devices import discover_camera_devices
from ..services.gpu_status import query_gpu_status
from ..services.media_detector import DetectionSummary, VideoDetectorThread, detect_image
from ..services.model_locator import find_best_model
from ..widgets.layout import Page, SectionCard, VideoPanel


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

        workspace = QHBoxLayout()
        workspace.setSpacing(16)

        self.video_panel = VideoPanel(
            "等待输入源",
            "选择摄像头、图片或视频后点击开始检测，结果会保存到 reports/desktop。",
            self,
        )
        workspace.addWidget(self.video_panel, 3)

        side = QVBoxLayout()
        side.setSpacing(14)

        self.source_card = SectionCard("输入源", self)
        source_grid = QGridLayout()
        source_grid.setHorizontalSpacing(10)
        source_grid.setVerticalSpacing(10)

        self.mode_selector = ComboBox(self.source_card)
        self.mode_selector.addItems(["摄像头检测", "图片检测", "视频检测"])

        self.camera_selector = ComboBox(self.source_card)
        self.camera_selector.addItems([device.label for device in self.camera_devices])

        self.file_path = LineEdit(self.source_card)
        self.file_path.setReadOnly(True)
        self.file_path.setPlaceholderText("选择图片或视频文件")

        self.device_selector = ComboBox(self.source_card)
        self.device_selector.addItems(["自动选择", "CUDA:0", "CPU"])

        self.model_path = LineEdit(self.source_card)
        self.model_path.setReadOnly(True)
        self.model_path.setText(str(self.model_candidate.path) if self.model_candidate else "未找到可用模型")

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
        source_grid.addWidget(self.camera_selector, 1, 1)
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
        self.conf_slider.setValue(35)

        self.iou_slider = Slider(Qt.Horizontal, self.params_card)
        self.iou_slider.setRange(1, 100)
        self.iou_slider.setValue(45)

        self.size_spin = SpinBox(self.params_card)
        self.size_spin.setRange(320, 1280)
        self.size_spin.setSingleStep(32)
        self.size_spin.setValue(640)

        params_grid.addWidget(BodyLabel("置信度", self.params_card), 0, 0)
        params_grid.addWidget(self.conf_slider, 0, 1)
        params_grid.addWidget(BodyLabel("IoU", self.params_card), 1, 0)
        params_grid.addWidget(self.iou_slider, 1, 1)
        params_grid.addWidget(BodyLabel("图像尺寸", self.params_card), 2, 0)
        params_grid.addWidget(self.size_spin, 2, 1)
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
        status_grid = QGridLayout()
        status_grid.setHorizontalSpacing(10)
        status_grid.setVerticalSpacing(12)

        self.status_value = QLabel("待机", self.status_card)
        self.status_value.setObjectName("statusPill")
        self.status_value.setAlignment(Qt.AlignCenter)

        self.fps_value = _MetricValue("0.0 FPS", self.status_card)
        self.total_value = _MetricValue("0", self.status_card)
        self.progress_value = BodyLabel("-", self.status_card)
        self.progress_value.setObjectName("mutedLabel")
        self.gpu_value = BodyLabel(query_gpu_status().text, self.status_card)
        self.gpu_value.setObjectName("mutedLabel")
        self.gpu_value.setWordWrap(True)

        status_grid.addWidget(BodyLabel("状态", self.status_card), 0, 0)
        status_grid.addWidget(self.status_value, 0, 1)
        status_grid.addWidget(BodyLabel("FPS", self.status_card), 1, 0)
        status_grid.addWidget(self.fps_value, 1, 1)
        status_grid.addWidget(BodyLabel("总数量", self.status_card), 2, 0)
        status_grid.addWidget(self.total_value, 2, 1)
        status_grid.addWidget(BodyLabel("进度", self.status_card), 3, 0)
        status_grid.addWidget(self.progress_value, 3, 1)
        status_grid.addWidget(BodyLabel("GPU", self.status_card), 4, 0)
        status_grid.addWidget(self.gpu_value, 4, 1)
        self.status_card.layout.addLayout(status_grid)

        self.classes_card = SectionCard("类别统计", self)
        self.class_counts = QVBoxLayout()
        self.class_counts.setSpacing(8)
        self.classes_card.layout.addLayout(self.class_counts)
        self._render_class_counts({})

        self.start_button = PrimaryPushButton(FIF.PLAY, "开始检测", self)
        self.stop_button = PushButton(FIF.PAUSE, "停止", self)
        self.open_output_button = PushButton(FIF.FOLDER, "打开结果", self)
        self.stop_button.setEnabled(False)
        self.open_output_button.setEnabled(False)

        run_buttons = QHBoxLayout()
        run_buttons.setSpacing(10)
        run_buttons.addWidget(self.start_button)
        run_buttons.addWidget(self.stop_button)
        run_buttons.addWidget(self.open_output_button)

        side.addWidget(self.source_card)
        side.addWidget(self.params_card)
        side.addWidget(self.options_card)
        side.addWidget(self.status_card)
        side.addWidget(self.classes_card)
        side.addLayout(run_buttons)
        side.addStretch(1)

        workspace.addLayout(side, 2)
        self.root_layout.addLayout(workspace, 1)

        self.mode_selector.currentIndexChanged.connect(self._update_mode_controls)
        self.pick_file_button.clicked.connect(self.pick_source_file)
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.open_output_button.clicked.connect(self.open_last_output)
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

        camera_index = self.camera_devices[self.camera_selector.currentIndex()].index
        self.camera_thread = CameraDetectorThread(
            model_path=self.model_candidate.path,
            camera_index=camera_index,
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
        self._render_class_counts(data.get("counts", {}))

    def apply_summary(self, summary: object) -> None:
        """Show final image or video summary."""
        result = summary if isinstance(summary, DetectionSummary) else None
        if not result:
            return
        if result.frame is not None:
            self.update_frame(result.frame)
        self.last_output_path = result.output_path
        self.open_output_button.setEnabled(True)
        self.fps_value.setText(f"{result.fps:.1f} FPS")
        self.total_value.setText(str(result.total_count))
        self.progress_value.setText(str(result.output_path))
        self.gpu_value.setText(query_gpu_status().text)
        self._render_class_counts(result.class_counts)

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

    def _set_running(self, running: bool, can_stop: bool = True) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running and can_stop)
        self.mode_selector.setEnabled(not running)
        self.camera_selector.setEnabled(not running)
        self.device_selector.setEnabled(not running)
        self.pick_file_button.setEnabled(not running)

    def _update_mode_controls(self) -> None:
        mode = self.mode_selector.currentText()
        is_camera = mode == "摄像头检测"
        self.camera_selector.setEnabled(is_camera)
        self.pick_file_button.setEnabled(not is_camera)
        self.file_path.setEnabled(not is_camera)
        if is_camera:
            self.file_path.clear()
        else:
            selected = self.selected_image_path if mode == "图片检测" else self.selected_video_path
            self.file_path.setText(str(selected) if selected else "")

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

        name_label = BodyLabel(name, self)
        value_label = QLabel(str(count), self)
        value_label.setObjectName("statusPill")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setFixedWidth(52)

        layout.addWidget(name_label, 1)
        layout.addWidget(value_label)
