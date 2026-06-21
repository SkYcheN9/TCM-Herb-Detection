"""Reusable layout widgets for the desktop client."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import BodyLabel, CaptionLabel, SimpleCardWidget, SubtitleLabel, TitleLabel


class Page(QWidget):
    """Base page with a consistent industrial software layout."""

    def __init__(self, route_key: str, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(route_key)

        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(36, 30, 36, 34)
        self.root_layout.setSpacing(22)

        header = QWidget(self)
        header.setObjectName("sectionHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        self.title_label = TitleLabel(title, header)
        self.subtitle_label = BodyLabel(subtitle, header)
        self.subtitle_label.setObjectName("mutedLabel")
        self.subtitle_label.setWordWrap(True)

        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.subtitle_label)
        self.root_layout.addWidget(header)


class MetricCard(SimpleCardWidget):
    """Compact KPI card."""

    def __init__(
        self,
        title: str,
        value: str,
        detail: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setBorderRadius(8)
        self.setMinimumHeight(126)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        title_label = CaptionLabel(title, self)
        title_label.setObjectName("metricTitle")

        value_label = QLabel(value, self)
        value_label.setObjectName("metricValue")

        detail_label = CaptionLabel(detail, self)
        detail_label.setObjectName("mutedLabel")
        detail_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addStretch(1)
        layout.addWidget(detail_label)


class SectionCard(SimpleCardWidget):
    """Card with a title and free-form content area."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setBorderRadius(8)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 18, 20, 20)
        self.layout.setSpacing(14)

        title_label = SubtitleLabel(title, self)
        self.layout.addWidget(title_label)


class HeroPanel(QFrame):
    """Large status panel used on the dashboard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("heroPanel")
        self.setMinimumHeight(210)


class PreviewPanel(QFrame):
    """Detection preview placeholder."""

    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("previewPanel")
        self.setMinimumHeight(420)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(26, 26, 26, 26)
        layout.setSpacing(10)

        glyph = QLabel("◇", self)
        glyph.setObjectName("previewGlyph")
        glyph.setAlignment(Qt.AlignCenter)

        title_label = QLabel(title, self)
        title_label.setObjectName("previewTitle")
        title_label.setAlignment(Qt.AlignCenter)

        subtitle_label = BodyLabel(subtitle, self)
        subtitle_label.setObjectName("mutedLabel")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setWordWrap(True)

        layout.addWidget(glyph)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)


class VideoPanel(QFrame):
    """Live video surface for camera detection."""

    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("previewPanel")
        self.setMinimumHeight(560)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = QPixmap()

        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(0)

        self.video_label = _FrameLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setScaledContents(False)
        self.video_label.hide()

        self.empty_title = QLabel(title, self)
        self.empty_title.setObjectName("previewTitle")
        self.empty_title.setAlignment(Qt.AlignCenter)
        self.empty_title.setStyleSheet("color: #F8FAFC; font-size: 18px; font-weight: 700;")

        self.empty_subtitle = BodyLabel(subtitle, self)
        self.empty_subtitle.setObjectName("mutedLabel")
        self.empty_subtitle.setAlignment(Qt.AlignCenter)
        self.empty_subtitle.setWordWrap(True)
        self.empty_subtitle.setStyleSheet("color: rgba(248, 250, 252, 0.72);")

        self.empty_container = QWidget(self)
        empty_layout = QVBoxLayout(self.empty_container)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        empty_layout.setSpacing(10)
        empty_layout.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(self.empty_title)
        empty_layout.addWidget(self.empty_subtitle)

        self.layout.addWidget(self.video_label, 1)
        self.layout.addWidget(self.empty_container, 1)

    def set_frame(self, pixmap: QPixmap) -> None:
        """Display a new camera frame."""
        self._pixmap = pixmap
        self.empty_container.hide()
        self.video_label.show()
        self._update_pixmap()

    def clear_frame(self) -> None:
        """Return the surface to its empty state."""
        self._pixmap = QPixmap()
        self.video_label.clear()
        self.video_label.hide()
        self.empty_container.show()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._pixmap.isNull():
            return

        target_size = self.video_label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            target_size = self.contentsRect().adjusted(8, 8, -8, -8).size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return

        scaled = self._pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)


class TagRow(QWidget):
    """Horizontal row of compact tags."""

    def __init__(self, tags: Iterable[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        for tag in tags:
            label = CaptionLabel(tag, self)
            label.setObjectName("tagLabel")
            layout.addWidget(label)

        layout.addStretch(1)


class _FrameLabel(QLabel):
    """Preview label whose layout size is independent from the current pixmap."""

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    def minimumSizeHint(self) -> QSize:
        return QSize(320, 180)


def add_metric_grid(layout: QVBoxLayout, metrics: list[tuple[str, str, str]], columns: int = 4) -> None:
    """Add a responsive-looking KPI grid to a vertical layout."""
    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(14)
    grid.setVerticalSpacing(14)

    for index, (title, value, detail) in enumerate(metrics):
        grid.addWidget(MetricCard(title, value, detail), index // columns, index % columns)

    layout.addLayout(grid)
