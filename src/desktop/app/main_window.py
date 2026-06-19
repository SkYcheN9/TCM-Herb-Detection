"""Main window and navigation for the desktop client."""

from __future__ import annotations

from PySide6.QtGui import QCloseEvent
from PySide6.QtCore import QSize
from qfluentwidgets import FluentIcon as FIF
from qfluentwidgets import FluentWindow, NavigationItemPosition

from .styles import APP_STYLE
from .views.about import AboutView
from .views.dashboard import DashboardView
from .views.detection import DetectionView
from .views.history import HistoryView
from .views.settings import SettingsView


class MainWindow(FluentWindow):
    """TCM-SliceAI desktop shell."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("mainWindow")
        self.setWindowTitle("TCM-SliceAI")
        self.resize(1480, 900)
        self.setMinimumSize(QSize(1180, 760))

        self.dashboard_view = DashboardView(self)
        self.detection_view = DetectionView(self)
        self.history_view = HistoryView(self)
        self.settings_view = SettingsView(self)
        self.about_view = AboutView(self)

        self._init_navigation()
        self._init_dashboard_actions()
        self.setStyleSheet(APP_STYLE)

    def _init_navigation(self) -> None:
        self.addSubInterface(self.dashboard_view, FIF.HOME, "Dashboard")
        self.addSubInterface(self.detection_view, FIF.CAMERA, "Detection")
        self.addSubInterface(self.history_view, FIF.HISTORY, "History")
        self.addSubInterface(
            self.settings_view,
            FIF.SETTING,
            "Settings",
            position=NavigationItemPosition.BOTTOM,
        )
        self.addSubInterface(
            self.about_view,
            FIF.INFO,
            "About",
            position=NavigationItemPosition.BOTTOM,
        )

    def _init_dashboard_actions(self) -> None:
        self.dashboard_view.detect_button.clicked.connect(lambda: self.switchTo(self.detection_view))
        self.dashboard_view.history_button.clicked.connect(lambda: self.switchTo(self.history_view))

    def closeEvent(self, event: QCloseEvent) -> None:
        self.detection_view.stop_detection()
        super().closeEvent(event)
