"""Application entry point for the TCM-SliceAI desktop client."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from qfluentwidgets import FluentTranslator, Theme, setTheme, setThemeColor

from .main_window import MainWindow


def main() -> int:
    """Start the desktop application."""
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)
    app.setApplicationName("TCM-SliceAI")
    app.setApplicationDisplayName("TCM-SliceAI")
    app.setOrganizationName("TCM-SliceAI")

    translator = FluentTranslator()
    app.installTranslator(translator)

    setTheme(Theme.DARK)
    setThemeColor("#15A3B8")

    window = MainWindow()
    window.show()

    return app.exec()

