"""Shared styling for the desktop client."""

from __future__ import annotations


APP_STYLE = """
QWidget {
    font-family: "Segoe UI", "Microsoft YaHei UI";
}

QWidget#dashboardView,
QWidget#detectionView,
QWidget#historyView,
QWidget#settingsView,
QWidget#aboutView {
    background: #101317;
}

QWidget#pageContainer {
    background: transparent;
}

QWidget#sectionHeader {
    background: transparent;
}

QFrame#heroPanel {
    background: #141A20;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
}

QFrame#previewPanel {
    background: #0B0E11;
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 8px;
}

QFrame#timelineRail {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 2px;
}

QLabel#eyebrowLabel {
    color: #77D8E8;
    font-size: 12px;
    font-weight: 600;
}

QLabel#mutedLabel,
QLabel#metaLabel {
    color: rgba(255, 255, 255, 0.62);
}

QLabel#metricValue {
    color: #F7FAFC;
    font-size: 28px;
    font-weight: 700;
}

QLabel#metricTitle {
    color: rgba(255, 255, 255, 0.68);
    font-size: 12px;
}

QLabel#statusPill {
    color: #89F2C8;
    background: rgba(31, 205, 128, 0.12);
    border: 1px solid rgba(31, 205, 128, 0.28);
    border-radius: 8px;
    padding: 5px 10px;
}

QLabel#warningPill {
    color: #FFD480;
    background: rgba(255, 181, 71, 0.12);
    border: 1px solid rgba(255, 181, 71, 0.26);
    border-radius: 8px;
    padding: 5px 10px;
}

QLabel#previewGlyph {
    color: rgba(255, 255, 255, 0.22);
    font-size: 64px;
}

QLabel#previewTitle {
    color: rgba(255, 255, 255, 0.82);
    font-size: 18px;
    font-weight: 600;
}

QLabel#tagLabel {
    color: rgba(255, 255, 255, 0.76);
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 6px 9px;
}

QTableWidget {
    background: transparent;
    color: #F7FAFC;
    border: none;
    gridline-color: rgba(255, 255, 255, 0.08);
}

QHeaderView::section {
    background: rgba(255, 255, 255, 0.06);
    color: rgba(255, 255, 255, 0.72);
    border: none;
    padding: 8px;
}
"""

