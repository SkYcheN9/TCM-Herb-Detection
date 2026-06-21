"""Shared styling for the desktop client."""

from __future__ import annotations


APP_STYLE = """
QWidget {
    font-family: "Segoe UI", "Microsoft YaHei UI";
    color: #111827;
}

QWidget#dashboardView,
QWidget#detectionView,
QWidget#historyView,
QWidget#settingsView,
QWidget#aboutView {
    background: #F3F6F8;
}

QWidget#pageContainer {
    background: transparent;
}

QWidget#sectionHeader {
    background: transparent;
}

QFrame#heroPanel {
    background: #111820;
    border: 1px solid rgba(17, 24, 39, 0.18);
    border-radius: 8px;
}

QFrame#heroPanel QLabel#heroTitle {
    color: #F8FAFC;
    font-size: 22px;
    font-weight: 700;
}

QFrame#heroPanel QLabel#heroBody {
    color: rgba(248, 250, 252, 0.86);
    font-size: 15px;
    line-height: 1.4;
}

QFrame#heroPanel QLabel#heroMeta {
    color: rgba(248, 250, 252, 0.78);
}

QFrame#previewPanel {
    background: #0B0E11;
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 8px;
}

SmoothScrollArea {
    background: transparent;
    border: none;
}

QSplitter::handle {
    background: #DCE5EA;
    border-radius: 3px;
}

QSplitter::handle:hover {
    background: #13A8B8;
}

QFrame#timelineRail {
    background: rgba(17, 24, 39, 0.10);
    border-radius: 2px;
}

QFrame#heroPanel QLabel,
QFrame#previewPanel QLabel {
    color: #F8FAFC;
}

QLabel#eyebrowLabel {
    color: #0E9BB0;
    font-size: 12px;
    font-weight: 600;
}

QLabel#mutedLabel,
QLabel#metaLabel {
    color: #5B6673;
}

QLabel#metricValue {
    color: #111827;
    font-size: 28px;
    font-weight: 700;
}

QLabel#metricTitle {
    color: #5B6673;
    font-size: 12px;
}

QLabel#valueBadge {
    color: #0F766E;
    background: #DDF8EA;
    border: 1px solid #9DE7C3;
    border-radius: 8px;
    padding: 4px 8px;
    font-weight: 600;
}

QLabel#statusPill {
    color: #047857;
    background: #DDF8EA;
    border: 1px solid #9DE7C3;
    border-radius: 8px;
    padding: 5px 10px;
}

QFrame#heroPanel QLabel#statusPill {
    color: #065F46;
    background: #D9FBEA;
    border: 1px solid #9DE7C3;
}

QLabel#warningPill {
    color: #9A5B00;
    background: #FFF2CC;
    border: 1px solid #F2C66D;
    border-radius: 8px;
    padding: 5px 10px;
}

QFrame#heroPanel QLabel#warningPill {
    color: #8A5A00;
    background: #FFF0C2;
    border: 1px solid #F3C86F;
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
    color: #334155;
    background: #EAF1F5;
    border: 1px solid #D6E0E7;
    border-radius: 8px;
    padding: 6px 9px;
}

QFrame#heroPanel QLabel#mutedLabel,
QFrame#previewPanel QLabel#mutedLabel {
    color: rgba(248, 250, 252, 0.74);
}

QFrame#heroPanel QLabel#eyebrowLabel {
    color: #7CE7F4;
}

QFrame#heroPanel QLabel#tagLabel,
QFrame#previewPanel QLabel#tagLabel {
    color: rgba(248, 250, 252, 0.84);
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

QTableWidget {
    background: transparent;
    color: #111827;
    border: none;
    gridline-color: #E2E8F0;
}

QHeaderView::section {
    background: #EAF1F5;
    color: #334155;
    border: none;
    padding: 8px;
}
"""

