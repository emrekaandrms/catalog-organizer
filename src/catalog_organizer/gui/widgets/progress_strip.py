"""Progress strip — bar + counter + ETA + recent-files thumbnail row."""
from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

_MAX_THUMBS = 8


class ProgressStrip(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._start_time: float | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(0, 0, 0, 0)

        # Top row: bar | counter | ETA
        top = QHBoxLayout()
        self._bar = QProgressBar()
        self._bar.setRange(0, 1)
        self._bar.setValue(0)
        top.addWidget(self._bar, stretch=1)

        self._counter = QLabel("0 / 0")
        self._counter.setMinimumWidth(80)
        self._counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self._counter)

        self._eta = QLabel("ETA: –")
        self._eta.setMinimumWidth(110)
        self._eta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._eta.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        top.addWidget(self._eta)
        root.addLayout(top)

        # Thumbnail strip (insert from the left; trailing stretch keeps left-anchored)
        thumb_widget = QWidget()
        thumb_widget.setMinimumHeight(72)
        self._thumb_row = QHBoxLayout(thumb_widget)
        self._thumb_row.setSpacing(4)
        self._thumb_row.setContentsMargins(0, 0, 0, 0)
        self._thumb_row.addStretch()
        root.addWidget(thumb_widget)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, total: int = 0) -> None:
        self._start_time = None
        self._bar.setRange(0, max(1, total))
        self._bar.setValue(0)
        self._counter.setText(f"0 / {total}")
        self._eta.setText("ETA: –")
        # Strip out all thumbnail labels, keep the trailing stretch.
        while self._thumb_row.count() > 1:
            item = self._thumb_row.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    def set_progress(self, done: int, total: int) -> None:
        if self._start_time is None and done > 0:
            self._start_time = time.monotonic()
        self._bar.setRange(0, max(1, total))
        self._bar.setValue(done)
        self._counter.setText(f"{done} / {total}")

        if self._start_time and done > 0 and done < total:
            elapsed = time.monotonic() - self._start_time
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            self._eta.setText(f"ETA: {self._fmt_duration(remaining)}")
        elif total > 0 and done >= total:
            self._eta.setText("Done")

    def add_thumbnail(self, image_path: Path | None) -> None:
        lbl = QLabel()
        lbl.setFixedSize(64, 64)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("border: 1px solid #3a3a5c; border-radius: 4px;")
        if image_path is not None and image_path.exists():
            pm = QPixmap(str(image_path)).scaled(
                QSize(60, 60),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl.setPixmap(pm)
        else:
            lbl.setText("?")
            lbl.setStyleSheet(lbl.styleSheet() + " color: #5a5a7a;")

        # Insert at index 0 (left); stretch stays last.
        self._thumb_row.insertWidget(0, lbl)

        # Cap at _MAX_THUMBS; +1 for the trailing stretch item.
        while self._thumb_row.count() > _MAX_THUMBS + 1:
            item = self._thumb_row.takeAt(_MAX_THUMBS)
            if item and item.widget():
                item.widget().deleteLater()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)} s"
        if seconds < 3600:
            return f"{int(seconds // 60)} m {int(seconds % 60)} s"
        return f"{int(seconds // 3600)} h {int((seconds % 3600) // 60)} m"

    # ── Test accessors ────────────────────────────────────────────────────────

    @property
    def counter_text(self) -> str:
        return self._counter.text()

    @property
    def thumbnail_count(self) -> int:
        # Subtract 1 for the trailing stretch
        return max(0, self._thumb_row.count() - 1)
