"""Logs panel — live tail of audit_log.jsonl with severity filter."""
from __future__ import annotations

import json

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from catalog_organizer.core.paths import data_dir

_LEVEL_COLOR: dict[str, str] = {
    "debug":   "#5a5a7a",
    "info":    "#9e9e9e",
    "warning": "#ff9800",
    "error":   "#f44336",
}
_MAX_DISPLAY = 500


class LogsPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._refresh()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(2_000)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Audit Log")
        tfont = QFont()
        tfont.setPointSize(16)
        tfont.setBold(True)
        title.setFont(tfont)
        root.addWidget(title)

        # Log list (defined before toolbar so clear_btn can reference it)
        self._list = QListWidget()
        self._list.setObjectName("LogView")

        # Toolbar
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Severity:"))
        self._filter = QComboBox()
        self._filter.addItems(["all", "debug", "info", "warning", "error"])
        self._filter.currentTextChanged.connect(self._refresh)
        bar.addWidget(self._filter)
        bar.addStretch()
        copy_btn = QPushButton("Copy All")
        copy_btn.clicked.connect(self._copy_all)
        bar.addWidget(copy_btn)
        clear_btn = QPushButton("Clear View")
        clear_btn.clicked.connect(self._list.clear)
        bar.addWidget(clear_btn)
        root.addLayout(bar)

        root.addWidget(self._list)

    def _refresh(self) -> None:
        path = data_dir() / "audit_log.jsonl"
        if not path.exists():
            return
        severity = self._filter.currentText()
        raw_lines = path.read_bytes().splitlines()[-_MAX_DISPLAY:]

        self._list.clear()
        for raw in reversed(raw_lines):
            try:
                ev    = json.loads(raw)
                level = ev.get("level", "info").lower()
                if severity != "all" and level != severity:
                    continue
                ts   = ev.get("timestamp", "")[:19].replace("T", " ")
                code = ev.get("code", "")
                msg  = ev.get("message", "")[:120]
                text = f"{ts}  [{level.upper():<7}]  {code:<32}  {msg}"
                item = QListWidgetItem(text)
                item.setForeground(QColor(_LEVEL_COLOR.get(level, "#9e9e9e")))
                self._list.addItem(item)
            except Exception:
                pass

    def _copy_all(self) -> None:
        lines = [self._list.item(i).text() for i in range(self._list.count())]
        QApplication.clipboard().setText("\n".join(lines))
