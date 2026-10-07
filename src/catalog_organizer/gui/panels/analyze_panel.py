"""Analyze panel — the production surface for geometric analysis.

Pick a file or a folder; get weights (8 alloys, net + sprue + final product),
stone sizes/counts/carats, dimensions, ring bore, and sprue detection — all
computed from geometry alone (no VLM / Ollama). Export the table to CSV.

This is the calibrated core (±4% vs workshop scale on test4); the VLM only
adds category/tags and lives in the separate Pilot path.
"""
from __future__ import annotations

import csv
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class _AnalyzeWorker(QThread):
    one_done = pyqtSignal(str, dict)      # report_text, flat_dict
    all_done = pyqtSignal(int)            # count
    progress = pyqtSignal(int, int)       # done, total

    def __init__(self, files: list[Path]) -> None:
        super().__init__()
        self._files = files

    def run(self) -> None:  # type: ignore[override]
        from catalog_organizer.orchestrator.analyze import (  # noqa: PLC0415
            analyze_file, render_text, to_flat_dict,
        )
        total = len(self._files)
        for i, f in enumerate(self._files, start=1):
            result = analyze_file(f)
            self.one_done.emit(render_text(result), to_flat_dict(result))
            self.progress.emit(i, total)
        self.all_done.emit(total)


class AnalyzePanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[dict] = []
        self._worker: _AnalyzeWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Analyze — weights · stones · dimensions · sprue")
        tf = QFont(); tf.setPointSize(16); tf.setBold(True)
        title.setFont(tf)
        root.addWidget(title)

        sub = QLabel("Geometry-only. No Ollama needed. Calibrated to ±4% vs scale.")
        sub.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        root.addWidget(sub)

        # Buttons
        row = QHBoxLayout()
        b_file = QPushButton("Analyze file…")
        b_folder = QPushButton("Analyze folder…")
        self._b_csv = QPushButton("Export CSV…")
        self._b_csv.setEnabled(False)
        b_file.clicked.connect(self._pick_file)
        b_folder.clicked.connect(self._pick_folder)
        self._b_csv.clicked.connect(self._export_csv)
        row.addWidget(b_file)
        row.addWidget(b_folder)
        row.addWidget(self._b_csv)
        row.addStretch()
        root.addLayout(row)

        self._bar = QProgressBar()
        self._bar.setVisible(False)
        root.addWidget(self._bar)

        self._status = QLabel("Pick a file or folder to begin.")
        self._status.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        root.addWidget(self._status)

        self._out = QTextEdit()
        self._out.setReadOnly(True)
        self._out.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 12px;"
            " background: #12121e; color: #e0e0e0;"
        )
        root.addWidget(self._out, stretch=1)

    # ── pickers ──────────────────────────────────────────────────────────────

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a CAD file", "", "CAD files (*.stl *.3dm)")
        if path:
            self._run([Path(path)])

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Pick a folder of CAD files")
        if not path:
            return
        files = sorted(
            p for p in Path(path).rglob("*") if p.suffix.lower() in (".stl", ".3dm"))
        if not files:
            self._status.setText("No .stl / .3dm files found.")
            return
        self._run(files)

    # ── run ──────────────────────────────────────────────────────────────────

    def _run(self, files: list[Path]) -> None:
        self._rows.clear()
        self._out.clear()
        self._b_csv.setEnabled(False)
        self._bar.setVisible(True)
        self._bar.setRange(0, len(files))
        self._bar.setValue(0)
        self._status.setText(f"Analyzing {len(files)} file(s)…")
        self._worker = _AnalyzeWorker(files)
        self._worker.one_done.connect(self._on_one)
        self._worker.progress.connect(lambda d, t: self._bar.setValue(d))
        self._worker.all_done.connect(self._on_all)
        self._worker.start()

    def _on_one(self, report: str, flat: dict) -> None:
        self._rows.append(flat)
        self._out.append(report)
        self._out.append("")

    def _on_all(self, count: int) -> None:
        self._bar.setVisible(False)
        self._b_csv.setEnabled(bool(self._rows))
        errs = sum(1 for r in self._rows if r.get("error"))
        self._status.setText(
            f"Done. {count} file(s) analyzed"
            + (f", {errs} with errors." if errs else "."))

    # ── export ───────────────────────────────────────────────────────────────

    def _export_csv(self) -> None:
        if not self._rows:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "analysis.csv", "CSV (*.csv)")
        if not path:
            return
        fields: list[str] = []
        for r in self._rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(self._rows)
        self._status.setText(f"CSV written: {path}")
