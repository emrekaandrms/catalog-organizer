"""
Startup requirements dialog.

Shown before the main window if any checks fail or warn. The user sees a table
of all checks with color-coded status icons. Critical failures disable the
"Launch" button; warnings allow launch with a confirmation.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from catalog_organizer.core.startup_check import (
    CheckResult,
    StartupReport,
    Status,
    run_all_checks,
)

# ── Status icons / colours ────────────────────────────────────────────────────
_STATUS_ICON = {Status.OK: "✓", Status.WARNING: "⚠", Status.CRITICAL: "✗"}
_STATUS_COLOR = {
    Status.OK: QColor("#4caf50"),
    Status.WARNING: QColor("#ff9800"),
    Status.CRITICAL: QColor("#f44336"),
}


# ── Background worker ─────────────────────────────────────────────────────────

class _CheckWorker(QThread):
    finished = pyqtSignal(object)   # StartupReport

    def __init__(
        self,
        ollama_host: str,
        ollama_port: int,
        vlm_model: str,
        check_ollama: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._host = ollama_host
        self._port = ollama_port
        self._model = vlm_model
        self._check_ollama = check_ollama

    def run(self) -> None:
        report = run_all_checks(
            ollama_host=self._host,
            ollama_port=self._port,
            vlm_model=self._model,
            check_ollama=self._check_ollama,
        )
        self.finished.emit(report)


# ── Dialog ────────────────────────────────────────────────────────────────────

class StartupDialog(QDialog):
    """
    Displays startup check results. Blocks launch if any critical check fails.
    Call `exec()` and inspect `result() == QDialog.DialogCode.Accepted` to
    decide whether to open the main window.
    """

    def __init__(
        self,
        ollama_host: str = "127.0.0.1",
        ollama_port: int = 11434,
        vlm_model: str = "qwen3.5:9b-q4_K_M",
        check_ollama: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Catalog Organizer — Startup Check")
        self.setMinimumSize(720, 460)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint
        )

        self._build_ui()
        self._run_checks(ollama_host, ollama_port, vlm_model, check_ollama)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Checking requirements…")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        root.addWidget(title)
        self._title_label = title

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        root.addWidget(self._progress)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Status", "Check", "Message / Hint"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 64)
        self._table.setColumnWidth(1, 220)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        root.addWidget(self._table)

        # Summary row
        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        root.addWidget(self._summary_label)

        # Buttons
        self._buttons = QDialogButtonBox()
        self._launch_btn = self._buttons.addButton(
            "Launch", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._launch_btn.setEnabled(False)
        self._buttons.addButton(
            "Quit", QDialogButtonBox.ButtonRole.RejectRole
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

    # ── Async check runner ────────────────────────────────────────────────────

    def _run_checks(
        self, host: str, port: int, model: str, check_ollama: bool = True,
    ) -> None:
        self._worker = _CheckWorker(host, port, model, check_ollama, parent=self)
        self._worker.finished.connect(self._on_checks_done)
        self._worker.start()

    def _on_checks_done(self, report: StartupReport) -> None:
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._progress.setVisible(False)

        self._populate_table(report.results)

        if report.ok:
            if report.has_warnings:
                self._title_label.setText("Ready with warnings — you may launch.")
                self._summary_label.setText(
                    "Some optional components are missing. The app will work but "
                    "some features may be unavailable."
                )
            else:
                self._title_label.setText("All checks passed.")
            self._launch_btn.setEnabled(True)
        else:
            critical = report.critical_messages()
            self._title_label.setText(
                f"{len(critical)} critical issue(s) must be resolved before launch."
            )
            self._summary_label.setText(
                "Fix the items marked ✗ above, then restart the application."
            )
            self._launch_btn.setEnabled(False)

    def _populate_table(self, results: list[CheckResult]) -> None:
        self._table.setRowCount(len(results))
        for row, result in enumerate(results):
            # Status icon cell
            icon_item = QTableWidgetItem(_STATUS_ICON[result.status])
            icon_item.setForeground(_STATUS_COLOR[result.status])
            icon_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, 0, icon_item)

            # Name cell
            name_item = QTableWidgetItem(result.name)
            self._table.setItem(row, 1, name_item)

            # Message + hint cell
            msg = result.message
            if result.hint:
                msg = f"{msg}\n→ {result.hint}"
            msg_item = QTableWidgetItem(msg)
            if result.status == Status.CRITICAL:
                msg_item.setForeground(_STATUS_COLOR[Status.CRITICAL])
            elif result.status == Status.WARNING:
                msg_item.setForeground(_STATUS_COLOR[Status.WARNING])
            self._table.setItem(row, 2, msg_item)

        self._table.resizeRowsToContents()


# ── Convenience function ──────────────────────────────────────────────────────

def run_startup_check(
    ollama_host: str = "127.0.0.1",
    ollama_port: int = 11434,
    vlm_model: str = "qwen3.5:9b-q4_K_M",
    check_ollama: bool = True,
) -> bool:
    """
    Show the startup check dialog. Returns True if the user clicked Launch,
    False if they clicked Quit or closed the dialog.

    Pass `check_ollama=False` when the configured provider is a cloud
    backend, so the dialog doesn't block launch on a local Ollama the user
    never installed.

    Call this from app.py before creating the main window:

        if not run_startup_check():
            sys.exit(0)
    """
    dlg = StartupDialog(
        ollama_host=ollama_host,
        ollama_port=ollama_port,
        vlm_model=vlm_model,
        check_ollama=check_ollama,
    )
    return dlg.exec() == QDialog.DialogCode.Accepted
