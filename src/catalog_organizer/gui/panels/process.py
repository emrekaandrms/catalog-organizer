"""Process panel — pick a folder, run the full pipeline on every CAD file
inside, export the catalog CSV.

Single-purpose surface that replaces the old Scan / Pilot / Catalog trio.
No manifest pre-step required: the panel walks the folder, hashes files,
allocates file_ids inline, then feeds entries to BatchRunner. CSV is
rewritten by CatalogWriter on every flush; the user clicks "Open CSV"
when the run finishes (or any time during the run to see partial output).
"""
from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from catalog_organizer.catalog.index import CatalogIndex
from catalog_organizer.catalog.tag_validator import TagValidator
from catalog_organizer.catalog.writer import CatalogWriter
from catalog_organizer.core.audit import AuditWriter
from catalog_organizer.core.config import load_metal_densities, load_pipeline_settings
from catalog_organizer.core.paths import data_dir
from catalog_organizer.core.schemas import ManifestEntry
from catalog_organizer.gui.widgets.progress_strip import ProgressStrip
from catalog_organizer.orchestrator.batch import BatchRunner
from catalog_organizer.orchestrator.pipeline import PipelineDeps
from catalog_organizer.scanner.manifest import scan
from catalog_organizer.scanner.walker import walk_for_cad_files


class _BatchWorker(QThread):
    """Run BatchRunner.run() off the GUI thread so signals can cross."""

    def __init__(
        self,
        runner: BatchRunner,
        entries: list[ManifestEntry],
        batch_name: str,
        skip_filter: bool = False,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._entries = entries
        self._batch_name = batch_name
        self._skip_filter = skip_filter

    def run(self) -> None:  # type: ignore[override]
        self._runner.run(
            self._entries,
            batch_name=self._batch_name,
            skip_filter=self._skip_filter,
        )


# ── File-choice dialog ─────────────────────────────────────────────────────

class _FileChoiceDialog(QDialog):
    """Shows every CAD file found on disk in the chosen folder. Each row is
    tagged 'new', 'already in catalog', or 'duplicate of <other file>'.

    Earlier the panel silently refused folders whose files were either
    duplicates (same SHA as something already cataloged from another path)
    or already cataloged. The user couldn't tell whether the pipeline was
    broken or just deduplicating, so we now always show the listing and let
    them pick what to do. Duplicates are still highlighted but not blocked.
    """

    CHOICE_NEW = "new"
    CHOICE_REPROCESS = "reprocess"
    CHOICE_CANCEL = "cancel"

    def __init__(
        self,
        folder: Path,
        all_entries: list[ManifestEntry],
        new_entries: list[ManifestEntry],
        already_entries: list[ManifestEntry],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Files in folder")
        self.setMinimumSize(680, 480)
        self.choice = self.CHOICE_CANCEL

        layout = QVBoxLayout(self)

        dup_count = sum(1 for e in all_entries if e.duplicate_of)
        summary = QLabel(
            f"<b>{folder}</b><br>"
            f"{len(all_entries)} CAD file(s) on disk  ·  "
            f"<span style='color:#4caf50'>{len(new_entries)} new</span>  ·  "
            f"<span style='color:#f0b132'>{len(already_entries)} already in catalog</span>"
            + (f"  ·  <span style='color:#9e9e9e'>{dup_count} duplicate(s) by content hash</span>"
               if dup_count else "")
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        # File listing — every file with a status tag.
        already_ids = {e.file_id for e in already_entries}
        listing = QListWidget()
        for e in all_entries:
            name = Path(e.source_path).name
            tag = "•  new                "
            if e.file_id in already_ids:
                tag = "✓  already in catalog"
            elif e.duplicate_of:
                tag = f"⚠  duplicate of {e.duplicate_of}"
            listing.addItem(QListWidgetItem(f"{tag}   —   {name}"))
        layout.addWidget(listing, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        new_btn = QPushButton(f"Process new only ({len(new_entries)})")
        reprocess_btn = QPushButton(f"Reprocess all ({len(all_entries)})")
        cancel_btn = QPushButton("Cancel")
        new_btn.setEnabled(bool(new_entries))
        reprocess_btn.setEnabled(bool(all_entries))
        new_btn.clicked.connect(lambda: self._set_and_accept(self.CHOICE_NEW))
        reprocess_btn.clicked.connect(lambda: self._set_and_accept(self.CHOICE_REPROCESS))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(new_btn)
        btn_row.addWidget(reprocess_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _set_and_accept(self, choice: str) -> None:
        self.choice = choice
        self.accept()


class ProcessPanel(QWidget):
    """Pick a folder → walk + hash → batch pipeline → CSV. One screen, one flow."""

    def __init__(self, index: CatalogIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._folder: Path | None = None
        self._worker: _BatchWorker | None = None
        self._runner: BatchRunner | None = None
        self._writer: CatalogWriter | None = None
        self._audit: AuditWriter | None = None
        self._csv_path: Path = data_dir() / "catalog_export.csv"
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Process — folder → catalog CSV")
        tf = QFont(); tf.setPointSize(16); tf.setBold(True)
        title.setFont(tf)
        root.addWidget(title)

        hint = QLabel(
            "Pick a folder of .stl / .3dm files. Every file is hashed, "
            "snapshotted, classified, measured, and written to "
            "data/catalog_export.csv. Resume-safe: files already in the "
            "catalog are skipped."
        )
        hint.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        # Folder picker
        fld_row = QHBoxLayout()
        self._folder_label = QLabel("(no folder selected)")
        self._folder_label.setStyleSheet("color: #9e9e9e;")
        pick_btn = QPushButton("Pick folder…")
        pick_btn.clicked.connect(self._pick_folder)
        fld_row.addWidget(QLabel("Folder:"))
        fld_row.addWidget(self._folder_label, stretch=1)
        fld_row.addWidget(pick_btn)
        root.addLayout(fld_row)

        # Controls
        btn_row = QHBoxLayout()
        self._start_btn  = QPushButton("Start")
        self._pause_btn  = QPushButton("Pause")
        self._cancel_btn = QPushButton("Cancel")
        self._open_csv_btn = QPushButton("Open CSV")
        for b in (self._start_btn, self._pause_btn, self._cancel_btn, self._open_csv_btn):
            b.setFixedHeight(36)
        self._start_btn.clicked.connect(self._on_start)
        self._pause_btn.clicked.connect(self._on_pause)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._open_csv_btn.clicked.connect(self._on_open_csv)
        self._pause_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._pause_btn)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._open_csv_btn)
        root.addLayout(btn_row)

        self._progress = ProgressStrip()
        root.addWidget(self._progress)

        root.addWidget(QLabel("Failures"))
        self._failure_list = QListWidget()
        self._failure_list.setMaximumHeight(160)
        root.addWidget(self._failure_list)

        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        root.addStretch()

    # ── Actions ──────────────────────────────────────────────────────────────

    def _pick_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder to process")
        if path:
            self._folder = Path(path)
            self._folder_label.setText(str(self._folder))
            self._folder_label.setStyleSheet("color: #e0e0e0;")

    def _on_start(self) -> None:
        if self._folder is None:
            QMessageBox.warning(self, "No folder", "Pick a folder first.")
            return

        # Step 1: check the filesystem directly — the manifest dict is shared
        # across all past scans and not a reliable "what's in this folder
        # right now" signal. walk_for_cad_files tells us the truth.
        walked = walk_for_cad_files([self._folder])
        if not walked:
            QMessageBox.information(
                self, "No CAD files",
                f"No .stl or .3dm files found under:\n{self._folder}",
            )
            self._status.setText(f"No .stl/.3dm files in {self._folder}.")
            return

        # Step 2: refresh / extend the manifest so every file on disk has an
        # entry (hash, file_id, duplicate detection). Idempotent.
        try:
            self._status.setText(f"Scanning {len(walked)} files in {self._folder} …")
            manifest = scan([self._folder], batch_name="process")
        except Exception as exc:
            QMessageBox.critical(self, "Scan failed", f"Could not scan folder:\n{exc}")
            self._status.setText("Scan failed.")
            return

        # Step 3: pick the manifest entries that correspond to disk files in
        # the chosen folder. Duplicate-of-other-path is NOT a reason to hide —
        # the user explicitly chose this folder; show every file and let them
        # decide. The dialog tags duplicates so it's still visible.
        folder_paths = {str(p) for p in walked}
        all_in_folder = [
            e for e in manifest.values() if e.source_path in folder_paths
        ]

        if not all_in_folder:
            # Filesystem has files but manifest mapping failed — show a
            # diagnostic message instead of silently doing nothing.
            QMessageBox.critical(
                self, "Manifest mismatch",
                "Files were found on disk but none mapped to manifest "
                "entries. This is a bug — please share the path list:\n\n"
                + "\n".join(sorted(folder_paths)[:20]),
            )
            return

        new_entries     = [e for e in all_in_folder if e.file_id not in self._index]
        already_entries = [e for e in all_in_folder if e.file_id in self._index]

        # Always show the dialog so the user sees what's about to run.
        dlg = _FileChoiceDialog(
            self._folder, all_in_folder, new_entries, already_entries, self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self._status.setText("Cancelled.")
            return

        # In both modes we pass entries straight to BatchRunner with
        # skip_filter=True: the panel's dialog is now the source of truth
        # for what to run, so we don't want filter_eligible to second-guess
        # the user (e.g. by dropping content-hash duplicates).
        if dlg.choice == _FileChoiceDialog.CHOICE_NEW:
            entries = new_entries
        else:  # CHOICE_REPROCESS
            entries = all_in_folder
        skip_filter = True

        if not entries:
            QMessageBox.information(
                self, "Nothing to process",
                "Selection ended up empty (no files match the chosen mode).",
            )
            return

        try:
            deps = self._build_deps()
        except Exception as exc:
            QMessageBox.critical(self, "Config error", str(exc))
            return

        self._writer = CatalogWriter()
        self._audit  = AuditWriter(data_dir() / "audit_log.jsonl")
        parallel_workers = int(
            load_pipeline_settings().get("vlm", {}).get("parallel_workers", 1)
        )
        self._runner = BatchRunner(
            deps, self._writer, self._index, self._audit,
            parallel_workers=parallel_workers,
        )

        self._runner.progressChanged.connect(self._progress.set_progress)
        self._runner.fileFailed.connect(self._on_file_failed)
        self._runner.batchFinished.connect(self._on_finished)

        self._failure_list.clear()
        self._progress.reset(len(entries))
        mode = "reprocess" if skip_filter else "new"

        # Reset the session cost tracker so the status bar starts at $0 for
        # this batch — otherwise yesterday's leftover totals would tag onto
        # tonight's run. Also pin the provider used at start, so the end-of-
        # run dialog can name it even if the user switches mid-run.
        from catalog_organizer.core.cost_tracker import tracker_singleton  # noqa: PLC0415
        tracker_singleton().reset()
        self._batch_provider = deps.vlm_client.display_name
        # Optimistic cost estimate (shown alongside running totals)
        try:
            self._batch_est_cost = deps.vlm_client.estimate_cost_usd(n_images=4) * len(entries)
        except Exception:
            self._batch_est_cost = 0.0

        self._status.setText(
            f"Processing {len(entries)} files ({mode}) via {self._batch_provider}  "
            f"·  est. ${self._batch_est_cost:.2f}  ·  → {self._csv_path}"
        )
        self._set_running_buttons(True)

        self._worker = _BatchWorker(
            self._runner, entries, batch_name="process", skip_filter=skip_filter,
        )
        self._worker.start()

    def _on_pause(self) -> None:
        if self._runner is None:
            return
        if self._pause_btn.text() == "Pause":
            self._runner.pause()
            self._pause_btn.setText("Resume")
            self._status.setText("Paused.")
        else:
            self._runner.resume()
            self._pause_btn.setText("Pause")
            self._status.setText("Running…")

    def _on_cancel(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
            self._status.setText("Cancel requested…")

    def _on_open_csv(self) -> None:
        if not self._csv_path.exists():
            self._status.setText(f"CSV not yet written ({self._csv_path}).")
            return
        try:
            os.startfile(str(self._csv_path))  # type: ignore[attr-defined]  # Windows-only
        except Exception as exc:
            self._status.setText(f"Could not open CSV: {exc}")

    # ── Runner slots ─────────────────────────────────────────────────────────

    def _on_file_failed(self, file_id: str, code: str) -> None:
        self._failure_list.addItem(QListWidgetItem(f"{file_id}  —  {code}"))

    def _on_finished(self, success: int, fail: int) -> None:
        self._set_running_buttons(False)
        self._status.setText(
            f"Done. success={success}  fail={fail}   →   {self._csv_path}"
        )
        self._show_run_summary(success, fail)
        if self._writer is not None:
            self._writer.close()
        if self._audit is not None:
            self._audit.close()
        self._index.reload()

    # ── End-of-run summary ──────────────────────────────────────────────────

    def _show_run_summary(self, success: int, fail: int) -> None:
        """Pop a one-shot dialog with what just happened: counts, cost,
        average latency, output path. For Ollama (cost=0) it doubles as a
        progress confirmation; for cloud providers it's how the user
        verifies their bill before the invoice arrives."""
        from catalog_organizer.core.cost_tracker import tracker_singleton  # noqa: PLC0415
        s = tracker_singleton().snapshot()

        avg_ms = (s.total_wall_ms / s.total_calls) if s.total_calls else 0.0
        provider = getattr(self, "_batch_provider", "—")
        est = getattr(self, "_batch_est_cost", 0.0)

        lines = [
            f"<b>{success} succeeded</b>, {fail} failed",
            "",
            f"Provider: {provider}",
            f"VLM calls: {s.total_calls}",
            f"Tokens: {s.total_prompt_tokens:,} in / {s.total_output_tokens:,} out",
            f"Average wall time: {avg_ms:,.0f} ms / call",
            "",
            f"<b>Cost this batch: ${s.total_cost_usd:.4f}</b>"
            + (f"  (estimated before run: ${est:.4f})" if est > 0 else ""),
            "",
            f"Output: {self._csv_path}",
        ]
        QMessageBox.information(self, "Batch complete", "<br>".join(lines))

    def _set_running_buttons(self, running: bool) -> None:
        self._start_btn.setEnabled(not running)
        self._pause_btn.setEnabled(running)
        self._cancel_btn.setEnabled(running)
        if not running:
            self._pause_btn.setText("Pause")

    # ── Deps ─────────────────────────────────────────────────────────────────

    def _build_deps(self) -> PipelineDeps:
        from catalog_organizer.vlm.providers import create_vlm_provider  # noqa: PLC0415
        settings = load_pipeline_settings()
        densities = load_metal_densities()
        vlm_cfg = settings.get("vlm", {})
        return PipelineDeps(
            vlm_client=create_vlm_provider(vlm_cfg),
            tag_validator=TagValidator(),
            metal_densities=densities,
            scale_thresholds=settings.get("scale_thresholds", {}),
            snapshot_resolution=settings.get("snapshot", {}).get("resolution", 1024),
            thumbnail_size=settings.get("snapshot", {}).get("thumbnail_size", 256),
        )
