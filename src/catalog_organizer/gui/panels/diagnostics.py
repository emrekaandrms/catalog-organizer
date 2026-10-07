"""Diagnostics panel — pick one file, run each pipeline stage by itself,
see how long it took.

Stages (in pipeline order):
  1. Mesh load            — trimesh / rhino3dm read
  2. Brep extraction      — .3dm only; BrepFace.GetMesh(Render)
  3. Snapshot render      — PyVista 4 views + webp thumbnail
  4. Measurements         — bbox + estimate_volume + category-specific
  5. VLM call             — Ollama /api/chat (one request)
  6. Finalize             — tag-validate, scale-check, sanitize, build record

Each button runs ONE stage and reports wall time + a short result summary.
The diagnostics panel keeps a `_state` dict so the result of stage N is
available as input to stage N+1 (e.g. snapshots → VLM, mesh → measurements).
'Run All' fires each in sequence and reports per-stage timing in the log.

This panel is a profiling tool, not a production code path. It deliberately
re-imports modules and creates fresh deps so timings reflect real cold/warm
costs (apart from Ollama keep-alive, which is the user's machine state).
"""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _diag_file_id(file: Path) -> str:
    """Stable, per-file id so each profiled file gets its own snapshots folder
    under cache/snapshots/. Without this, folder-mode runs overwrite each other.

    The `DIAG-` prefix keeps these visibly separate from real catalog file_ids
    (which are content-hash based) so cache cleanup is unambiguous.
    """
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in file.stem)
    return f"DIAG-{safe[:48]}"


# ── Background worker ────────────────────────────────────────────────────────

class _StageWorker(QThread):
    """Runs a single callable off the GUI thread; reports (label, ms, ok, text)."""
    finished_with = pyqtSignal(str, float, bool, str)   # stage, ms, ok, summary

    def __init__(self, stage: str, callable_) -> None:
        super().__init__()
        self._stage = stage
        self._callable = callable_

    def run(self) -> None:  # type: ignore[override]
        t0 = time.monotonic()
        ok = True
        summary = ""
        try:
            summary = self._callable()
        except Exception:
            ok = False
            summary = traceback.format_exc()
        dt_ms = (time.monotonic() - t0) * 1000.0
        self.finished_with.emit(self._stage, dt_ms, ok, summary)


# ── Stage row ────────────────────────────────────────────────────────────────

class _StageRow:
    """A label + Run button + last-result label triple, laid out into a grid."""
    def __init__(self, parent: QWidget, row: int, name: str,
                 on_run, grid: QGridLayout) -> None:
        self.name = name
        self.label = QLabel(name)
        self.run_btn = QPushButton("Run")
        self.run_btn.setFixedWidth(60)
        self.result = QLabel("(not run)")
        self.result.setStyleSheet("color: #9e9e9e; font-family: Consolas, monospace;")

        self.run_btn.clicked.connect(lambda: on_run(self.name))

        grid.addWidget(self.label,    row, 0)
        grid.addWidget(self.run_btn,  row, 1)
        grid.addWidget(self.result,   row, 2)

    def set_running(self) -> None:
        self.result.setText("running…")
        self.result.setStyleSheet("color: #1e88e5; font-family: Consolas, monospace;")
        self.run_btn.setEnabled(False)

    def set_result(self, dt_ms: float, ok: bool, summary: str) -> None:
        prefix = "OK " if ok else "FAIL "
        # Show only the first line of summary inline; full text goes to the log
        first_line = summary.splitlines()[0] if summary else ""
        self.result.setText(f"{prefix}{dt_ms:>8.1f} ms   {first_line[:60]}")
        self.result.setStyleSheet(
            "color: #4caf50; font-family: Consolas, monospace;" if ok
            else "color: #f44336; font-family: Consolas, monospace;"
        )
        self.run_btn.setEnabled(True)


# ── Main panel ───────────────────────────────────────────────────────────────

_STAGES = [
    "1. Load mesh",
    "2. Extract Brep meshes (.3dm only)",
    "3. Render snapshots (4 views + thumbnail)",
    "4. Compute measurements",
    "5. VLM call (Ollama)",
    "6. Finalize record",
]


class DiagnosticsPanel(QWidget):
    """Per-stage pipeline profiler."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._file: Path | None = None
        self._folder_files: list[Path] = []      # remaining files from folder mode
        self._folder_results: list[tuple[Path, list[tuple[str, float, bool]]]] = []
        self._state: dict[str, Any] = {}         # carries data between stages
        self._workers: list[_StageWorker] = []
        self._run_all_queue: list[str] = []
        self._batch_active = False               # True when walking a folder
        self._build_ui()

    # ── UI ───────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Diagnostics — per-stage pipeline profiler")
        tf = QFont(); tf.setPointSize(16); tf.setBold(True)
        title.setFont(tf)
        root.addWidget(title)

        # Scope reminder — Diagnostics is timing-only; it does NOT write
        # catalog JSON / index CSV. Those come from the Pilot panel
        # (production BatchRunner). Without this line users assume the
        # profiler also persists records and get confused when nothing shows
        # up in data/catalog/.
        scope = QLabel(
            "Profiling only — measures per-stage wall time and writes snapshots "
            "to cache/snapshots/DIAG-<name>/. Does NOT write catalog records or "
            "index CSV — use the Pilot panel for that."
        )
        scope.setStyleSheet("color: #f0b132; font-size: 11px;")
        scope.setWordWrap(True)
        root.addWidget(scope)

        # File / folder picker row
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Target:"))
        self._file_label = QLabel("(none)")
        self._file_label.setStyleSheet("color: #9e9e9e;")
        pick_file_btn = QPushButton("Pick file…")
        pick_file_btn.clicked.connect(self._pick_file)
        pick_folder_btn = QPushButton("Pick folder…")
        pick_folder_btn.clicked.connect(self._pick_folder)
        file_row.addWidget(self._file_label, stretch=1)
        file_row.addWidget(pick_file_btn)
        file_row.addWidget(pick_folder_btn)
        root.addLayout(file_row)

        # Batch progress (only meaningful in folder mode)
        self._batch_label = QLabel("")
        self._batch_label.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        root.addWidget(self._batch_label)

        # Stage grid
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(frame)
        grid.setColumnStretch(2, 1)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        self._rows: dict[str, _StageRow] = {}
        for i, name in enumerate(_STAGES):
            self._rows[name] = _StageRow(self, i, name, self._run_stage, grid)
        root.addWidget(frame)

        # Run-all row + reset
        action_row = QHBoxLayout()
        self._run_all_btn = QPushButton("Run All (sequential)")
        self._run_all_btn.setFixedHeight(36)
        self._run_all_btn.clicked.connect(self._run_all)
        reset_btn = QPushButton("Reset state")
        reset_btn.clicked.connect(self._reset_state)
        open_snaps_btn = QPushButton("Open snapshots folder")
        open_snaps_btn.clicked.connect(self._open_snapshots_folder)
        action_row.addWidget(self._run_all_btn)
        action_row.addWidget(reset_btn)
        action_row.addWidget(open_snaps_btn)
        action_row.addStretch()
        root.addLayout(action_row)

        # Log
        root.addWidget(QLabel("Log:"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            "font-family: Consolas, monospace; font-size: 11px;"
            " background: #12121e; color: #e0e0e0;"
        )
        root.addWidget(self._log, stretch=1)

    # ── File picker ──────────────────────────────────────────────────────────

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a CAD file to profile", "",
            "CAD files (*.stl *.3dm)"
        )
        if path:
            self._file = Path(path)
            self._folder_files.clear()
            self._batch_active = False
            self._batch_label.setText("")
            self._file_label.setText(str(self._file))
            self._file_label.setStyleSheet("color: #e0e0e0;")
            self._reset_state()
            self._set_per_stage_run_enabled(True)
            self._log.append(f"\n=== File: {self._file.name} ({self._file.stat().st_size / 1024:.1f} KB) ===")

    def _set_per_stage_run_enabled(self, enabled: bool) -> None:
        """Disable per-stage 'Run' buttons in folder mode — walking a folder
        only makes sense via 'Run All'. The Run All button stays enabled
        either way (it's the meaningful action in both modes)."""
        for row in self._rows.values():
            row.run_btn.setEnabled(enabled)

    def _in_folder_mode(self) -> bool:
        """True when a folder is queued (running or waiting for Run All)."""
        return self._batch_active or len(self._folder_files) > 0

    def _pick_folder(self) -> None:
        """Folder mode — discover all .stl / .3dm and queue them up.

        Folder mode disables the per-stage 'Run' buttons: walking a folder
        only makes sense end-to-end via 'Run All'. Without this gate users
        click per-stage 'Run' and see just one file processed, then assume
        the folder mode is broken.
        """
        path = QFileDialog.getExistingDirectory(
            self, "Pick a folder of CAD files to profile",
        )
        if not path:
            return
        folder = Path(path)
        files = sorted(
            [p for p in folder.rglob("*") if p.suffix.lower() in (".stl", ".3dm")]
        )
        if not files:
            self._log.append(f"\n(no .stl/.3dm files found under {folder})")
            return

        self._folder_files = files[1:]   # remaining (first is current)
        self._file = files[0]
        self._batch_active = False       # only flips True when Run All starts
        self._file_label.setText(f"{folder}  ·  {len(files)} files")
        self._file_label.setStyleSheet("color: #e0e0e0;")
        self._batch_label.setText(
            f"Folder mode: {len(files)} files queued. "
            "Per-stage 'Run' buttons are disabled — click 'Run All' to walk "
            "every file end-to-end."
        )
        self._folder_results.clear()
        self._reset_state()
        self._set_per_stage_run_enabled(False)
        self._log.append(
            f"\n=== Folder: {folder} "
            f"({len(files)} files; current: {self._file.name}) ==="
        )

    def _open_snapshots_folder(self) -> None:
        """Open the current file's snapshots folder (or the parent cache dir
        if no file is loaded yet) in Explorer."""
        from catalog_organizer.core.paths import cache_dir, snapshots_dir  # noqa: PLC0415
        if self._file is not None:
            folder = snapshots_dir(_diag_file_id(self._file))
        else:
            folder = cache_dir() / "snapshots"
        if not folder.exists() or not any(folder.iterdir()):
            self._log.append(
                f"(no snapshots yet — run stage 3 first; expected at {folder})"
            )
            return
        try:
            os.startfile(str(folder))  # type: ignore[attr-defined]  # Windows-only
        except Exception as exc:
            self._log.append(f"Could not open folder: {exc}\n  path: {folder}")

    def _reset_state(self) -> None:
        self._state.clear()
        for row in self._rows.values():
            row.result.setText("(not run)")
            row.result.setStyleSheet(
                "color: #9e9e9e; font-family: Consolas, monospace;"
            )
        self._log.append("(state reset)")

    # ── Stage dispatch ───────────────────────────────────────────────────────

    def _run_stage(self, name: str) -> None:
        if self._file is None:
            self._log.append("Pick a file first.")
            return

        if name not in _STAGE_CALLABLES:
            self._log.append(f"Unknown stage: {name}")
            return

        self._rows[name].set_running()
        self._log.append(f"→ {name} ...")

        callable_ = _STAGE_CALLABLES[name](self._file, self._state)
        worker = _StageWorker(name, callable_)
        worker.finished_with.connect(self._on_stage_done)
        worker.start()
        self._workers.append(worker)

    def _on_stage_done(self, stage: str, dt_ms: float, ok: bool, summary: str) -> None:
        self._rows[stage].set_result(dt_ms, ok, summary)
        # set_result() re-enables the per-stage Run button. In folder mode
        # we don't want users to fire individual stages mid-batch — keep
        # them disabled until batch finishes.
        if self._in_folder_mode():
            self._set_per_stage_run_enabled(False)
        marker = "✔" if ok else "✘"
        self._log.append(f"   {marker} {stage}: {dt_ms:.1f} ms")
        # Append multi-line summaries below the timing line, indented
        for line in summary.splitlines():
            if line.strip():
                self._log.append(f"      {line}")

        # Track this stage's timing for the current file (folder-mode summary)
        if self._batch_active and self._file is not None:
            if not self._folder_results or self._folder_results[-1][0] != self._file:
                self._folder_results.append((self._file, []))
            self._folder_results[-1][1].append((stage, dt_ms, ok))

        # Continue current file's queue
        if self._run_all_queue:
            if ok:
                next_name = self._run_all_queue.pop(0)
                self._run_stage(next_name)
            else:
                self._log.append(f"   (run-all aborted after {stage} failed)")
                self._run_all_queue.clear()
                if self._batch_active:
                    # Skip rest of this file's stages; move on to next file.
                    self._advance_to_next_file()
                else:
                    self._run_all_btn.setEnabled(True)
            return

        # Current file done — in batch mode, advance to next file
        if self._batch_active:
            self._advance_to_next_file()
        else:
            self._run_all_btn.setEnabled(True)

    def _advance_to_next_file(self) -> None:
        """Folder-mode helper: pop the next file and run its stage queue."""
        if not self._folder_files:
            # Batch is done — print summary
            self._batch_active = False
            self._run_all_btn.setEnabled(True)
            self._print_folder_summary()
            return

        self._file = self._folder_files.pop(0)
        idx_done = len(self._folder_results)
        total = idx_done + 1 + len(self._folder_files)
        self._batch_label.setText(
            f"Folder mode: file {idx_done + 1} / {total} — {self._file.name}"
        )
        self._reset_state()
        self._log.append(f"\n=== [{idx_done + 1}/{total}] {self._file.name} ===")
        self._run_all_queue = self._stages_for_current_file()
        first = self._run_all_queue.pop(0)
        self._run_stage(first)

    def _print_folder_summary(self) -> None:
        """At the end of a folder run, dump a compact per-file timing table."""
        self._log.append("\n=== Folder summary (ms per stage) ===")
        # Column header
        stage_keys = ["load", "brep", "render", "measure", "vlm", "final", "TOTAL"]
        header = f"  {'file':<32} " + " ".join(f"{k:>8}" for k in stage_keys)
        self._log.append(header)
        for file, stage_results in self._folder_results:
            row = {
                "1. Load mesh": "load",
                "2. Extract Brep meshes (.3dm only)": "brep",
                "3. Render snapshots (4 views + thumbnail)": "render",
                "4. Compute measurements": "measure",
                "5. VLM call (Ollama)": "vlm",
                "6. Finalize record": "final",
            }
            timings = {abbr: 0.0 for abbr in row.values()}
            ok_all = True
            for stage_name, dt_ms, ok in stage_results:
                abbr = row.get(stage_name)
                if abbr:
                    timings[abbr] = dt_ms
                ok_all = ok_all and ok
            total_ms = sum(timings.values())
            marker = "✔" if ok_all else "✘"
            cells = " ".join(f"{timings.get(k, 0):>8.0f}" for k in stage_keys[:-1])
            self._log.append(f"  {marker} {file.name:<30} {cells} {total_ms:>8.0f}")
        self._log.append("=" * 60)

    def _stages_for_current_file(self) -> list[str]:
        """Skip Brep extraction for STL files."""
        if self._file is not None and self._file.suffix.lower() == ".stl":
            return [s for s in _STAGES if "Brep" not in s]
        return list(_STAGES)

    def _run_all(self) -> None:
        if self._file is None:
            self._log.append("Pick a file or folder first.")
            return

        self._reset_state()
        self._run_all_btn.setEnabled(False)

        # Folder mode: this file is the first of many; activate batch mode.
        if self._folder_files:
            self._batch_active = True
            self._folder_results.clear()
            total = 1 + len(self._folder_files)
            self._batch_label.setText(
                f"Folder mode: file 1 / {total} — {self._file.name}"
            )
            self._log.append(f"\n=== [1/{total}] {self._file.name} ===")
        else:
            self._batch_active = False
            self._log.append(f"\n=== Run All on {self._file.name} ===")

        self._run_all_queue = self._stages_for_current_file()
        first = self._run_all_queue.pop(0)
        self._run_stage(first)


# ── Stage callables ───────────────────────────────────────────────────────────
# Each entry is a factory: (file, state) -> () -> summary_str

def _make_load_mesh(file: Path, state: dict):
    def go():
        from catalog_organizer.orchestrator.pipeline import _load_mesh  # noqa: PLC0415
        mesh = _load_mesh(file, file.suffix.lower())
        state["mesh"] = mesh
        return (
            f"vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, "
            f"bounds={mesh.bounds.tolist()}"
        )
    return go


def _make_brep_extract(file: Path, state: dict):
    def go():
        if file.suffix.lower() != ".3dm":
            return "skipped (not a .3dm file)"
        from catalog_organizer.snapshotter.threedm import (  # noqa: PLC0415
            load_3dm_as_trimesh_arrays,
        )
        verts, faces = load_3dm_as_trimesh_arrays(file)
        state["brep_verts"] = verts
        state["brep_faces"] = faces
        return f"verts={len(verts)}, faces={len(faces)}"
    return go


def _make_render_snapshots(file: Path, state: dict):
    def go():
        # Use a temp file_id so snapshots go to a known location
        from catalog_organizer.core.config import load_pipeline_settings  # noqa: PLC0415
        from catalog_organizer.orchestrator.pipeline import _take_snapshots  # noqa: PLC0415
        s = load_pipeline_settings()
        resolution = int(s.get("snapshot", {}).get("resolution", 512))
        thumb = int(s.get("snapshot", {}).get("thumbnail_size", 256))
        snaps = _take_snapshots(
            file, _diag_file_id(file), file.suffix.lower(),
            resolution=resolution, thumbnail_size=thumb,
        )
        state["snapshots"] = snaps
        sizes = ", ".join(
            f"{k}={snaps[k].stat().st_size // 1024}KB"
            for k in ("front", "side", "top", "iso") if k in snaps
        )
        return f"resolution={resolution}, {sizes}"
    return go


def _make_measurements(file: Path, state: dict):
    def go():
        from catalog_organizer.cad.measurements import (  # noqa: PLC0415
            estimate_volume, measure_bbox,
        )
        mesh = state.get("mesh")
        if mesh is None:
            return "no mesh in state — run Load Mesh first"
        bbox = measure_bbox(mesh)
        vol, conf = estimate_volume(mesh)
        return (
            f"bbox={bbox.width:.1f}x{bbox.height:.1f}x{bbox.depth:.1f}mm  "
            f"vol={vol:.0f}mm³ ({conf})"
        )
    return go


def _make_vlm_call(file: Path, state: dict):
    def go():
        from catalog_organizer.core.config import load_pipeline_settings  # noqa: PLC0415
        from catalog_organizer.vlm.providers import create_vlm_provider  # noqa: PLC0415
        snaps = state.get("snapshots")
        if snaps is None:
            return "no snapshots in state — run Render Snapshots first"
        s = load_pipeline_settings().get("vlm", {})
        client = create_vlm_provider(s)
        snapshot_paths = [snaps[v] for v in ("front", "side", "top", "iso")]
        result = client.classify(snapshot_paths)
        state["vlm"] = result
        t = client.last_timings or {}
        # Multi-line summary: first line stays compact (shown in the row label),
        # extra lines are picked up by _on_stage_done and appended to the log.
        first = (
            f"{result.main_category}/{result.subcategory}  "
            f"main_conf={result.main_category_confidence:.2f}  "
            f"tags={len(result.controlled_tags)}  "
            f"review={result.needs_manual_review}"
        )
        if t:
            breakdown = (
                f"timings: load={t['load_ms']:.0f}ms  "
                f"prompt_eval={t['prompt_eval_ms']:.0f}ms ({t['prompt_eval_count']} tok)  "
                f"eval={t['eval_ms']:.0f}ms ({t['eval_count']} tok)  "
                f"total={t['total_ms']:.0f}ms  "
                f"images={t['image_bytes']//1024}KB"
            )
            attempts = t.get("attempts") or []
            if len(attempts) > 1 or (attempts and attempts[0]["outcome"] != "ok"):
                # Show per-attempt outcomes so retries / timeouts are visible.
                # If there was only one successful attempt, this line is noise — skip it.
                parts = [
                    f"#{a['i']+1}={a['outcome']}({a['wall_ms']:.0f}ms)"
                    for a in attempts
                ]
                breakdown += f"\nattempts: {' → '.join(parts)}"
            return f"{first}\n{breakdown}"
        return first
    return go


def _make_finalize(file: Path, state: dict):
    def go():
        # We don't have a full ManifestEntry/PreparedItem here, so synthesise a minimal one
        from datetime import datetime, timezone
        from catalog_organizer.cad.measurements import (  # noqa: PLC0415
            estimate_volume, measure_bbox,
        )
        from catalog_organizer.core.config import (  # noqa: PLC0415
            load_metal_densities, load_pipeline_settings,
        )
        from catalog_organizer.core.schemas import ManifestEntry, Measurements  # noqa: PLC0415
        from catalog_organizer.catalog.tag_validator import TagValidator  # noqa: PLC0415
        from catalog_organizer.orchestrator.pipeline import (  # noqa: PLC0415
            PipelineDeps, PreparedItem, finalize_from_vlm,
        )

        mesh = state.get("mesh")
        snaps = state.get("snapshots")
        vlm = state.get("vlm")
        if not (mesh and snaps and vlm):
            return "need mesh + snapshots + vlm — run earlier stages first"

        settings = load_pipeline_settings()
        bbox = measure_bbox(mesh)
        vol, vol_conf = estimate_volume(mesh)
        bbox_m = Measurements(
            bbox_width_mm=bbox.width, bbox_height_mm=bbox.height,
            bbox_depth_mm=bbox.depth, volume_mm3=vol,
            geometry_source=("trimesh" if file.suffix.lower() == ".stl" else "rhino"),
        )
        # Minimal ManifestEntry stand-in
        from hashlib import sha256
        entry = ManifestEntry(
            file_id=_diag_file_id(file),
            source_path=str(file),
            file_extension=file.suffix.lower(),  # type: ignore[arg-type]
            file_size_bytes=file.stat().st_size,
            created_at=None, modified_at=None,
            sha256=sha256(file.read_bytes()).hexdigest(),
            duplicate_of=None, state="new",
            processing_batch="diag",
            scan_timestamp=datetime.now(tz=timezone.utc),
        )
        prepared = PreparedItem(
            entry=entry, snapshot_paths=[snaps[v] for v in ("front","side","top","iso")],
            mesh=mesh, bbox_measurements=bbox_m, vol_conf=vol_conf,
            geometry_source=bbox_m.geometry_source,
        )
        deps = PipelineDeps(
            vlm_client=None,    # not used by finalize
            tag_validator=TagValidator(),
            metal_densities=load_metal_densities(),
            scale_thresholds=settings.get("scale_thresholds", {}),
            snapshot_resolution=int(settings.get("snapshot", {}).get("resolution", 512)),
            thumbnail_size=int(settings.get("snapshot", {}).get("thumbnail_size", 256)),
        )
        record = finalize_from_vlm(prepared, vlm, deps)
        state["record"] = record
        return f"state={record.state}  cat={record.main_category}/{record.subcategory}  tags={len(record.controlled_tags)}"
    return go


_STAGE_CALLABLES = {
    "1. Load mesh":                              _make_load_mesh,
    "2. Extract Brep meshes (.3dm only)":        _make_brep_extract,
    "3. Render snapshots (4 views + thumbnail)": _make_render_snapshots,
    "4. Compute measurements":                   _make_measurements,
    "5. VLM call (Ollama)":                      _make_vlm_call,
    "6. Finalize record":                        _make_finalize,
}
