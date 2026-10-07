"""Batch orchestrator — producer/consumer parallel pipeline (D.16).

Architecture (locked in `docs/superpowers/specs/2026-05-15-parallel-pipeline-design.md`):

  ┌─ Snapshot worker (1 thread, VTK-locked) ──────────┐
  │  prepare_snapshots(entry) → push to ready_queue   │
  └───────────────────────────────────────────────────┘
                         │
                         ▼  bounded Queue (size 6)
  ┌─ VLM worker pool (N=parallel_workers, default 2) ┐
  │  classify(snaps) → finalize_from_vlm → write     │
  │  serialised by single threading.Lock              │
  └───────────────────────────────────────────────────┘

VTK is single-threaded; that's why snapshots are produced by one thread.
The VLM is the dominant cost (~80% per file), so we parallelise it
instead. The user must set `OLLAMA_NUM_PARALLEL` on their Ollama server
to a value ≥ `parallel_workers` to actually get concurrent inference.

`parallel_workers=1` produces a degenerate 1+1 pipeline that still
overlaps CPU work with the VLM call (no behaviour change at the API
level — all signals fire, completion order may be arbitrary at N>1).
"""
from __future__ import annotations

import queue
import threading
import uuid
from datetime import datetime, timezone

from PyQt6.QtCore import QObject, pyqtSignal

from catalog_organizer.catalog.index import CatalogIndex
from catalog_organizer.catalog.writer import CatalogWriter
from catalog_organizer.core.audit import AuditWriter
from catalog_organizer.core.schemas import AuditEvent, ManifestEntry
from catalog_organizer.orchestrator.pipeline import (
    PipelineDeps,
    finalize_from_vlm,
    prepare_snapshots,
)
from catalog_organizer.orchestrator.resume import filter_eligible

_SENTINEL = object()
_QUEUE_MAXSIZE = 6


def _record_cost_from_client(client) -> None:
    """Push the just-completed call's timings to the session cost tracker.

    Safe to call on any provider (Ollama returns cost_usd=0). Wrapped in a
    try/except because the tracker is Qt-aware and we don't want to crash
    the batch if the GUI is somehow torn down mid-run.
    """
    try:
        from catalog_organizer.core.cost_tracker import tracker_singleton  # noqa: PLC0415
        tracker_singleton().record_call(getattr(client, "last_timings", None))
    except Exception:
        pass


class BatchRunner(QObject):
    progressChanged = pyqtSignal(int, int)     # done, total
    fileDone = pyqtSignal(str)                 # file_id
    fileFailed = pyqtSignal(str, str)          # file_id, error_code
    batchFinished = pyqtSignal(int, int)       # success_count, fail_count

    def __init__(
        self,
        deps: PipelineDeps,
        writer: CatalogWriter,
        index: CatalogIndex,
        audit: AuditWriter | None = None,
        parent: QObject | None = None,
        *,
        parallel_workers: int = 1,
    ) -> None:
        super().__init__(parent)
        if parallel_workers < 1 or parallel_workers > 8:
            raise ValueError(
                f"parallel_workers must be 1..8, got {parallel_workers}"
            )
        self._deps = deps
        self._writer = writer
        self._index = index
        self._audit = audit
        self._cancel = threading.Event()
        self._pause = threading.Event()
        self._write_lock = threading.Lock()
        self._parallel_workers = parallel_workers

    # ── Public control API ───────────────────────────────────────────────

    def cancel(self) -> None:
        self._cancel.set()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    # ── Entry point ──────────────────────────────────────────────────────

    def run(
        self,
        entries: list[ManifestEntry],
        batch_name: str = "pilot",
        *,
        skip_filter: bool = False,
    ) -> tuple[int, int]:
        """Run a batch.

        `skip_filter=True` bypasses `filter_eligible` (which normally drops
        items already in the catalog). Use this when the user explicitly
        chose to reprocess a folder — the catalog index is allowed to be
        overwritten by a later record with the same file_id.
        """
        # Eagerly boot Rhino.Inside HERE, synchronously, before any worker
        # thread exists. `cad.rhino_engine._boot()` can only attempt the
        # real rhinoinside.load()/CLR-hosting bootstrap ONCE per process; if
        # that first attempt happens to fire from a freshly-spawned worker
        # thread (or races against another thread's first call), the .NET
        # CLR host fails and Rhino is marked permanently unavailable for the
        # rest of the process — silently downgrading every subsequent .3dm
        # cutter-volume measurement to the old heuristic with NO error
        # surfaced (reproduced via `tests/test_batch_runner_parallel.py`
        # corrupting a later golden-file test in the same pytest process,
        # 2026-06-30). Booting once here, before `snapshot_thread`/the VLM
        # pool are spawned, makes every later call from any thread a cheap
        # already-booted check instead of a race.
        from catalog_organizer.cad import rhino_engine  # noqa: PLC0415
        rhino_engine.is_available()

        # Single-worker path runs in the calling thread, preserving the
        # synchronous signal semantics existing callers (and tests) rely on.
        if self._parallel_workers == 1:
            return self._run_sequential(entries, batch_name, skip_filter=skip_filter)

        eligible = entries if skip_filter else filter_eligible(entries, self._index)
        total = len(eligible)
        ready_queue: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        counters = {"done": 0, "success": 0, "fail": 0}

        def snapshot_loop() -> None:
            """Producer: prepare each file's snapshots and push to queue."""
            try:
                for entry in eligible:
                    if self._cancel.is_set():
                        break
                    while self._pause.is_set() and not self._cancel.is_set():
                        self._cancel.wait(timeout=0.1)
                    if self._cancel.is_set():
                        break

                    try:
                        prepared = prepare_snapshots(entry, self._deps)
                        ready_queue.put((entry, prepared, None))
                    except Exception as exc:
                        # Snapshot itself failed; let a VLM worker emit the
                        # failure signal so all reporting flows through one place.
                        ready_queue.put((entry, None, exc))
            finally:
                # Always release every VLM worker, even on cancel/exception.
                for _ in range(self._parallel_workers):
                    ready_queue.put(_SENTINEL)

        def vlm_loop() -> None:
            """Consumer: pull from queue, classify, finalize, write."""
            while True:
                item = ready_queue.get()
                if item is _SENTINEL:
                    return
                entry, prepared, snap_exc = item

                if self._cancel.is_set():
                    continue

                if snap_exc is not None:
                    code = getattr(snap_exc, "code", "snapshot.failed")
                    self._handle_failure(entry.file_id, code, str(snap_exc),
                                         counters, total)
                    continue

                try:
                    vlm = self._deps.vlm_client.classify(prepared.snapshot_paths)
                    # Record per-call cost into the session tracker. This
                    # has to fire even when finalize_from_vlm() throws —
                    # the API call already cost the user money, so it should
                    # show up in the live status bar regardless.
                    _record_cost_from_client(self._deps.vlm_client)
                    record = finalize_from_vlm(prepared, vlm, self._deps)
                except Exception as exc:
                    code = getattr(exc, "code", "pipeline.failed")
                    self._handle_failure(entry.file_id, code, str(exc),
                                         counters, total)
                    continue

                with self._write_lock:
                    self._writer.append(record)
                    self._index.upsert(record)
                    counters["done"] += 1
                    counters["success"] += 1
                    self._emit_audit(
                        entry.file_id, "review", "info", "pipeline.done",
                        f"processed {entry.file_id} → {record.state}",
                    )
                    self.fileDone.emit(entry.file_id)
                    self.progressChanged.emit(counters["done"], total)

        # ── Spin up threads ──────────────────────────────────────────────
        snapshot_thread = threading.Thread(
            target=snapshot_loop, name="snapshot-worker", daemon=True,
        )
        vlm_threads = [
            threading.Thread(target=vlm_loop, name=f"vlm-worker-{i}", daemon=True)
            for i in range(self._parallel_workers)
        ]
        snapshot_thread.start()
        for t in vlm_threads:
            t.start()

        snapshot_thread.join()
        for t in vlm_threads:
            t.join()

        with self._write_lock:
            self._writer.flush()
            success = counters["success"]
            fail = counters["fail"]

        self.batchFinished.emit(success, fail)
        return success, fail

    # ── Single-worker legacy path (preserves caller-thread signal semantics)

    def _run_sequential(
        self,
        entries: list[ManifestEntry],
        batch_name: str,
        *,
        skip_filter: bool = False,
    ) -> tuple[int, int]:
        from catalog_organizer.orchestrator.pipeline import process_one  # noqa: PLC0415

        eligible = entries if skip_filter else filter_eligible(entries, self._index)
        total = len(eligible)
        success = 0
        fail = 0
        for i, entry in enumerate(eligible, start=1):
            if self._cancel.is_set():
                break
            while self._pause.is_set() and not self._cancel.is_set():
                self._cancel.wait(timeout=0.1)
            if self._cancel.is_set():
                break

            try:
                record = process_one(entry, self._deps)
                _record_cost_from_client(self._deps.vlm_client)
                self._writer.append(record)
                self._index.upsert(record)
                self._emit_audit(
                    entry.file_id, "review", "info", "pipeline.done",
                    f"processed {entry.file_id} → {record.state}",
                )
                self.fileDone.emit(entry.file_id)
                success += 1
            except Exception as exc:
                # Still record cost — the VLM call already happened and the
                # API charge is real regardless of downstream failure.
                _record_cost_from_client(self._deps.vlm_client)
                code = getattr(exc, "code", "pipeline.failed")
                self.fileFailed.emit(entry.file_id, code)
                self._emit_audit(entry.file_id, "review", "error", code, str(exc))
                fail += 1

            self.progressChanged.emit(i, total)

        self._writer.flush()
        self.batchFinished.emit(success, fail)
        return success, fail

    # ── Failure helper (serialises counter + signal updates) ─────────────

    def _handle_failure(
        self,
        file_id: str,
        code: str,
        message: str,
        counters: dict,
        total: int,
    ) -> None:
        with self._write_lock:
            counters["done"] += 1
            counters["fail"] += 1
            self.fileFailed.emit(file_id, code)
            self._emit_audit(file_id, "review", "error", code, message)
            self.progressChanged.emit(counters["done"], total)

    # ── Audit helper ─────────────────────────────────────────────────────

    def _emit_audit(
        self,
        file_id: str,
        phase: str,
        level: str,
        code: str,
        message: str,
    ) -> None:
        if self._audit is None:
            return
        self._audit.write(AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(tz=timezone.utc),
            file_id=file_id,
            phase=phase,          # type: ignore[arg-type]
            level=level,          # type: ignore[arg-type]
            code=code,
            message=message,
        ))
