"""Parallel BatchRunner tests (D.16).

Cross-thread signal delivery via Qt requires an event loop. The parallel
path emits from worker threads, so we use queued signal delivery and a
short busy-wait on counter state instead of asserting on signal events.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import pytest

from catalog_organizer.catalog.index import CatalogIndex
from catalog_organizer.catalog.tag_validator import TagValidator
from catalog_organizer.catalog.writer import CatalogWriter
from catalog_organizer.core.ids import IdAllocator
from catalog_organizer.core.schemas import VLMResult
from catalog_organizer.orchestrator.batch import BatchRunner
from catalog_organizer.orchestrator.pipeline import PipelineDeps
from catalog_organizer.scanner.manifest import scan

FIXTURES = Path(__file__).parent / "fixtures"

_DENSITIES = {
    "silver_925":      {"density_g_cm3": 10.36},
    "gold_10k_yellow": {"density_g_cm3": 11.57},
    "gold_10k_white":  {"density_g_cm3": 11.01},
    "gold_14k_yellow": {"density_g_cm3": 13.07},
    "gold_14k_white":  {"density_g_cm3": 12.56},
    "gold_18k_yellow": {"density_g_cm3": 15.58},
    "gold_18k_white":  {"density_g_cm3": 14.66},
    "platinum_950":    {"density_g_cm3": 20.13},
}
_THRESHOLDS = {
    "ring":     {"min_bbox_mm": 5, "max_bbox_mm": 80, "min_inner_mm": 10, "max_inner_mm": 30},
    "earring":  {"min_bbox_mm": 3, "max_bbox_mm": 120},
    "bracelet": {"min_bbox_mm": 25, "max_bbox_mm": 140},
    "pendant":  {"min_bbox_mm": 3, "max_bbox_mm": 100},
}
_DICT = ["ring", "band", "closed_form", "polished", "round_form",
         "single_piece", "geometric", "earring", "pair", "drop_structure"]
_FB = {
    "ring":    ["ring", "band", "closed_form", "polished", "round_form"],
    "unknown": ["single_piece", "closed_form", "polished", "geometric", "single_piece"],
}


class FakeVLM:
    """Deterministic VLM stub. Classifies every input as a polished ring band."""
    def classify(self, snapshot_paths, second_pass=None):
        return VLMResult(
            main_category="ring",
            main_category_confidence=0.91,
            subcategory="band",
            subcategory_confidence=0.85,
            controlled_tags=["ring", "band", "polished", "round_form", "closed_form"],
            tag_confidence=0.88,
            stone_presence="no_stone",
            estimated_visible_stone_count=0,
            polished_status="polished",
            visible_sprue_or_casting_stem=False,
            sprue_confidence=0.9,
        )


def _make_deps() -> PipelineDeps:
    return PipelineDeps(
        vlm_client=FakeVLM(),
        tag_validator=TagValidator(dictionary=_DICT, fallbacks=_FB),
        metal_densities=_DENSITIES,
        scale_thresholds=_THRESHOLDS,
        snapshot_resolution=128,
        thumbnail_size=64,
    )


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    def fake_cache_dir():
        d = tmp_path / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d
    monkeypatch.setattr("catalog_organizer.snapshotter.stl.snapshots_dir",
                        lambda fid: fake_cache_dir() / "snapshots" / fid)
    monkeypatch.setattr("catalog_organizer.snapshotter.stl.thumbnails_dir",
                        lambda: fake_cache_dir() / "thumbnails")
    monkeypatch.setattr("catalog_organizer.snapshotter.threedm.snapshots_dir",
                        lambda fid: fake_cache_dir() / "snapshots" / fid)
    monkeypatch.setattr("catalog_organizer.snapshotter.threedm.thumbnails_dir",
                        lambda: fake_cache_dir() / "thumbnails")
    return tmp_path


# ── Tests ────────────────────────────────────────────────────────────────────

def test_batch_runner_parallel_workers_processes_all(isolated_cache):
    """5 fixtures + parallel_workers=2: all 5 records present, no losses."""
    allocator = IdAllocator(isolated_cache / "next_id.txt")
    manifest = scan([FIXTURES], batch_name="t",
                    manifest_path=isolated_cache / "manifest.jsonl",
                    allocator=allocator)
    entries = list(manifest.values())
    assert len(entries) == 5

    writer = CatalogWriter(
        jsonl_path=isolated_cache / "catalog_master.jsonl",
        csv_path=isolated_cache / "catalog_export.csv",
        flush_every=10,
    )
    index = CatalogIndex(jsonl_path=isolated_cache / "catalog_master.jsonl")
    runner = BatchRunner(_make_deps(), writer, index, audit=None, parallel_workers=2)

    success, fail = runner.run(entries, batch_name="pilot")
    writer.close()

    assert success == 5
    assert fail == 0
    records = list(CatalogIndex(jsonl_path=isolated_cache / "catalog_master.jsonl").all())
    assert len(records) == 5
    assert {r.file_id for r in records} == {e.file_id for e in entries}


def test_batch_runner_parallel_writer_lock_serialises_writes(isolated_cache, monkeypatch):
    """3 workers × 5 fixtures: writer.append must never be called concurrently."""
    allocator = IdAllocator(isolated_cache / "next_id.txt")
    manifest = scan([FIXTURES], batch_name="t",
                    manifest_path=isolated_cache / "manifest.jsonl",
                    allocator=allocator)
    entries = list(manifest.values())

    writer = CatalogWriter(
        jsonl_path=isolated_cache / "catalog_master.jsonl",
        csv_path=isolated_cache / "catalog_export.csv",
        flush_every=10,
    )
    index = CatalogIndex(jsonl_path=isolated_cache / "catalog_master.jsonl")

    in_flight = [0]
    max_in_flight = [0]
    overlap_lock = Lock()
    orig_append = writer.append

    def tracking_append(record):
        with overlap_lock:
            in_flight[0] += 1
            max_in_flight[0] = max(max_in_flight[0], in_flight[0])
        try:
            return orig_append(record)
        finally:
            with overlap_lock:
                in_flight[0] -= 1

    monkeypatch.setattr(writer, "append", tracking_append)

    runner = BatchRunner(_make_deps(), writer, index, audit=None, parallel_workers=3)
    runner.run(entries)
    writer.close()

    assert max_in_flight[0] == 1, (
        f"writer.append was called concurrently (max overlap={max_in_flight[0]})"
    )


def test_batch_runner_parallel_cancel_drains_cleanly(isolated_cache):
    """Cancel after first record completes; remaining workers exit cleanly.

    NOTE: cancellation is triggered from inside the fake VLM, NOT via a Qt
    signal connection. Connecting a lambda to `fileDone` here caused an
    intermittent access-violation crash: the signal is emitted from a worker
    thread, delivery is queued on the QApplication event loop, and the queued
    event could fire during the *next* test's setup against a torn-down
    runner object.
    """
    allocator = IdAllocator(isolated_cache / "next_id.txt")
    manifest = scan([FIXTURES], batch_name="t",
                    manifest_path=isolated_cache / "manifest.jsonl",
                    allocator=allocator)
    entries = list(manifest.values())

    writer = CatalogWriter(
        jsonl_path=isolated_cache / "catalog_master.jsonl",
        csv_path=isolated_cache / "catalog_export.csv",
        flush_every=10,
    )
    index = CatalogIndex(jsonl_path=isolated_cache / "catalog_master.jsonl")

    class CancellingVLM(FakeVLM):
        """Cancels the batch right after producing its first result."""
        def classify(self, snapshot_paths, second_pass=None):
            result = super().classify(snapshot_paths, second_pass)
            runner.cancel()      # threading.Event.set() — thread-safe, no Qt
            return result

    deps = _make_deps()
    deps.vlm_client = CancellingVLM()
    runner = BatchRunner(deps, writer, index, audit=None, parallel_workers=2)

    success, fail = runner.run(entries)
    writer.close()

    assert success + fail <= len(entries)
    # Every persisted record must have a terminal state.
    records = list(CatalogIndex(jsonl_path=isolated_cache / "catalog_master.jsonl").all())
    assert all(r.state in ("final", "needs_review", "failed") for r in records)


def test_batch_runner_parallel_workers_one_works(isolated_cache):
    """parallel_workers=1 still produces all 5 records (sequential legacy path)."""
    allocator = IdAllocator(isolated_cache / "next_id.txt")
    manifest = scan([FIXTURES], batch_name="t",
                    manifest_path=isolated_cache / "manifest.jsonl",
                    allocator=allocator)
    entries = list(manifest.values())

    writer = CatalogWriter(
        jsonl_path=isolated_cache / "catalog_master.jsonl",
        csv_path=isolated_cache / "catalog_export.csv",
        flush_every=10,
    )
    index = CatalogIndex(jsonl_path=isolated_cache / "catalog_master.jsonl")
    runner = BatchRunner(_make_deps(), writer, index, audit=None, parallel_workers=1)

    success, fail = runner.run(entries)
    writer.close()
    assert success == 5
    assert fail == 0


def test_batch_runner_rejects_bad_parallel_workers_value(isolated_cache):
    """Validation: parallel_workers must be 1..8."""
    writer = CatalogWriter(
        jsonl_path=isolated_cache / "catalog_master.jsonl",
        csv_path=isolated_cache / "catalog_export.csv",
        flush_every=10,
    )
    index = CatalogIndex(jsonl_path=isolated_cache / "catalog_master.jsonl")
    with pytest.raises(ValueError):
        BatchRunner(_make_deps(), writer, index, audit=None, parallel_workers=0)
    with pytest.raises(ValueError):
        BatchRunner(_make_deps(), writer, index, audit=None, parallel_workers=99)
