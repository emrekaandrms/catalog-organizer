"""End-to-end orchestrator tests with a deterministic fake VLM client."""
from __future__ import annotations

from pathlib import Path

import pytest

from catalog_organizer.catalog.index import CatalogIndex
from catalog_organizer.catalog.tag_validator import TagValidator
from catalog_organizer.catalog.writer import CatalogWriter
from catalog_organizer.core.ids import IdAllocator
from catalog_organizer.core.schemas import VLMResult
from catalog_organizer.orchestrator.batch import BatchRunner
from catalog_organizer.orchestrator.pipeline import PipelineDeps, process_one
from catalog_organizer.orchestrator.resume import filter_eligible
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

_DICT = [
    "ring", "band", "closed_form", "polished", "round_form",
    "single_piece", "geometric", "earring", "pair", "drop_structure",
]
_FB = {
    "ring": ["ring", "band", "closed_form", "polished", "round_form"],
    "unknown": ["single_piece", "closed_form", "polished", "geometric", "single_piece"],
}


class FakeVLM:
    """Deterministic VLM that classifies every file as a ring with stable values."""
    def classify(self, snapshot_paths: list[Path], second_pass: dict | None = None) -> VLMResult:
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
            uncertainty_notes=[],
            needs_second_pass=False,
            needs_manual_review=False,
        )


class _DesignVLM(FakeVLM):
    """FakeVLM variant that lets a test control design_complete/confidence,
    for testing pipeline.py's sellability computation
    (the scale/sales design notes §2, 2026-07-24)."""
    def __init__(self, design_complete: bool, confidence: float) -> None:
        self._design_complete = design_complete
        self._confidence = confidence

    def classify(self, snapshot_paths, second_pass=None):
        result = super().classify(snapshot_paths, second_pass)
        return result.model_copy(update={
            "design_complete": self._design_complete,
            "design_complete_confidence": self._confidence,
        })


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch):
    def fake_cache_dir() -> Path:
        d = tmp_path / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr("catalog_organizer.snapshotter.stl.snapshots_dir",
                        lambda file_id: fake_cache_dir() / "snapshots" / file_id)
    monkeypatch.setattr("catalog_organizer.snapshotter.stl.thumbnails_dir",
                        lambda: fake_cache_dir() / "thumbnails")
    monkeypatch.setattr("catalog_organizer.snapshotter.threedm.snapshots_dir",
                        lambda file_id: fake_cache_dir() / "snapshots" / file_id)
    monkeypatch.setattr("catalog_organizer.snapshotter.threedm.thumbnails_dir",
                        lambda: fake_cache_dir() / "thumbnails")
    return tmp_path


def _make_deps() -> PipelineDeps:
    return PipelineDeps(
        vlm_client=FakeVLM(),
        tag_validator=TagValidator(dictionary=_DICT, fallbacks=_FB),
        metal_densities=_DENSITIES,
        scale_thresholds=_THRESHOLDS,
        snapshot_resolution=128,
        thumbnail_size=64,
    )


def test_process_one_stl_end_to_end(isolated_cache: Path):
    allocator = IdAllocator(isolated_cache / "next_id.txt")
    manifest = scan(
        [FIXTURES],
        batch_name="t",
        manifest_path=isolated_cache / "manifest.jsonl",
        allocator=allocator,
    )
    # Pick a fixture with non-degenerate bbox (sample_ring.stl spans (1,1,0)).
    stl_entry = next(
        e for e in manifest.values()
        if e.file_extension == ".stl" and "ring" in e.source_path.lower()
    )
    record = process_one(stl_entry, _make_deps())
    assert record.main_category == "ring"
    assert len(record.controlled_tags) >= 5
    assert record.measurements.bbox_width_mm > 0
    assert record.measurements.geometry_source == "trimesh"
    assert record.state in ("final", "needs_review")
    assert record.schema_version == 1
    assert len(record.snapshot_paths) == 4


def _stl_ring_entry(isolated_cache: Path):
    allocator = IdAllocator(isolated_cache / "next_id.txt")
    manifest = scan(
        [FIXTURES],
        batch_name="t",
        manifest_path=isolated_cache / "manifest.jsonl",
        allocator=allocator,
    )
    return next(
        e for e in manifest.values()
        if e.file_extension == ".stl" and "ring" in e.source_path.lower()
    )


def test_sellability_sellable_when_design_complete_and_confident(isolated_cache: Path):
    deps = _make_deps()
    deps.vlm_client = _DesignVLM(design_complete=True, confidence=0.95)
    record = process_one(_stl_ring_entry(isolated_cache), deps)
    assert record.design_complete is True
    assert record.sellability == "sellable"
    assert record.sellability_reason is None


def test_sellability_not_sellable_when_design_incomplete(isolated_cache: Path):
    """Regression target for the scale/sales design notes §2: a design the
    VLM confidently judges incomplete (e.g. a sprue stub, a missing stone
    seat, a part that doesn't sit flush) must be flagged not_sellable."""
    deps = _make_deps()
    deps.vlm_client = _DesignVLM(design_complete=False, confidence=0.95)
    record = process_one(_stl_ring_entry(isolated_cache), deps)
    assert record.design_complete is False
    assert record.sellability == "not_sellable"
    assert record.sellability_reason == "design_incomplete"


def test_sellability_needs_review_when_design_complete_confidence_low(isolated_cache: Path):
    """Low design_complete_confidence must route to needs_review regardless
    of which way the boolean leans — don't trust an unconfident verdict
    either way (same discipline as every other confidence field)."""
    deps = _make_deps()
    deps.vlm_client = _DesignVLM(design_complete=True, confidence=0.4)
    record = process_one(_stl_ring_entry(isolated_cache), deps)
    assert record.sellability == "needs_review"
    assert record.sellability_reason == "design_complete_uncertain"


def test_sellability_ignores_brand_entirely(isolated_cache: Path):
    """the scale/sales design notes §2.1: this is a professional design
    workshop drawing commissioned/brand-referenced pieces as a paid
    service — brand must NEVER gate sellability, only design_complete."""
    class _BrandedCompleteVLM(_DesignVLM):
        def classify(self, snapshot_paths, second_pass=None):
            result = super().classify(snapshot_paths, second_pass)
            return result.model_copy(update={
                "brand_guess": "cartier", "brand_confidence": 0.95,
            })

    deps = _make_deps()
    deps.vlm_client = _BrandedCompleteVLM(design_complete=True, confidence=0.95)
    record = process_one(_stl_ring_entry(isolated_cache), deps)
    assert record.brand == "cartier"
    assert record.sellability == "sellable"


def test_batch_runner_processes_all_fixtures(isolated_cache: Path):
    allocator = IdAllocator(isolated_cache / "next_id.txt")
    manifest = scan(
        [FIXTURES],
        batch_name="pilot",
        manifest_path=isolated_cache / "manifest.jsonl",
        allocator=allocator,
    )
    entries = list(manifest.values())
    assert len(entries) == 5

    writer = CatalogWriter(
        jsonl_path=isolated_cache / "catalog_master.jsonl",
        csv_path=isolated_cache / "catalog_export.csv",
        flush_every=10,
    )
    index = CatalogIndex(jsonl_path=isolated_cache / "catalog_master.jsonl")

    runner = BatchRunner(_make_deps(), writer, index, audit=None)

    progress_events: list[tuple[int, int]] = []
    runner.progressChanged.connect(lambda d, t: progress_events.append((d, t)))

    success, fail = runner.run(entries, batch_name="pilot")
    writer.close()

    assert success == 5
    assert fail == 0
    # Verify catalog_master.jsonl has 5 records, csv too.
    records = list(CatalogIndex(jsonl_path=isolated_cache / "catalog_master.jsonl").all())
    assert len(records) == 5
    # Final progress event reports 5/5
    assert progress_events[-1] == (5, 5)


def test_resume_skips_already_finished(isolated_cache: Path):
    allocator = IdAllocator(isolated_cache / "next_id.txt")
    manifest = scan(
        [FIXTURES],
        batch_name="pilot",
        manifest_path=isolated_cache / "manifest.jsonl",
        allocator=allocator,
    )
    entries = list(manifest.values())

    writer = CatalogWriter(
        jsonl_path=isolated_cache / "catalog_master.jsonl",
        csv_path=isolated_cache / "catalog_export.csv",
        flush_every=10,
    )
    index = CatalogIndex(jsonl_path=isolated_cache / "catalog_master.jsonl")

    # First run: process everything.
    BatchRunner(_make_deps(), writer, index).run(entries)
    writer.close()

    # Second run: filter_eligible should report zero remaining.
    index.reload()
    remaining = filter_eligible(entries, index)
    assert remaining == []
