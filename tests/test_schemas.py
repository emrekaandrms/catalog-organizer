from __future__ import annotations

from datetime import datetime, timezone

import orjson
import pytest
from pydantic import ValidationError

from catalog_organizer.core.schemas import (
    AuditEvent,
    CatalogRecord,
    ManifestEntry,
    Measurements,
    MetalWeights,
    SpruInfo,
    StoneSummary,
)

_NOW = datetime.now(tz=timezone.utc)


def _manifest_entry(**overrides) -> dict:
    base = {
        "file_id": "JCAD-000000001",
        "source_path": "D:\\archive\\ring.stl",
        "file_extension": ".stl",
        "file_size_bytes": 102400,
        "created_at": _NOW,
        "modified_at": _NOW,
        "sha256": "a" * 64,
        "duplicate_of": None,
        "state": "new",
        "processing_batch": "pilot_300",
        "scan_timestamp": _NOW,
    }
    base.update(overrides)
    return base


def _measurements() -> Measurements:
    return Measurements(
        bbox_width_mm=20.0,
        bbox_height_mm=22.0,
        bbox_depth_mm=8.0,
        volume_mm3=3500.0,
        geometry_source="trimesh",
    )


def _metal_weights() -> MetalWeights:
    return MetalWeights(
        silver_925_g=36.3,
        gold_10k_g=40.5,
        gold_14k_g=45.7,
        gold_18k_g=54.5,
        platinum_g=70.5,
    )


def _stone_summary() -> StoneSummary:
    return StoneSummary(status="polished", center_stone=None, side_stones=[], total_estimated_carat=0.0)


def _spru_info() -> SpruInfo:
    return SpruInfo(detected=False, source="none", confidence=0.95, estimated_volume_mm3=None, estimated_weight_by_metal=None)


def test_manifest_entry_valid():
    entry = ManifestEntry(**_manifest_entry())
    assert entry.file_id == "JCAD-000000001"
    assert entry.duplicate_of is None
    assert entry.state == "new"


def test_manifest_entry_duplicate_nullable():
    entry = ManifestEntry(**_manifest_entry(duplicate_of=None))
    assert entry.duplicate_of is None

    entry2 = ManifestEntry(**_manifest_entry(duplicate_of="JCAD-000000002"))
    assert entry2.duplicate_of == "JCAD-000000002"


def test_catalog_record_requires_5_tags():
    base = dict(
        file_id="JCAD-000000001",
        source_path="D:\\archive\\ring.stl",
        file_extension=".stl",
        sha256="a" * 64,
        main_category="ring",
        main_category_confidence=0.95,
        subcategory="solitaire",
        subcategory_confidence=0.90,
        controlled_tags=["ring", "band", "polished"],   # only 3 — should fail
        tag_confidence=0.85,
        polished_status="polished",
        measurements=_measurements(),
        metal_weights=_metal_weights(),
        weight_source="stl_volume_density",
        weight_confidence="medium",
        stone_summary=_stone_summary(),
        sprue=_spru_info(),
        scale_warning=False,
        scale_warning_reason=None,
        snapshot_paths=["cache/snapshots/JCAD-000000001/front.png"],
        state="final",
        needs_manual_review=False,
        review_reason=None,
        manual_correction_applied=False,
        processing_log_ref="evt-001",
        processed_at=_NOW,
    )
    with pytest.raises(ValidationError):
        CatalogRecord(**base)


def test_audit_event_round_trip():
    event = AuditEvent(
        event_id="evt-001",
        timestamp=_NOW,
        file_id="JCAD-000000001",
        phase="snapshot",
        level="info",
        code="snapshot.render_done",
        message="Snapshot complete",
        extra={"duration_ms": 320},
    )
    raw = orjson.dumps(event.model_dump(mode="json"))
    restored = AuditEvent.model_validate(orjson.loads(raw))
    assert restored.event_id == event.event_id
    assert restored.extra["duration_ms"] == 320


def test_catalog_record_brand_and_rich_description_default_for_legacy_rows():
    """Pre-existing JSONL rows (written before the catalogue-search design notes,
    2026-07-24) have no brand/brand_confidence/rich_description keys at
    all. They must still load — with brand unset and description empty —
    rather than raising, since re-processing the whole catalog to backfill
    them is explicitly a separate, opt-in step (plan §2.4)."""
    base = dict(
        file_id="JCAD-000000001",
        source_path="D:\\archive\\ring.stl",
        file_extension=".stl",
        sha256="a" * 64,
        main_category="ring",
        main_category_confidence=0.95,
        subcategory="solitaire",
        subcategory_confidence=0.90,
        controlled_tags=["ring", "band", "polished", "round_form", "single_piece"],
        tag_confidence=0.85,
        polished_status="polished",
        measurements=_measurements(),
        metal_weights=_metal_weights(),
        weight_source="stl_volume_density",
        weight_confidence="medium",
        stone_summary=_stone_summary(),
        sprue=_spru_info(),
        scale_warning=False,
        scale_warning_reason=None,
        snapshot_paths=["cache/snapshots/JCAD-000000001/front.png"],
        state="final",
        needs_manual_review=False,
        review_reason=None,
        manual_correction_applied=False,
        processing_log_ref="evt-001",
        processed_at=_NOW,
    )
    record = CatalogRecord(**base)
    assert record.brand is None
    assert record.brand_confidence == 0.0
    assert record.rich_description == ""


def test_vlm_result_defaults_brand_unknown_and_empty_description():
    from catalog_organizer.core.schemas import VLMResult
    result = VLMResult(
        main_category="ring",
        controlled_tags=["ring", "band", "polished", "round_form", "single_piece"],
    )
    assert result.brand_guess == "unknown"
    assert result.brand_confidence == 0.0
    assert result.rich_description == ""


def test_vlm_result_normalizes_polished_model_alias_in_polished_status():
    """Regression (2026-07-24, real 65-file PilotBatch pilot): the VLM
    sometimes emits "polished_model" (a controlled_tags production tag) in
    the polished_status field instead of "polished". Before this fix that
    raised vlm.schema_violation and discarded the ENTIRE response — 6 of 65
    real files lost their category/tags/brand/description over one field.
    """
    from catalog_organizer.core.schemas import VLMResult
    result = VLMResult(
        main_category="ring",
        controlled_tags=["ring", "band", "polished", "round_form", "single_piece"],
        polished_status="polished_model",
    )
    assert result.polished_status == "polished"


def test_vlm_result_design_complete_confidence_zero_flags_review():
    """design_complete_confidence at its 0.0 default must trigger
    needs_manual_review, same as the other core confidence fields
    (the scale/sales design notes §1/§5, 2026-07-24)."""
    from catalog_organizer.core.schemas import VLMResult
    result = VLMResult(
        main_category="ring",
        main_category_confidence=0.9,
        subcategory_confidence=0.9,
        tag_confidence=0.9,
        controlled_tags=["ring", "band", "polished", "round_form", "single_piece"],
    )
    assert result.design_complete_confidence == 0.0
    assert result.needs_manual_review is True


def test_catalog_record_sellability_defaults_needs_review_for_legacy_rows():
    """Pre-existing JSONL rows written before this field existed have no
    sellability at all. They must load with the conservative default
    (needs_review), not silently claim sellable."""
    base = dict(
        file_id="JCAD-000000001",
        source_path="D:\\archive\\ring.stl",
        file_extension=".stl",
        sha256="a" * 64,
        main_category="ring",
        main_category_confidence=0.95,
        subcategory="solitaire",
        subcategory_confidence=0.90,
        controlled_tags=["ring", "band", "polished", "round_form", "single_piece"],
        tag_confidence=0.85,
        polished_status="polished",
        measurements=_measurements(),
        metal_weights=_metal_weights(),
        weight_source="stl_volume_density",
        weight_confidence="medium",
        stone_summary=_stone_summary(),
        sprue=_spru_info(),
        scale_warning=False,
        scale_warning_reason=None,
        snapshot_paths=["cache/snapshots/JCAD-000000001/front.png"],
        state="final",
        needs_manual_review=False,
        review_reason=None,
        manual_correction_applied=False,
        processing_log_ref="evt-001",
        processed_at=_NOW,
    )
    record = CatalogRecord(**base)
    assert record.design_complete is True
    assert record.sellability == "needs_review"
    assert record.sales_channel == "undecided"


def test_measurements_ring_fields_nullable():
    m = Measurements(
        bbox_width_mm=20.0,
        bbox_height_mm=22.0,
        bbox_depth_mm=8.0,
        volume_mm3=3500.0,
        geometry_source="trimesh",
    )
    assert m.ring_inner_diameter_mm is None
    assert m.earring_total_height_mm is None
    assert m.bracelet_width_mm is None
    assert m.pendant_height_mm is None
