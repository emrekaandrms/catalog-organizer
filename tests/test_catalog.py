from __future__ import annotations

import csv
from pathlib import Path

from catalog_organizer.catalog.index import CatalogIndex
from catalog_organizer.catalog.writer import CatalogWriter


def test_writer_appends_jsonl_and_writes_csv(tmp_path: Path, record_factory):
    jsonl = tmp_path / "catalog_master.jsonl"
    csv_path = tmp_path / "catalog_export.csv"
    writer = CatalogWriter(jsonl_path=jsonl, csv_path=csv_path, flush_every=2)
    writer.append(record_factory(file_id="JCAD-000000001"))
    writer.append(record_factory(file_id="JCAD-000000002"))
    writer.close()

    assert jsonl.exists()
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["file_id"] == "JCAD-000000001"
    assert rows[1]["file_id"] == "JCAD-000000002"


def test_csv_export_flattens_every_field_group(tmp_path: Path, record_factory):
    """The CSV is the user-facing catalog export. Every field group from
    CatalogRecord must surface as columns: identity, classification,
    measurements (core + category-specific), per-metal weights, stones,
    sprue, and review state. This test pins the header so a future refactor
    that drops a group is caught immediately.
    """
    from catalog_organizer.core.schemas import (
        Measurements, StoneEntry, StoneSummary, SpruInfo,
    )
    jsonl = tmp_path / "catalog_master.jsonl"
    csv_path = tmp_path / "catalog_export.csv"
    writer = CatalogWriter(jsonl_path=jsonl, csv_path=csv_path, flush_every=1)

    # A richly-populated ring record — covers ring measurements, metal
    # weights, a center stone + side stones, and a detected sprue with
    # weight-per-metal estimates.
    writer.append(record_factory(
        file_id="JCAD-RING-01",
        main_category="ring",
        measurements=Measurements(
            bbox_width_mm=20.0, bbox_height_mm=22.0, bbox_depth_mm=8.0,
            volume_mm3=3500.0, geometry_source="rhino",
            ring_inner_diameter_mm=17.2, ring_inner_circumference_mm=54.0,
            ring_size_eu=54.0, ring_band_width_mm=2.5,
            ring_top_width_mm=8.0, ring_top_height_mm=6.0,
        ),
        stone_summary=StoneSummary(
            status="stone",
            center_stone=StoneEntry(
                shape="round", size_mm="6.5", quantity=1,
                estimated_carat_each=1.0, estimated_total_carat=1.0,
                source="vlm", confidence=0.85,
            ),
            side_stones=[
                StoneEntry(shape="round", size_mm="1.5", quantity=12,
                           estimated_carat_each=0.015, estimated_total_carat=0.18,
                           source="geometry", confidence=0.7),
            ],
            total_estimated_carat=1.18,
        ),
        sprue=SpruInfo(
            detected=True, source="vlm", confidence=0.9,
            estimated_volume_mm3=150.0,
        ),
    ))
    writer.close()

    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        headers = reader.fieldnames or []

    # Every field group must be represented.
    expected_must_have = {
        # identity
        "file_id", "source_path", "file_extension", "sha256",
        # classification
        "main_category", "subcategory", "controlled_tags", "tag_confidence",
        # core measurements
        "bbox_width_mm", "bbox_height_mm", "bbox_depth_mm", "volume_mm3",
        # category-specific
        "ring_inner_diameter_mm", "ring_size_eu",
        "earring_total_height_mm", "bracelet_inner_diameter_x_mm",
        "pendant_height_mm",
        # per-metal weights (yellow/white split for gold)
        "silver_925_g",
        "gold_10k_yellow_g", "gold_10k_white_g",
        "gold_14k_yellow_g", "gold_14k_white_g",
        "gold_18k_yellow_g", "gold_18k_white_g",
        "platinum_g",
        # stones
        "stone_status", "center_stone_shape", "center_stone_estimated_carat",
        "side_stones_count", "side_stones_detail", "total_estimated_carat",
        # final product (metal + stones)
        "total_stone_weight_g",
        "product_silver_925_g",
        "product_gold_18k_white_g",
        "product_platinum_g",
        # sprue: volume AND per-metal mass. The metal_* columns are already
        # net of the sprue, so these give the other half a workshop needs.
        "sprue_detected", "sprue_estimated_volume_mm3",
        "sprue_silver_925_g", "sprue_gold_14k_yellow_g", "sprue_platinum_g",
        # review state
        "state", "needs_manual_review", "processed_at",
    }
    missing = expected_must_have - set(headers)
    assert not missing, f"CSV is missing required columns: {sorted(missing)}"

    row = rows[0]
    # Ring-specific cell populated
    assert row["ring_inner_diameter_mm"] == "17.200"
    assert row["ring_size_eu"] == "54.00"
    # Earring/bracelet/pendant columns for a ring → must be empty, not "0.000"
    assert row["earring_total_height_mm"] == ""
    assert row["bracelet_inner_diameter_x_mm"] == ""
    assert row["pendant_height_mm"] == ""
    # Center stone + side stones (side_stones_count is the SUM of per-group
    # quantities, not the number of distinct groups — a pavé of 12 stones
    # in one group must show as 12, not 1)
    assert row["center_stone_shape"] == "round"
    assert row["center_stone_estimated_carat"] == "1.000"
    assert row["side_stones_count"] == "12"
    assert row["side_stones_total_carat"] == "0.180"
    assert row["total_estimated_carat"] == "1.180"
    # Side-stones detail is a compact, pipe-separated breakdown per group
    assert "round:1.5:qty=12:0.180ct" == row["side_stones_detail"]
    # Stone weight in grams = total_carat × 0.2 (1 ct = 0.2 g by definition)
    # 1.18 ct × 0.2 = 0.236 g
    assert row["total_stone_weight_g"] == "0.236"
    # Product weight = metal + stones, per metal. silver = 36.3 + 0.236 = 36.536
    assert row["product_silver_925_g"] == "36.536"
    # Sprue volume AND its mass per metal. A wedding-band support bar is
    # 4-14 % of the piece, so the workshop needs the runner's own weight
    # next to the (already net) piece weight, not just a volume to convert
    # by hand — 2026-07-30.
    assert row["sprue_detected"] == "1"
    assert row["sprue_estimated_volume_mm3"] == "150.0"
    # 150 mm3 × 20.13 g/cm3 (platinum) = 3.0195 g -> 3.019 at 3 dp
    assert row["sprue_platinum_g"] == "3.019"
    # 150 mm3 × 10.36 g/cm3 (925 silver) = 1.554 g
    assert row["sprue_silver_925_g"] == "1.554"
    # Controlled tags joined with pipes
    assert "|" in row["controlled_tags"]


def test_csv_blanks_when_metal_weights_unavailable(tmp_path: Path, record_factory):
    """Records with weight_source='unavailable' have metal_weights=None.
    CSV must render those cells as blank, not '0.000' (which would look like
    'this item weighs zero')."""
    jsonl = tmp_path / "catalog_master.jsonl"
    csv_path = tmp_path / "catalog_export.csv"
    writer = CatalogWriter(jsonl_path=jsonl, csv_path=csv_path, flush_every=1)
    writer.append(record_factory(
        file_id="JCAD-NOWEIGHT-01",
        metal_weights=None,
        weight_source="unavailable",
        weight_confidence="unavailable",
    ))
    writer.close()

    with csv_path.open(encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    assert row["silver_925_g"] == ""
    assert row["gold_18k_yellow_g"] == ""
    assert row["platinum_g"] == ""
    assert row["weight_source"] == "unavailable"


def test_index_reloads_from_jsonl(tmp_path: Path, record_factory):
    jsonl = tmp_path / "catalog_master.jsonl"
    csv_path = tmp_path / "catalog_export.csv"
    writer = CatalogWriter(jsonl_path=jsonl, csv_path=csv_path, flush_every=1)
    writer.append(record_factory(file_id="JCAD-000000001", main_category="ring"))
    writer.append(record_factory(file_id="JCAD-000000002", main_category="earring"))
    writer.close()

    index = CatalogIndex(jsonl_path=jsonl)
    assert len(index) == 2
    assert index.get("JCAD-000000001").main_category == "ring"
    assert index.get("JCAD-000000002").main_category == "earring"


