"""D.13 — Validation report tests."""
from __future__ import annotations

from pathlib import Path

from catalog_organizer.validation.report import (
    TruthRow,
    compare,
    load_truth_csv,
    render_markdown,
)


def test_load_truth_csv_round_trip(tmp_path):
    csv_path = tmp_path / "truth.csv"
    csv_path.write_text(
        "file_id,main_category,subcategory,polished_status,tags\n"
        "JCAD-000000001,ring,solitaire,polished,\"ring;band;polished\"\n"
        "JCAD-000000002,earring,hoop,stone,\"earring;pair\"\n"
        "JCAD-000000003,ring,,,\n",
        encoding="utf-8",
    )
    truth = load_truth_csv(csv_path)
    assert set(truth.keys()) == {"JCAD-000000001", "JCAD-000000002", "JCAD-000000003"}
    assert truth["JCAD-000000001"].tags == ["ring", "band", "polished"]
    # Empty cells become None
    assert truth["JCAD-000000003"].subcategory is None
    assert truth["JCAD-000000003"].polished_status is None
    assert truth["JCAD-000000003"].tags == []


def test_compare_computes_accuracy(record_factory):
    truth = {
        "JCAD-000000001": TruthRow("JCAD-000000001", "ring",    "solitaire", "polished", ["ring", "band"]),
        "JCAD-000000002": TruthRow("JCAD-000000002", "earring", "hoop",      "stone",    []),
        "JCAD-000000003": TruthRow("JCAD-000000003", "pendant", "single",    "polished", ["pendant"]),
    }
    records = [
        record_factory(file_id="JCAD-000000001", main_category="ring", tags=["ring", "band", "polished", "round_form", "single_piece"]),
        # Subcategory differs but main matches
        record_factory(file_id="JCAD-000000002", main_category="earring", tags=["earring", "single", "polished", "round_form", "single_piece"]),
        # Main category differs (predicted ring, truth pendant)
        record_factory(file_id="JCAD-000000003", main_category="ring", tags=["ring", "band", "polished", "round_form", "single_piece"]),
    ]
    # record_factory's default subcategory is "solitaire"
    report = compare(records, truth)

    assert report.total_compared == 3
    assert report.main_correct == 2          # records 1 and 2
    assert abs(report.main_accuracy - 2/3) < 1e-6

    # Subcategory: only record 1 matches (truth=solitaire, pred=solitaire)
    # Record 2: main matches but sub differs (truth=hoop, pred=solitaire)
    assert report.sub_eligible == 2
    assert report.sub_correct == 1

    # Polished: defaults are "polished" on all records.
    # truth: r1=polished (match), r2=stone (no match), r3=polished (match) → 2/3
    assert report.polished_eligible == 3
    assert report.polished_correct == 2

    # Mispredictions: only record 3 has a wrong main_category
    assert len(report.mispredictions) == 1
    fid, pred, exp = report.mispredictions[0]
    assert fid == "JCAD-000000003"
    assert pred.startswith("ring/")
    assert exp.startswith("pendant/")


def test_render_markdown_contains_metrics(record_factory):
    truth = {"JCAD-000000001": TruthRow("JCAD-000000001", "ring", "solitaire", "polished", ["ring"])}
    records = [record_factory(file_id="JCAD-000000001")]
    report = compare(records, truth)
    md = render_markdown(report)
    assert "# Pilot Validation Report" in md
    assert "Main-category accuracy" in md
    assert "100" in md  # 100% on a single matching record
