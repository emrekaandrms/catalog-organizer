from __future__ import annotations

from catalog_organizer.catalog.search import UNKNOWN_BRAND, distinct_brands, search_records
from catalog_organizer.core.config import load_brand_names


def test_search_filters_by_category_and_subcategory(record_factory):
    records = [
        record_factory(file_id="A", main_category="ring", subcategory="solitaire"),
        record_factory(file_id="B", main_category="earring", subcategory="stud"),
    ]
    result = search_records(records, category="ring")
    assert [r.file_id for r in result] == ["A"]

    result = search_records(records, category="ring", subcategory="solitaire")
    assert [r.file_id for r in result] == ["A"]

    result = search_records(records, category="ring", subcategory="halo")
    assert result == []


def test_search_filters_by_stone_status(record_factory):
    """Must filter on stone_summary.status (geometry-verified), NOT the
    top-level polished_status (VLM's raw, un-cross-checked opinion) — a
    real 68-file pilot (2026-07-24) showed the two disagree on nearly
    every file. Records here deliberately set mismatched values for both
    fields to prove the filter reads the right one."""
    from catalog_organizer.core.schemas import StoneSummary
    records = [
        record_factory(
            file_id="A", polished_status="unclear",
            stone_summary=StoneSummary(
                status="stone", center_stone=None, side_stones=[],
                total_estimated_carat=0.0,
            ),
        ),
        record_factory(
            file_id="B", polished_status="stone",
            stone_summary=StoneSummary(
                status="polished", center_stone=None, side_stones=[],
                total_estimated_carat=0.0,
            ),
        ),
    ]
    assert [r.file_id for r in search_records(records, stone_status="stone")] == ["A"]


def test_search_filters_by_brand_and_unknown(record_factory):
    records = [
        record_factory(file_id="A", brand="cartier"),
        record_factory(file_id="B", brand=None),
    ]
    assert [r.file_id for r in search_records(records, brand="cartier")] == ["A"]
    assert [r.file_id for r in search_records(records, brand=UNKNOWN_BRAND)] == ["B"]


def test_search_filters_by_sprue_detected(record_factory):
    from catalog_organizer.core.schemas import SpruInfo
    records = [
        record_factory(file_id="A", sprue=SpruInfo(
            detected=True, source="vlm", confidence=0.9, estimated_volume_mm3=10.0)),
        record_factory(file_id="B", sprue=SpruInfo(
            detected=False, source="none", confidence=0.95, estimated_volume_mm3=None)),
    ]
    assert [r.file_id for r in search_records(records, sprue_detected=True)] == ["A"]
    assert [r.file_id for r in search_records(records, sprue_detected=False)] == ["B"]


def test_search_free_text_matches_tags_and_rich_description(record_factory):
    records = [
        record_factory(
            file_id="A",
            controlled_tags=["ring", "band", "polished", "round_form", "cross_motif"],
            rich_description="taşlı haç yüzük",
        ),
        record_factory(
            file_id="B",
            controlled_tags=["chain", "chain_structure", "link_structure",
                             "closed_form", "single_piece"],
            rich_description="garibaldi zincir ortası seramik taşlı",
        ),
    ]
    # word from controlled_tags dictionary
    assert [r.file_id for r in search_records(records, free_text="cross_motif")] == ["A"]
    # word only present in free-text rich_description
    assert [r.file_id for r in search_records(records, free_text="garibaldi")] == ["B"]
    # multiple words, AND semantics, case-insensitive (mixed-case input,
    # deliberately avoiding Turkish "İ" — Python's locale-independent
    # str.lower() turns it into "i" + combining dot above, not plain "i")
    assert [r.file_id for r in search_records(records, free_text="Taşlı Zincir")] == ["B"]


def test_search_free_text_matches_turkish_dotted_capital_i(record_factory):
    """'İnci' (pearl) starts with Turkish dotted capital İ. A query typed
    lowercase ('inci') must still match — Python's locale-independent
    str.lower() would otherwise turn 'İ' into 'i' + a combining dot above,
    which never equals a plain 'i' and silently breaks the match."""
    records = [
        record_factory(file_id="A", rich_description="İnci kolye"),
        record_factory(file_id="B", rich_description="altın kolye"),
    ]
    assert [r.file_id for r in search_records(records, free_text="inci")] == ["A"]
    # and the reverse direction: dotless I in the query must match content
    # spelled with Turkish ı (e.g. "kolye" typed as "KOLYE" via caps-lock,
    # which on a Turkish layout can produce dotless I only for the letter I)
    assert [r.file_id for r in search_records(records, free_text="KOLYE")] == ["A", "B"]


def test_search_combines_all_filters_with_and(record_factory):
    from catalog_organizer.core.schemas import StoneSummary
    records = [
        record_factory(
            file_id="A", main_category="ring", brand="cartier",
            rich_description="taşlı çiçek yüzük",
            stone_summary=StoneSummary(
                status="stone", center_stone=None, side_stones=[],
                total_estimated_carat=0.0,
            ),
        ),
        record_factory(
            file_id="B", main_category="ring", brand="cartier",
            rich_description="cilalı yüzük",
            stone_summary=StoneSummary(
                status="polished", center_stone=None, side_stones=[],
                total_estimated_carat=0.0,
            ),
        ),
    ]
    result = search_records(
        records, category="ring", brand="cartier",
        stone_status="stone", free_text="çiçek",
    )
    assert [r.file_id for r in result] == ["A"]


def test_distinct_brands_sorted_and_excludes_none(record_factory):
    records = [
        record_factory(file_id="A", brand="van_cleef_arpels"),
        record_factory(file_id="B", brand="cartier"),
        record_factory(file_id="C", brand=None),
    ]
    assert distinct_brands(records) == ["cartier", "van_cleef_arpels"]


def test_brand_names_config_loads_starter_list():
    brands = load_brand_names()
    assert "Cartier" in brands
    assert "Van Cleef & Arpels" in brands
