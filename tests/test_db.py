from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import orjson
import pytest

from catalog_organizer.core.schemas import (
    CatalogRecord, MetalWeights, SpruInfo, StoneSummary,
)
from catalog_organizer.db.connection import Database
from catalog_organizer.db.products import (
    UNKNOWN_BRAND, clear_override, count_products, distinct_brands, get_overrides,
    get_product, get_products, search_product_ids, search_products, set_override,
)
from catalog_organizer.db.schema import SCHEMA_VERSION, apply_schema, current_version
from catalog_organizer.db.selections import (
    DuplicateSelectionName, add_items, create_selection, delete_selection,
    get_selection, list_selections, remove_items, rename_selection,
    selection_file_ids, selections_containing,
)
from catalog_organizer.db.sync import sync_from_jsonl
from catalog_organizer.db.text import to_fts_query, tr_lower

EXPECTED_TABLES = {
    "products", "products_fts", "product_overrides", "selections",
    "selection_items", "listings", "listing_variants", "exports", "schema_meta",
}


def _write_jsonl(path: Path, records) -> None:
    with path.open("wb") as fh:
        for rec in records:
            fh.write(orjson.dumps(rec.model_dump(mode="json")) + b"\n")


@pytest.fixture
def db(tmp_path: Path):
    d = Database(tmp_path / "c.db")
    yield d
    d.close()


@pytest.fixture
def db_with(tmp_path: Path):
    def _make(records):
        jsonl = tmp_path / "m.jsonl"
        _write_jsonl(jsonl, records)
        d = Database(tmp_path / "c.db")
        sync_from_jsonl(d.conn, jsonl)
        return d, jsonl
    return _make


# ---------------------------------------------------------------- şema

def test_schema_creates_every_table(db):
    names = {r[0] for r in db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    assert EXPECTED_TABLES <= names


def test_schema_version_recorded_and_idempotent(db):
    apply_schema(db.conn)
    apply_schema(db.conn)
    assert current_version(db.conn) == SCHEMA_VERSION
    assert db.conn.execute("SELECT COUNT(*) FROM schema_meta").fetchone()[0] == 1


def test_wal_and_foreign_keys_on(db):
    assert db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


# ---------------------------------------------------------------- senkron

def test_sync_inserts(db_with, record_factory):
    d, _ = db_with([record_factory(file_id="A"), record_factory(file_id="B")])
    assert [r.file_id for r in search_products(d.conn)] == ["A", "B"]
    d.close()


def test_sync_is_idempotent(db_with, record_factory):
    d, jsonl = db_with([record_factory(file_id="A")])
    stats = sync_from_jsonl(d.conn, jsonl)
    assert (stats.inserted, stats.updated, stats.unchanged) == (0, 0, 1)
    d.close()


def test_sync_updates_on_new_processed_at(db_with, record_factory):
    d, jsonl = db_with([record_factory(file_id="A", main_category="ring")])
    _write_jsonl(jsonl, [record_factory(
        file_id="A", main_category="pendant",
        processed_at=datetime.now(tz=timezone.utc) + timedelta(hours=1))])
    stats = sync_from_jsonl(d.conn, jsonl)
    assert stats.updated == 1
    assert get_product(d.conn, "A").main_category == "pendant"
    d.close()


def test_later_jsonl_line_wins(db_with, record_factory):
    t0 = datetime.now(tz=timezone.utc)
    d, _ = db_with([
        record_factory(file_id="A", main_category="ring", processed_at=t0),
        record_factory(file_id="A", main_category="bracelet",
                       processed_at=t0 + timedelta(minutes=5)),
    ])
    assert count_products(d.conn) == 1
    assert get_product(d.conn, "A").main_category == "bracelet"
    d.close()


def test_sync_preserves_overrides(db_with, record_factory):
    """Yol 2'nin bel kemiği: senkron elle düzeltmeyi SİLMEZ."""
    d, jsonl = db_with([record_factory(file_id="A", main_category="ring")])
    set_override(d.conn, "A", "main_category", "pendant")
    sync_from_jsonl(d.conn, jsonl)
    assert get_product(d.conn, "A").main_category == "pendant"
    d.close()


def test_record_json_round_trips(db_with, record_factory):
    d, _ = db_with([record_factory(file_id="A")])
    raw = d.conn.execute(
        "SELECT record_json FROM products WHERE file_id='A'").fetchone()[0]
    assert CatalogRecord.model_validate(json.loads(raw)).file_id == "A"
    d.close()


def test_sync_flattens_weights(db_with, record_factory):
    d, _ = db_with([record_factory(
        file_id="A", metal_weights=MetalWeights(silver_925_g=3.5,
                                                gold_14k_yellow_g=4.4))])
    row = d.conn.execute(
        "SELECT silver_925_g, gold_14k_yellow_g FROM products WHERE file_id='A'"
    ).fetchone()
    assert (row[0], row[1]) == (3.5, 4.4)
    d.close()


def test_sync_handles_null_weights(db_with, record_factory):
    d, _ = db_with([record_factory(
        file_id="A", metal_weights=None, weight_source="unavailable",
        weight_confidence="unavailable")])
    assert d.conn.execute(
        "SELECT silver_925_g FROM products WHERE file_id='A'").fetchone()[0] is None
    d.close()


def test_sync_missing_file_is_noop(db):
    assert sync_from_jsonl(db.conn, Path("yok.jsonl")).total == 0


# ---------------------------------------------------------------- metin

def test_tr_lower():
    assert tr_lower("İSTANBUL") == "istanbul"
    assert tr_lower("ISPARTA") == "ısparta"
    assert tr_lower("GÜMÜŞ ÇİÇEK") == "gümüş çiçek"


def test_fts_query_quotes_and_ands():
    assert to_fts_query("kalp kolye") == '"kalp" AND "kolye"'
    assert to_fts_query('taş "özel"') == '"taş" AND "özel"'
    assert to_fts_query("   ") == ""


# ---------------------------------------------------------------- arama

def test_filter_category_and_subcategory(db_with, record_factory):
    d, _ = db_with([
        record_factory(file_id="A", main_category="ring", subcategory="solitaire"),
        record_factory(file_id="B", main_category="earring", subcategory="stud"),
    ])
    assert [r.file_id for r in search_products(d.conn, category="ring")] == ["A"]
    assert search_products(d.conn, category="ring", subcategory="halo") == []
    d.close()


def test_stone_filter_uses_stone_summary_not_polished_status(db_with, record_factory):
    d, _ = db_with([
        record_factory(file_id="A", polished_status="unclear",
                       stone_summary=StoneSummary(status="stone", center_stone=None,
                                                  side_stones=[],
                                                  total_estimated_carat=0.0)),
        record_factory(file_id="B", polished_status="stone",
                       stone_summary=StoneSummary(status="polished", center_stone=None,
                                                  side_stones=[],
                                                  total_estimated_carat=0.0)),
    ])
    assert [r.file_id for r in search_products(d.conn, stone_status="stone")] == ["A"]
    d.close()


def test_brand_and_unknown_brand(db_with, record_factory):
    d, _ = db_with([record_factory(file_id="A", brand="cartier"),
                    record_factory(file_id="B", brand=None)])
    assert [r.file_id for r in search_products(d.conn, brand="cartier")] == ["A"]
    assert [r.file_id for r in search_products(d.conn, brand=UNKNOWN_BRAND)] == ["B"]
    d.close()


def test_sprue_filter(db_with, record_factory):
    d, _ = db_with([
        record_factory(file_id="A", sprue=SpruInfo(
            detected=True, source="geometry", confidence=0.9,
            estimated_volume_mm3=20.0)),
        record_factory(file_id="B", sprue=SpruInfo(
            detected=False, source="none", confidence=0.0,
            estimated_volume_mm3=None)),
    ])
    assert [r.file_id for r in search_products(d.conn, sprue_detected=True)] == ["A"]
    assert [r.file_id for r in search_products(d.conn, sprue_detected=False)] == ["B"]
    d.close()


def test_free_text_across_tags_and_description(db_with, record_factory):
    d, _ = db_with([
        record_factory(file_id="A", main_category="pendant",
                       rich_description="taşlı haç kolye ucu",
                       tags=["cross", "stone_set", "pendant", "religious", "silver"]),
        record_factory(file_id="B", main_category="ring",
                       rich_description="sade alyans"),
    ])
    assert [r.file_id for r in search_products(d.conn, free_text="haç")] == ["A"]
    assert [r.file_id for r in search_products(d.conn, free_text="taşlı haç")] == ["A"]
    assert search_products(d.conn, free_text="haç alyans") == []
    d.close()


def test_free_text_turkish_case_insensitive(db_with, record_factory):
    d, _ = db_with([record_factory(file_id="A", rich_description="İnce zincir")])
    for typed in ("İNCE", "ince", "İnce"):
        assert [r.file_id for r in search_products(d.conn, free_text=typed)] == ["A"]
    d.close()


def test_fts_drops_stale_text_on_resync(db_with, record_factory):
    d, jsonl = db_with([record_factory(file_id="A", rich_description="eski metin")])
    _write_jsonl(jsonl, [record_factory(
        file_id="A", rich_description="yeni metin",
        processed_at=datetime.now(tz=timezone.utc) + timedelta(hours=1))])
    sync_from_jsonl(d.conn, jsonl)
    assert d.conn.execute("SELECT COUNT(*) FROM products_fts").fetchone()[0] == 1
    assert search_products(d.conn, free_text="eski") == []
    assert [r.file_id for r in search_products(d.conn, free_text="yeni")] == ["A"]
    d.close()


def test_filters_combine_with_and(db_with, record_factory):
    d, _ = db_with([record_factory(file_id="A", main_category="ring", brand="atasay"),
                    record_factory(file_id="B", main_category="ring", brand="sozer")])
    assert [r.file_id for r in
            search_products(d.conn, category="ring", brand="atasay")] == ["A"]
    d.close()


def test_limit_offset(db_with, record_factory):
    d, _ = db_with([record_factory(file_id=f"{i:03d}") for i in range(10)])
    page = search_products(d.conn, limit=3, offset=3)
    assert [r.file_id for r in page] == ["003", "004", "005"]
    d.close()


def test_get_products_preserves_order(db_with, record_factory):
    d, _ = db_with([record_factory(file_id=x) for x in ("A", "B", "C")])
    assert [r.file_id for r in get_products(d.conn, ["C", "A"])] == ["C", "A"]
    d.close()


# ---------------------------------------------------------------- override

def test_override_applies_and_is_searchable(db_with, record_factory):
    d, _ = db_with([record_factory(file_id="A", main_category="ring")])
    set_override(d.conn, "A", "main_category", "pendant")
    assert get_product(d.conn, "A").main_category == "pendant"
    assert [r.file_id for r in search_products(d.conn, category="pendant")] == ["A"]
    assert search_products(d.conn, category="ring") == []
    d.close()


def test_clear_override_restores(db_with, record_factory):
    d, _ = db_with([record_factory(file_id="A", main_category="ring")])
    set_override(d.conn, "A", "main_category", "pendant")
    clear_override(d.conn, "A", "main_category")
    assert get_product(d.conn, "A").main_category == "ring"
    assert get_overrides(d.conn, "A") == {}
    d.close()


def test_override_rejects_unknown_field(db_with, record_factory):
    d, _ = db_with([record_factory(file_id="A")])
    with pytest.raises(ValueError, match="main_category"):
        set_override(d.conn, "A", "sellability", "sellable")
    d.close()


def test_distinct_brands_includes_overrides(db_with, record_factory):
    d, _ = db_with([record_factory(file_id="A", brand=None)])
    set_override(d.conn, "A", "brand", "cartier")
    assert distinct_brands(d.conn) == ["cartier"]
    d.close()


def test_search_product_ids(db_with, record_factory):
    d, _ = db_with([record_factory(file_id="A", main_category="ring"),
                    record_factory(file_id="B", main_category="ring"),
                    record_factory(file_id="C", main_category="earring")])
    assert search_product_ids(d.conn, category="ring") == ["A", "B"]
    d.close()


# ---------------------------------------------------------------- seçimler

def test_create_and_list_selections(db):
    sid = create_selection(db.conn, "Kış 2026", note="ilk parti")
    rows = list_selections(db.conn)
    assert (rows[0].selection_id, rows[0].name, rows[0].item_count) == (sid, "Kış 2026", 0)


def test_duplicate_selection_name_rejected(db):
    create_selection(db.conn, "S")
    with pytest.raises(DuplicateSelectionName):
        create_selection(db.conn, "S")


def test_add_items_counts_only_new_and_is_idempotent(db):
    sid = create_selection(db.conn, "S")
    assert add_items(db.conn, sid, ["A", "B"]) == 2
    assert add_items(db.conn, sid, ["B", "C"]) == 1
    assert add_items(db.conn, sid, ["A", "B"]) == 0
    assert selection_file_ids(db.conn, sid) == ["A", "B", "C"]


def test_add_empty_is_noop(db):
    sid = create_selection(db.conn, "S")
    assert add_items(db.conn, sid, []) == 0


def test_remove_items(db):
    sid = create_selection(db.conn, "S")
    add_items(db.conn, sid, ["A", "B", "C"])
    assert remove_items(db.conn, sid, ["B"]) == 1
    assert selection_file_ids(db.conn, sid) == ["A", "C"]


def test_item_count(db):
    sid = create_selection(db.conn, "S")
    add_items(db.conn, sid, ["A", "B", "C"])
    assert get_selection(db.conn, sid).item_count == 3


def test_delete_cascades(db):
    sid = create_selection(db.conn, "S")
    add_items(db.conn, sid, ["A"])
    delete_selection(db.conn, sid)
    assert list_selections(db.conn) == []
    assert db.conn.execute("SELECT COUNT(*) FROM selection_items").fetchone()[0] == 0


def test_rename_and_duplicate_rename(db):
    create_selection(db.conn, "A")
    sid = create_selection(db.conn, "B")
    rename_selection(db.conn, sid, "C")
    assert get_selection(db.conn, sid).name == "C"
    with pytest.raises(DuplicateSelectionName):
        rename_selection(db.conn, sid, "A")


def test_product_in_multiple_selections(db):
    s1 = create_selection(db.conn, "Kış 2026")
    s2 = create_selection(db.conn, "Etsy ilk parti")
    add_items(db.conn, s1, ["A"])
    add_items(db.conn, s2, ["A"])
    assert sorted(selections_containing(db.conn, "A")) == sorted([s1, s2])


def test_get_missing_selection(db):
    assert get_selection(db.conn, 999) is None
