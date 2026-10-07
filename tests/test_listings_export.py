from __future__ import annotations

import csv
from pathlib import Path

import orjson
import pytest

from catalog_organizer.db.connection import Database
from catalog_organizer.db.listings import (
    get_listing, get_variants, list_exports, list_listings, mark_exported,
    record_export, replace_variants, set_status, stale_priced_listings,
    update_text, upsert_listing,
)
from catalog_organizer.export import etsy as etsy_export
from catalog_organizer.export import woocommerce as woo_export
from catalog_organizer.listing.pricing import VariantPrice, load_pricing_config
from catalog_organizer.listing.schemas import (
    EtsyListingDraft, WooListingDraft, slugify,
)
from catalog_organizer.listing.validate import (
    check_unique_slugs, partition, validate_listing,
)


@pytest.fixture
def db(tmp_path: Path):
    d = Database(tmp_path / "c.db")
    yield d
    d.close()


def _etsy_fields(**over):
    base = dict(
        title="Sterling Silver Heart Pendant",
        description="Handmade heart pendant.",
        tags=["heart pendant", "silver"],
        materials=["sterling silver"],
        who_made="i_did",
        when_made="made_to_order",
        is_supply=0,
        etsy_taxonomy_id=1265,
        category_path="Jewelry > Necklaces",
    )
    base.update(over)
    return base


def _woo_fields(**over):
    base = dict(
        title="Kalp Kolye Ucu",
        short_description="Zarif kalp kolye ucu.",
        description="<p>Zarif kalp kolye ucu.</p>",
        tags=["kalp", "kolye ucu"],
        category_path="Kolye > Kolye Ucu",
        slug="kalp-kolye-ucu",
        meta_description="Zarif kalp kolye ucu, gümüş ve altın seçenekleriyle.",
        sustainability_note="Geri dönüştürülmüş gümüş kullanılır.",
    )
    base.update(over)
    return base


def _silver(price=1485.0, sku="A-AG", currency="TRY"):
    return VariantPrice(variant_key="silver_925", sku=sku, price=price,
                        currency=currency, weight_g=5.0, is_digital=False)


def _woo_six(file_id="A"):
    return [
        VariantPrice("silver_925", f"{file_id}-AG", 1485.0, "TRY", 5.0, False),
        VariantPrice("gold_8k", f"{file_id}-8K", 16317.0, "TRY", 5.0, False),
        VariantPrice("gold_10k", f"{file_id}-10K", 20433.0, "TRY", 5.0, False),
        VariantPrice("gold_14k", f"{file_id}-14K", 28665.0, "TRY", 5.0, False),
        VariantPrice("gold_18k", f"{file_id}-18K", 36750.0, "TRY", 5.0, False),
        VariantPrice("stl_digital", f"{file_id}-STL", 50.0, "TRY", None, True),
    ]


# ---------------------------------------------------------------- şemalar

def test_etsy_title_over_140_rejected():
    with pytest.raises(ValueError, match="140"):
        EtsyListingDraft(title="x" * 141, description="d")


def test_etsy_tags_trimmed_to_13():
    draft = EtsyListingDraft(
        title="t", description="d", tags=[f"tag{i}" for i in range(20)])
    assert len(draft.tags) == 13


def test_etsy_overlong_tag_dropped():
    draft = EtsyListingDraft(title="t", description="d",
                             tags=["ok tag", "x" * 21])
    assert draft.tags == ["ok tag"]


def test_etsy_duplicate_tags_removed_case_insensitively():
    draft = EtsyListingDraft(title="t", description="d",
                             tags=["Silver", "silver", "gold"])
    assert draft.tags == ["Silver", "gold"]


def test_etsy_materials_limited():
    draft = EtsyListingDraft(title="t", description="d",
                             materials=[f"m{i}" for i in range(20)] + ["y" * 46])
    assert len(draft.materials) == 13
    assert "y" * 46 not in draft.materials


def test_empty_title_rejected():
    with pytest.raises(ValueError):
        EtsyListingDraft(title="   ", description="d")
    with pytest.raises(ValueError):
        WooListingDraft(title="", description="d")


def test_woo_meta_description_truncated():
    draft = WooListingDraft(title="t", description="d", meta_description="x" * 200)
    assert len(draft.meta_description) == 160


def test_slugify_transliterates_turkish():
    assert slugify("Kalp Kolye Ucu") == "kalp-kolye-ucu"
    assert slugify("Işıl Gümüş Çiçek") == "isil-gumus-cicek"
    assert slugify("İnce  Zincir!") == "ince-zincir"


# ---------------------------------------------------------------- depo

def test_upsert_and_get_listing(db):
    lid = upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields())
    listing = get_listing(db.conn, "A", "etsy")
    assert listing.listing_id == lid
    assert listing.title == "Sterling Silver Heart Pendant"
    assert listing.tags == ["heart pendant", "silver"]
    assert listing.status == "draft"


def test_upsert_twice_updates_not_duplicates(db):
    upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields())
    upsert_listing(db.conn, "A", "etsy", language="en",
                   fields=_etsy_fields(title="New Title"))
    assert db.conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0] == 1
    assert get_listing(db.conn, "A", "etsy").title == "New Title"


def test_regeneration_resets_status_to_draft(db):
    """Yeni metin onaylanmış sayılmaz."""
    lid = upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields())
    set_status(db.conn, lid, "approved")
    upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields())
    assert get_listing(db.conn, "A", "etsy").status == "draft"


def test_update_text_marks_human_edited(db):
    lid = upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields())
    update_text(db.conn, lid, title="Elle Yazılmış")
    listing = get_listing(db.conn, "A", "etsy")
    assert listing.title == "Elle Yazılmış"
    assert listing.human_edited is True
    assert listing.edited_at is not None


def test_update_text_rejects_unknown_field(db):
    lid = upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields())
    with pytest.raises(ValueError, match="düzenlenemez"):
        update_text(db.conn, lid, status="approved")


def test_set_status_rejects_bad_value(db):
    lid = upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields())
    with pytest.raises(ValueError):
        set_status(db.conn, lid, "yayında")


def test_replace_variants_and_read_back(db):
    lid = upsert_listing(db.conn, "A", "woocommerce", language="tr",
                         fields=_woo_fields())
    cfg = load_pricing_config()
    assert replace_variants(db.conn, lid, _woo_six(), cfg.snapshot()) == 6
    variants = get_variants(db.conn, lid)
    assert [v.variant_key for v in variants] == [
        "silver_925", "gold_8k", "gold_10k", "gold_14k", "gold_18k", "stl_digital"]
    assert variants[-1].is_digital is True


def test_replace_variants_is_not_additive(db):
    lid = upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields())
    snap = load_pricing_config().snapshot()
    replace_variants(db.conn, lid, [_silver()], snap)
    replace_variants(db.conn, lid, [_silver(price=200.0)], snap)
    variants = get_variants(db.conn, lid)
    assert len(variants) == 1 and variants[0].price == 200.0


def test_variants_deleted_with_listing(db):
    from catalog_organizer.db.listings import delete_listing
    lid = upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields())
    replace_variants(db.conn, lid, [_silver()], {})
    delete_listing(db.conn, lid)
    assert db.conn.execute("SELECT COUNT(*) FROM listing_variants").fetchone()[0] == 0


def test_stale_priced_listings_detects_rate_change(db):
    lid = upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields())
    replace_variants(db.conn, lid, [_silver()], {"gold": 7000})
    assert stale_priced_listings(db.conn, {"gold": 7000}) == []
    assert stale_priced_listings(db.conn, {"gold": 9000}) == [lid]


def test_list_listings_filters(db):
    upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields(),
                   selection_id=None)
    lid = upsert_listing(db.conn, "B", "woocommerce", language="tr",
                         fields=_woo_fields())
    set_status(db.conn, lid, "approved")
    assert [x.file_id for x in list_listings(db.conn, channel="etsy")] == ["A"]
    assert [x.file_id for x in
            list_listings(db.conn, statuses=("approved",))] == ["B"]
    assert list_listings(db.conn, file_ids=[]) == []


def test_export_record_and_mark(db):
    lid = upsert_listing(db.conn, "A", "etsy", language="en", fields=_etsy_fields())
    record_export(db.conn, channel="etsy", selection_id=None,
                  file_path="x.csv", row_count=1)
    assert list_exports(db.conn)[0]["row_count"] == 1
    assert mark_exported(db.conn, [lid]) == 1
    assert get_listing(db.conn, "A", "etsy").status == "exported"


# ---------------------------------------------------------------- doğrulama

def _row(db, file_id, channel, fields, variants):
    lid = upsert_listing(db.conn, file_id, channel,
                         language="en" if channel == "etsy" else "tr", fields=fields)
    replace_variants(db.conn, lid, variants, {})
    return get_listing(db.conn, file_id, channel)


def test_valid_listing_has_no_issues(db):
    listing = _row(db, "A", "etsy", _etsy_fields(), [_silver(105.24, "A-AG", "USD")])
    assert validate_listing(listing) == []


def test_missing_variants_flagged(db):
    listing = _row(db, "A", "etsy", _etsy_fields(), [])
    assert any(i.field == "variants" for i in validate_listing(listing))


def test_zero_price_flagged(db):
    listing = _row(db, "A", "etsy", _etsy_fields(),
                   [_silver(price=0.0, currency="USD")])
    assert any("sıfır" in i.message for i in validate_listing(listing))


def test_missing_taxonomy_flagged(db):
    listing = _row(db, "A", "etsy", _etsy_fields(etsy_taxonomy_id=None),
                   [_silver(105.24, "A-AG", "USD")])
    assert any(i.field == "etsy_taxonomy_id" for i in validate_listing(listing))


def test_empty_slug_flagged(db):
    listing = _row(db, "A", "woocommerce", _woo_fields(slug=""), _woo_six())
    assert any(i.field == "slug" for i in validate_listing(listing))


def test_duplicate_slug_detected(db):
    a = _row(db, "A", "woocommerce", _woo_fields(), _woo_six("A"))
    b = _row(db, "B", "woocommerce", _woo_fields(), _woo_six("B"))
    issues = check_unique_slugs([a, b])
    assert len(issues) == 1 and issues[0].file_id == "B"


def test_partition_splits_good_from_bad(db):
    good = _row(db, "A", "etsy", _etsy_fields(), [_silver(105.24, "A-AG", "USD")])
    bad = _row(db, "B", "etsy", _etsy_fields(title=""), [])
    ok, problems = partition([good, bad])
    assert [x.file_id for x in ok] == ["A"]
    assert "B" in problems


# ---------------------------------------------------------------- Woo CSV

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_woo_export_writes_parent_and_six_variations(db, tmp_path):
    listing = _row(db, "A", "woocommerce", _woo_fields(), _woo_six())
    out = tmp_path / "woo.csv"
    result = woo_export.export_woocommerce([listing], out)

    rows = _read_csv(out)
    assert result.row_count == 7
    assert rows[0]["Type"] == "variable"
    assert rows[0]["SKU"] == "A"
    assert [r["Type"] for r in rows[1:]] == (
        ["variation"] * 5 + ["variation, virtual, downloadable"])


def test_woo_stl_variation_is_virtual_and_downloadable(db, tmp_path):
    listing = _row(db, "A", "woocommerce", _woo_fields(), _woo_six())
    out = tmp_path / "woo.csv"
    woo_export.export_woocommerce([listing], out)
    stl = [r for r in _read_csv(out) if r["SKU"] == "A-STL"][0]
    assert "virtual" in stl["Type"] and "downloadable" in stl["Type"]
    assert stl["Download 1 URL"].endswith("A.stl")
    assert stl["Regular price"] == "50.00"
    assert stl["Weight (kg)"] == ""


def test_woo_parent_lists_all_attribute_values(db, tmp_path):
    listing = _row(db, "A", "woocommerce", _woo_fields(), _woo_six())
    out = tmp_path / "woo.csv"
    woo_export.export_woocommerce([listing], out)
    parent = _read_csv(out)[0]
    assert parent["Attribute 1 name"] == "Materyal"
    assert parent["Attribute 1 value(s)"] == (
        "Gümüş, 8 Ayar Altın, 10 Ayar Altın, 14 Ayar Altın, "
        "18 Ayar Altın, STL Dosyası")
    assert parent["Attribute 1 visible"] == "1"


def test_woo_variations_reference_parent_sku(db, tmp_path):
    listing = _row(db, "A", "woocommerce", _woo_fields(), _woo_six())
    out = tmp_path / "woo.csv"
    woo_export.export_woocommerce([listing], out)
    for row in _read_csv(out)[1:]:
        assert row["Parent"] == "A"


def test_woo_prices_match_variants(db, tmp_path):
    listing = _row(db, "A", "woocommerce", _woo_fields(), _woo_six())
    out = tmp_path / "woo.csv"
    woo_export.export_woocommerce([listing], out)
    prices = {r["SKU"]: r["Regular price"] for r in _read_csv(out) if r["SKU"] != "A"}
    assert prices == {
        "A-AG": "1485.00", "A-8K": "16317.00", "A-10K": "20433.00",
        "A-14K": "28665.00", "A-18K": "36750.00", "A-STL": "50.00",
    }


def test_woo_weight_unit_kg_converts(db, tmp_path):
    listing = _row(db, "A", "woocommerce", _woo_fields(), _woo_six())
    out = tmp_path / "woo.csv"
    woo_export.export_woocommerce([listing], out, weight_unit="kg")
    ag = [r for r in _read_csv(out) if r["SKU"] == "A-AG"][0]
    assert ag["Weight (kg)"] == "0.005000"


def test_woo_file_is_utf8_bom(db, tmp_path):
    listing = _row(db, "A", "woocommerce", _woo_fields(), _woo_six())
    out = tmp_path / "woo.csv"
    woo_export.export_woocommerce([listing], out)
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")


def test_woo_turkish_characters_survive(db, tmp_path):
    listing = _row(db, "A", "woocommerce",
                   _woo_fields(title="Işıl Gümüş Çiçek"), _woo_six())
    out = tmp_path / "woo.csv"
    woo_export.export_woocommerce([listing], out)
    assert _read_csv(out)[0]["Name"] == "Işıl Gümüş Çiçek"


def test_woo_images_column_left_empty(db, tmp_path):
    """Gri clay render'lar ilana konmaz — ürünü satılamaz gösterir."""
    listing = _row(db, "A", "woocommerce", _woo_fields(), _woo_six())
    out = tmp_path / "woo.csv"
    woo_export.export_woocommerce([listing], out)
    assert all(r["Images"] == "" for r in _read_csv(out))


def test_woo_skips_invalid_listing(db, tmp_path):
    good = _row(db, "A", "woocommerce", _woo_fields(), _woo_six("A"))
    bad = _row(db, "B", "woocommerce", _woo_fields(slug="", title=""), [])
    out = tmp_path / "woo.csv"
    result = woo_export.export_woocommerce([good, bad], out)
    assert result.listing_count == 1
    assert "B" in result.skipped
    assert all(r["Meta: _catalog_file_id"] == "A" for r in _read_csv(out))


# ---------------------------------------------------------------- Etsy

def test_etsy_csv_one_row_per_listing(db, tmp_path):
    listing = _row(db, "A", "etsy", _etsy_fields(), [_silver(105.24, "A-AG", "USD")])
    out = tmp_path / "etsy.csv"
    result = etsy_export.export_etsy_csv([listing], out)
    rows = _read_csv(out)
    assert result.row_count == 1
    assert rows[0]["TITLE"] == "Sterling Silver Heart Pendant"
    assert rows[0]["PRICE"] == "105.24"
    assert rows[0]["CURRENCY_CODE"] == "USD"
    assert rows[0]["TAGS"] == "heart pendant,silver"
    assert rows[0]["SKU"] == "A-AG"


def test_etsy_csv_images_empty(db, tmp_path):
    listing = _row(db, "A", "etsy", _etsy_fields(), [_silver(105.24, "A-AG", "USD")])
    out = tmp_path / "etsy.csv"
    etsy_export.export_etsy_csv([listing], out)
    assert _read_csv(out)[0]["IMAGE1"] == ""


def test_etsy_json_matches_create_draft_listing(db, tmp_path):
    listing = _row(db, "A", "etsy", _etsy_fields(), [_silver(105.24, "A-AG", "USD")])
    out = tmp_path / "etsy.json"
    etsy_export.export_etsy_json([listing], out)
    payload = orjson.loads(out.read_bytes())[0]
    assert payload["title"] == "Sterling Silver Heart Pendant"
    assert payload["price"] == 105.24
    assert payload["who_made"] == "i_did"
    assert payload["when_made"] == "made_to_order"
    assert payload["taxonomy_id"] == 1265
    assert payload["is_supply"] is False
    assert payload["sku"] == ["A-AG"]


def test_etsy_json_flags_missing_shop_ids(db, tmp_path):
    """shipping_profile_id mağazaya özgü — uydurmuyoruz, eksik diye
    işaretliyoruz."""
    listing = _row(db, "A", "etsy", _etsy_fields(), [_silver(105.24, "A-AG", "USD")])
    out = tmp_path / "etsy.json"
    etsy_export.export_etsy_json([listing], out)
    payload = orjson.loads(out.read_bytes())[0]
    assert payload["shipping_profile_id"] is None
    assert "shipping_profile_id" in payload["_missing"]


def test_etsy_skips_listing_without_physical_variant(db, tmp_path):
    listing = _row(db, "A", "etsy", _etsy_fields(),
                   [VariantPrice("stl_digital", "A-STL", 50.0, "TRY", None, True)])
    out = tmp_path / "etsy.csv"
    result = etsy_export.export_etsy_csv([listing], out)
    assert result.row_count == 0
    assert "A" in result.skipped
