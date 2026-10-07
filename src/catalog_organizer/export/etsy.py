"""Etsy dışa aktarımı — CSV + API JSON.

Etsy'de ilan OLUŞTURMAK için birinci parti CSV içe aktarma yok (yalnızca
mevcut ilanların dışa aktarımı var). Bu yüzden iki çıktı üretiyoruz:

1. CSV — Etsy'nin kendi ilan-dışa-aktarım kolon düzenine uyumlu; üçüncü
   parti yükleyiciler (Vela vb.) bunu kabul ediyor.
2. JSON — Etsy Open API v3 `createDraftListing` gövdesine uyumlu. Asıl
   çıktı bu; kullanıcı API modülünü kendisi yazacak.

`shipping_profile_id` ve `return_policy_id` mağazaya özgü sayılar; elimizde
yok ve UYDURULMUYOR — JSON'da `null` bırakılıp `_missing` listesinde
belirtiliyor, böylece API'ye gönderilmeden önce doldurulması gerektiği
görülüyor.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import orjson

from catalog_organizer.db.listings import ListingRow
from catalog_organizer.listing.validate import ValidationIssue, partition

CSV_COLUMNS = (
    "TITLE", "DESCRIPTION", "PRICE", "CURRENCY_CODE", "QUANTITY",
    "TAGS", "MATERIALS", "SKU",
    "IMAGE1", "IMAGE2", "IMAGE3", "IMAGE4", "IMAGE5",
    "VARIATION 1 TYPE", "VARIATION 1 NAME", "VARIATION 1 VALUES",
    "VARIATION 2 TYPE", "VARIATION 2 NAME", "VARIATION 2 VALUES",
)

DEFAULT_QUANTITY = 1
REQUIRED_SHOP_FIELDS = ("shipping_profile_id", "return_policy_id")


@dataclass
class ExportResult:
    file_path: Path
    row_count: int
    listing_count: int
    skipped: dict[str, list[ValidationIssue]] = field(default_factory=dict)
    exported_listing_ids: list[int] = field(default_factory=list)


def _primary_variant(listing: ListingRow):
    """Etsy'de tek varyant var (gümüş) — fiyat ondan gelir."""
    physical = [v for v in listing.variants if not v.is_digital]
    return physical[0] if physical else None


def _csv_row(listing: ListingRow) -> dict[str, str] | None:
    variant = _primary_variant(listing)
    if variant is None:
        return None
    row = {c: "" for c in CSV_COLUMNS}
    row.update({
        "TITLE": listing.title or "",
        "DESCRIPTION": listing.description or "",
        "PRICE": f"{variant.price:.2f}",
        "CURRENCY_CODE": variant.currency,
        "QUANTITY": str(DEFAULT_QUANTITY),
        "TAGS": ",".join(listing.tags),
        "MATERIALS": ",".join(listing.materials),
        "SKU": variant.sku,
    })
    return row


def export_etsy_csv(listings: list[ListingRow], out_path: Path) -> ExportResult:
    ok, skipped = partition(listings)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    exported: list[int] = []
    for listing in ok:
        row = _csv_row(listing)
        if row is None:
            skipped.setdefault(listing.file_id, []).append(ValidationIssue(
                listing.file_id, "etsy", "variants",
                "fiziksel varyant yok — Etsy ilanı fiyatsız olamaz"))
            continue
        rows.append(row)
        exported.append(listing.listing_id)

    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    return ExportResult(
        file_path=out_path, row_count=len(rows), listing_count=len(rows),
        skipped=skipped, exported_listing_ids=exported,
    )


def to_api_payload(listing: ListingRow) -> dict[str, object]:
    """Etsy Open API v3 `createDraftListing` gövdesi."""
    variant = _primary_variant(listing)
    return {
        "quantity": DEFAULT_QUANTITY,
        "title": listing.title or "",
        "description": listing.description or "",
        "price": round(variant.price, 2) if variant else None,
        "who_made": listing.who_made,
        "when_made": listing.when_made,
        "taxonomy_id": listing.etsy_taxonomy_id,
        "is_supply": bool(listing.is_supply),
        "tags": list(listing.tags),
        "materials": list(listing.materials),
        "sku": [variant.sku] if variant else [],
        "shipping_profile_id": None,
        "return_policy_id": None,
        "_file_id": listing.file_id,
        "_currency": variant.currency if variant else None,
        "_missing": list(REQUIRED_SHOP_FIELDS),
    }


def export_etsy_json(listings: list[ListingRow], out_path: Path) -> ExportResult:
    ok, skipped = partition(listings)

    payloads: list[dict[str, object]] = []
    exported: list[int] = []
    for listing in ok:
        if _primary_variant(listing) is None:
            skipped.setdefault(listing.file_id, []).append(ValidationIssue(
                listing.file_id, "etsy", "variants",
                "fiziksel varyant yok — Etsy ilanı fiyatsız olamaz"))
            continue
        payloads.append(to_api_payload(listing))
        exported.append(listing.listing_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(orjson.dumps(payloads, option=orjson.OPT_INDENT_2))

    return ExportResult(
        file_path=out_path, row_count=len(payloads), listing_count=len(payloads),
        skipped=skipped, exported_listing_ids=exported,
    )
