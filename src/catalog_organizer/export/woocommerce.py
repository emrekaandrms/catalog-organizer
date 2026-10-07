"""WooCommerce ürün CSV'si — birinci parti içe aktarıcı formatı.

Varyasyonlu ürün = 1 ana satır (`Type=variable`) + N varyasyon satırı
(`Type=variation`). STL varyantı `Type="variation, virtual, downloadable"`
olarak yazılır; bu tek satır kargodan muaf kalır ve dosyayı otomatik teslim
eder, diğerleri fiziksel kalır.

Kodlama UTF-8-BOM: BOM'suz UTF-8 CSV'yi Excel'de açtığında ş/ğ/ı bozuluyor.
WooCommerce içe aktarıcısı BOM'u sorunsuz yutuyor.

SKU'lar deterministik ve kalıcı — içe aktarıcı bunlardan eşleştirip mevcut
ürünü GÜNCELLER, kopya açmaz. Bu olmadan her yeniden aktarım sitede yüzlerce
çift ürün bırakır.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from catalog_organizer.db.listings import ListingRow
from catalog_organizer.listing.pricing import VARIANT_LABEL_TR
from catalog_organizer.listing.validate import ValidationIssue, check_unique_slugs, partition

ATTRIBUTE_NAME = "Materyal"

COLUMNS = (
    "ID", "Type", "SKU", "Name", "Published", "Is featured?",
    "Visibility in catalog", "Short description", "Description",
    "Tax status", "In stock?", "Stock", "Weight (kg)",
    "Regular price", "Categories", "Tags", "Images", "Parent",
    "Position", "Attribute 1 name", "Attribute 1 value(s)",
    "Attribute 1 visible", "Attribute 1 global",
    "Download 1 name", "Download 1 URL",
    "Meta: _yoast_wpseo_metadesc", "Meta: _slug",
    "Meta: _surdurulebilirlik", "Meta: _catalog_file_id",
)


@dataclass
class ExportResult:
    file_path: Path
    row_count: int
    listing_count: int
    skipped: dict[str, list[ValidationIssue]] = field(default_factory=dict)
    exported_listing_ids: list[int] = field(default_factory=list)


def _weight(grams: float | None, unit: str) -> str:
    if grams is None:
        return ""
    return f"{grams / 1000.0:.6f}" if unit == "kg" else f"{grams:.3f}"


def _variant_rows(listing: ListingRow, weight_unit: str, download_url_template: str):
    parent_sku = f"{listing.file_id}"
    for position, variant in enumerate(listing.variants):
        label = VARIANT_LABEL_TR.get(variant.variant_key, variant.variant_key)
        row = {c: "" for c in COLUMNS}
        row.update({
            "Type": ("variation, virtual, downloadable" if variant.is_digital
                     else "variation"),
            "SKU": variant.sku,
            "Name": f"{listing.title} - {label}",
            "Published": "1",
            "Tax status": "taxable",
            "In stock?": "1",
            "Regular price": f"{variant.price:.2f}",
            "Parent": parent_sku,
            "Position": str(position),
            "Attribute 1 name": ATTRIBUTE_NAME,
            "Attribute 1 value(s)": label,
            "Weight (kg)": _weight(variant.weight_g, weight_unit),
            "Meta: _catalog_file_id": listing.file_id,
        })
        if variant.is_digital:
            row["Download 1 name"] = "STL"
            row["Download 1 URL"] = download_url_template.format(
                file_id=listing.file_id)
        yield row


def _parent_row(listing: ListingRow, include_images: bool) -> dict[str, str]:
    labels = [VARIANT_LABEL_TR.get(v.variant_key, v.variant_key)
              for v in listing.variants]
    row = {c: "" for c in COLUMNS}
    row.update({
        "Type": "variable",
        "SKU": listing.file_id,
        "Name": listing.title or "",
        "Published": "1",
        "Is featured?": "0",
        "Visibility in catalog": "visible",
        "Short description": listing.short_description or "",
        "Description": listing.description or "",
        "Tax status": "taxable",
        "In stock?": "1",
        "Categories": listing.category_path or "",
        "Tags": ", ".join(listing.tags),
        "Attribute 1 name": ATTRIBUTE_NAME,
        "Attribute 1 value(s)": ", ".join(labels),
        "Attribute 1 visible": "1",
        "Attribute 1 global": "1",
        "Meta: _yoast_wpseo_metadesc": listing.meta_description or "",
        "Meta: _slug": listing.slug or "",
        "Meta: _surdurulebilirlik": listing.sustainability_note or "",
        "Meta: _catalog_file_id": listing.file_id,
    })
    # Images kolonu bilerek boş: mevcut snapshot'lar VLM sınıflandırması için
    # üretilmiş gri clay render'lar, pazarlama görseli değil.
    if include_images:
        row["Images"] = ""
    return row


def export_woocommerce(
    listings: list[ListingRow],
    out_path: Path,
    *,
    weight_unit: str = "g",
    download_url_template: str = "https://example.invalid/stl/{file_id}.stl",
    include_images: bool = False,
) -> ExportResult:
    ok, skipped = partition(listings)
    for issue in check_unique_slugs(ok):
        skipped.setdefault(issue.file_id, []).append(issue)
    ok = [x for x in ok if x.file_id not in skipped]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for listing in ok:
        rows.append(_parent_row(listing, include_images))
        rows.extend(_variant_rows(listing, weight_unit, download_url_template))

    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    return ExportResult(
        file_path=out_path,
        row_count=len(rows),
        listing_count=len(ok),
        skipped=skipped,
        exported_listing_ids=[x.listing_id for x in ok],
    )
