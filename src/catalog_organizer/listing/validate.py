"""Aktarım öncesi doğrulama.

Sorunlu satır dosyaya GİRMEZ, ekranda gerekçesiyle listelenir. Etsy'nin
kendi arayüzünde 100 ilanın 12'sinin neden reddedildiğini aramaktansa
burada görmek daha iyi.
"""
from __future__ import annotations

from dataclasses import dataclass

from catalog_organizer.db.listings import ListingRow
from catalog_organizer.listing.schemas import (
    ETSY_MATERIAL_MAX_COUNT, ETSY_MATERIAL_MAX_LEN, ETSY_TAG_MAX_COUNT,
    ETSY_TAG_MAX_LEN, ETSY_TITLE_MAX,
)


@dataclass(frozen=True)
class ValidationIssue:
    file_id: str
    channel: str
    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.file_id} [{self.channel}] {self.field}: {self.message}"


def _common_issues(listing: ListingRow) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    add = lambda f, m: issues.append(  # noqa: E731
        ValidationIssue(listing.file_id, listing.channel, f, m))

    if not (listing.title or "").strip():
        add("title", "başlık boş")
    if not (listing.description or "").strip():
        add("description", "açıklama boş")

    if not listing.variants:
        add("variants", "hiç varyant yok — ağırlık ölçülemediyse fiyat çıkmaz")
    for v in listing.variants:
        if v.price <= 0:
            add(f"variant:{v.variant_key}", f"fiyat sıfır ya da negatif ({v.price})")
        if not v.sku:
            add(f"variant:{v.variant_key}", "SKU boş")
    skus = [v.sku for v in listing.variants]
    if len(skus) != len(set(skus)):
        add("variants", "aynı ilanda tekrarlanan SKU var")
    return issues


def validate_listing(listing: ListingRow) -> list[ValidationIssue]:
    issues = _common_issues(listing)
    add = lambda f, m: issues.append(  # noqa: E731
        ValidationIssue(listing.file_id, listing.channel, f, m))

    if listing.channel == "etsy":
        title = listing.title or ""
        if len(title) > ETSY_TITLE_MAX:
            add("title", f"{len(title)} karakter, Etsy sınırı {ETSY_TITLE_MAX}")
        if len(listing.tags) > ETSY_TAG_MAX_COUNT:
            add("tags", f"{len(listing.tags)} tag, Etsy sınırı {ETSY_TAG_MAX_COUNT}")
        for tag in listing.tags:
            if len(tag) > ETSY_TAG_MAX_LEN:
                add("tags", f"'{tag}' {len(tag)} karakter, sınır {ETSY_TAG_MAX_LEN}")
        if len(listing.materials) > ETSY_MATERIAL_MAX_COUNT:
            add("materials",
                f"{len(listing.materials)} materyal, sınır {ETSY_MATERIAL_MAX_COUNT}")
        for mat in listing.materials:
            if len(mat) > ETSY_MATERIAL_MAX_LEN:
                add("materials",
                    f"'{mat}' {len(mat)} karakter, sınır {ETSY_MATERIAL_MAX_LEN}")
        if not listing.who_made:
            add("who_made", "Etsy zorunlu alanı boş")
        if not listing.when_made:
            add("when_made", "Etsy zorunlu alanı boş")
        if listing.etsy_taxonomy_id is None:
            add("etsy_taxonomy_id",
                "Etsy taksonomi ID'si yok — config/etsy_taxonomy_map.yaml doldurulmalı")

    elif listing.channel == "woocommerce":
        if not (listing.slug or "").strip():
            add("slug", "slug boş")
        if not (listing.category_path or "").strip():
            add("category_path", "kategori boş")

    return issues


def validate_many(listings: list[ListingRow]) -> dict[str, list[ValidationIssue]]:
    """file_id → sorunlar. Sorunsuz ilanlar sözlükte hiç görünmez."""
    out: dict[str, list[ValidationIssue]] = {}
    for listing in listings:
        issues = validate_listing(listing)
        if issues:
            out[listing.file_id] = issues
    return out


def partition(
    listings: list[ListingRow],
) -> tuple[list[ListingRow], dict[str, list[ValidationIssue]]]:
    """(aktarılabilirler, sorunlular) — dışa aktarıcıların kullandığı ayrım."""
    problems = validate_many(listings)
    ok = [x for x in listings if x.file_id not in problems]
    return ok, problems


def check_unique_slugs(listings: list[ListingRow]) -> list[ValidationIssue]:
    """WooCommerce'te iki ürün aynı slug'ı alamaz — ikincisi sessizce
    '-2' ekiyle yayınlanır ve SEO'da beklenen URL kaybolur."""
    seen: dict[str, str] = {}
    issues: list[ValidationIssue] = []
    for listing in listings:
        slug = (listing.slug or "").strip()
        if not slug:
            continue
        if slug in seen:
            issues.append(ValidationIssue(
                listing.file_id, listing.channel, "slug",
                f"'{slug}' zaten {seen[slug]} tarafından kullanılıyor",
            ))
        else:
            seen[slug] = listing.file_id
    return issues
