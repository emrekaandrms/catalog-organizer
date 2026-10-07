"""İlan metni üretimi.

Kanal başına bir LLM çağrısı (ürün başına iki). Etsy ve Woo'nun alan setleri,
uzunluk sınırları ve tag kuralları farklı; tek çağrıya ikisini birden
yaptırmak JSON şemasını iki katına çıkarır. Kesik JSON'un bir cevabın
tamamını çöpe attığını bu projede zaten yaşadık (polished_status olayı,
65 dosyanın 6'sı).

Sayısal alanlar modele YAZDIRILMAZ — `build_facts()` onları DB'den derleyip
prompt'a bağlayıcı bilgi olarak basar, model yalnızca etrafına metin yazar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catalog_organizer.core.schemas import CatalogRecord
from catalog_organizer.listing.pricing import (
    VARIANT_LABEL_TR, PricingConfig, VariantPrice, price_variants,
)
from catalog_organizer.listing.schemas import (
    EtsyListingDraft, WooListingDraft, slugify,
)

CHANNEL_LANGUAGE = {"etsy": "en", "woocommerce": "tr"}

_CATEGORY_EN = {
    "ring": "Ring", "earring": "Earrings", "pendant": "Pendant",
    "necklace": "Necklace", "bracelet": "Bracelet", "chain": "Chain",
    "brooch": "Brooch", "charm": "Charm", "set": "Jewellery Set",
}
_CATEGORY_TR = {
    "ring": "Yüzük", "earring": "Küpe", "pendant": "Kolye Ucu",
    "necklace": "Kolye", "bracelet": "Bileklik", "chain": "Zincir",
    "brooch": "Broş", "charm": "Charm", "set": "Takı Seti",
}


class ListingGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedListing:
    file_id: str
    channel: str
    language: str
    fields: dict[str, Any]
    variants: list[VariantPrice]
    provider: str
    model: str
    prompt_version: str


def _dimension_line(rec: CatalogRecord) -> str:
    m = rec.measurements
    return (f"{m.bbox_width_mm:.1f} x {m.bbox_height_mm:.1f} x "
            f"{m.bbox_depth_mm:.1f} mm")


def build_facts(
    rec: CatalogRecord, channel: str, variants: list[VariantPrice]
) -> str:
    """Modele verilecek bağlayıcı bilgi bloğu.

    Buradaki her satır ölçülmüş ya da hesaplanmış veridir. Model bunları
    doğrulamaz, çelişmez ve birebir tekrar etmez — etrafına metin yazar.
    """
    stones = rec.stone_summary
    tr = channel == "woocommerce"
    names = _CATEGORY_TR if tr else _CATEGORY_EN
    category = names.get(rec.main_category, rec.main_category)

    lines = [
        f"category: {category}",
        f"subcategory: {rec.subcategory}",
        f"tags_from_analysis: {', '.join(rec.controlled_tags)}",
        f"bounding_box: {_dimension_line(rec)}",
        f"has_stone_settings: {'yes' if stones.status == 'stone' else 'no'}"
        if stones.status in ("stone", "polished") else "has_stone_settings: unknown",
    ]
    if stones.status == "stone":
        n = sum(s.quantity for s in stones.side_stones)
        if stones.center_stone is not None:
            n += stones.center_stone.quantity
        if n:
            lines.append(f"stone_seat_count: {n}")
    if rec.rich_description:
        lines.append(f"visual_description: {rec.rich_description}")

    if channel == "etsy":
        lines.append("materials_offered: sterling silver 925 only")
        silver = next((v for v in variants if v.variant_key == "silver_925"), None)
        if silver and silver.weight_g:
            lines.append(f"silver_weight_g: {silver.weight_g:.2f}")
    else:
        offered = [VARIANT_LABEL_TR.get(v.variant_key, v.variant_key)
                   for v in variants]
        lines.append(f"satis_secenekleri: {', '.join(offered)}")
        lines.append("uretim: siparise ozel dokum (made to order)")

    return "\n".join(lines)


def _fallback_category_path(rec: CatalogRecord, channel: str) -> str:
    if channel == "etsy":
        return f"Jewelry > {_CATEGORY_EN.get(rec.main_category, 'Other')}"
    return _CATEGORY_TR.get(rec.main_category, "Diğer")


def _etsy_fields(
    rec: CatalogRecord, draft: EtsyListingDraft, cfg: PricingConfig
) -> dict[str, Any]:
    return {
        "title": draft.title,
        "description": draft.description,
        "tags": draft.tags,
        "materials": draft.materials or ["sterling silver"],
        "category_path": draft.taxonomy_hint or _fallback_category_path(rec, "etsy"),
        # Taksonomi ID'si uydurulmuyor — eşleme tablosundan gelir, yoksa None
        # kalır ve doğrulama bunu eksik diye işaretler.
        "etsy_taxonomy_id": _taxonomy_id(rec),
        "who_made": cfg.etsy_who_made,
        "when_made": cfg.etsy_when_made,
        "is_supply": 0,
        "made_to_order": 1,
        "sustainability_note": draft.sustainability_note,
        "slug": None,
        "short_description": None,
        "meta_description": None,
    }


def _woo_fields(rec: CatalogRecord, draft: WooListingDraft) -> dict[str, Any]:
    slug = draft.slug.strip() or slugify(draft.title)
    if not slug:
        slug = slugify(rec.file_id)
    return {
        "title": draft.title,
        "short_description": draft.short_description,
        "description": draft.description,
        "tags": draft.tags,
        "materials": [],
        "category_path": (
            " > ".join(draft.categories) if draft.categories
            else _fallback_category_path(rec, "woocommerce")
        ),
        "etsy_taxonomy_id": None,
        "slug": slug,
        "meta_description": draft.meta_description,
        "who_made": None,
        "when_made": None,
        "is_supply": 0,
        "made_to_order": 1,
        "sustainability_note": draft.sustainability_note,
    }


def _taxonomy_id(rec: CatalogRecord) -> int | None:
    """config/etsy_taxonomy_map.yaml varsa oradan; yoksa None.

    Etsy'nin sayısal taksonomi ID'leri mağazaya değil platforma ait sabitler
    ama elimizde güvenilir bir liste yok. Uydurmak yerine boş bırakıp
    doğrulamada işaretliyoruz.
    """
    try:
        from catalog_organizer.core.config import _load_yaml
        mapping = _load_yaml("etsy_taxonomy_map.yaml").get("categories", {})
    except Exception:
        return None
    value = mapping.get(rec.subcategory) or mapping.get(rec.main_category)
    return int(value) if value is not None else None


def generate_listing(
    rec: CatalogRecord,
    channel: str,
    provider,
    templates: dict[str, Any],
    cfg: PricingConfig,
) -> GeneratedListing:
    if channel not in CHANNEL_LANGUAGE:
        raise ListingGenerationError(f"bilinmeyen kanal: {channel}")
    block = templates.get(channel)
    if not block:
        raise ListingGenerationError(
            f"listing_prompt_templates.yaml içinde '{channel}' bloğu yok")

    variants = price_variants(rec, channel, cfg)
    facts = build_facts(rec, channel, variants)
    payload = provider.complete_json(
        block["system"], block["user"].format(facts=facts)
    )

    try:
        if channel == "etsy":
            draft = EtsyListingDraft.model_validate(payload)
            fields = _etsy_fields(rec, draft, cfg)
        else:
            draft = WooListingDraft.model_validate(payload)
            fields = _woo_fields(rec, draft)
    except Exception as exc:
        raise ListingGenerationError(
            f"{rec.file_id} [{channel}] modelin cevabı şemaya uymadı: {exc}"
        ) from exc

    return GeneratedListing(
        file_id=rec.file_id,
        channel=channel,
        language=CHANNEL_LANGUAGE[channel],
        fields=fields,
        variants=variants,
        provider=getattr(provider, "display_name", "unknown"),
        model=getattr(provider, "model_name", "unknown"),
        prompt_version=str(templates.get("prompt_version", "unknown")),
    )


def persist(conn, generated: GeneratedListing, cfg: PricingConfig,
            selection_id: int | None = None) -> int:
    """Üretilen metni ve fiyatları DB'ye yazar; listing_id döndürür."""
    from catalog_organizer.db.listings import replace_variants, upsert_listing

    listing_id = upsert_listing(
        conn, generated.file_id, generated.channel,
        language=generated.language,
        fields=generated.fields,
        selection_id=selection_id,
        provider=generated.provider,
        model=generated.model,
        prompt_version=generated.prompt_version,
    )
    replace_variants(conn, listing_id, generated.variants, cfg.snapshot())
    return listing_id


def reprice(conn, rec: CatalogRecord, channel: str, cfg: PricingConfig) -> int:
    """Metne dokunmadan fiyatları yeniden hesaplar. Kur değiştiğinde
    çalıştırılır — 100 ilanı yeniden yazdırmaya gerek yok."""
    from catalog_organizer.db.listings import get_listing, replace_variants

    listing = get_listing(conn, rec.file_id, channel, with_variants=False)
    if listing is None:
        return 0
    variants = price_variants(rec, channel, cfg)
    return replace_variants(conn, listing.listing_id, variants, cfg.snapshot())
