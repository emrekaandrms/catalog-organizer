"""In-memory multi-filter search over a CatalogIndex snapshot.

At 10-50k records a plain Python filter over CatalogIndex.all() runs in
milliseconds (the catalogue-search design notes §4/§5) — no search engine or database
needed at this scale.
"""
from __future__ import annotations

from catalog_organizer.core.schemas import CatalogRecord

UNKNOWN_BRAND = "unknown"

# Python's default str.lower() is locale-independent and follows the
# "international" I/i mapping (İ -> i + combining dot above, I -> i), not
# Turkish rules (İ -> i, I -> ı). Left uncorrected, a query typed with a
# Turkish dotted/dotless I would silently fail to match catalog text
# containing the other form. Every other Turkish letter (Ş/Ğ/Ü/Ö/Ç) already
# has a single, unambiguous lowercase mapping so only I/İ need remapping.
_TR_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})


def _tr_lower(text: str) -> str:
    return text.translate(_TR_LOWER_MAP).lower()


def search_records(
    records: list[CatalogRecord],
    *,
    category: str | None = None,
    subcategory: str | None = None,
    stone_status: str | None = None,
    brand: str | None = None,
    sprue_detected: bool | None = None,
    free_text: str | None = None,
) -> list[CatalogRecord]:
    """Filter records by controlled fields (AND) plus a free-text match
    against controlled_tags + rich_description + category/subcategory.

    `brand=UNKNOWN_BRAND` matches records with no brand assigned.
    `free_text` is split into whitespace-separated words; every word must
    appear as a substring somewhere in the searchable text (AND, case-
    insensitive) — this is what lets "taşlı haç jesus gurmet zincir" match
    a record via a mix of controlled_tags and rich_description.
    """
    words = _tr_lower(free_text).split() if free_text else []

    out: list[CatalogRecord] = []
    for rec in records:
        if category and rec.main_category != category:
            continue
        if subcategory and rec.subcategory != subcategory:
            continue
        # rec.stone_summary.status, NOT rec.polished_status: the latter is
        # the VLM's raw, un-cross-checked single-field opinion and — per a
        # real 68-file pilot (2026-07-24, PilotBatch) — it disagreed
        # with reality on nearly every file (50/68 "unclear", 0 "stone",
        # even though most filenames and stone_summary.status agreed the
        # piece has stones). stone_summary.status is built by
        # _build_stone_summary() from actual gem-layer geometry for .3dm
        # (extract_stone_summary_from_3dm — the proxy-stone-filtered
        # extractor validated earlier this session) / seat geometry for
        # .stl, falling back to the VLM opinion only when geometry yields
        # nothing — much more trustworthy for "taşlı/taşsız" search.
        if stone_status and rec.stone_summary.status != stone_status:
            continue
        if brand:
            if brand == UNKNOWN_BRAND:
                if rec.brand is not None:
                    continue
            elif rec.brand != brand:
                continue
        if sprue_detected is not None and rec.sprue.detected != sprue_detected:
            continue
        if words:
            haystack = " ".join((
                *rec.controlled_tags,
                rec.rich_description,
                rec.main_category,
                rec.subcategory,
            ))
            haystack = _tr_lower(haystack)
            if not all(w in haystack for w in words):
                continue
        out.append(rec)
    return out


def distinct_brands(records: list[CatalogRecord]) -> list[str]:
    """Sorted distinct brand values actually present in the catalog (for
    populating the search screen's brand dropdown)."""
    return sorted({rec.brand for rec in records if rec.brand})
