"""Ürün sorguları — `catalog/search.py`'deki RAM döngüsünün SQL karşılığı.

Override bindirme burada yapılır: `product_overrides`'taki değerler hem
döndürülen `CatalogRecord`'a hem de filtrelere uygulanır. Filtrelere de
uygulanması şart — yoksa düzeltilmiş bir kategoriyle arama yapılamaz.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import orjson

from catalog_organizer.core.schemas import CatalogRecord
from catalog_organizer.db.text import to_fts_query

OVERRIDABLE_FIELDS: tuple[str, ...] = ("main_category", "subcategory", "brand")
UNKNOWN_BRAND = "unknown"

_EFFECTIVE = {
    "main_category": "COALESCE(o_cat.value, p.main_category)",
    "subcategory": "COALESCE(o_sub.value, p.subcategory)",
    "brand": "COALESCE(o_brand.value, p.brand)",
}

_FROM = """
FROM products p
LEFT JOIN product_overrides o_cat
       ON o_cat.file_id = p.file_id AND o_cat.field = 'main_category'
LEFT JOIN product_overrides o_sub
       ON o_sub.file_id = p.file_id AND o_sub.field = 'subcategory'
LEFT JOIN product_overrides o_brand
       ON o_brand.file_id = p.file_id AND o_brand.field = 'brand'
"""


def _build_where(
    *,
    category: str | None,
    subcategory: str | None,
    stone_status: str | None,
    brand: str | None,
    sprue_detected: bool | None,
    free_text: str | None,
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []

    if category:
        clauses.append(f"{_EFFECTIVE['main_category']} = ?")
        params.append(category)
    if subcategory:
        clauses.append(f"{_EFFECTIVE['subcategory']} = ?")
        params.append(subcategory)
    if stone_status:
        # stone_summary.status — polished_status DEĞİL. 68 dosyalık gerçek
        # pilotta ikisi neredeyse her dosyada çelişmişti.
        clauses.append("p.stone_status = ?")
        params.append(stone_status)
    if brand:
        if brand == UNKNOWN_BRAND:
            clauses.append(f"{_EFFECTIVE['brand']} IS NULL")
        else:
            clauses.append(f"{_EFFECTIVE['brand']} = ?")
            params.append(brand)
    if sprue_detected is not None:
        clauses.append("p.sprue_detected = ?")
        params.append(int(sprue_detected))
    if free_text:
        query = to_fts_query(free_text)
        if query:
            clauses.append(
                "p.file_id IN (SELECT file_id FROM products_fts "
                "WHERE products_fts MATCH ?)"
            )
            params.append(query)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


_SELECT_COLS = (
    "SELECT p.record_json, "
    f"{_EFFECTIVE['main_category']} AS eff_category, "
    f"{_EFFECTIVE['subcategory']} AS eff_subcategory, "
    f"{_EFFECTIVE['brand']} AS eff_brand "
)


def _hydrate(row: sqlite3.Row) -> CatalogRecord:
    rec = CatalogRecord.model_validate(orjson.loads(row["record_json"]))
    patch: dict[str, object] = {}
    for field, alias in (
        ("main_category", "eff_category"),
        ("subcategory", "eff_subcategory"),
        ("brand", "eff_brand"),
    ):
        value = row[alias]
        if value != getattr(rec, field):
            patch[field] = value
    return rec.model_copy(update=patch) if patch else rec


def search_product_ids(
    conn: sqlite3.Connection,
    *,
    category: str | None = None,
    subcategory: str | None = None,
    stone_status: str | None = None,
    brand: str | None = None,
    sprue_detected: bool | None = None,
    free_text: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[str]:
    where, params = _build_where(
        category=category, subcategory=subcategory, stone_status=stone_status,
        brand=brand, sprue_detected=sprue_detected, free_text=free_text,
    )
    sql = f"SELECT p.file_id {_FROM}{where} ORDER BY p.file_id"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params = [*params, limit, offset]
    return [r[0] for r in conn.execute(sql, params)]


def search_products(
    conn: sqlite3.Connection,
    *,
    category: str | None = None,
    subcategory: str | None = None,
    stone_status: str | None = None,
    brand: str | None = None,
    sprue_detected: bool | None = None,
    free_text: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[CatalogRecord]:
    where, params = _build_where(
        category=category, subcategory=subcategory, stone_status=stone_status,
        brand=brand, sprue_detected=sprue_detected, free_text=free_text,
    )
    sql = f"{_SELECT_COLS}{_FROM}{where} ORDER BY p.file_id"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params = [*params, limit, offset]
    return [_hydrate(row) for row in conn.execute(sql, params)]


def get_product(conn: sqlite3.Connection, file_id: str) -> CatalogRecord | None:
    row = conn.execute(
        f"{_SELECT_COLS}{_FROM} WHERE p.file_id = ?", (file_id,)
    ).fetchone()
    return _hydrate(row) if row else None


def get_products(conn: sqlite3.Connection, file_ids: list[str]) -> list[CatalogRecord]:
    """Verilen sırayı koruyarak toplu okuma (seçim listeleri için)."""
    if not file_ids:
        return []
    marks = ",".join("?" * len(file_ids))
    rows = conn.execute(
        f"{_SELECT_COLS}{_FROM} WHERE p.file_id IN ({marks})", file_ids
    ).fetchall()
    by_id = {orjson.loads(r["record_json"])["file_id"]: _hydrate(r) for r in rows}
    return [by_id[fid] for fid in file_ids if fid in by_id]


def count_products(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM products").fetchone()[0])


def distinct_brands(conn: sqlite3.Connection) -> list[str]:
    sql = (
        f"SELECT DISTINCT {_EFFECTIVE['brand']} AS b {_FROM} "
        f"WHERE b IS NOT NULL AND b != '' ORDER BY b"
    )
    return [r[0] for r in conn.execute(sql)]


def distinct_categories(conn: sqlite3.Connection) -> list[str]:
    sql = (
        f"SELECT DISTINCT {_EFFECTIVE['main_category']} AS c {_FROM} "
        f"WHERE c IS NOT NULL AND c != '' ORDER BY c"
    )
    return [r[0] for r in conn.execute(sql)]


def set_override(
    conn: sqlite3.Connection,
    file_id: str,
    field: str,
    value: str | None,
    note: str | None = None,
) -> None:
    if field not in OVERRIDABLE_FIELDS:
        raise ValueError(
            f"'{field}' override edilemez; izin verilenler: {OVERRIDABLE_FIELDS}"
        )
    conn.execute(
        "INSERT INTO product_overrides (file_id, field, value, note, updated_at) "
        "VALUES (?,?,?,?,?) "
        "ON CONFLICT(file_id, field) DO UPDATE SET "
        "value=excluded.value, note=excluded.note, updated_at=excluded.updated_at",
        (file_id, field, value, note, datetime.now(tz=timezone.utc).isoformat()),
    )
    conn.commit()


def clear_override(conn: sqlite3.Connection, file_id: str, field: str) -> None:
    conn.execute(
        "DELETE FROM product_overrides WHERE file_id = ? AND field = ?",
        (file_id, field),
    )
    conn.commit()


def get_overrides(conn: sqlite3.Connection, file_id: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT field, value FROM product_overrides WHERE file_id = ?", (file_id,)
    ).fetchall()
    return {r["field"]: r["value"] for r in rows}
