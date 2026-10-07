"""İlan ve varyant CRUD — DB'nin sahibi olduğu veri.

Bir ürünün her kanalda en fazla bir ilanı olur (UNIQUE file_id+channel).
Metin yeniden üretilirse üzerine yazılır; kullanıcı elle düzenlerse
`human_edited=1` işaretlenir ve bir daha üretim onu ezmez (üretici
`skip_human_edited` ile korur).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import orjson

from catalog_organizer.listing.pricing import VariantPrice

STATUSES = ("draft", "approved", "exported")

_TEXT_FIELDS = (
    "title", "short_description", "description", "category_path",
    "etsy_taxonomy_id", "slug", "meta_description", "who_made", "when_made",
    "is_supply", "made_to_order", "sustainability_note",
)


@dataclass(frozen=True)
class ListingRow:
    listing_id: int
    file_id: str
    channel: str
    selection_id: int | None
    language: str
    status: str
    title: str | None
    short_description: str | None
    description: str | None
    tags: list[str]
    materials: list[str]
    category_path: str | None
    etsy_taxonomy_id: int | None
    slug: str | None
    meta_description: str | None
    who_made: str | None
    when_made: str | None
    is_supply: bool
    made_to_order: bool
    sustainability_note: str | None
    generated_provider: str | None
    generated_model: str | None
    prompt_version: str | None
    generated_at: str | None
    human_edited: bool
    edited_at: str | None
    variants: list[VariantPrice] = field(default_factory=list)


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _loads(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _row_to_listing(row: sqlite3.Row) -> ListingRow:
    return ListingRow(
        listing_id=int(row["listing_id"]),
        file_id=row["file_id"],
        channel=row["channel"],
        selection_id=row["selection_id"],
        language=row["language"],
        status=row["status"],
        title=row["title"],
        short_description=row["short_description"],
        description=row["description"],
        tags=_loads(row["tags"]),
        materials=_loads(row["materials"]),
        category_path=row["category_path"],
        etsy_taxonomy_id=row["etsy_taxonomy_id"],
        slug=row["slug"],
        meta_description=row["meta_description"],
        who_made=row["who_made"],
        when_made=row["when_made"],
        is_supply=bool(row["is_supply"]),
        made_to_order=bool(row["made_to_order"]),
        sustainability_note=row["sustainability_note"],
        generated_provider=row["generated_provider"],
        generated_model=row["generated_model"],
        prompt_version=row["prompt_version"],
        generated_at=row["generated_at"],
        human_edited=bool(row["human_edited"]),
        edited_at=row["edited_at"],
    )


def upsert_listing(
    conn: sqlite3.Connection,
    file_id: str,
    channel: str,
    *,
    language: str,
    fields: dict[str, Any],
    selection_id: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
) -> int:
    """Üretilen metni yazar. Mevcut ilan varsa üzerine yazar ve durumu
    `draft`'a döndürür — yeni metin onaylanmış sayılmaz."""
    payload: dict[str, Any] = {
        "file_id": file_id,
        "channel": channel,
        "selection_id": selection_id,
        "language": language,
        "status": "draft",
        "tags": orjson.dumps(fields.get("tags", [])).decode(),
        "materials": orjson.dumps(fields.get("materials", [])).decode(),
        "generated_provider": provider,
        "generated_model": model,
        "prompt_version": prompt_version,
        "generated_at": _now(),
        "human_edited": 0,
        "edited_at": None,
    }
    for key in _TEXT_FIELDS:
        payload[key] = fields.get(key)

    cols = list(payload)
    sql = (
        f"INSERT INTO listings ({', '.join(cols)}) "
        f"VALUES ({', '.join(':' + c for c in cols)}) "
        f"ON CONFLICT(file_id, channel) DO UPDATE SET "
        + ", ".join(f"{c}=excluded.{c}" for c in cols
                    if c not in ("file_id", "channel"))
    )
    conn.execute(sql, payload)
    conn.commit()
    row = conn.execute(
        "SELECT listing_id FROM listings WHERE file_id=? AND channel=?",
        (file_id, channel),
    ).fetchone()
    return int(row[0])


def get_listing(
    conn: sqlite3.Connection, file_id: str, channel: str, *, with_variants: bool = True
) -> ListingRow | None:
    row = conn.execute(
        "SELECT * FROM listings WHERE file_id=? AND channel=?", (file_id, channel)
    ).fetchone()
    if row is None:
        return None
    listing = _row_to_listing(row)
    if with_variants:
        listing.variants.extend(get_variants(conn, listing.listing_id))
    return listing


def list_listings(
    conn: sqlite3.Connection,
    *,
    selection_id: int | None = None,
    channel: str | None = None,
    statuses: tuple[str, ...] | None = None,
    file_ids: list[str] | None = None,
) -> list[ListingRow]:
    clauses: list[str] = []
    params: list[Any] = []
    if selection_id is not None:
        clauses.append("selection_id = ?")
        params.append(selection_id)
    if channel:
        clauses.append("channel = ?")
        params.append(channel)
    if statuses:
        clauses.append(f"status IN ({','.join('?' * len(statuses))})")
        params.extend(statuses)
    if file_ids is not None:
        if not file_ids:
            return []
        clauses.append(f"file_id IN ({','.join('?' * len(file_ids))})")
        params.extend(file_ids)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(f"SELECT * FROM listings{where} ORDER BY file_id", params)

    out: list[ListingRow] = []
    for row in rows:
        listing = _row_to_listing(row)
        listing.variants.extend(get_variants(conn, listing.listing_id))
        out.append(listing)
    return out


def set_status(conn: sqlite3.Connection, listing_id: int, status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"geçersiz durum: {status}")
    conn.execute(
        "UPDATE listings SET status = ? WHERE listing_id = ?", (status, listing_id)
    )
    conn.commit()


def update_text(conn: sqlite3.Connection, listing_id: int, **fields: Any) -> None:
    """Kullanıcının elle düzenlemesi. `human_edited` işaretlenir; yeniden
    üretim bu ilanı atlar (aksi halde kullanıcının yazdığı metin sessizce
    kaybolurdu)."""
    allowed = set(_TEXT_FIELDS) | {"tags", "materials"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"düzenlenemez alan(lar): {sorted(unknown)}")

    payload: dict[str, Any] = {}
    for key, value in fields.items():
        payload[key] = (
            orjson.dumps(value).decode() if key in ("tags", "materials") else value
        )
    payload["human_edited"] = 1
    payload["edited_at"] = _now()

    sets = ", ".join(f"{k} = :{k}" for k in payload)
    payload["listing_id"] = listing_id
    conn.execute(f"UPDATE listings SET {sets} WHERE listing_id = :listing_id", payload)
    conn.commit()


def delete_listing(conn: sqlite3.Connection, listing_id: int) -> None:
    conn.execute("DELETE FROM listings WHERE listing_id = ?", (listing_id,))
    conn.commit()


# ---------------------------------------------------------------- varyantlar

def replace_variants(
    conn: sqlite3.Connection,
    listing_id: int,
    variants: list[VariantPrice],
    snapshot: dict[str, Any],
) -> int:
    """Varyantları tamamen değiştirir (fiyat yeniden hesaplaması bunu çağırır).

    `snapshot` hesapta kullanılan sabitlerin kaydı — kur değişince hangi
    fiyatın bayat kaldığı buradan anlaşılır.
    """
    conn.execute("DELETE FROM listing_variants WHERE listing_id = ?", (listing_id,))
    if variants:
        blob = orjson.dumps(snapshot).decode()
        now = _now()
        conn.executemany(
            "INSERT INTO listing_variants "
            "(listing_id, variant_key, sku, price, currency, weight_g, "
            " is_digital, pricing_snapshot, computed_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(listing_id, v.variant_key, v.sku, v.price, v.currency,
              v.weight_g, int(v.is_digital), blob, now) for v in variants],
        )
    conn.commit()
    return len(variants)


def get_variants(conn: sqlite3.Connection, listing_id: int) -> list[VariantPrice]:
    rows = conn.execute(
        "SELECT variant_key, sku, price, currency, weight_g, is_digital "
        "FROM listing_variants WHERE listing_id = ? ORDER BY variant_id",
        (listing_id,),
    ).fetchall()
    return [
        VariantPrice(
            variant_key=r["variant_key"], sku=r["sku"], price=float(r["price"]),
            currency=r["currency"], weight_g=r["weight_g"],
            is_digital=bool(r["is_digital"]),
        )
        for r in rows
    ]


def stale_priced_listings(
    conn: sqlite3.Connection, current_snapshot: dict[str, Any]
) -> list[int]:
    """Fiyatı güncel sabitlerden farklı sabitlerle hesaplanmış ilanlar."""
    blob = orjson.dumps(current_snapshot).decode()
    rows = conn.execute(
        "SELECT DISTINCT listing_id FROM listing_variants WHERE pricing_snapshot != ?",
        (blob,),
    ).fetchall()
    return [int(r[0]) for r in rows]


# ---------------------------------------------------------------- aktarımlar

def record_export(
    conn: sqlite3.Connection,
    *,
    channel: str,
    selection_id: int | None,
    file_path: str,
    row_count: int,
    skipped_count: int = 0,
) -> int:
    cur = conn.execute(
        "INSERT INTO exports (channel, selection_id, file_path, row_count, "
        "skipped_count, created_at) VALUES (?,?,?,?,?,?)",
        (channel, selection_id, file_path, row_count, skipped_count, _now()),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_exports(conn: sqlite3.Connection, limit: int = 50) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM exports ORDER BY export_id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def mark_exported(conn: sqlite3.Connection, listing_ids: list[int]) -> int:
    if not listing_ids:
        return 0
    marks = ",".join("?" * len(listing_ids))
    cur = conn.execute(
        f"UPDATE listings SET status='exported' WHERE listing_id IN ({marks})",
        listing_ids,
    )
    conn.commit()
    return int(cur.rowcount)
