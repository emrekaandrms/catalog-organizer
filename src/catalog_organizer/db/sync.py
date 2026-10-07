"""JSONL → SQLite senkronu.

`products` JSONL'in aynasıdır: her senkronda üzerine yazılır. Kullanıcının
elle düzeltmeleri `product_overrides`'ta ayrı durur ve bu modül ona ASLA
dokunmaz.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import orjson

from catalog_organizer.catalog.writer import read_all_records
from catalog_organizer.core.paths import data_dir
from catalog_organizer.core.schemas import CatalogRecord
from catalog_organizer.db.text import normalise_for_index


@dataclass(frozen=True)
class SyncStats:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    total: int = 0


def build_search_text(rec: CatalogRecord) -> str:
    return " ".join((
        *rec.controlled_tags,
        rec.rich_description or "",
        rec.main_category or "",
        rec.subcategory or "",
        rec.brand or "",
        rec.file_id,
    ))


def record_to_row(rec: CatalogRecord) -> dict[str, object]:
    mw = rec.metal_weights
    stones = rec.stone_summary
    n_stones = sum(s.quantity for s in stones.side_stones)
    if stones.center_stone is not None:
        n_stones += stones.center_stone.quantity
    return {
        "file_id": rec.file_id,
        "source_path": rec.source_path,
        "file_extension": rec.file_extension,
        "sha256": rec.sha256,
        "main_category": rec.main_category,
        "subcategory": rec.subcategory,
        "brand": rec.brand,
        "controlled_tags": orjson.dumps(rec.controlled_tags).decode(),
        "rich_description": rec.rich_description,
        "stone_status": stones.status,
        "total_carat": stones.total_estimated_carat,
        "stone_count": n_stones,
        "sprue_detected": int(rec.sprue.detected),
        "sprue_volume_mm3": rec.sprue.estimated_volume_mm3,
        "silver_925_g": mw.silver_925_g if mw else None,
        "gold_10k_yellow_g": mw.gold_10k_yellow_g if mw else None,
        "gold_14k_yellow_g": mw.gold_14k_yellow_g if mw else None,
        "gold_18k_yellow_g": mw.gold_18k_yellow_g if mw else None,
        "platinum_g": mw.platinum_g if mw else None,
        "bbox_width_mm": rec.measurements.bbox_width_mm,
        "bbox_height_mm": rec.measurements.bbox_height_mm,
        "bbox_depth_mm": rec.measurements.bbox_depth_mm,
        "volume_mm3": rec.measurements.volume_mm3,
        "design_complete": int(rec.design_complete),
        "sellability": rec.sellability,
        "state": rec.state,
        "needs_manual_review": int(rec.needs_manual_review),
        "snapshot_dir": (
            str(Path(rec.snapshot_paths[0]).parent) if rec.snapshot_paths else None
        ),
        "processed_at": rec.processed_at.isoformat(),
        "record_json": orjson.dumps(rec.model_dump(mode="json")).decode(),
    }


_COLUMNS = (
    "file_id", "source_path", "file_extension", "sha256", "main_category",
    "subcategory", "brand", "controlled_tags", "rich_description",
    "stone_status", "total_carat", "stone_count", "sprue_detected",
    "sprue_volume_mm3", "silver_925_g", "gold_10k_yellow_g",
    "gold_14k_yellow_g", "gold_18k_yellow_g", "platinum_g", "bbox_width_mm",
    "bbox_height_mm", "bbox_depth_mm", "volume_mm3", "design_complete",
    "sellability", "state", "needs_manual_review", "snapshot_dir",
    "processed_at", "record_json", "synced_at",
)

_UPSERT = (
    f"INSERT INTO products ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join(':' + c for c in _COLUMNS)}) "
    f"ON CONFLICT(file_id) DO UPDATE SET "
    + ", ".join(f"{c}=excluded.{c}" for c in _COLUMNS if c != "file_id")
)


def refresh_fts(conn: sqlite3.Connection, file_ids: list[str] | None = None) -> int:
    """Verilen ürünlerin FTS satırlarını yeniden yazar (None ise hepsini).

    FTS5 sanal tablosunda UPDATE yok — eski satır silinip yenisi yazılır.
    Yapılmazsa yeniden işlenen üründe eski açıklama indekste kalır.
    """
    if file_ids is None:
        rows = conn.execute("SELECT file_id, record_json FROM products").fetchall()
    else:
        if not file_ids:
            return 0
        marks = ",".join("?" * len(file_ids))
        rows = conn.execute(
            f"SELECT file_id, record_json FROM products WHERE file_id IN ({marks})",
            file_ids,
        ).fetchall()

    payload: list[tuple[str, str]] = []
    for row in rows:
        rec = CatalogRecord.model_validate(orjson.loads(row["record_json"]))
        payload.append((row["file_id"], normalise_for_index(build_search_text(rec))))

    if payload:
        ids = [p[0] for p in payload]
        marks = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM products_fts WHERE file_id IN ({marks})", ids)
        conn.executemany(
            "INSERT INTO products_fts (file_id, search_text) VALUES (?, ?)", payload
        )
        conn.commit()
    return len(payload)


def sync_from_jsonl(
    conn: sqlite3.Connection,
    jsonl_path: Path | None = None,
) -> SyncStats:
    path = Path(jsonl_path) if jsonl_path else (data_dir() / "catalog_master.jsonl")
    records = read_all_records(path)
    if not records:
        return SyncStats()

    # Aynı file_id birden çok satırda olabilir (append-only); sonuncusu kazanır.
    latest: dict[str, CatalogRecord] = {rec.file_id: rec for rec in records}

    existing = {
        r["file_id"]: r["processed_at"]
        for r in conn.execute("SELECT file_id, processed_at FROM products")
    }
    now = datetime.now(tz=timezone.utc).isoformat()

    inserted = updated = unchanged = 0
    rows: list[dict[str, object]] = []
    for file_id, rec in latest.items():
        row = record_to_row(rec)
        prev = existing.get(file_id)
        if prev is None:
            inserted += 1
        elif prev == row["processed_at"]:
            unchanged += 1
            continue
        else:
            updated += 1
        row["synced_at"] = now
        rows.append(row)

    if rows:
        conn.executemany(_UPSERT, rows)
        conn.commit()
        refresh_fts(conn, [str(r["file_id"]) for r in rows])

    return SyncStats(
        inserted=inserted, updated=updated, unchanged=unchanged, total=len(latest)
    )
