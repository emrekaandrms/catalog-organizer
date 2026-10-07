"""Seçim listeleri — 1000 üründen 100'ünün ayrıldığı yer.

Bir ürün birden çok listede olabilir. Liste silinince üyelikleri de silinir
(ON DELETE CASCADE — `foreign_keys` pragma'sı açık olmalı).
"""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone


class DuplicateSelectionName(ValueError):
    """Aynı isimde ikinci liste açılamaz — kullanıcı ayırt edemez."""


@dataclass(frozen=True)
class Selection:
    selection_id: int
    name: str
    note: str | None
    item_count: int
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _touch(conn: sqlite3.Connection, selection_id: int) -> None:
    conn.execute(
        "UPDATE selections SET updated_at = ? WHERE selection_id = ?",
        (_now(), selection_id),
    )


def create_selection(
    conn: sqlite3.Connection, name: str, note: str | None = None
) -> int:
    now = _now()
    try:
        cur = conn.execute(
            "INSERT INTO selections (name, note, created_at, updated_at) "
            "VALUES (?,?,?,?)",
            (name, note, now, now),
        )
    except sqlite3.IntegrityError as exc:
        raise DuplicateSelectionName(f"'{name}' adlı bir liste zaten var") from exc
    conn.commit()
    return int(cur.lastrowid)


_SELECT = """
SELECT s.selection_id, s.name, s.note, s.created_at, s.updated_at,
       (SELECT COUNT(*) FROM selection_items i
         WHERE i.selection_id = s.selection_id) AS item_count
FROM selections s
"""


def _row_to_selection(row: sqlite3.Row) -> Selection:
    return Selection(
        selection_id=int(row["selection_id"]),
        name=row["name"],
        note=row["note"],
        item_count=int(row["item_count"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_selections(conn: sqlite3.Connection) -> list[Selection]:
    return [_row_to_selection(r) for r in conn.execute(_SELECT + " ORDER BY s.name")]


def get_selection(conn: sqlite3.Connection, selection_id: int) -> Selection | None:
    row = conn.execute(_SELECT + " WHERE s.selection_id = ?", (selection_id,)).fetchone()
    return _row_to_selection(row) if row else None


def rename_selection(conn: sqlite3.Connection, selection_id: int, name: str) -> None:
    try:
        conn.execute(
            "UPDATE selections SET name = ?, updated_at = ? WHERE selection_id = ?",
            (name, _now(), selection_id),
        )
    except sqlite3.IntegrityError as exc:
        raise DuplicateSelectionName(f"'{name}' adlı bir liste zaten var") from exc
    conn.commit()


def delete_selection(conn: sqlite3.Connection, selection_id: int) -> None:
    conn.execute("DELETE FROM selections WHERE selection_id = ?", (selection_id,))
    conn.commit()


def add_items(
    conn: sqlite3.Connection, selection_id: int, file_ids: Sequence[str]
) -> int:
    """Eklenen YENİ satır sayısını döndürür (zaten üye olanlar sayılmaz)."""
    if not file_ids:
        return 0
    start = conn.execute(
        "SELECT COALESCE(MAX(position), -1) FROM selection_items WHERE selection_id = ?",
        (selection_id,),
    ).fetchone()[0] + 1
    before = conn.execute(
        "SELECT COUNT(*) FROM selection_items WHERE selection_id = ?", (selection_id,)
    ).fetchone()[0]
    now = _now()
    conn.executemany(
        "INSERT OR IGNORE INTO selection_items "
        "(selection_id, file_id, position, added_at) VALUES (?,?,?,?)",
        [(selection_id, fid, start + i, now) for i, fid in enumerate(file_ids)],
    )
    after = conn.execute(
        "SELECT COUNT(*) FROM selection_items WHERE selection_id = ?", (selection_id,)
    ).fetchone()[0]
    _touch(conn, selection_id)
    conn.commit()
    return int(after - before)


def remove_items(
    conn: sqlite3.Connection, selection_id: int, file_ids: Sequence[str]
) -> int:
    if not file_ids:
        return 0
    marks = ",".join("?" * len(file_ids))
    cur = conn.execute(
        f"DELETE FROM selection_items WHERE selection_id = ? AND file_id IN ({marks})",
        (selection_id, *file_ids),
    )
    _touch(conn, selection_id)
    conn.commit()
    return int(cur.rowcount)


def selection_file_ids(conn: sqlite3.Connection, selection_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT file_id FROM selection_items WHERE selection_id = ? ORDER BY position",
        (selection_id,),
    ).fetchall()
    return [r[0] for r in rows]


def selections_containing(conn: sqlite3.Connection, file_id: str) -> list[int]:
    rows = conn.execute(
        "SELECT selection_id FROM selection_items WHERE file_id = ?", (file_id,)
    ).fetchall()
    return [int(r[0]) for r in rows]
