"""Per-product render choices the user makes by hand.

Today one: whether to put stones in the seats of a stoneless STL (`webview.seats`). An STL cannot
say whether its holes were meant for stones, so the program must not decide alone: the product's
own record (its analysis says "stone" or not) is the default, and the user's explicit choice,
stored here, always wins. The Render tab sets it; the PDF catalogue reads the same answer, so a
picture and the printed page never disagree about whether the piece has stones.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def get_place_stones(conn: sqlite3.Connection, file_id: str) -> bool | None:
    """The user's explicit choice, or None if they never made one."""
    row = conn.execute("SELECT place_stones FROM render_options WHERE file_id = ?",
                       (file_id,)).fetchone()
    return None if row is None else bool(row[0])


def set_place_stones(conn: sqlite3.Connection, file_id: str, value: bool) -> None:
    conn.execute(
        "INSERT INTO render_options (file_id, place_stones, updated_at) VALUES (?,?,?) "
        "ON CONFLICT(file_id) DO UPDATE SET place_stones = excluded.place_stones, "
        "updated_at = excluded.updated_at",
        (file_id, 1 if value else 0, datetime.now(tz=timezone.utc).isoformat()))
    conn.commit()


def clear_place_stones(conn: sqlite3.Connection, file_id: str) -> None:
    conn.execute("DELETE FROM render_options WHERE file_id = ?", (file_id,))
    conn.commit()


def wants_placed_stones(conn: sqlite3.Connection | None, record) -> bool:
    """Should this product's render get stones put in its seats?

    The user's choice if there is one; otherwise only when the product's own record says it has
    stones. A product recorded as stoneless ("polished") never gets stones unless the user asks.
    """
    if conn is not None:
        explicit = get_place_stones(conn, record.file_id)
        if explicit is not None:
            return explicit
    return record.stone_summary.status == "stone"
