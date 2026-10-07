"""Hand-set camera angles from the Render tab's manual-orbit dialog.

`camera_axes()` in `render/product_render.py` guesses the front/iso direction
from surface geometry, and gets it wrong often enough on plain ring bands
(the bore-avoidance fix in the same session is one such case) that the user
needs a per-piece way to override it. This is that override: keyed by
file_id + view, so "front" and "iso" can each be customised independently,
and a piece with no override just keeps using the automatic guess.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

VIEWS = ("front", "iso")


@dataclass(frozen=True)
class CameraOverride:
    file_id: str
    view: str
    direction: tuple[float, float, float]
    up: tuple[float, float, float]
    zoom: float | None
    updated_at: str


def set_camera_override(
    conn: sqlite3.Connection,
    file_id: str,
    view: str,
    direction: tuple[float, float, float],
    up: tuple[float, float, float],
    zoom: float | None = None,
) -> None:
    if view not in VIEWS:
        raise ValueError(f"geçersiz görünüm: {view!r}; olmalı: {VIEWS}")
    dx, dy, dz = (float(v) for v in direction)
    ux, uy, uz = (float(v) for v in up)
    conn.execute(
        "INSERT INTO camera_overrides "
        "(file_id, view, dir_x, dir_y, dir_z, up_x, up_y, up_z, zoom, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(file_id, view) DO UPDATE SET "
        "dir_x=excluded.dir_x, dir_y=excluded.dir_y, dir_z=excluded.dir_z, "
        "up_x=excluded.up_x, up_y=excluded.up_y, up_z=excluded.up_z, "
        "zoom=excluded.zoom, updated_at=excluded.updated_at",
        (file_id, view, dx, dy, dz, ux, uy, uz, zoom,
         datetime.now(tz=timezone.utc).isoformat()),
    )
    conn.commit()


def get_camera_override(
    conn: sqlite3.Connection, file_id: str, view: str
) -> CameraOverride | None:
    row = conn.execute(
        "SELECT * FROM camera_overrides WHERE file_id = ? AND view = ?",
        (file_id, view),
    ).fetchone()
    if row is None:
        return None
    return CameraOverride(
        file_id=row["file_id"], view=row["view"],
        direction=(row["dir_x"], row["dir_y"], row["dir_z"]),
        up=(row["up_x"], row["up_y"], row["up_z"]),
        zoom=row["zoom"], updated_at=row["updated_at"],
    )


def get_camera_overrides(conn: sqlite3.Connection, file_id: str
                         ) -> dict[str, CameraOverride]:
    """{view: override} for every view this product has a custom angle for."""
    rows = conn.execute(
        "SELECT * FROM camera_overrides WHERE file_id = ?", (file_id,)
    ).fetchall()
    out: dict[str, CameraOverride] = {}
    for row in rows:
        out[row["view"]] = CameraOverride(
            file_id=row["file_id"], view=row["view"],
            direction=(row["dir_x"], row["dir_y"], row["dir_z"]),
            up=(row["up_x"], row["up_y"], row["up_z"]),
            zoom=row["zoom"], updated_at=row["updated_at"],
        )
    return out


def clear_camera_override(conn: sqlite3.Connection, file_id: str,
                          view: str | None = None) -> int:
    """Remove one view's override, or every override for this product when
    `view` is omitted. Returns how many rows were removed."""
    if view is None:
        cur = conn.execute(
            "DELETE FROM camera_overrides WHERE file_id = ?", (file_id,))
    else:
        cur = conn.execute(
            "DELETE FROM camera_overrides WHERE file_id = ? AND view = ?",
            (file_id, view))
    conn.commit()
    return int(cur.rowcount)
