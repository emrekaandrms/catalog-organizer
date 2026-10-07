from __future__ import annotations

from pathlib import Path

import pytest

from catalog_organizer.db.camera_overrides import (
    clear_camera_override,
    get_camera_override,
    get_camera_overrides,
    set_camera_override,
)
from catalog_organizer.db.connection import Database


@pytest.fixture
def db(tmp_path: Path):
    d = Database(tmp_path / "c.db")
    yield d
    d.close()


def test_set_and_get_round_trips(db):
    set_camera_override(db.conn, "A", "front", (1.0, 0.2, 0.3), (0.0, 0.0, 1.0))
    got = get_camera_override(db.conn, "A", "front")
    assert got is not None
    assert got.direction == (1.0, 0.2, 0.3)
    assert got.up == (0.0, 0.0, 1.0)


def test_missing_override_is_none(db):
    assert get_camera_override(db.conn, "NOPE", "front") is None


def test_front_and_iso_are_independent(db):
    set_camera_override(db.conn, "A", "front", (1, 0, 0), (0, 0, 1))
    set_camera_override(db.conn, "A", "iso", (0, 1, 0), (0, 0, 1))
    both = get_camera_overrides(db.conn, "A")
    assert set(both) == {"front", "iso"}
    assert both["front"].direction == (1.0, 0.0, 0.0)
    assert both["iso"].direction == (0.0, 1.0, 0.0)


def test_setting_again_overwrites_not_duplicates(db):
    set_camera_override(db.conn, "A", "front", (1, 0, 0), (0, 0, 1))
    set_camera_override(db.conn, "A", "front", (0, 1, 0), (0, 0, 1))
    assert get_camera_override(db.conn, "A", "front").direction == (0.0, 1.0, 0.0)
    n = db.conn.execute(
        "SELECT COUNT(*) FROM camera_overrides WHERE file_id='A'").fetchone()[0]
    assert n == 1


def test_rejects_unknown_view(db):
    with pytest.raises(ValueError, match="front"):
        set_camera_override(db.conn, "A", "side", (1, 0, 0), (0, 0, 1))


def test_clear_one_view_leaves_the_other(db):
    set_camera_override(db.conn, "A", "front", (1, 0, 0), (0, 0, 1))
    set_camera_override(db.conn, "A", "iso", (0, 1, 0), (0, 0, 1))
    assert clear_camera_override(db.conn, "A", "front") == 1
    remaining = get_camera_overrides(db.conn, "A")
    assert set(remaining) == {"iso"}


def test_clear_without_view_removes_everything(db):
    set_camera_override(db.conn, "A", "front", (1, 0, 0), (0, 0, 1))
    set_camera_override(db.conn, "A", "iso", (0, 1, 0), (0, 0, 1))
    assert clear_camera_override(db.conn, "A") == 2
    assert get_camera_overrides(db.conn, "A") == {}


def test_clear_on_product_with_no_overrides_is_a_noop(db):
    assert clear_camera_override(db.conn, "NOTHING") == 0


def test_overrides_are_isolated_per_product(db):
    set_camera_override(db.conn, "A", "front", (1, 0, 0), (0, 0, 1))
    set_camera_override(db.conn, "B", "front", (0, 1, 0), (0, 0, 1))
    assert get_camera_override(db.conn, "A", "front").direction == (1.0, 0.0, 0.0)
    assert get_camera_override(db.conn, "B", "front").direction == (0.0, 1.0, 0.0)
    clear_camera_override(db.conn, "A")
    assert get_camera_override(db.conn, "B", "front") is not None
