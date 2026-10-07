"""Whether stones are put in the seats of a stoneless STL is the user's call, not the program's.

An STL cannot say whether its holes were meant for stones. The first version placed stones
whenever the scan found seats, and gave a stone to a cocktail ring and a tennis bracelet that the
catalogue records as stoneless ("polished"). The product's own record is now the default and the
user's explicit choice always wins; the Render tab and the PDF catalogue read the same answer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catalog_organizer.core.schemas import StoneSummary
from catalog_organizer.db.connection import Database
from catalog_organizer.db.render_options import (
    clear_place_stones, get_place_stones, set_place_stones, wants_placed_stones,
)


def _record(record_factory, status, file_id="P1"):
    return record_factory(file_id=file_id, stone_summary=StoneSummary(
        status=status, center_stone=None, side_stones=[], total_estimated_carat=0.0))


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "c.db")
    yield database
    database.close()


def test_with_no_choice_the_products_own_record_decides(db, record_factory):
    assert wants_placed_stones(db.conn, _record(record_factory, "stone"))
    assert not wants_placed_stones(db.conn, _record(record_factory, "polished"))
    assert not wants_placed_stones(db.conn, _record(record_factory, "unclear"))


def test_the_users_choice_beats_the_record_both_ways(db, record_factory):
    stoneless = _record(record_factory, "polished", "A")
    stoned = _record(record_factory, "stone", "B")
    set_place_stones(db.conn, "A", True)
    set_place_stones(db.conn, "B", False)
    assert wants_placed_stones(db.conn, stoneless)
    assert not wants_placed_stones(db.conn, stoned)
    clear_place_stones(db.conn, "A")
    assert get_place_stones(db.conn, "A") is None
    assert not wants_placed_stones(db.conn, stoneless), "back to what the record says"


def test_a_choice_is_stored_and_changed_in_place(db):
    assert get_place_stones(db.conn, "X") is None
    set_place_stones(db.conn, "X", True)
    set_place_stones(db.conn, "X", False)
    assert get_place_stones(db.conn, "X") is False
    assert db.conn.execute("SELECT COUNT(*) FROM render_options").fetchone()[0] == 1


def test_without_a_database_only_the_record_speaks(record_factory):
    assert wants_placed_stones(None, _record(record_factory, "stone"))
    assert not wants_placed_stones(None, _record(record_factory, "polished"))


# ------------------------------------------------------------------ the scene

def _slab_with_cone_seat():
    import trimesh
    slab = trimesh.creation.box(extents=(20, 20, 6))
    slab.apply_translation((0, 0, 3))
    cutter = trimesh.creation.cone(radius=3.2, height=3.4, sections=64)
    cutter.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (1, 0, 0)))
    cutter.apply_translation((0, 0, 6.4))
    mesh = trimesh.boolean.difference([slab, cutter], engine="manifold")
    return (np.asarray(mesh.vertices), np.asarray(mesh.faces)), None


def test_an_stl_gets_no_stones_unless_they_are_asked_for(tmp_path):
    from catalog_organizer.webview import export

    arrays = _slab_with_cone_seat()
    asked = export.export_product(Path("c.stl"), "ASKED", tmp_path, category="pendant", arrays=arrays,
                                  place_stones=True)
    default = export.export_product(Path("c.stl"), "DEFAULT", tmp_path, category="pendant", arrays=arrays)
    assert asked.placed_stones and asked.stones == 1
    assert not default.placed_stones and default.stones == 0, "placement is a request, never a default"


def test_a_scene_with_stones_and_one_without_do_not_share_a_cache_slot(tmp_path, monkeypatch):
    from catalog_organizer.webview import scene_cache

    monkeypatch.setattr(scene_cache, "cache_dir", lambda: tmp_path)
    arrays = _slab_with_cone_seat()
    plain = scene_cache.ensure_scene(Path("c.stl"), "SLOT", arrays=arrays)
    placed = scene_cache.ensure_scene(Path("c.stl"), "SLOT", arrays=arrays, place_stones=True)
    again = scene_cache.ensure_scene(Path("c.stl"), "SLOT", arrays=arrays, place_stones=True)
    assert plain.stones == 0 and not plain.placed
    assert placed.stones == 1 and placed.placed and not placed.from_cache
    assert again.from_cache


# ------------------------------------------------------------------ the catalogue PDF

def test_the_pdf_asks_for_stones_only_for_products_that_have_them(tmp_path, monkeypatch, qtbot, record_factory, db):
    """The picture in the Render tab and the printed page must agree."""
    import catalog_organizer.render.engine as chooser
    from catalog_organizer.export import catalog_pdf
    from catalog_organizer.render.product_render import RenderResult

    asked: dict[str, bool] = {}

    def fake(source, *, file_id, metal_key, stone_key, **kw):
        asked[file_id] = kw["place_stones"]
        return RenderResult(file_id, metal_key, stone_key, {}, False, "web")

    monkeypatch.setattr(chooser, "load_and_render", fake)
    recs = []
    for fid, status in (("WITH", "stone"), ("WITHOUT", "polished"), ("FORCED", "polished"), ("HELD", "stone")):
        source = tmp_path / f"{fid}.stl"
        source.write_bytes(b"x")
        recs.append(record_factory(file_id=fid, source_path=str(source), stone_summary=StoneSummary(
            status=status, center_stone=None, side_stones=[], total_estimated_carat=0.0)))
    set_place_stones(db.conn, "FORCED", True)
    set_place_stones(db.conn, "HELD", False)

    catalog_pdf.render_and_export(recs, tmp_path / "k.pdf", metal_key="yellow_gold", stone_key="white",
                                  title="t", db_conn=db.conn)
    assert asked == {"WITH": True, "WITHOUT": False, "FORCED": True, "HELD": False}
