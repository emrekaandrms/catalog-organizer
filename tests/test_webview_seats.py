"""Stones placed in the seats of a stoneless casting model.

A casting-model STL has no stones, only the seats cut for them. Every filter in `webview.seats`
exists because of a measured false positive on the catalogue's own files (a ring's finger bore read
as a 7.9 mm seat; pockets in a sculpted pendant's cloth read as seats; the detector once measured
seat centres in the frame of each slice), so each is pinned here.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from catalog_organizer.webview import seats


def _candidate(**kw):
    base = dict(entrance=np.array([1.0, 2.0, 10.0]), axis=np.array([0.0, 0.0, 1.0]),
                diameter_mm=3.0, circularity=0.97, n_slices=12, depth_mm=1.2, taper=0.3,
                open_above=True, on_metal=True)
    base.update(kw)
    return seats.Candidate(**base)


def _placed(centre=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), diameter_mm=3.0):
    return seats.PlacedSeat(np.array(centre, dtype=float), np.array(axis, dtype=float), diameter_mm, 1.0)


# ------------------------------------------------------------------ what is and is not a seat

def test_a_round_tapering_open_seat_takes_a_stone():
    assert seats.accept(_candidate())


@pytest.mark.parametrize("why, candidate", [
    ("a ring's finger bore: far deeper than it is wide", _candidate(diameter_mm=7.9, depth_mm=19.6)),
    ("a film, not a seat", _candidate(depth_mm=0.01)),
    ("an irregular pocket (relief in cloth), not a round seat", _candidate(circularity=0.7)),
    ("a straight tube does not narrow like a cone", _candidate(taper=1.0)),
    ("too few slices to be sure it is a cone", _candidate(n_slices=2)),
    ("a hole drilled from inside a band, covered on top", _candidate(open_above=False)),
    ("a hole with no metal near it: a measurement in the wrong place", _candidate(on_metal=False)),
])
def test_what_is_not_a_seat_gets_no_stone(why, candidate):
    assert not seats.accept(candidate), why


def test_a_seat_seen_from_two_axes_is_one_stone():
    twice = [_candidate(entrance=np.array([5.0, 5.0, 5.0]), circularity=0.90),
             _candidate(entrance=np.array([5.2, 5.0, 5.0]), axis=np.array([1.0, 0.0, 0.0]), circularity=0.99)]
    chosen = seats.choose(twice)
    assert len(chosen) == 1 and chosen[0].circularity == 0.99, "the rounder reading wins"
    apart = [_candidate(entrance=np.array([5.0, 5.0, 5.0])), _candidate(entrance=np.array([9.0, 5.0, 5.0]))]
    assert len(seats.choose(apart)) == 2


# ------------------------------------------------------------------ the stone itself

def test_a_stone_stands_on_its_seat_with_the_girdle_at_the_entrance():
    verts, _faces = seats.stones_for([_placed(centre=(3.0, 4.0, 10.0), diameter_mm=6.0)])
    girdle = verts[np.abs(verts[:, 2] - 10.0) <= 0.12]       # the girdle band: 3 % of the diameter thick
    assert np.isclose(np.hypot(girdle[:, 0] - 3.0, girdle[:, 1] - 4.0).max(), 3.0, atol=0.05)
    assert verts[:, 2].max() > 10.0 + 0.9, "the crown stands above the metal"
    assert verts[:, 2].min() < 10.0 - 2.4, "the pavilion goes down into the seat"
    assert np.allclose([verts[:, 0].mean(), verts[:, 1].mean()], [3.0, 4.0], atol=0.2)


def test_a_stone_on_a_sideways_seat_leans_the_way_the_seat_opens():
    verts, _ = seats.stones_for([_placed(centre=(10.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), diameter_mm=6.0)])
    assert verts[:, 0].max() > 10.0 + 0.9, "the crown stands out of the face the seat is in"
    assert verts[:, 0].min() < 10.0 - 2.4, "the pavilion goes into the metal"
    assert np.allclose([verts[:, 1].mean(), verts[:, 2].mean()], [0.0, 0.0], atol=0.2)


def test_no_seats_no_stones():
    assert seats.stones_for([]) is None


# ------------------------------------------------------------------ real geometry end to end

def _slab_with_cone_seat():
    """20 x 20 x 6 slab with a cone pocket (6 mm across at the top, tapering) cut from the top."""
    slab = trimesh.creation.box(extents=(20, 20, 6))
    slab.apply_translation((0, 0, 3))                                   # top face at z = 6
    cutter = trimesh.creation.cone(radius=3.2, height=3.4, sections=64)   # base at z=0, apex at +z
    cutter.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (1, 0, 0)))   # apex down
    cutter.apply_translation((0, 0, 6.4))                               # base above the top, apex at 3.0
    return trimesh.boolean.difference([slab, cutter], engine="manifold")


def _arrays(mesh):
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def test_the_detector_finds_a_real_cone_seat_and_a_stone_is_placed_in_it():
    metal = _slab_with_cone_seat()
    found = seats.detect(_arrays(metal))
    assert len(found) == 1, "one seat, found once"
    assert 5.3 <= found[0].diameter_mm <= 6.0, found[0].diameter_mm
    assert np.allclose(found[0].centre[:2], (0.0, 0.0), atol=0.1)
    assert found[0].centre[2] == pytest.approx(6.0, abs=0.25)
    assert np.allclose(found[0].axis, (0.0, 0.0, 1.0))
    verts, _faces = seats.seat_stones(_arrays(metal))
    assert verts[:, 2].max() > 6.0, "the crown stands above the slab"


def test_a_straight_hole_through_a_slab_is_not_a_seat():
    slab = trimesh.creation.box(extents=(20, 20, 6))
    hole = trimesh.creation.cylinder(radius=3.0, height=10, sections=64)
    metal = trimesh.boolean.difference([slab, hole], engine="manifold")
    assert seats.seat_stones(_arrays(metal)) is None


def test_a_piece_away_from_the_origin_gets_its_seat_where_it_really_is():
    """Regression. The detector read each seat's centre in the 2D frame of its own slice, which
    `to_planar` centres on the section: a piece 50 mm from the origin reported its seats near (0, 0)
    and a stone was placed in mid-air. Only a piece centred on the origin came out right by luck."""
    metal = _slab_with_cone_seat()
    metal.apply_translation((37.0, 52.0, -4.0))
    found = seats.detect(_arrays(metal))
    assert len(found) == 1
    assert np.allclose(found[0].centre, (37.0, 52.0, 2.0), atol=0.25), found[0].centre


def test_a_seat_with_no_metal_near_it_gets_no_stone(monkeypatch):
    """Belt and braces for the frame bug above: whatever the detector says, a stone is only placed
    where the metal really is."""
    from catalog_organizer.cad import stl_stone_seats

    metal = _slab_with_cone_seat()
    real = stl_stone_seats.find_stone_seats(metal)
    assert real, "the fixture must have a seat to move"
    moved = SimpleNamespace(**{**real[0].__dict__, "centre_xy": (500.0, 500.0)})
    monkeypatch.setattr(stl_stone_seats, "find_stone_seats", lambda mesh, **kw: [moved])
    assert seats.detect(_arrays(metal)) == []


def test_a_seat_that_opens_sideways_is_found_and_stoned_along_its_own_axis():
    """A ring's side or a pendant's edge: the seat opens toward +X, not +Z."""
    metal = _slab_with_cone_seat()
    metal.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))   # +Z -> +X
    found = seats.detect(_arrays(metal))
    assert len(found) == 1
    assert np.allclose(found[0].axis, (1.0, 0.0, 0.0))
    assert found[0].centre[0] == pytest.approx(6.0, abs=0.25)
    verts, _ = seats.seat_stones(_arrays(metal))
    assert verts[:, 0].max() > 6.0, "the crown stands out of the +X face"


def test_a_seat_on_the_underside_is_found_too():
    metal = _slab_with_cone_seat()
    metal.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (1, 0, 0)))      # opens toward -Z
    found = seats.detect(_arrays(metal))
    assert len(found) == 1 and np.allclose(found[0].axis, (0.0, 0.0, -1.0))


def test_a_hole_covered_from_above_is_not_a_seat():
    """The gallery holes of a perforated band: the same cone, but a roof of metal over it."""
    metal = _slab_with_cone_seat()
    roof = trimesh.creation.box(extents=(20, 20, 2))
    roof.apply_translation((0, 0, 7.5))                                  # a plate 0.5 mm above the opening
    both = trimesh.util.concatenate([metal, roof])
    found = [s for s in seats.detect(_arrays(both)) if np.allclose(s.axis, (0, 0, 1))]
    assert found == []


# ------------------------------------------------------------------ the scene

def test_a_stoneless_stl_gets_its_stones_in_the_scene_and_a_3dm_does_not(tmp_path, monkeypatch):
    from catalog_organizer.webview import export, scene_cache

    monkeypatch.setattr(scene_cache, "cache_dir", lambda: tmp_path)
    arrays = (_arrays(_slab_with_cone_seat()), None)

    stl = export.export_product(Path("cast.stl"), "SEAT_STL", tmp_path, category="pendant", arrays=arrays,
                                place_stones=True)
    assert stl.placed_stones and stl.stones == 1

    # a .3dm with no gem layer has no seats to read this way: nothing is invented
    dm = export.export_product(Path("cast.3dm"), "SEAT_3DM", tmp_path, category="pendant", arrays=arrays,
                               place_stones=True)
    assert not dm.placed_stones and dm.stones == 0


def test_the_scene_says_when_its_stones_were_placed(tmp_path):
    from catalog_organizer.webview import export
    from catalog_organizer.webview.glb import read_json_chunk

    export.export_product(Path("cast.stl"), "SEAT_FLAG", tmp_path, category="pendant",
                          arrays=(_arrays(_slab_with_cone_seat()), None), place_stones=True)
    extras = read_json_chunk(tmp_path / "data" / "SEAT_FLAG" / "piece.glb")["scenes"][0]["extras"]
    assert extras["placedStones"] is True


def test_a_scan_is_remembered_per_source_file(tmp_path, monkeypatch):
    """The scan is the slow part; a scene rebuilt for another reason must not repeat it."""
    from catalog_organizer.core import paths

    monkeypatch.setattr(paths, "cache_dir", lambda: tmp_path)
    source = tmp_path / "piece.stl"
    source.write_bytes(b"x")
    metal = _arrays(_slab_with_cone_seat())
    first = seats.scan_cached(source, metal)
    assert first

    def boom(*a, **k):
        raise AssertionError("scanned again")

    monkeypatch.setattr(seats, "scan", boom)
    again = seats.scan_cached(source, metal)
    assert len(again) == len(first)
    assert np.allclose(again[0].entrance, first[0].entrance)
    source.write_bytes(b"changed")                      # the CAD file was saved again
    with pytest.raises(AssertionError):
        seats.scan_cached(source, metal)
