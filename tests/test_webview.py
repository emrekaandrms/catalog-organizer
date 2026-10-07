"""The web-viewer export: scene files a browser will read, built from catalogue geometry.

No GPU and no browser here: these pin what the exporter WRITES. The things worth pinning are
the ones that fail silently in a viewer -- a stone whose frame is not an orthonormal basis, a
camera on the wrong side of the piece, an occlusion value that is NaN -- because a viewer
shows all of those as "looks a bit off" rather than as an error.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from catalog_organizer.render import facets
from catalog_organizer.webview import ao, export
from catalog_organizer.webview.glb import read_json_chunk


def _box(size, offset=(0.0, 0.0, 0.0)):
    sx, sy, sz = size
    v = np.array([[0, 0, 0], [sx, 0, 0], [sx, sy, 0], [0, sy, 0],
                  [0, 0, sz], [sx, 0, sz], [sx, sy, sz], [0, sy, sz]], dtype=float)
    v = v + np.asarray(offset, dtype=float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    return v, f


def _piece():
    """A 20 x 12 x 4 slab with one real brilliant sitting on it."""
    metal = _box((20.0, 12.0, 4.0))
    sv, sf = facets._brilliant_mesh()
    stone = (sv * 2.0 + np.array([10.0, 6.0, 6.0]), sf)
    return metal, stone


def _parse(path: Path):
    doc = read_json_chunk(path)
    return doc, doc["scenes"][0]["extras"]


def test_export_writes_a_scene_a_viewer_can_read(tmp_path):
    result = export.export_product(None, "T1", tmp_path, category="pendant",
                                   label="deneme", arrays=_piece())
    glb = tmp_path / "data" / "T1" / "piece.glb"
    assert glb.exists() and result.bytes == glb.stat().st_size
    doc, extras = _parse(glb)

    # one metal body, one node per stone, each stone carrying its own frame
    roles = [n["extras"]["role"] for n in doc["nodes"]]
    assert roles == ["metal", "gem"]
    assert result.stones == 1 and result.planes > 50
    assert extras["cut"]["planes"] == result.planes

    # the metal ships the occlusion attribute the viewer's shader reads
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    assert set(attrs) == {"POSITION", "NORMAL", "_AO"}
    info = json.loads((tmp_path / "data" / "T1" / "info.json").read_text(encoding="utf-8"))
    assert info["file_id"] == "T1" and info["triangles"] == result.triangles


def test_the_camera_sits_in_front_of_the_piece_looking_down_minus_z(tmp_path):
    """Every product is turned into its own camera's frame; the viewer relies on that."""
    export.export_product(None, "T2", tmp_path, category="pendant", arrays=_piece())
    _doc, extras = _parse(tmp_path / "data" / "T2" / "piece.glb")
    pose = extras["pose"]
    x, y, z = pose["cameraPos"]
    assert (x, y) == (0.0, 0.0) and z > 0, "camera must be on +Z"
    assert pose["up"] == [0.0, 1.0, 0.0]
    assert pose["fov"] == 10.0


def test_a_stones_frame_is_an_orthonormal_basis_in_the_new_frame(tmp_path):
    """The tracer turns rays into the stone's frame with a transpose. That is only the inverse
    if the three axes are orthonormal -- rotating the piece must not have skewed them."""
    export.export_product(None, "T3", tmp_path, category="ring", arrays=_piece())
    doc, _ = _parse(tmp_path / "data" / "T3" / "piece.glb")
    stone = next(n for n in doc["nodes"] if n["extras"]["role"] == "gem")
    basis = np.asarray(stone["extras"]["frame"]["basis"])
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-4)
    assert stone["extras"]["frame"]["radius"] > 0


def test_rotating_into_the_camera_frame_preserves_where_the_stone_sits(tmp_path):
    """Distances are what a rotation must not change: the stone's centre keeps its distance
    from the piece's centre."""
    metal, stone = _piece()
    centre = 0.5 * (np.minimum(metal[0].min(0), stone[0].min(0))
                    + np.maximum(metal[0].max(0), stone[0].max(0)))
    frames = facets.stone_frames(np.asarray(stone[0]), np.asarray(stone[1]))
    want = float(np.linalg.norm(frames[0]["centre"] - centre))
    export.export_product(None, "T4", tmp_path, category="pendant", arrays=(metal, stone))
    doc, _ = _parse(tmp_path / "data" / "T4" / "piece.glb")
    got = next(n for n in doc["nodes"] if n["extras"]["role"] == "gem")["extras"]["frame"]["centre"]
    assert float(np.linalg.norm(got)) == pytest.approx(want, rel=1e-4)


def test_a_piece_over_the_triangle_budget_is_decimated_and_says_so(tmp_path):
    """The catalogue holds meshes of 1.8 million triangles; a web viewer wants tens of thousands.
    Over budget the metal is simplified, and the result records what it came from."""
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=5, radius=6.0)     # 20 480 triangles
    arrays = ((np.asarray(sphere.vertices), np.asarray(sphere.faces)), None)
    result = export.export_product(None, "T5", tmp_path, category="pendant",
                                   arrays=arrays, max_triangles=8000)
    assert result.decimated_from == len(sphere.faces)
    # the budget is on what SHIPS: splitting long edges after the simplification must not
    # push the file back over it (a decimated brooch once went 120k -> 143k)
    assert result.triangles <= 8000 * 1.1
    assert result.stones == 0

    untouched = export.export_product(None, "T6", tmp_path, category="pendant", arrays=arrays)
    assert untouched.decimated_from is None


def test_occlusion_is_open_on_a_plain_face_and_closed_in_a_slot():
    """Baked occlusion is what gives the metal its crevices. A point on an open face sees the
    sky; a point at the bottom of a narrow slot does not."""
    left = _box((10.0, 10.0, 6.0))
    right = _box((10.0, 10.0, 6.0), offset=(10.6, 0.0, 0.0))        # a 0.6 mm slot between them
    verts = np.vstack([left[0], right[0]])
    faces = np.vstack([left[1], right[1] + 8])
    open_face = np.array([[5.0, 5.0, 6.0]])
    slot_floor = np.array([[10.3, 5.0, 4.0]])
    pts = np.vstack([open_face, slot_floor])
    nrm = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    out = ao.bake_vertex_ao(verts, faces, pts, nrm, extent=21.0)
    assert out[0] > 0.9, f"an open face reads {out[0]:.2f}"
    assert out[1] < 0.7, f"the bottom of a slot reads {out[1]:.2f}"
    assert np.isfinite(out).all()


def test_a_vertex_with_no_normal_gets_no_occlusion_rather_than_nan():
    verts, faces = _box((10.0, 10.0, 4.0))
    out = ao.bake_vertex_ao(verts, faces, np.array([[5.0, 5.0, 8.0]]),
                            np.array([[0.0, 0.0, 0.0]]), extent=10.0)
    assert out[0] == 1.0


def test_one_bad_product_does_not_stop_the_rest(tmp_path):
    """export_many reports per product. A broken file in a list of sixty must not cost the
    other fifty-nine."""
    good = tmp_path / "good.stl"
    import trimesh
    trimesh.creation.box(extents=(20, 12, 4)).export(good)
    missing = tmp_path / "missing.stl"
    results, errors = export.export_many(
        [("GOOD", good, "pendant", "iyi"), ("BAD", missing, "pendant", "kotu")],
        tmp_path / "site")
    assert [r.file_id for r in results] == ["GOOD"]
    assert "BAD" in errors and errors["BAD"]
    # the gallery lists what succeeded, and only that
    listed = json.loads((tmp_path / "site" / "data" / "products.json").read_text(encoding="utf-8"))
    assert [p["file_id"] for p in listed] == ["GOOD"]
    for name in ("index.html", "viewer.html", "viewer.js"):
        assert (tmp_path / "site" / name).exists()


def test_the_shared_environments_are_written_once_at_the_size_the_viewer_expects(tmp_path):
    from catalog_organizer.webview import environment

    sizes = environment.write_shared(tmp_path)
    metal = tmp_path / "env_metal.f16"
    gem = tmp_path / "env_gem.f16"
    # RGBA half-float: 4 channels x 2 bytes
    assert metal.stat().st_size == sizes["envMetal"]["w"] * sizes["envMetal"]["h"] * 8
    assert gem.stat().st_size == sizes["envGem"]["w"] * sizes["envGem"]["h"] * 8
    data = np.frombuffer(metal.read_bytes(), dtype="<f2").reshape(-1, 4)
    assert np.isfinite(data).all() and float(data[:, :3].max()) > 0


def test_the_glb_container_is_well_formed(tmp_path):
    """Header, then a JSON chunk, then a BIN chunk, lengths consistent."""
    export.export_product(None, "T7", tmp_path, category="pendant", arrays=_piece())
    raw = (tmp_path / "data" / "T7" / "piece.glb").read_bytes()
    magic, version, total = struct.unpack("<4sII", raw[:12])
    assert (magic, version) == (b"glTF", 2) and total == len(raw)
    json_len, json_kind = struct.unpack("<I4s", raw[12:20])
    assert json_kind == b"JSON" and json_len % 4 == 0
    bin_len, bin_kind = struct.unpack("<I4s", raw[20 + json_len:28 + json_len])
    assert bin_kind == b"BIN\0" and 28 + json_len + bin_len == len(raw)


def _accessor(path: Path, index: int) -> np.ndarray:
    """Decode one accessor of a GLB (float32 / uint32 only, which is all the exporter writes)."""
    raw = path.read_bytes()
    json_len, _ = struct.unpack("<I4s", raw[12:20])
    doc = json.loads(raw[20:20 + json_len])
    bin_start = 28 + json_len
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    width = {"SCALAR": 1, "VEC3": 3}[acc["type"]]
    dtype = np.float32 if acc["componentType"] == 5126 else np.uint32
    start = bin_start + view["byteOffset"] + acc.get("byteOffset", 0)
    data = np.frombuffer(raw, dtype=dtype, count=acc["count"] * width, offset=start)
    return data.reshape(-1, width) if width > 1 else data


def test_a_big_flat_face_is_not_darkened_by_its_own_rim(tmp_path):
    """The brooch bug. Occlusion is baked per VERTEX, so a large flat face with vertices only
    on its rim inherits the rim's darkness across its whole interior: a brooch's disc, the one
    open flat area of the piece, rendered chocolate-brown against the server render's olive
    gold, with the AO values AT the disc's own vertices reading ~0.8. Long edges are split
    before the bake so the middle of a big face has vertices of its own.

    A tray: a 20 x 20 plate (two triangles) with walls 5 mm high round its edge.
    """
    from catalog_organizer.webview.pose import product_pose

    parts = [_box((20.0, 20.0, 1.0))]
    for off, size in (((0, 0, 1), (20, 1, 5)), ((0, 19, 1), (20, 1, 5)),
                      ((0, 0, 1), (1, 20, 5)), ((19, 0, 1), (1, 20, 5))):
        parts.append(_box(size, offset=off))
    verts = np.vstack([p[0] for p in parts])
    faces = np.vstack([p[1] + 8 * i for i, p in enumerate(parts)])
    arrays = ((verts, faces), None)

    export.export_product(None, "TRAY", tmp_path, category="pendant", arrays=arrays)
    glb = tmp_path / "data" / "TRAY" / "piece.glb"
    doc = read_json_chunk(glb)
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    pos = _accessor(glb, attrs["POSITION"]).astype(float)
    occlusion = _accessor(glb, attrs["_AO"])

    pose = product_pose(arrays[0], None, category="pendant", source=Path("tray.3dm"))
    middle = pose.rotation @ (np.array([10.0, 10.0, 1.0]) - pose.centre)     # plate-top centre
    near = np.linalg.norm(pos - middle, axis=1) < 1.6
    assert near.any(), "no vertex anywhere near the middle of the big face"
    assert float(occlusion[near].min()) > 0.5, (
        f"the middle of the plate reads {occlusion[near].min():.2f}: still the rim's darkness")


def test_splitting_long_edges_keeps_the_surface_and_leaves_no_long_edge():
    """The split is for the occlusion bake, not for the look: it must not move or lose any surface."""
    verts, faces = _box((40.0, 12.0, 4.0))
    area = lambda v, f: 0.5 * np.linalg.norm(                                   # noqa: E731
        np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum()
    new_v, new_f = export._split_long_edges(verts, faces, extent=40.0)
    limit = export.MAX_EDGE_FRACTION * 40.0
    tri = new_v[new_f]
    longest = np.linalg.norm(np.roll(tri, -1, axis=1) - tri, axis=2).max()
    assert longest <= limit + 1e-9
    assert area(new_v, new_f) == pytest.approx(area(verts, faces), rel=1e-9)
    assert np.allclose(new_v.min(0), verts.min(0)) and np.allclose(new_v.max(0), verts.max(0))
    # the two triangles on each side of a split edge share its midpoint
    assert len(new_v) < 3 * len(new_f) / 2


def test_a_thin_sliver_costs_far_fewer_triangles_than_splitting_every_face_in_four():
    """CAD meshes are full of long thin triangles. Splitting each 1 -> 4 (what trimesh's
    subdivide_to_size does) turned such meshes 10-80 % bigger; bisecting the longest edge cuts a
    sliver across its length only."""
    import trimesh

    verts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 0.5, 0.0]])
    faces = np.array([[0, 1, 2]])
    limit = export.MAX_EDGE_FRACTION * 100.0
    _v, bisected = export._split_long_edges(verts, faces, extent=100.0)
    _v, quartered = trimesh.remesh.subdivide_to_size(verts, faces, max_edge=limit, max_iter=8)
    assert len(bisected) < 0.5 * len(quartered), (len(bisected), len(quartered))


def test_a_vertex_buried_inside_another_body_does_not_bake_dark():
    """Catalogue models are often overlapping solids. A vertex of the inner body is 'blocked' on
    every ray and would bake to ~0, and its triangles show through the outer body's surface as
    dark patches (a necklace had 7,395 of 64,180 such vertices). Buried means neutral."""
    outer = _box((10.0, 10.0, 10.0))
    inner = _box((4.0, 4.0, 4.0), offset=(3.0, 3.0, 5.5))   # its top is 0.5 mm under the outer top
    verts = np.vstack([outer[0], inner[0]])
    faces = np.vstack([outer[1], inner[1] + 8])
    inner_corner = np.array([[5.0, 5.0, 9.5]])                       # inside the outer box
    outer_face = np.array([[5.0, 5.0, 10.0]])                        # open sky above the top
    pts = np.vstack([inner_corner, outer_face])
    nrm = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    out = ao.bake_vertex_ao(verts, faces, pts, nrm, extent=10.0)
    assert out[0] == 1.0, f"a buried vertex reads {out[0]:.2f}"
    assert out[1] > 0.8, f"an exposed face reads {out[1]:.2f}"
