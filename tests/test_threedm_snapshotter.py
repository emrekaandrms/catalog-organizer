from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from catalog_organizer.snapshotter import threedm as threedm_mod

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch):
    def fake_cache_dir() -> Path:
        d = tmp_path / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr("catalog_organizer.snapshotter.threedm.snapshots_dir",
                        lambda file_id: fake_cache_dir() / "snapshots" / file_id)
    monkeypatch.setattr("catalog_organizer.snapshotter.threedm.thumbnails_dir",
                        lambda: fake_cache_dir() / "thumbnails")
    return tmp_path


def test_snapshot_3dm_produces_4_pngs_and_thumbnail(isolated_cache: Path):
    results = threedm_mod.snapshot_3dm(
        FIXTURES / "sample_ring.3dm",
        file_id="JCAD-3DM-0001",
        resolution=256,
        thumbnail_size=128,
    )
    for view in ("front", "side", "top", "iso"):
        assert results[view].exists()
    assert results["thumbnail"].exists()
    assert results["thumbnail"].suffix == ".webp"


def test_snapshot_3dm_render_not_blank(isolated_cache: Path):
    results = threedm_mod.snapshot_3dm(
        FIXTURES / "sample_pendant.3dm",
        file_id="JCAD-3DM-0002",
        resolution=256,
        thumbnail_size=128,
    )
    img = Image.open(results["iso"]).convert("RGB")
    non_white = sum(1 for r, g, b in img.getdata() if (r, g, b) != (255, 255, 255))
    assert non_white > 100


def test_snapshot_3dm_raises_on_empty_file(isolated_cache: Path, tmp_path: Path):
    empty = tmp_path / "empty.3dm"
    empty.write_bytes(b"not a real 3dm file")
    with pytest.raises(ValueError):
        threedm_mod.snapshot_3dm(empty, file_id="JCAD-3DM-BAD", resolution=128)


_TEST4_DIR = Path(__file__).resolve().parents[1] / "test4"
_TEST_FILES2_DIR = Path(__file__).resolve().parents[1] / "samples"


def test_metal_volume_confidence_low_when_creation_curves_contribute():
    """When metal_volume_mm3()'s Creation-Curves reinstatement pass (pass 2)
    actually adds volume, confidence must downgrade to "low" — the real-vs-
    construction-junk classification for that reinstated geometry is an
    unresolved judgment call (the accuracy-remediation notes, item 2, 2026-06-
    30: two independent geometric signals were tried and both failed to
    generalise across real files), so the uncertainty must be surfaced
    rather than reported as a confident "medium" alongside a clean file.

    Real-file regression (skips if the workshop fixtures aren't present on
    this machine, same pattern as the golden-file tests): sampleB_15_21.3dm
    and sampleC_14.3dm are known to have real Creation-Curves
    contribution; sampleC_13.3dm has none.
    """
    from catalog_organizer.snapshotter.threedm import metal_volume_mm3

    sampleB = _TEST4_DIR / "sampleB_15_21.3dm"
    sampleC13 = _TEST4_DIR / "sampleC_13.3dm"
    sampleC14 = _TEST4_DIR / "sampleC_14.3dm"
    if not (sampleB.exists() and sampleC13.exists() and sampleC14.exists()):
        pytest.skip("samples/ workshop fixtures not present on this machine")

    _, conf_sampleB = metal_volume_mm3(sampleB)
    _, conf_sampleC13 = metal_volume_mm3(sampleC13)
    _, conf_sampleC14 = metal_volume_mm3(sampleC14)

    assert conf_sampleB == "low"
    assert conf_sampleC13 == "medium"
    assert conf_sampleC14 == "low"


def test_is_cutter_layer_name_matches_cutting_objects():
    """Cutter-layer detection must fire for MatrixGold's 'Cutting Objects'
    layer (the canonical name) plus the generic 'Cutter*' variants, but
    NOT for Heads / Metal / Gem layers."""
    from catalog_organizer.snapshotter.threedm import _is_cutter_layer_name
    for name in ("Cutting Objects", "Cutters", "MainCutter", "stone_cutter_01"):
        assert _is_cutter_layer_name(name), f"expected cutter: {name!r}"
    for name in ("Metal 01", "Heads", "Gem 02", "User Layer 01", "Lights"):
        assert not _is_cutter_layer_name(name), f"not a cutter: {name!r}"


def test_is_non_metal_layer_name_matches_matrixgold_conventions():
    """Pin the layer-name substrings the metal-only loader strips.
    Calibrated against real MatrixGold exports (samples/*.3dm,
    2026-05-19). Heads (stone-setting prongs) are intentionally NOT
    stripped because they're physically part of the metal piece."""
    from catalog_organizer.snapshotter.threedm import _is_non_metal_layer_name

    # MUST be flagged as non-metal:
    for name in (
        "Gem", "Gem 01", "Gem 02", "Gem 03", "Gem 04",
        "Cutting Objects",
        "Finger Sizes",
        "Creation Curves",
        "Lights",
        "stone seats",
        "MainCutter",
    ):
        assert _is_non_metal_layer_name(name), f"expected {name!r} non-metal"

    # MUST stay in (real metal layers from sampled files):
    for name in (
        "Metal 01", "Metal 03", "Metal 04",
        "User Layer 01", "User Layer 02", "User Layer 17",
        "Heads",
        "Default",
    ):
        assert not _is_non_metal_layer_name(name), f"expected {name!r} metal"


def test_brep_extraction_uses_render_mesh_not_bbox():
    """Regression: when a Brep has cached render meshes, _brep_to_arrays must
    produce many more triangles than the 12-triangle bbox fallback.

    A 3DM file with NURBS geometry (typical Rhino/MatrixGold output) was rendered
    as 12 bounding-box triangles before this fix, which destroyed VLM accuracy.
    """
    import rhino3dm as r3
    fixture = FIXTURES / "sample_ring.3dm"
    f3 = r3.File3dm.Read(str(fixture))
    breps = [o.Geometry for o in f3.Objects if isinstance(o.Geometry, r3.Brep)]
    if not breps:
        pytest.skip("fixture has no Brep geometry")
    verts, faces = threedm_mod._brep_to_arrays(breps[0])
    # Real render mesh: should have substantially more than 8 verts / 12 faces.
    # Bbox fallback would return exactly (8, 12).
    assert faces.shape[0] > 12 or verts.shape[0] > 8, (
        f"Brep extraction fell back to bbox: {verts.shape[0]} verts, {faces.shape[0]} faces"
    )


def _make_cube_mesh(mn: tuple, mx: tuple):
    import rhino3dm as r3
    m = r3.Mesh()
    x0, y0, z0 = mn
    x1, y1, z1 = mx
    for p in ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
              (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)):
        m.Vertices.Add(*p)
    for f in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        m.Faces.AddFace(*f)
    return m


def test_load_3dm_as_trimesh_arrays_drops_floating_oversized_gem_proxy(tmp_path: Path):
    """Regression (2026-07-24, reported directly by the user against a real
    68-file pilot batch — PilotBatch/): a leftover placeholder "Gem"
    object left in the file by MatrixGold/the designer — floating far from
    the metal and/or grossly oversized in some axis — rendered as a giant
    faceted blob dead-center of the piece in the VLM snapshot, visually
    obscuring the actual design (a letter pendant became unrecognisable;
    confirmed on 8 real files across 4 different designer folders plus
    3 historical golden files). cad/stone_extraction.py already proved and
    filtered this same class of object out of the stone COUNT via a
    distance-to-metal + per-axis-size check (the accuracy-remediation notes
    item 1); load_3dm_as_trimesh_arrays(only_metal=False) must apply the
    same filter to the RENDER, while still keeping a real, correctly
    positioned small gem untouched.
    """
    import rhino3dm as r3

    f3 = r3.File3dm()
    li_metal = f3.Layers.AddLayer("Metal 01", (200, 200, 200, 255))
    li_gem = f3.Layers.AddLayer("Gem", (255, 255, 255, 255))

    attrs_metal = r3.ObjectAttributes()
    attrs_metal.LayerIndex = li_metal
    f3.Objects.AddMesh(_make_cube_mesh((0, 0, 0), (5, 5, 5)), attrs_metal)

    # Real stone: small, flush against the metal (a genuine prong-set gem).
    attrs_real_gem = r3.ObjectAttributes()
    attrs_real_gem.LayerIndex = li_gem
    f3.Objects.AddMesh(_make_cube_mesh((5, 0, 0), (6, 1, 1)), attrs_real_gem)

    # Bogus proxy: floats far away AND is grossly oversized vs the piece.
    attrs_proxy = r3.ObjectAttributes()
    attrs_proxy.LayerIndex = li_gem
    f3.Objects.AddMesh(_make_cube_mesh((100, 100, 100), (120, 120, 102)), attrs_proxy)

    path = tmp_path / "synthetic_proxy_gem.3dm"
    f3.Write(str(path), 7)

    verts, _faces = threedm_mod.load_3dm_as_trimesh_arrays(path, only_metal=False)
    # 8 verts (metal cube) + 8 verts (real gem cube) = 16; proxy's 8 verts dropped.
    assert verts.shape[0] == 16, f"expected proxy gem dropped, got {verts.shape[0]} verts"
    assert verts.max() < 100, "proxy gem geometry (at x/y/z=100-120) leaked into the render"


# ------------------------------------------------- ağırlık: çift sayım

def _box(size, offset=(0.0, 0.0, 0.0)):
    import numpy as np
    sx, sy, sz = size
    v = np.array([[0, 0, 0], [sx, 0, 0], [sx, sy, 0], [0, sy, 0],
                  [0, 0, sz], [sx, 0, sz], [sx, sy, sz], [0, sy, sz]],
                 dtype=float) + np.asarray(offset, dtype=float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                  [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    return v, f


def test_overlapping_parts_are_weighed_once_not_twice():
    """The "gramlar %20-25 yüksek çıkıyor" report, at its root.

    A cast piece weighs the volume of the UNION of its parts. Jewellery CAD
    overlaps constantly — a prong sits inside the head it grows from, a bezel
    wall runs into the shank — and summing per-object volumes counts every one
    of those overlaps twice. Measured over the workshop files the sum runs 0.7
    to 24.7 % above the union, worst on exactly the dense, many-pronged pieces.

    Two 10 mm cubes sharing a 5 mm slab: sum 2000 mm³, union 1500 mm³.
    """
    import trimesh
    from catalog_organizer.snapshotter.threedm import _union_volume_mm3

    a = trimesh.Trimesh(*_box((10.0, 10.0, 10.0)), process=False)
    b = trimesh.Trimesh(*_box((10.0, 10.0, 10.0), (5.0, 0.0, 0.0)),
                        process=False)
    assert abs(abs(a.volume) + abs(b.volume) - 2000.0) < 1.0
    union = _union_volume_mm3([a, b])
    assert union is not None
    assert abs(union - 1500.0) < 1.0, f"union came out {union:.1f} mm³"


def test_a_brep_face_soup_is_welded_before_it_is_unioned():
    """Rhino writes a separate mesh per Brep face, so nothing arrives
    watertight: vertices along a shared edge are duplicated and the windings
    between faces disagree. Without welding the union engine refuses the file
    outright ("Not all meshes are volumes") and every piece falls back to the
    sum it was supposed to replace."""
    import numpy as np
    from catalog_organizer.snapshotter.threedm import _as_solid

    verts, faces = _box((4.0, 6.0, 8.0))
    # Explode it the way a Brep does: every triangle gets its own vertices.
    loose_v = verts[faces].reshape(-1, 3)
    loose_f = np.arange(len(loose_v)).reshape(-1, 3)
    raw = __import__("trimesh").Trimesh(vertices=loose_v, faces=loose_f,
                                        process=False)
    assert not raw.is_watertight

    solid = _as_solid(loose_v, loose_f)
    assert solid is not None and solid.is_watertight
    assert abs(abs(solid.volume) - 192.0) < 1e-6
