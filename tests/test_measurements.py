from __future__ import annotations

import numpy as np
import trimesh

from catalog_organizer.cad.measurements import (
    estimate_volume,
    measure_bbox,
    measure_bracelet,
    measure_earring,
    measure_pendant,
    measure_ring,
)
from catalog_organizer.cad.scale_check import check_scale
from catalog_organizer.cad.sprue import detect_sprue
from catalog_organizer.core.schemas import Measurements


def make_ring_mesh(major_radius: float = 9.0, minor_radius: float = 1.0) -> trimesh.Trimesh:
    """Torus with hole along Z. Outer diameter = 2*(R+r), inner diameter = 2*(R-r)."""
    return trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius)


def test_measure_bbox_correct_within_tolerance():
    mesh = make_ring_mesh(major_radius=9.0, minor_radius=1.0)
    bbox = measure_bbox(mesh)
    assert abs(bbox.width - 20.0) < 0.1
    assert abs(bbox.height - 20.0) < 0.1
    assert abs(bbox.depth - 2.0) < 0.1
    # Theoretical volume: 2 * pi^2 * R * r^2 = 177.65 mm^3 (discrete mesh ≈ 175).
    assert bbox.volume_mm3 > 150 and bbox.volume_mm3 < 200


def test_measure_ring_inner_diameter_within_half_mm():
    mesh = make_ring_mesh(major_radius=9.0, minor_radius=1.0)
    ring = measure_ring(mesh)
    # True inner diameter = 2 * (R - r) = 16.0 mm. Stop point: ±0.5 mm.
    assert abs(ring.inner_diameter_mm - 16.0) < 0.5
    assert abs(ring.inner_circumference_mm - (np.pi * 16.0)) < 0.5 * np.pi
    # Band width along finger axis ≈ 2r = 2.0 mm.
    assert abs(ring.band_width_mm - 2.0) < 0.2


def test_measure_ring_works_on_larger_band():
    # Verify on a different size to be sure the algorithm isn't hard-coded.
    mesh = make_ring_mesh(major_radius=12.0, minor_radius=1.5)
    ring = measure_ring(mesh)
    # True inner diameter = 2 * (12 - 1.5) = 21.0 mm.
    assert abs(ring.inner_diameter_mm - 21.0) < 0.5


def test_measure_ring_ignores_floating_centre_stone():
    """Regression (sampleA_43.3dm): a centre stone modelled as a separate
    component hanging into the finger bore must NOT collapse the measured
    inner diameter. The old centroid-min algorithm returned ~0.9 mm here.

    Build a 16 mm-bore band plus a small floating sphere ("diamond") sitting
    in the middle of the bore as an independent component. The structural
    filter must drop the sphere and still report ~16 mm.
    """
    band = make_ring_mesh(major_radius=9.0, minor_radius=1.0)   # bore = 16 mm
    stone = trimesh.creation.icosphere(subdivisions=2, radius=1.5)
    # Place the stone at the bore centre (origin), as a separate body.
    combined = trimesh.util.concatenate([band, stone])
    ring = measure_ring(combined)
    # Must stay near 16 mm, not collapse toward the 3 mm stone.
    assert ring.inner_diameter_mm > 12.0, (
        f"floating stone corrupted bore: got {ring.inner_diameter_mm:.2f} mm"
    )
    assert abs(ring.inner_diameter_mm - 16.0) < 2.0


def test_measure_earring_pair_detection():
    left = make_ring_mesh(major_radius=5.0, minor_radius=0.5)
    right = left.copy()
    right.apply_translation([20.0, 0.0, 0.0])
    both = trimesh.util.concatenate([left, right])
    res = measure_earring(both)
    assert res.pair_or_single == "pair"


def test_measure_earring_single_detection():
    mesh = trimesh.creation.cylinder(radius=2.0, height=10.0)
    res = measure_earring(mesh)
    assert res.pair_or_single == "single"


def test_bracelet_uses_ring_algorithm():
    mesh = make_ring_mesh(major_radius=30.0, minor_radius=2.0)
    res = measure_bracelet(mesh)
    # True inner diameter = 56.0 mm
    assert abs(res.inner_diameter_x_mm - 56.0) < 1.0


def test_pendant_bbox():
    mesh = trimesh.creation.box(extents=[10.0, 15.0, 4.0])
    res = measure_pendant(mesh)
    assert abs(res.width_mm - 10.0) < 0.01
    assert abs(res.height_mm - 15.0) < 0.01
    assert abs(res.depth_mm - 4.0) < 0.01


def test_scale_check_flags_oversize_ring():
    m = Measurements(
        bbox_width_mm=120.0, bbox_height_mm=120.0, bbox_depth_mm=8.0,
        volume_mm3=1000.0, geometry_source="trimesh",
        ring_inner_diameter_mm=15.0,
    )
    thresholds = {"ring": {"min_bbox_mm": 5, "max_bbox_mm": 80, "min_inner_mm": 10, "max_inner_mm": 30}}
    warn, reason = check_scale(m, "ring", thresholds)
    assert warn is True
    assert "above" in reason


def test_scale_check_passes_normal_ring():
    m = Measurements(
        bbox_width_mm=20.0, bbox_height_mm=20.0, bbox_depth_mm=2.0,
        volume_mm3=175.0, geometry_source="trimesh",
        ring_inner_diameter_mm=16.0,
    )
    thresholds = {"ring": {"min_bbox_mm": 5, "max_bbox_mm": 80, "min_inner_mm": 10, "max_inner_mm": 30}}
    warn, reason = check_scale(m, "ring", thresholds)
    assert warn is False
    assert reason is None


def test_sprue_vlm_only_when_no_mesh():
    """Without a mesh, detect_sprue falls back to the VLM yes/no — same as
    the original stub behaviour. Volume stays None (no estimate available)."""
    s_true = detect_sprue(vlm_says_sprue=True, vlm_confidence=0.9)
    assert s_true.detected is True
    assert s_true.source == "vlm"
    # Confidence dampened ×0.6.
    assert abs(s_true.confidence - 0.54) < 0.001
    assert s_true.estimated_volume_mm3 is None

    s_false = detect_sprue(vlm_says_sprue=False, vlm_confidence=0.0)
    assert s_false.detected is False
    assert s_false.source == "none"


def test_sprue_geometric_finds_neck_and_estimates_volume():
    """A torus (main piece) glued to a thin cylindrical stub (the sprue)
    must trigger geometric detection along the sprue axis and produce a
    positive volume estimate."""
    ring = trimesh.creation.torus(major_radius=9.0, minor_radius=1.0)
    # Slim cylinder hanging off the bottom of the ring along -Z
    sprue = trimesh.creation.cylinder(radius=1.0, height=8.0)
    sprue.apply_translation([0.0, 0.0, -5.0])
    glued = trimesh.util.concatenate([ring, sprue])

    info = detect_sprue(vlm_says_sprue=False, vlm_confidence=0.0, mesh=glued)
    assert info.detected is True
    assert info.source == "geometry"
    assert info.estimated_volume_mm3 is not None
    # The cylinder's true volume is pi * 1^2 * 8 = 25.13 mm^3; allow generous
    # tolerance because the slab approximation is rough.
    assert 10.0 < info.estimated_volume_mm3 < 80.0


def test_sprue_geometric_no_false_positive_on_plain_ring():
    """A clean torus must NOT trigger sprue detection — there's no neck."""
    ring = trimesh.creation.torus(major_radius=9.0, minor_radius=1.0)
    info = detect_sprue(vlm_says_sprue=False, vlm_confidence=0.0, mesh=ring)
    assert info.detected is False
    assert info.source == "none"
    assert info.estimated_volume_mm3 is None


def test_sprue_hybrid_blends_vlm_and_geometric_confidence():
    """When both signals agree, source = vlm_geometry_combined and the
    confidence is at least as high as the geometric-only value."""
    ring = trimesh.creation.torus(major_radius=9.0, minor_radius=1.0)
    sprue = trimesh.creation.cylinder(radius=1.0, height=8.0)
    sprue.apply_translation([0.0, 0.0, -5.0])
    glued = trimesh.util.concatenate([ring, sprue])

    geom_only = detect_sprue(False, 0.0, mesh=glued)
    hybrid = detect_sprue(True, 0.9, mesh=glued)
    assert hybrid.source == "vlm_geometry_combined"
    assert hybrid.confidence >= geom_only.confidence


def test_estimate_volume_watertight_high_confidence():
    """Watertight torus → exact volume, confidence='high'."""
    mesh = make_ring_mesh(major_radius=9.0, minor_radius=1.0)
    assert mesh.is_volume
    vol, conf = estimate_volume(mesh)
    assert conf == "high"
    assert 150 < vol < 200


def test_estimate_volume_non_watertight_but_winding_consistent_uses_signed_volume():
    """When the surface has gaps but face winding is still consistent, the
    signed `mesh.volume` (divergence-theorem integral) is still accurate.
    Confidence drops to 'medium' but the value must NOT fall through to
    the convex hull — which would over-estimate by 2-3× on real jewelry.

    Verified against MatrixGold + scale on a real pendant (pave-pendant.stl,
    2026-05-18): signed mesh.volume = 423.95 mm³ vs ground truth 425.7 mm³.
    """
    mesh = make_ring_mesh(major_radius=9.0, minor_radius=1.0)
    # Drop one triangle to break watertightness without breaking winding.
    mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                          faces=mesh.faces[:-1].copy(),
                          process=False)
    assert not mesh.is_watertight
    assert mesh.is_winding_consistent

    vol, conf = estimate_volume(mesh)
    assert conf == "medium"
    # signed volume should still be ~theoretical 177 mm³ (one missing face
    # barely moves the integral).
    assert 150 < vol < 200


def test_estimate_volume_winding_inconsistent_falls_back_to_hull():
    """When face winding is inconsistent, signed volume is unreliable —
    we must drop to the convex-hull fallback."""
    mesh = make_ring_mesh(major_radius=9.0, minor_radius=1.0)
    # Flip half the faces' winding so trimesh marks it inconsistent.
    faces = mesh.faces.copy()
    faces[: len(faces) // 2] = faces[: len(faces) // 2, ::-1]
    broken = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=faces, process=False)
    assert not broken.is_winding_consistent

    vol, conf = estimate_volume(broken)
    assert conf == "low"
    # Hull of a torus is roughly a disc; volume well below the bbox 800.
    assert 0 < vol <= 800


def test_capped_cutter_volume_never_exceeds_30pct_of_gross(monkeypatch):
    """Regression (sampleA_43.3dm): a halo ring's 11 mm centre-stone cutter
    produced a raw cutter estimate LARGER than the metal body itself
    (1049 mm3 vs 783 mm3 gross), zeroing the net volume and killing all
    weights. The cap clamps subtraction to 30% of gross.

    This is the LEGACY-HEURISTIC fallback path only (2026-06-29c: the
    primary path is now an exact Rhino.Inside boolean measurement, which is
    never capped — a real measurement can't over-subtract by construction).
    Mock `rhino_engine.exact_cutter_volume_mm3` to return None (Rhino
    unavailable) so this test exercises the fallback+cap logic it was
    written to protect, same as it always has on a machine without Rhino."""
    from catalog_organizer.cad import rhino_engine
    from catalog_organizer.snapshotter import threedm

    monkeypatch.setattr(rhino_engine, "exact_cutter_volume_mm3", lambda p: None)

    monkeypatch.setattr(threedm, "estimate_cutter_volume_mm3", lambda p: 1049.3)
    capped = threedm.capped_cutter_volume_mm3("fake.3dm", 782.9)
    assert abs(capped - 0.30 * 782.9) < 1e-6

    # Small cutters pass through unclamped.
    monkeypatch.setattr(threedm, "estimate_cutter_volume_mm3", lambda p: 90.0)
    assert threedm.capped_cutter_volume_mm3("fake.3dm", 782.9) == 90.0

    # Zero gross volume -> no subtraction at all.
    assert threedm.capped_cutter_volume_mm3("fake.3dm", 0.0) == 0.0


def test_detect_casting_rails_finds_runner_bars():
    """Casting-tree sprue (5 MM KOLYE.stl pattern): pieces hung on long
    slender runner bars. The slab detector can't see it (sprue is fatter
    than the pieces); the component rail detector must."""
    from catalog_organizer.cad.sprue import detect_casting_rails

    rail = trimesh.creation.box(extents=[80.0, 1.6, 1.6])
    cubes = []
    for i in range(6):
        c = trimesh.creation.box(extents=[6.0, 6.0, 2.0])
        c.apply_translation([-30 + i * 12, 4.0, 0.0])
        cubes.append(c)
    tree = trimesh.util.concatenate([rail] + cubes)

    result = detect_casting_rails(tree)
    assert result is not None, "rails not detected"
    rails_vol, overlap, n_rails = result
    assert n_rails == 1
    # True rail volume = 80*1.6*1.6 = 204.8
    assert abs(rails_vol - 204.8) < 5.0
    # Cubes don't touch the rail here -> overlap ~ 0
    assert overlap < 5.0


def test_detect_casting_rails_none_on_single_piece():
    """A plain ring has no separate rail components -> None (falls back
    to the slab detector)."""
    from catalog_organizer.cad.sprue import detect_casting_rails

    ring = trimesh.creation.torus(major_radius=9.0, minor_radius=1.0)
    assert detect_casting_rails(ring) is None


def test_detect_note_objects_flags_separated_text_beside_piece():
    """Engraved monogram beside the piece (sampleA_12 customer-monogram pattern): a small
    blob sitting OUTSIDE the main body's bounding box is a note → flagged.
    A small blob INSIDE the body envelope (a stone seat) is kept."""
    from catalog_organizer.snapshotter.threedm import detect_note_object_indices

    # Main body: a ring-sized box centred at origin.
    body = trimesh.creation.box(extents=[20.0, 20.0, 4.0])
    # A "stone seat": small box INSIDE the body footprint (kept).
    seat = trimesh.creation.box(extents=[1.5, 1.5, 4.2])
    seat.apply_translation([5.0, 5.0, 0.0])
    # Three "letter" slabs sitting in a row BESIDE the body (outside bbox).
    letters = []
    for i in range(3):
        lt = trimesh.creation.box(extents=[2.0, 3.0, 0.8])
        lt.apply_translation([-4.0 + i * 4.0, -16.0, 0.0])  # y=-16, outside body (±10)
        letters.append(lt)

    objs = [(0, body.vertices), (1, seat.vertices)]
    objs += [(2 + i, lt.vertices) for i, lt in enumerate(letters)]

    flagged = detect_note_object_indices(objs)
    # The three letters (indices 2,3,4) are flagged; body and seat are not.
    assert flagged == {2, 3, 4}, f"got {flagged}"


def test_detect_note_objects_keeps_everything_when_single_blob():
    """A piece whose parts all touch (one spatial blob) flags nothing."""
    from catalog_organizer.snapshotter.threedm import detect_note_object_indices

    a = trimesh.creation.box(extents=[10.0, 10.0, 3.0])
    b = trimesh.creation.box(extents=[3.0, 3.0, 3.0])
    b.apply_translation([6.0, 0.0, 0.0])  # touches/overlaps a
    c = trimesh.creation.box(extents=[3.0, 3.0, 3.0])
    c.apply_translation([0.0, 6.0, 0.0])
    objs = [(0, a.vertices), (1, b.vertices), (2, c.vertices)]
    assert detect_note_object_indices(objs) == set()
