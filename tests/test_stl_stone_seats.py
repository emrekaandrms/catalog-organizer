"""Unit tests for the STL stone-seat detector.

We don't ship gigabyte fixtures into the test suite — the end-to-end
calibration on `samples/*.stl` (cross-pendant, plain-band/2, pave-pendant) lives in the manual smoke script. Here we pin the *small, pure*
pieces of the algorithm:

  * `_classify_ring` filters: size, circularity, area.
  * `_cluster_by_xy` collapses repeated detections of the same seat
    across consecutive Z slices.
  * End-to-end on a synthetic mesh with a known cylindrical hole.
"""
from __future__ import annotations

import math


def test_classify_ring_accepts_round_seat():
    from shapely.geometry import Polygon
    from catalog_organizer.cad.stl_stone_seats import _classify_ring

    # A 2 mm circle approximated by a 32-vertex polygon
    angles = [2 * math.pi * i / 32 for i in range(32)]
    ring = [(math.cos(a), math.sin(a)) for a in angles]
    cand = _classify_ring(Polygon(ring).exterior.coords, z=0.5,
                          min_d=0.5, max_d=10.0, min_circ=0.65)
    assert cand is not None
    assert 1.8 < cand.diameter_mm < 2.1
    assert cand.circularity > 0.95


def test_classify_ring_rejects_elongated_slot():
    """A long thin slot (e.g. a letter cut-out or chain hole) must NOT be
    flagged as a stone seat."""
    from shapely.geometry import Polygon
    from catalog_organizer.cad.stl_stone_seats import _classify_ring

    # 4 mm × 0.5 mm rectangle — area gives d_eq ~1.6 but circularity is 0.125
    rect = [(0, 0), (4, 0), (4, 0.5), (0, 0.5)]
    cand = _classify_ring(Polygon(rect).exterior.coords, z=0.5,
                          min_d=0.5, max_d=10.0, min_circ=0.65)
    assert cand is None


def test_classify_ring_rejects_too_small():
    """Tiny holes (< 0.6 mm) are mesh noise (degenerate slivers), not
    stones."""
    from shapely.geometry import Polygon
    from catalog_organizer.cad.stl_stone_seats import _classify_ring

    angles = [2 * math.pi * i / 16 for i in range(16)]
    ring = [(0.2 * math.cos(a), 0.2 * math.sin(a)) for a in angles]  # 0.4 mm dia
    cand = _classify_ring(Polygon(ring).exterior.coords, z=0.5,
                          min_d=0.6, max_d=10.0, min_circ=0.65)
    assert cand is None


def test_cluster_collapses_repeated_slices_into_one_seat():
    """Six consecutive Z slices of the same 2 mm seat collapse into one
    `DetectedSeat`, and the reported diameter is the max observed (the
    widest cross-section = the entrance opening)."""
    from catalog_organizer.cad.stl_stone_seats import _SeatCandidate, _cluster_by_xy

    cands = []
    # Same physical seat at (5.0, 3.0); diameter grows toward the top
    for i, d in enumerate([1.40, 1.65, 1.90, 2.00, 1.80, 1.55]):
        cands.append(_SeatCandidate(z=1.0 - i * 0.1, cx=5.0, cy=3.0,
                                     diameter_mm=d, circularity=0.95))
    # A second seat 4 mm away — must NOT merge with the first
    for d in [1.10, 1.15, 1.20]:
        cands.append(_SeatCandidate(z=1.0, cx=9.0, cy=3.0,
                                     diameter_mm=d, circularity=0.92))

    seats = _cluster_by_xy(cands, cluster_xy_mm=1.5, min_supporting_slices=2)
    assert len(seats) == 2
    # First seat: max diameter is 2.00
    assert abs(seats[0].diameter_mm - 2.00) < 1e-6
    # Centres should round-trip approximately
    assert abs(seats[0].centre_xy[0] - 5.0) < 0.05
    assert abs(seats[1].centre_xy[0] - 9.0) < 0.05


def test_entrance_flat_diameter_run_detects_constant_cylinder():
    """A straight-walled through-hole (honeycomb/filigree cutout) reports
    the same diameter starting at its very first (entrance) slice — the
    run length should equal the slice count."""
    from catalog_organizer.cad.stl_stone_seats import (
        _SeatCandidate, _entrance_flat_diameter_run,
    )
    cluster = [
        _SeatCandidate(z=1.0 - i * 0.1, cx=0.0, cy=0.0,
                       diameter_mm=3.38, circularity=0.86)
        for i in range(8)
    ]
    assert _entrance_flat_diameter_run(cluster) == 8


def test_entrance_flat_diameter_run_detects_taper():
    """A real cone-shaped seat's diameter changes at every slice near the
    entrance — the run should be 1 (the very first step already differs)."""
    from catalog_organizer.cad.stl_stone_seats import (
        _SeatCandidate, _entrance_flat_diameter_run,
    )
    cluster = [
        _SeatCandidate(z=1.0 - i * 0.1, cx=0.0, cy=0.0,
                       diameter_mm=d, circularity=1.0)
        for i, d in enumerate([1.23, 1.01, 0.78, 0.67, 0.50])
    ]
    assert _entrance_flat_diameter_run(cluster) == 1


def test_entrance_flat_diameter_run_partial_plateau():
    """A short plateau right at the entrance (real seats sometimes repeat
    2-3 slices, e.g. from mesh-resolution rounding) must NOT trip the
    honeycomb rejection — only long runs (>=5) should."""
    from catalog_organizer.cad.stl_stone_seats import (
        _SeatCandidate, _entrance_flat_diameter_run,
    )
    cluster = [
        _SeatCandidate(z=1.0 - i * 0.1, cx=0.0, cy=0.0,
                       diameter_mm=d, circularity=1.0)
        for i, d in enumerate([1.23, 1.23, 1.23, 0.78, 0.67])
    ]
    assert _entrance_flat_diameter_run(cluster) == 3


def test_entrance_flat_diameter_run_ignores_flat_tip():
    """Regression (pave-bracelet.stl, 2026-06-30): a real cone taper that
    goes flat near the CULET TIP (mesh triangulation can't resolve a
    perfect point, so the smallest few slices often repeat the same
    near-minimum diameter) must NOT be rejected — only a flat run starting
    at the ENTRANCE (the widest/topmost slice) is suspicious. A first cut
    of this filter checked for a flat run anywhere in the cluster and
    wrongly dropped 31 of 65 real seats on this file because their culet
    tips bottomed out at the same diameter for several slices."""
    from catalog_organizer.cad.stl_stone_seats import (
        _SeatCandidate, _entrance_flat_diameter_run,
    )
    # Real taper from the entrance (1.235->1.059->0.835), THEN the tip
    # bottoms out at the minimum-detectable diameter for several slices.
    cluster = [
        _SeatCandidate(z=1.0 - i * 0.1, cx=0.0, cy=0.0, diameter_mm=d,
                       circularity=1.0)
        for i, d in enumerate([1.235, 1.059, 0.835, 0.616, 0.616, 0.616, 0.616, 0.616])
    ]
    assert _entrance_flat_diameter_run(cluster) == 1


def test_cluster_by_xy_rejects_entrance_flat_diameter_run():
    """End-to-end: a cluster whose diameter never changes STARTING AT THE
    ENTRANCE (a honeycomb/lattice through-hole slicing straight down) must
    be dropped, even though it would otherwise pass the circularity/size/
    slice-count gates. Regression for samples/petek taşları
    kapatıldı.stl (2026-06-30): a 264-hole honeycomb bangle was being
    reported as 264 'stones'. A real seat whose culet tip happens to go
    flat (not its entrance) must be KEPT — see
    test_entrance_flat_diameter_run_ignores_flat_tip above."""
    from catalog_organizer.cad.stl_stone_seats import _SeatCandidate, _cluster_by_xy

    honeycomb_hole = [
        _SeatCandidate(z=1.0 - i * 0.1, cx=10.0, cy=0.0,
                       diameter_mm=3.38, circularity=0.86)
        for i in range(10)
    ]
    real_seat_flat_tip = [
        _SeatCandidate(z=1.0 - i * 0.1, cx=0.0, cy=0.0, diameter_mm=d,
                       circularity=1.0)
        for i, d in enumerate([1.235, 1.059, 0.835, 0.616, 0.616, 0.616, 0.616, 0.616])
    ]
    seats = _cluster_by_xy(honeycomb_hole + real_seat_flat_tip,
                           cluster_xy_mm=1.5, min_supporting_slices=2)
    assert len(seats) == 1
    assert abs(seats[0].centre_xy[0] - 0.0) < 0.05


def test_cluster_drops_single_slice_noise():
    """A 'seat' that appears in only one Z slice is almost always mesh
    noise — we require at least two supporting slices."""
    from catalog_organizer.cad.stl_stone_seats import _SeatCandidate, _cluster_by_xy

    cands = [
        _SeatCandidate(z=0.5, cx=5.0, cy=3.0, diameter_mm=1.5, circularity=0.9),
    ]
    seats = _cluster_by_xy(cands, cluster_xy_mm=1.5, min_supporting_slices=2)
    assert seats == []


def test_find_stone_seats_on_synthetic_block_with_hole():
    """A solid 10 x 10 x 3 mm block with a 3 mm CONE-shaped seat cut 2 mm
    into one face (wide entrance at the surface, narrowing to a point —
    a real gem seat's shape) must yield exactly one detected seat near the
    drilled diameter.

    Uses a cone, not a cylinder: 2026-06-30's honeycomb/through-hole filter
    (`_longest_flat_diameter_run`, see stl_stone_seats.py) rejects any
    cluster reporting the same diameter across >=5 consecutive slices — a
    perfect cylinder IS a constant-diameter shape by construction, so a
    cylindrical fixture would now be correctly rejected as indistinguishable
    from a honeycomb/lattice cutout. A cone is what a real seat actually is."""
    import trimesh
    from catalog_organizer.cad.stl_stone_seats import find_stone_seats

    block = trimesh.creation.box(extents=[10, 10, 3])
    # Base (wide, radius 1.5 -> 3mm diameter) at z=0, apex at z=+2 by
    # default; flip so the apex points DOWN into the block and the wide
    # entrance sits flush with the block's top surface (z=+1.5).
    drill = trimesh.creation.cone(radius=1.5, height=2.0, sections=64)
    drill.apply_transform(trimesh.transformations.rotation_matrix(
        angle=3.14159265, direction=[1, 0, 0]))
    drill.apply_translation([0, 0, 1.5])
    # Need shapely-friendly boolean. trimesh.boolean requires a backend
    # ('manifold' or 'blender'); both may be absent — skip gracefully.
    try:
        bored = trimesh.boolean.difference([block, drill])
    except Exception:
        import pytest
        pytest.skip("No boolean backend available in this environment")
    seats = find_stone_seats(bored, z_step=0.1, min_d=0.5)
    assert len(seats) >= 1
    # The cone's wide end was 3 mm diameter; cluster reports the max
    # cross-section (the entrance opening), within a few percent.
    assert 2.7 < seats[0].diameter_mm < 3.3
