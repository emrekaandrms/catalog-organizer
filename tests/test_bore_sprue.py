"""Support bar threaded through a ring's finger bore (cad/sprue.py).

Reported by the user 2026-07-30: a freshly added `alyans*` (wedding band)
series where every file carries a casting support bar, but only 1 of 14 was
flagged — and even that one came from the VLM with no volume, so no weight
correction happened at all. The bar is 4-14 % of the piece, which is far too
much to fold silently into the metal weight.

Synthetic fixtures here mirror the real topology: a torus with a thin
cylinder spanning its hole, fused into one mesh.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from catalog_organizer.cad.sprue import (
    _bore_radius_by_angular_coverage,
    _ring_axis,
    detect_bore_sprue,
    detect_sprue,
    detect_sprue_geometric,
)


def _plain_ring(outer_r: float = 10.0, inner_r: float = 8.0, width: float = 6.0):
    """A hollow band lying in the XY plane, threaded on Z."""
    outer = trimesh.creation.cylinder(radius=outer_r, height=width, sections=192)
    inner = trimesh.creation.cylinder(radius=inner_r, height=width * 2, sections=192)
    return trimesh.boolean.difference([outer, inner], engine="manifold")


def _bar(radius: float, length: float, axis: str = "x"):
    bar = trimesh.creation.cylinder(radius=radius, height=length, sections=64)
    if axis == "x":
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        bar.apply_transform(rot)
    return bar


def _ring_with_bar(bar_radius: float = 0.6, **kw):
    """Ring plus a bar across its bore, unioned into a single solid — the
    real files split into exactly one connected component."""
    ring = _plain_ring(**kw)
    inner_r = kw.get("inner_r", 8.0)
    bar = _bar(bar_radius, inner_r * 2 + 1.0)
    return trimesh.boolean.union([ring, bar], engine="manifold")


def test_ring_axis_picks_the_axis_the_ring_is_threaded_on():
    """Seen down the finger axis a ring is circular, so the other two
    extents match. Chosen over "smallest extent", which flips on wide bands."""
    ring = _plain_ring(outer_r=10.0, inner_r=8.0, width=6.0)
    assert _ring_axis(ring) == 2  # built around Z


def test_bore_radius_recovered_on_a_plain_ring():
    ring = _plain_ring(outer_r=10.0, inner_r=8.0, width=6.0)
    r_in, r_out, _ = _bore_radius_by_angular_coverage(ring, _ring_axis(ring))
    assert r_out == pytest.approx(10.0, abs=0.2)
    assert r_in == pytest.approx(8.0, abs=0.35)


def test_bore_radius_survives_a_bar_across_the_hole():
    """Regression on the core trap: a bar splits the hole in two, so an
    "largest empty circle" search returns about half the true bore, and a
    coverage scan run inwards-out returns nearly zero (the bar wraps every
    angle near the centre). Neither may happen here."""
    ring = _ring_with_bar(bar_radius=0.6, outer_r=10.0, inner_r=8.0, width=6.0)
    r_in, _r_out, _ = _bore_radius_by_angular_coverage(ring, _ring_axis(ring))
    assert r_in == pytest.approx(8.0, abs=0.35), (
        f"bore collapsed to {r_in:.2f} mm with a bar present"
    )


def test_detects_bar_and_measures_it_within_a_few_percent():
    """Volume is an exact boolean, not an estimate — two cheaper
    approximations were rejected for 43 % and 97 % errors on real files."""
    bar_r, inner_r = 0.6, 8.0
    ring = _ring_with_bar(bar_radius=bar_r, outer_r=10.0, inner_r=inner_r, width=6.0)
    detected, volume, confidence = detect_bore_sprue(ring)

    assert detected is True
    assert confidence > 0.5
    # The captured stem is the part inside the bore: pi*r^2 * 2*inner_r.
    expected = np.pi * bar_r**2 * 2 * inner_r
    assert volume == pytest.approx(expected, rel=0.10)


def test_plain_ring_is_not_flagged():
    """The costly failure mode: a false positive silently shaves weight off
    a ring that has no sprue at all."""
    detected, volume, _ = detect_bore_sprue(_plain_ring())
    assert detected is False
    assert volume is None


def test_solid_disc_is_not_flagged():
    """No bore at all — r_inner/r_outer falls far below the guard, the same
    way `ring 17.stl` (0.02) and `pave-bracelet.stl` (0.09) do."""
    disc = trimesh.creation.cylinder(radius=10.0, height=6.0, sections=192)
    assert detect_bore_sprue(disc)[0] is False


def test_material_filling_most_of_the_bore_is_not_a_support_bar():
    """`FMR-32 11 boy (3 ADET MUM).stl` — three waxes around a runner — puts
    52 % of its volume in the gap between them. A support bar is a thin
    stick (4-14 % on the real alyans files), so a dominant in-bore solid
    must be rejected rather than reported as an enormous sprue."""
    fat = _ring_with_bar(bar_radius=5.0, outer_r=10.0, inner_r=8.0, width=6.0)
    assert detect_bore_sprue(fat)[0] is False


def test_slab_detector_alone_would_miss_this_topology():
    """Documents why a third detector was needed: the bar sits in the middle
    of the bounding box, so there is no low-area neck at either end of any
    axis for `detect_sprue_geometric` to find."""
    ring = _ring_with_bar()
    assert detect_sprue_geometric(ring)[0] is False


def test_bar_is_one_component_so_rail_detection_cannot_see_it_either():
    """`detect_casting_rails` needs separate components; the bar is fused."""
    ring = _ring_with_bar()
    assert len(ring.split(only_watertight=False)) == 1


def test_detect_sprue_reports_volume_so_weight_can_be_corrected():
    """End-to-end through the public entry point. A VLM-only hit carries no
    volume and silently skips the weight correction — that is exactly what
    happened to the one alyans file that *was* flagged before this fix."""
    ring = _ring_with_bar()
    info = detect_sprue(vlm_says_sprue=False, vlm_confidence=0.0, mesh=ring)
    assert info.detected is True
    assert info.source in ("geometry", "vlm_geometry_combined")
    assert info.estimated_volume_mm3 is not None
    assert info.estimated_volume_mm3 > 0


def test_detect_sprue_stays_quiet_on_a_plain_ring():
    info = detect_sprue(vlm_says_sprue=False, vlm_confidence=0.0, mesh=_plain_ring())
    assert info.detected is False
    assert info.estimated_volume_mm3 is None
