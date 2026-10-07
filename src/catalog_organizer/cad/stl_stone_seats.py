"""Geometric stone-seat extraction for STL files.

STL meshes don't contain stones (designers strip them before exporting
to the printer), but the **seats** they sit in are still cut into the
metal — empty cylindrical/conical depressions on the top face.

Algorithm (multi-slice + XY-cluster):

  1. Slice the mesh horizontally every ~0.1 mm along Z.
  2. In each slice, every shapely *interior ring* on a metal polygon is
     a hole = a candidate seat profile. (Slicing a solid pendant where
     seats have been opened produces metal cross-sections that contain
     small hole rings exactly where each seat sits.)
  3. Filter: equivalent circular diameter ∈ [min_d, max_d], bbox
     circularity ≥ min_circ so we drop elongated slots (chain holes,
     letter cut-outs, …).
  4. Cluster candidates across slices by XY proximity — the same seat
     appears in several consecutive slices.
  5. Per cluster, report the MAX observed diameter. That's the entrance
     opening at the top of the cone, which matches the stone's girdle.

Verified on `samples/cross-pendant.stl` (2026-05-21): finds 11 seats
including the 4 main 2-3.5 mm centre stones and the perimeter 0.7-0.85 mm
melee pavé. `samples/plain-band.stl` (plain ring, no stones) returns
zero — no false positives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon


@dataclass(frozen=True)
class _SeatCandidate:
    z: float
    cx: float
    cy: float
    diameter_mm: float
    circularity: float


@dataclass(frozen=True)
class DetectedSeat:
    diameter_mm: float
    centre_xy: tuple[float, float]
    n_slices: int
    avg_circularity: float
    # Where the seat opens and ends, for placing a stone in it (the analysis needs only the diameter):
    # Z of the topmost / lowest supporting slice, and the diameter at the lowest one.
    z_top: float | None = None
    z_bottom: float | None = None
    d_bottom: float | None = None


_DEFAULT_Z_STEP_MM = 0.10
_DEFAULT_MIN_D_MM = 0.60        # below this it's mesh noise, not a stone
_DEFAULT_MAX_D_MM = 12.0        # bigger than this is almost never a CZ
_DEFAULT_MIN_CIRC = 0.65        # bbox short/long; <0.65 is elongated slot
_DEFAULT_CLUSTER_XY_MM = 1.5    # same seat across slices stays within this
_DEFAULT_MIN_SUPPORTING_SLICES = 2

# A real gem seat is a CONE — table/entrance wide near the metal surface,
# narrowing toward the culet — so its cross-section diameter changes at
# essentially every Z slice near the ENTRANCE. A decorative through-hole
# (honeycomb / lattice / filigree cutout) has straight, parallel walls, so
# its diameter is constant right from the surface. Ground-truthed
# 2026-06-30 on `samples/honeycomb-bangle.stl` ("honeycomb,
# stones removed" — a bangle with 264 lattice holes and zero real stones):
# every hole reports the EXACT SAME diameter for 10 consecutive slices
# starting at its very first (topmost/entrance) slice.
#
# IMPORTANT: a real seat's diameter often ALSO goes flat near its culet —
# a cone doesn't triangulate to a perfect point, so once the cross-section
# shrinks close to the minimum detectable size the reported diameter can
# repeat for several slices near the TIP. That's normal mesh-resolution
# noise at the bottom, not a straight bore. Confirmed on `pave-bracelet.stl`
# (2026-05-21 golden, 65 real seats): dense pavé seats taper cleanly from
# the entrance (1.235→1.059→0.835 mm) then bottom out at ~0.616 mm for
# several slices near the tip — a first cut of this filter that looked for
# a flat run ANYWHERE in the cluster wrongly rejected 31 of those 65 real
# seats. The fix: only flag a flat run that STARTS AT THE ENTRANCE (the
# cluster's topmost/widest slice) — a straight bore is flat from the
# surface down; a cone is never flat at the top, only (sometimes) at the
# tip.
_MAX_FLAT_DIAMETER_RUN_SLICES = 5
_FLAT_DIAMETER_EPSILON_MM = 0.02

# A second honeycomb signature — a rounded lattice node slicing into a
# symmetric "waist" (diameter narrows then widens back, e.g. 0.77→0.71→
# 0.77 mm, since it's a dome/fillet, not a cone) — was tried as a
# "monotonicity ratio" filter (total up+down diameter movement vs net
# first-to-last change) on 2026-06-30 and REVERTED: it also rejected
# confirmed-real seats (cross-pendant.stl lost its 3.44 mm centre stone;
# pave-pendant.stl dropped 41→33). The naive per-cluster diameter
# sequence this would need to reason about isn't clean enough — real
# multi-slice clusters mix in nearby-but-distinct candidates that make a
# "one clean taper" assumption unreliable. Left as a known residual: the
# flat-run filter above catches straight through-holes (the dominant
# false-positive case) but not rounded lattice nodes/waists. See
# the accuracy-remediation notes, item 3.


def find_stone_seats(
    mesh: trimesh.Trimesh,
    *,
    z_step: float = _DEFAULT_Z_STEP_MM,
    min_d: float = _DEFAULT_MIN_D_MM,
    max_d: float = _DEFAULT_MAX_D_MM,
    min_circ: float = _DEFAULT_MIN_CIRC,
    cluster_xy_mm: float = _DEFAULT_CLUSTER_XY_MM,
    min_supporting_slices: int = _DEFAULT_MIN_SUPPORTING_SLICES,
) -> list[DetectedSeat]:
    """Walk the mesh in Z slices and return one `DetectedSeat` per
    open stone seat. Returns an empty list when the piece has no
    detectable seats (a plain ring or a solid pendant without stones).
    """
    bmin, bmax = mesh.bounds
    z_top, z_bot = float(bmax[2]), float(bmin[2])
    if z_top - z_bot < z_step * 4:
        return []   # too thin to slice meaningfully

    candidates: list[_SeatCandidate] = []
    z = z_top - z_step
    while z > z_bot + z_step:
        section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section is None:
            z -= z_step
            continue

        # Shapely sometimes refuses to repair pathological cross-sections
        # (self-intersecting boundaries near degenerate geometry). Skip
        # those slices rather than crashing the whole pipeline.
        try:
            path2d, to_3d = section.to_2D() if hasattr(section, "to_2D") else section.to_planar()
            polygons = list(path2d.polygons_full)
        except Exception:
            z -= z_step
            continue

        for poly in polygons:
            for ring in poly.interiors:
                cand = _classify_ring(ring, z, min_d, max_d, min_circ)
                if cand is not None:
                    # `to_planar` re-centres every slice on its OWN section, so cx/cy are relative to
                    # that, not to the file: a piece away from the origin reported seats near (0, 0),
                    # and two slices of one seat could disagree. Back to the file's frame:
                    x, y, *_ = np.asarray(to_3d, dtype=float) @ np.array([cand.cx, cand.cy, 0.0, 1.0])
                    candidates.append(_SeatCandidate(z=cand.z, cx=float(x), cy=float(y),
                                                     diameter_mm=cand.diameter_mm,
                                                     circularity=cand.circularity))
        z -= z_step

    return _cluster_by_xy(candidates, cluster_xy_mm, min_supporting_slices)


def _classify_ring(
    ring,
    z: float,
    min_d: float,
    max_d: float,
    min_circ: float,
) -> _SeatCandidate | None:
    """Build a Polygon from an interior ring and apply size + circularity
    filters. Returns the candidate or None if it doesn't pass."""
    try:
        hole = Polygon(ring)
        area = hole.area
        if area <= 0:
            return None
        d_eq = 2.0 * math.sqrt(area / math.pi)
        if not (min_d <= d_eq <= max_d):
            return None
        minx, miny, maxx, maxy = hole.bounds
        ex, ey = maxx - minx, maxy - miny
        max_side = max(ex, ey)
        if max_side <= 0:
            return None
        circ = min(ex, ey) / max_side
        if circ < min_circ:
            return None
        cx, cy = hole.centroid.x, hole.centroid.y
        return _SeatCandidate(z=z, cx=cx, cy=cy, diameter_mm=d_eq, circularity=circ)
    except Exception:
        return None


def _entrance_flat_diameter_run(cluster: list[_SeatCandidate]) -> int:
    """Run of consecutive (by Z) slices, STARTING AT THE ENTRANCE (the
    cluster's topmost/widest slice), whose diameter barely changes — the
    signature of a straight cylindrical through-hole rather than a
    tapering cone seat. A flat run elsewhere (e.g. the culet tip) is normal
    mesh-resolution noise, not a straight bore — only a flat ENTRANCE is
    suspicious. See `_MAX_FLAT_DIAMETER_RUN_SLICES` above."""
    ordered = sorted(cluster, key=lambda c: -c.z)
    run = 1
    for prev, cur in zip(ordered, ordered[1:]):
        if abs(cur.diameter_mm - prev.diameter_mm) <= _FLAT_DIAMETER_EPSILON_MM:
            run += 1
        else:
            break
    return run


def _cluster_by_xy(
    candidates: list[_SeatCandidate],
    cluster_xy_mm: float,
    min_supporting_slices: int,
) -> list[DetectedSeat]:
    """Group candidates into seats by XY proximity. The same physical seat
    appears in multiple consecutive Z slices — collapse those into one
    `DetectedSeat` and keep the *maximum* diameter (the entrance opening,
    which matches the stone girdle)."""
    clusters: list[list[_SeatCandidate]] = []
    for c in candidates:
        matched = False
        for cluster in clusters:
            mx = np.mean([item.cx for item in cluster])
            my = np.mean([item.cy for item in cluster])
            if math.hypot(c.cx - mx, c.cy - my) <= cluster_xy_mm:
                cluster.append(c)
                matched = True
                break
        if not matched:
            clusters.append([c])

    seats: list[DetectedSeat] = []
    for cluster in clusters:
        if len(cluster) < min_supporting_slices:
            continue
        if _entrance_flat_diameter_run(cluster) >= _MAX_FLAT_DIAMETER_RUN_SLICES:
            continue  # straight-walled through-hole (honeycomb/filigree), not a tapering seat
        diam = max(item.diameter_mm for item in cluster)
        cx = float(np.mean([item.cx for item in cluster]))
        cy = float(np.mean([item.cy for item in cluster]))
        avg_circ = float(np.mean([item.circularity for item in cluster]))
        seats.append(DetectedSeat(
            diameter_mm=float(diam),
            centre_xy=(cx, cy),
            n_slices=len(cluster),
            avg_circularity=avg_circ,
            z_top=float(max(item.z for item in cluster)),
            z_bottom=float(min(item.z for item in cluster)),
            d_bottom=float(min(cluster, key=lambda item: item.z).diameter_mm),
        ))
    seats.sort(key=lambda s: -s.diameter_mm)
    return seats


# ── StoneSummary builder ────────────────────────────────────────────────────

def build_stone_summary_for_stl(
    mesh: trimesh.Trimesh,
    weight_table,
):
    """Run `find_stone_seats(mesh)` and turn the result into a populated
    `StoneSummary`. Diameter → CZ carat via the weight table. Stones of
    identical size (within 0.05 mm) are grouped, matching the 3DM
    extractor's reporting style.

    Returns `None` when no seats are detected so callers can decide
    whether to fall back to VLM-only or to mark the piece as 'no stones'.
    """
    from catalog_organizer.core.schemas import StoneEntry, StoneSummary  # noqa: PLC0415

    seats = find_stone_seats(mesh)
    if not seats:
        return None

    # Group by quantised diameter (0.05 mm rounding) so a 12-seat pavé of
    # identical 1.50 mm openings is one StoneEntry with quantity=12.
    from collections import defaultdict
    buckets: dict[float, list[DetectedSeat]] = defaultdict(list)
    for s in seats:
        bucket_key = round(s.diameter_mm * 20) / 20.0
        buckets[bucket_key].append(s)

    entries: list = []
    for d, members in buckets.items():
        carat, _ = weight_table.estimate("round", d)
        qty = len(members)
        entries.append(StoneEntry(
            shape="round",
            size_mm=f"{d:.2f}",
            quantity=qty,
            estimated_carat_each=float(carat),
            estimated_total_carat=float(carat * qty),
            source="geometry",
            # Confidence reflects how well-supported the seat cluster is:
            # avg_circularity is a good proxy. Drop a notch vs 3DM block-
            # name detection because we're inferring from shape only.
            confidence=min(0.85, float(np.mean([m.avg_circularity for m in members]))),
        ))

    # Center stone = single largest (quantity-1) bucket
    qty1 = [e for e in entries if e.quantity == 1]
    if qty1:
        center = max(qty1, key=lambda e: e.estimated_carat_each)
        sides = [e for e in entries if e is not center]
    else:
        center = None
        sides = list(entries)

    total = sum(e.estimated_total_carat for e in entries)
    return StoneSummary(
        status="stone",
        center_stone=center,
        side_stones=sides,
        total_estimated_carat=float(total),
    )
