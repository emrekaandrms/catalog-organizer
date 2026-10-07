"""Stones for a piece whose file has none: put a brilliant in every seat the geometry shows.

A casting-model STL is the piece before it goes to the machine: the stones are not in it, but the
seats are cut into the metal. The analysis already measures those seats (`cad.stl_stone_seats`)
to estimate carats; this places a stone in each so the render shows what the piece is for.

The detector slices the mesh along +Z, so it sees the seats that open toward +Z. To see the others
the piece is turned so that each of the six axis directions in turn becomes +Z, the detector runs
again, and what it finds is turned back. Every filter below exists because of a measured false
positive on the catalogue's own files:

  * a ring's finger bore read as a 7.9 mm "seat" whose cluster was 19.6 mm deep -> depth limit;
  * relief pockets in a sculpted pendant (cloth folds) read as seats -> circularity and taper;
  * holes drilled from inside a band, covered by the band on top -> the sky must be open above
    the entrance (this also removes anything facing the middle of a ring: the other side of the
    band is in the way);
  * seats reported far from any metal (the detector once measured every centre in the frame of
    its own slice, so a piece away from the origin got stones in mid-air) -> must lie on metal.
Where the geometry does not support a stone, none is placed: a missing stone is a smaller lie than
a stone on a fold of cloth. A seat that opens at an angle between the axes is seen only if it is
within about 45 degrees of one of them.

The scan is the slow part (six passes over the mesh) and its raw findings are cached per source
file; the filters run on the cached findings, so tuning them never needs a rescan.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_SLICES = 3              # a cone seat is several tenths of a millimetre deep at least
MIN_DEPTH_RATIO = 0.10      # depth / diameter: a seat is not a film
MAX_DEPTH_RATIO = 1.5       # ... and not a bore: a cone seat is ~0.45 D deep, a ring bore is > 2 D
MIN_CIRCULARITY = 0.85      # bbox short/long side of the slice, averaged over the seat's slices
MAX_TAPER = 0.97            # diameter at the bottom / at the entrance: a seat narrows, a tube does not
MAX_SEATS = 800
REACH_RADII = 1.5           # the nearest metal must be within this many seat radii of the entrance
REACH_SLACK_MM = 0.3
SAME_SEAT = 0.6             # two finds closer than this many diameters are one seat seen twice
SCAN_VERSION = 2            # bump when the scan itself changes (not the filters): cached scans are dropped

AXES = tuple(np.array(v, dtype=float) for v in (
    (0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)))


@dataclass(frozen=True)
class Candidate:
    """What the scan saw: one seat-like hole, with everything the filters look at."""
    entrance: np.ndarray        # centre of the opening, in the file's frame
    axis: np.ndarray            # unit vector out of the seat (the stone's crown points this way)
    diameter_mm: float          # at the entrance
    circularity: float
    n_slices: int
    depth_mm: float
    taper: float                # diameter at the bottom / at the entrance
    open_above: bool            # nothing covers the entrance along the axis
    on_metal: bool              # metal lies within about a radius of the entrance


@dataclass(frozen=True)
class PlacedSeat:
    centre: np.ndarray          # the entrance, in the file's frame
    axis: np.ndarray
    diameter_mm: float
    circularity: float


def accept(c: Candidate) -> bool:
    """Is this a seat a stone can sit in?"""
    return (c.open_above and c.on_metal
            and c.n_slices >= MIN_SLICES
            and MIN_DEPTH_RATIO * c.diameter_mm <= c.depth_mm <= MAX_DEPTH_RATIO * c.diameter_mm
            and c.circularity >= MIN_CIRCULARITY
            and c.taper <= MAX_TAPER)


def _open_flags(verts, faces, seats) -> list[bool]:
    """For each seat: is the sky above its entrance clear of metal?"""
    from catalog_organizer.webview import ao

    extent = float(np.ptp(verts, axis=0).max())
    pitch = extent / 300.0
    grid, origin = ao._surface_voxels(verts, faces, pitch)
    limit = np.array(grid.shape) - 1
    flags = []
    for s in seats:
        r = 0.25 * s.diameter_mm
        lo = np.floor((np.array([s.centre_xy[0] - r, s.centre_xy[1] - r, s.z_top + 3 * pitch]) - origin) / pitch)
        hi = np.floor((np.array([s.centre_xy[0] + r, s.centre_xy[1] + r, 1e9]) - origin) / pitch)
        lo = np.clip(lo.astype(int), 0, limit)
        hi = np.clip(hi.astype(int), 0, limit)
        flags.append(bool(np.any(lo > hi)
                          or not grid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1].any()))
    return flags


def _on_metal(verts, seats) -> list[bool]:
    """A seat is a hole IN the metal: its wall is within about a radius of its entrance."""
    from scipy.spatial import cKDTree

    tree = cKDTree(verts)
    out = []
    for s in seats:
        gap, _ = tree.query([s.centre_xy[0], s.centre_xy[1], s.z_top])
        out.append(bool(gap <= 0.5 * s.diameter_mm * REACH_RADII + REACH_SLACK_MM))
    return out


def rotation_to_z(axis) -> np.ndarray:
    """3x3 rotation that turns `axis` into +Z."""
    import trimesh

    return trimesh.geometry.align_vectors(np.asarray(axis, dtype=float),
                                          np.array([0.0, 0.0, 1.0]))[:3, :3]


def scan(metal, *, axes=AXES) -> list[Candidate]:
    """Every seat-like hole the detector sees along the given directions, unfiltered."""
    import trimesh

    from catalog_organizer.cad.stl_stone_seats import find_stone_seats

    verts, faces = np.asarray(metal[0], dtype=float), np.asarray(metal[1])
    body_centre = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    out: list[Candidate] = []
    for axis in axes:
        rot = rotation_to_z(axis)
        local = (verts - body_centre) @ rot.T                  # the piece, with `axis` turned to +Z
        seats = find_stone_seats(trimesh.Trimesh(local, faces, process=False))
        if not seats:
            continue
        open_flags = _open_flags(local, faces, seats)
        near_metal = _on_metal(local, seats)
        for s, is_open, near in zip(seats, open_flags, near_metal):
            if s.z_top is None or s.z_bottom is None or s.d_bottom is None:
                continue
            entrance = rot.T @ np.array([s.centre_xy[0], s.centre_xy[1], s.z_top]) + body_centre
            out.append(Candidate(entrance, np.asarray(axis, dtype=float), float(s.diameter_mm),
                                 float(s.avg_circularity), int(s.n_slices), float(s.z_top - s.z_bottom),
                                 float(s.d_bottom / s.diameter_mm), is_open, near))
    return out


def choose(candidates: list[Candidate]) -> list[PlacedSeat]:
    """The accepted candidates, with a seat seen from two axes kept once (the rounder reading)."""
    kept: list[PlacedSeat] = []
    for c in sorted((c for c in candidates if accept(c)), key=lambda c: (-c.diameter_mm, -c.circularity)):
        if any(np.linalg.norm(c.entrance - k.centre) < SAME_SEAT * max(c.diameter_mm, k.diameter_mm)
               for k in kept):
            continue
        kept.append(PlacedSeat(c.entrance, c.axis, c.diameter_mm, c.circularity))
    return kept[:MAX_SEATS]


def detect(metal) -> list[PlacedSeat]:
    """The seats a stone can sit in, in the file's own frame, largest first."""
    return choose(scan(metal))


def scan_cached(source: Path | None, metal) -> list[Candidate]:
    """`scan`, remembered per source file (the scan is the slow part of opening a stoneless STL)."""
    from catalog_organizer.core.paths import cache_dir

    if source is None or not Path(source).exists():
        return scan(metal)
    st = Path(source).stat()
    key = hashlib.sha1(f"{Path(source).resolve()}|{st.st_size}|{st.st_mtime_ns}|{SCAN_VERSION}".encode()).hexdigest()[:20]
    folder = cache_dir() / "web" / "seats"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{key}.json"
    if path.exists():
        try:
            return [Candidate(np.array(r["e"]), np.array(r["a"]), r["d"], r["q"], r["n"], r["z"],
                              r["t"], r["o"], r["m"]) for r in json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, ValueError, KeyError):
            pass
    found = scan(metal)
    path.write_text(json.dumps([{"e": c.entrance.tolist(), "a": c.axis.tolist(), "d": c.diameter_mm,
                                 "q": c.circularity, "n": c.n_slices, "z": c.depth_mm, "t": c.taper,
                                 "o": c.open_above, "m": c.on_metal} for c in found]), encoding="utf-8")
    return found


def stones_for(seats: list[PlacedSeat]):
    """(verts, faces) of one standard brilliant per seat, or None if there are none."""
    from catalog_organizer.render import facets

    if not seats:
        return None
    unit_verts, unit_faces = facets._brilliant_mesh()          # girdle radius 1, crown toward +Z
    out_v, out_f, offset = [], [], 0
    for seat in seats:
        # the girdle sits at the seat's entrance: the crown stands above the metal, the pavilion
        # goes down into the cone
        turn = rotation_to_z(seat.axis).T                       # +Z -> the seat's axis
        out_v.append((unit_verts * (0.5 * seat.diameter_mm)) @ turn.T + seat.centre)
        out_f.append(unit_faces + offset)
        offset += len(unit_verts)
    return np.vstack(out_v), np.vstack(out_f)


def seat_stones(metal, source: Path | None = None):
    """Scan (cached), filter, place: (verts, faces) of the stones, or None."""
    return stones_for(choose(scan_cached(source, metal)))
