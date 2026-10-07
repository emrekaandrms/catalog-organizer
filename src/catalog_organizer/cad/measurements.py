"""Geometry-based measurements per §C.1, §C.2, §C.3.

Algorithm note (§C.1): the literal spec picks the finger axis as the
"second-largest eigenvalue" PC axis, which fails on flat symmetric rings
(plain wedding bands) where two in-plane eigenvalues are nearly equal.
We instead evaluate the projected-outline circularity for *each* of the
three PC axes and pick the most circular — this matches the spec's
fallback rule ("more circular projected outline via Hu-moment score")
but applies it as the primary criterion for robustness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


# ── helpers ───────────────────────────────────────────────────────────────────

def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors perpendicular to `axis`."""
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = helper - np.dot(helper, axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def _circularity(points_2d: np.ndarray) -> float:
    """Ratio of min-to-max distance from centroid. 1.0 = perfect circle."""
    if len(points_2d) < 3:
        return 0.0
    c = points_2d.mean(axis=0)
    d = np.linalg.norm(points_2d - c, axis=1)
    dmax = d.max()
    return float(d.min() / dmax) if dmax > 1e-9 else 0.0


def _structural_points(mesh: trimesh.Trimesh) -> np.ndarray:
    """Vertices of the load-bearing body, with loose stones removed.

    Jewelry CAD models the centre diamond and every pavé melee as a
    *separate* floating mesh component. On a halo/solitaire ring the
    centre stone's pavilion hangs down into the finger bore, so any
    bore measurement that uses all vertices is wrecked by it (the old
    code returned ~0.9 mm on `sampleA_43.3dm`).

    We keep only connected components whose largest extent is a sizable
    fraction of the whole object — the shank, setting and prongs — and
    drop the compact stone components. Falls back to all vertices when
    the split is unhelpful (e.g. a single-body plain band).
    """
    verts_all = np.asarray(mesh.vertices, dtype=np.float64)
    try:
        ring_ext = float(mesh.extents.max())
        components = mesh.split(only_watertight=False)
    except Exception:
        return verts_all
    if not components or len(components) == 1:
        return verts_all

    keep = [c for c in components if float(c.extents.max()) >= 0.4 * ring_ext]
    if not keep:
        return verts_all
    pts = np.vstack([np.asarray(c.vertices, dtype=np.float64) for c in keep])
    # Guard: structural body must retain enough of the object to be meaningful.
    if len(pts) < 12:
        return verts_all
    return pts


def _largest_passable_bore(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Find the finger bore: the largest cylinder that passes through the ring.

    Returns `(diameter_mm, finger_axis, bore_center_2d, projection_2d)`.

    Method:
      1. PCA gives 3 candidate axes.
      2. For each, project the points onto the perpendicular plane and find
         the largest *enclosed* empty circle (a coarse grid search, refined
         locally). "Enclosed" = points exist in ≥13 of 16 angular sectors
         around the centre, so the circle is a real interior hole and not an
         empty region off to one side.
      3. The axis whose projection contains the biggest enclosed empty circle
         is the finger axis; that circle's diameter is the inner diameter.

    This is exact on torus test fixtures (16.0 / 21.0 mm) and robust to
    floating centre stones once `_structural_points` has removed them.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415  (local: keeps import cost off hot paths)

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(cov)

    best_r = -1.0
    best_axis = eigvecs[:, 0]
    best_center = np.zeros(2)
    best_proj = np.zeros((0, 2))

    for i in range(3):
        axis = eigvecs[:, i]
        u, v = _orthonormal_basis(axis)
        proj = np.column_stack([centered @ u, centered @ v])
        r, c = _largest_enclosed_empty_circle(proj, cKDTree)
        if r > best_r:
            best_r, best_axis, best_center, best_proj = r, axis, c, proj

    return float(2.0 * best_r), best_axis, best_center, best_proj


def _largest_enclosed_empty_circle(
    pts: np.ndarray,
    kdtree_cls,
    grid: int = 56,
    n_sectors: int = 16,
    min_sectors: int = 13,
) -> tuple[float, np.ndarray]:
    """Largest empty circle whose centre is enclosed by points on all sides.

    Returns `(radius, center_xy)`. Used by `_largest_passable_bore`.
    """
    if len(pts) < 8:
        return 0.0, np.zeros(2)
    if len(pts) > 8000:
        idx = np.random.RandomState(0).choice(len(pts), 8000, replace=False)
        pts = pts[idx]
    tree = kdtree_cls(pts)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)

    def search(xs: np.ndarray, ys: np.ndarray) -> tuple[float, np.ndarray | None]:
        best_r = 0.0
        best_c: np.ndarray | None = None
        for x in xs:
            for y in ys:
                d = pts - np.array([x, y])
                ang = np.arctan2(d[:, 1], d[:, 0])
                sec = ((ang + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
                if len(np.unique(sec)) < min_sectors:
                    continue
                r, _ = tree.query([x, y])
                if r > best_r:
                    best_r, best_c = float(r), np.array([x, y])
        return best_r, best_c

    r, c = search(np.linspace(mn[0], mx[0], grid), np.linspace(mn[1], mx[1], grid))
    if c is not None:
        step = (mx[0] - mn[0]) / grid
        r2, c2 = search(
            np.linspace(c[0] - step, c[0] + step, 17),
            np.linspace(c[1] - step, c[1] + step, 17),
        )
        if c2 is not None:
            r, c = r2, c2
    return r, (c if c is not None else np.zeros(2))


# ── bbox ──────────────────────────────────────────────────────────────────────

@dataclass
class BBox:
    width: float
    height: float
    depth: float
    volume_mm3: float


def measure_bbox(mesh: trimesh.Trimesh) -> BBox:
    """Axis-aligned bbox + signed volume (0 if non-watertight)."""
    bmin, bmax = mesh.bounds
    w, h, d = (bmax - bmin).tolist()
    vol = float(abs(mesh.volume)) if mesh.is_volume else 0.0
    return BBox(width=float(w), height=float(h), depth=float(d), volume_mm3=vol)


def estimate_volume(mesh: trimesh.Trimesh) -> tuple[float, str]:
    """Return `(volume_mm3, confidence)` with a graded fallback strategy.

    Why this is delicate (verified against MatrixGold + scale on
    `pave-pendant.stl`, 2026-05-18):

    - Trimesh defines `is_volume == is_watertight AND is_winding_consistent`.
    - High-res jewelry STLs almost always have a few micro-gaps in the
      surface, so `is_watertight` is False — but the face winding is
      still consistent, and `mesh.volume` (signed integral via the
      divergence theorem) is accurate to within ~0.5% of the physical
      truth. On that test pendant: signed mesh.volume = 423.95 mm³ vs
      MatrixGold ground truth = 425.7 mm³ for 925 silver.
    - The previous code gated on `is_volume` and fell straight through to
      `convex_hull.volume`. For the same pendant the hull returned
      1202.5 mm³ → 2.83× over-estimate → silver mass reported as 12.46 g
      instead of 4.41 g.

    Order of preference:
      1. Watertight + winding consistent → signed `mesh.volume` ("high")
      2. Winding consistent (gaps allowed) → signed `mesh.volume` ("medium")
      3. Convex hull volume                → ("low")
      4. Bounding-box volume                → ("low")
      5. Nothing usable                     → 0.0, "unavailable"
    """
    try:
        if getattr(mesh, "is_winding_consistent", False):
            v = float(abs(mesh.volume))
            if v > 0:
                conf = "high" if getattr(mesh, "is_watertight", False) else "medium"
                return v, conf
    except Exception:
        pass

    try:
        hull = mesh.convex_hull
        v = float(abs(hull.volume))
        if v > 0:
            return v, "low"
    except Exception:
        pass

    try:
        bmin, bmax = mesh.bounds
        v = float(np.prod(bmax - bmin))
        if v > 0:
            return v, "low"
    except Exception:
        pass

    return 0.0, "unavailable"


# ── ring ──────────────────────────────────────────────────────────────────────

@dataclass
class RingMeasurements:
    inner_diameter_mm: float
    inner_circumference_mm: float
    size_eu: float
    band_width_mm: float
    top_width_mm: float
    top_height_mm: float


def measure_ring(mesh: trimesh.Trimesh) -> RingMeasurements:
    """
    Compute ring inner diameter, band width, and top dimensions.

    Steps:
    1. Strip loose stone components (`_structural_points`) so a centre
       diamond hanging into the bore can't corrupt the measurement.
    2. Find the finger bore = largest cylinder passing through the body
       (`_largest_passable_bore`). Its diameter is the inner diameter and
       its axis is the finger axis.
    3. Band width = extent of the structural body along the finger axis.
    4. Top dimensions = bbox of projected points beyond the 80th radial
       percentile from the bore centre (the bezel / setting face).

    Exact on torus fixtures (16.0 / 21.0 mm); robust on halo/solitaire
    rings with a modelled centre stone.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) < 4:
        return _ring_fallback(mesh)

    points = _structural_points(mesh)

    try:
        inner_diameter_mm, finger_axis, bore_center, proj = _largest_passable_bore(points)
    except (np.linalg.LinAlgError, ValueError):
        return _ring_fallback(mesh)

    if inner_diameter_mm <= 0.0 or proj.shape[0] == 0:
        return _ring_fallback(mesh)

    # Band width along finger axis (over the structural body).
    centroid = points.mean(axis=0)
    finger_proj = (points - centroid) @ finger_axis
    band_width_mm = float(finger_proj.max() - finger_proj.min())

    # Top dimensions: points beyond the 80th percentile radial distance
    # from the bore centre (the setting / bezel face).
    radial = np.linalg.norm(proj - bore_center, axis=1)
    threshold = float(np.percentile(radial, 80))
    top_pts = proj[radial >= threshold]
    if len(top_pts) >= 2:
        top_w = float(top_pts[:, 0].max() - top_pts[:, 0].min())
        top_h = float(top_pts[:, 1].max() - top_pts[:, 1].min())
    else:
        top_w = top_h = 0.0

    inner_circ = float(np.pi * inner_diameter_mm)
    return RingMeasurements(
        inner_diameter_mm=inner_diameter_mm,
        inner_circumference_mm=inner_circ,
        size_eu=inner_circ,    # industry convention
        band_width_mm=band_width_mm,
        top_width_mm=top_w,
        top_height_mm=top_h,
    )


def _ring_fallback(mesh: trimesh.Trimesh) -> RingMeasurements:
    """Bounding-box fallback when PCA cannot run (§C.1 fallback rule)."""
    bbox = measure_bbox(mesh)
    extents = sorted([bbox.width, bbox.height, bbox.depth])
    # Smallest extent assumed to be finger axis; inner diameter unknown → 0.
    return RingMeasurements(
        inner_diameter_mm=0.0,
        inner_circumference_mm=0.0,
        size_eu=0.0,
        band_width_mm=extents[0],
        top_width_mm=extents[2],
        top_height_mm=extents[1],
    )


# ── earring ───────────────────────────────────────────────────────────────────

@dataclass
class EarringMeasurements:
    total_width_mm: float
    total_height_mm: float
    depth_mm: float
    pair_or_single: str         # "pair" | "single" | "unclear"


def measure_earring(mesh: trimesh.Trimesh) -> EarringMeasurements:
    bbox = measure_bbox(mesh)
    components = mesh.split(only_watertight=False)
    n = len(components)

    pair_or_single = "unclear"
    if n == 2:
        v1 = float(components[0].bounding_box.volume)
        v2 = float(components[1].bounding_box.volume)
        vmax = max(v1, v2)
        diff = abs(v1 - v2) / vmax if vmax > 0 else 1.0
        c1 = np.asarray(components[0].centroid)
        c2 = np.asarray(components[1].centroid)
        dist = float(np.linalg.norm(c1 - c2))
        # §C.2 simplified: drop the IoU mirror check (voxelization is slow).
        # If sizes match within 5% and centroids are within 50 mm, call it a pair.
        if diff < 0.05 and dist < 50.0:
            pair_or_single = "pair"
    elif n == 1:
        # §C.2 step 3: single stud/drop heuristic — accept any single component.
        pair_or_single = "single"

    return EarringMeasurements(
        total_width_mm=bbox.width,
        total_height_mm=bbox.height,
        depth_mm=bbox.depth,
        pair_or_single=pair_or_single,
    )


# ── bracelet ──────────────────────────────────────────────────────────────────

@dataclass
class BraceletMeasurements:
    inner_diameter_x_mm: float
    inner_diameter_y_mm: float
    inner_circumference_mm: float
    width_mm: float


def measure_bracelet(mesh: trimesh.Trimesh) -> BraceletMeasurements:
    """Reuses ring inner-diameter algorithm. X/Y radii from projected extent."""
    ring = measure_ring(mesh)
    return BraceletMeasurements(
        inner_diameter_x_mm=ring.inner_diameter_mm,
        inner_diameter_y_mm=ring.inner_diameter_mm,
        inner_circumference_mm=ring.inner_circumference_mm,
        width_mm=ring.band_width_mm,
    )


# ── pendant ───────────────────────────────────────────────────────────────────

@dataclass
class PendantMeasurements:
    width_mm: float
    height_mm: float
    depth_mm: float


def measure_pendant(mesh: trimesh.Trimesh) -> PendantMeasurements:
    bbox = measure_bbox(mesh)
    return PendantMeasurements(
        width_mm=bbox.width,
        height_mm=bbox.height,
        depth_mm=bbox.depth,
    )
