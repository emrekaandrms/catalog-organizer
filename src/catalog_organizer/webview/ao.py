"""Per-vertex ambient occlusion, baked by marching rays through a voxel grid.

Luxury-retail viewers rely on a BAKED occlusion map (a texture on a second UV set). The
catalogue's meshes have no UVs, so occlusion is baked per vertex instead. This is what
took the web render's edge contrast from 37 to 56 against a target of 55 (see
docs/RENDER_ENGINE.md); without it the metal reads flat and its edges go soft.

A trimesh ray intersector manages ~430 rays/s on a 70k-triangle piece -- about an hour for
one product -- so the rays are marched through a voxel grid with numpy instead. That takes
seconds, and scales with the vertex count rather than the triangle count.

Everything is expressed relative to the piece's size, because the catalogue holds a 3 mm
pendant and a 60 mm ring and a fixed radius would be wrong for one of them.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

# Reference piece: ray radius 0.35 on an extent of 3.0, voxel 0.01, start offset 0.03.
RADIUS_FRACTION = 0.117
VOXELS_ACROSS = 300
START_FRACTION = 0.01

# Rays are cast in vertex blocks so the (vertices x steps x 3) array stays small.
_BLOCK = 12000
# Surface samples taken per batch while voxelising (bounds the (triangles x samples x 3) array).
_SAMPLES_PER_CHUNK = 2_000_000


def _surface_voxels(verts: np.ndarray, faces: np.ndarray, pitch: float):
    """Occupancy grid of the surface: every voxel a triangle passes through. Returns (grid, origin).

    Triangles are sampled on a barycentric lattice at half-voxel spacing and each sample marks
    its voxel; no gap can open in the shell. trimesh's `voxelized` does the same by subdividing
    until every edge is under a voxel, and on a 128k-triangle ring that took 41 of the 49
    seconds an export needed and built arrays of 72 million entries (a MemoryError once, when
    other jobs were running). Marking the grid directly needs neither the sort nor the copies.
    """
    verts = np.asarray(verts, dtype=np.float64)
    origin = verts.min(axis=0) - 2.0 * pitch
    shape = np.ceil((verts.max(axis=0) + 2.0 * pitch - origin) / pitch).astype(int) + 1
    grid = np.zeros(shape, dtype=bool)
    tri = verts[np.asarray(faces, dtype=np.int64)]
    longest = np.linalg.norm(np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1],
                                       tri[:, 0] - tri[:, 2]], axis=1), axis=2).max(axis=1)
    steps = np.maximum(1, np.ceil(longest / (0.5 * pitch)).astype(int))
    for k in np.unique(steps):
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = i + j <= k
        weights = np.stack([1.0 - (i[keep] + j[keep]) / k, i[keep] / k, j[keep] / k], axis=1)
        which = np.nonzero(steps == k)[0]
        per_chunk = max(1, int(_SAMPLES_PER_CHUNK // len(weights)))
        for lo in range(0, len(which), per_chunk):
            pts = np.einsum("mw,twc->tmc", weights, tri[which[lo:lo + per_chunk]]).reshape(-1, 3)
            at = np.floor((pts - origin) / pitch).astype(np.int64)
            grid[at[:, 0], at[:, 1], at[:, 2]] = True
    return grid, origin


def _buried(occ: np.ndarray, origin: np.ndarray, pitch: float, points: np.ndarray) -> np.ndarray:
    """True for points with no open air within a voxel: they lie inside the piece's own material.

    Catalogue models are often several overlapping solids (a plate and a bezel through it; the
    weight over-count came from the same thing). A vertex of one body that sits inside another
    is correctly 'blocked' on every ray and bakes to ~0, but its triangles poke through the
    visible surface and show up as dark patches shaped like those triangles: on a necklace,
    7,395 of 64,180 vertices were buried, with mean occlusion 0.17 against 0.68 for the rest.
    Air reachable from outside the grid is flood-filled; a point no outside air touches is buried.
    An open (non-watertight) shell lets the flood in everywhere, so nothing is buried there.
    """
    air, _count = ndimage.label(np.pad(~occ, 1, constant_values=True))
    outside = (air == air[0, 0, 0])[1:-1, 1:-1, 1:-1]
    exposed = ndimage.binary_dilation(outside, structure=np.ones((3, 3, 3), dtype=bool))
    idx = np.floor((points - origin) / pitch).astype(np.int64)
    inside_grid = np.all((idx >= 0) & (idx < np.array(occ.shape)), axis=1)
    buried = np.zeros(len(points), dtype=bool)
    i = idx[inside_grid]
    buried[inside_grid] = ~exposed[i[:, 0], i[:, 1], i[:, 2]]
    return buried


def bake_vertex_ao(verts: np.ndarray, faces: np.ndarray, points: np.ndarray,
                   normals: np.ndarray, *, extent: float, rays: int = 48,
                   steps: int = 22) -> np.ndarray:
    """One value per point: 1 = open sky, 0 = fully enclosed.

    verts, faces   the geometry to occlude WITH (voxelised; open shells are fine)
    points,normals where to measure; need not be the same vertices (split normals)
    extent         the piece's largest dimension; sets ray length and voxel size
    """
    radius = RADIUS_FRACTION * extent
    pitch = extent / VOXELS_ACROSS
    start = START_FRACTION * extent

    # Surface voxels only, NOT filled. Filling floods any region the voxel shell happens to
    # close -- a dish inside a ring, the gap between two wings -- and turns the air above a
    # flat face into 'solid', which drove a brooch's whole disc to zero occlusion and made it
    # render chocolate-brown against the server render's olive gold. Rays start on the
    # outside of a surface, so a shell is all they ever need to hit.
    occ, origin = _surface_voxels(verts, faces, pitch)
    shape = np.array(occ.shape)
    buried = _buried(occ, origin, pitch, np.asarray(points, dtype=np.float64))

    # cosine-weighted hemisphere directions in tangent space (golden-angle spiral)
    k = np.arange(rays) + 0.5
    r = np.sqrt(k / rays)
    theta = np.pi * (1 + 5 ** 0.5) * k
    local = np.stack([r * np.cos(theta), r * np.sin(theta),
                      np.sqrt(np.maximum(0.0, 1.0 - r * r))], axis=-1)
    dist = start + (radius - start) * (np.arange(steps) / (steps - 1)) ** 1.6

    out = np.ones(len(points))
    for lo in range(0, len(points), _BLOCK):
        p = points[lo:lo + _BLOCK]
        nrm = normals[lo:lo + _BLOCK]
        # a few vertices carry no usable normal; they get no occlusion rather than garbage
        degenerate = np.linalg.norm(nrm, axis=1) < 0.5
        nrm = np.where(degenerate[:, None], np.array([[0.0, 0.0, 1.0]]), nrm)
        n = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        helper = np.where(np.abs(n[:, :1]) > 0.9, np.array([[0.0, 1.0, 0.0]]),
                          np.array([[1.0, 0.0, 0.0]]))
        t = np.cross(helper, n)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        b = np.cross(n, t)

        blocked = np.zeros(len(p))
        for d in local:
            w = d[0] * t + d[1] * b + d[2] * n
            pts = p[:, None, :] + w[:, None, :] * dist[None, :, None]
            idx = np.floor((pts - origin) / pitch).astype(np.int64)
            inside = np.all((idx >= 0) & (idx < shape), axis=-1)
            idx = np.clip(idx, 0, shape - 1)
            hit = occ[idx[..., 0], idx[..., 1], idx[..., 2]] & inside
            blocked += hit.any(axis=1)
        ao = 1.0 - blocked / rays
        ao[degenerate] = 1.0
        out[lo:lo + _BLOCK] = ao
    out[buried] = 1.0        # inside another body: invisible itself, so it must not darken its triangles
    return out
