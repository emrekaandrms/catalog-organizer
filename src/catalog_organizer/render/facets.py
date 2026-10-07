"""Reduce a cut stone to the set of planes its facets lie on.

The data is a flat list of facets, each one a point on the facet and its outward
normal. The shader generator bakes every one of them into the GLSL as a compile-time
constant and unrolls the intersection loop over the lot. No texture, no acceleration
structure, no proxy: the ray is intersected against the stone's actual facet planes,
analytically, on every bounce.

That is the piece this renderer was missing. A cone proxy was tried and made
things measurably worse (local contrast 4.48 -> 1.15) because a smooth surface
makes neighbouring rays CONVERGE; real facets make them diverge, and that
divergence is the flash. A screen-space back-face buffer did better (2.30 ->
4.36) but it can only ever see the one surface nearest the camera.

A cut stone is a tiny number of planes -- a round brilliant is 57 facets plus
a girdle -- so the exact thing is affordable.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# Two facets count as the same plane when their normals agree to about a
# degree and their offsets to a thousandth of the stone's radius. Rhino's
# tessellation splits one facet into several triangles whose normals differ in
# the last few digits, and without merging those the shader would carry 300
# duplicate planes instead of 60.
_NORMAL_TOLERANCE = 0.9998          # cos of ~1.1 degrees
_OFFSET_TOLERANCE = 1e-3

# Slivers contribute nothing but noise to the plane set.
_MIN_AREA_FRACTION = 1e-4


@dataclass(frozen=True)
class StoneShape:
    """A stone in its own normalised frame: centred, radius 1, table up."""
    points: np.ndarray          # (n, 3) a point on each facet
    normals: np.ndarray         # (n, 3) outward unit normal of each facet
    axis: np.ndarray            # table direction, in the ORIGINAL frame
    centre: np.ndarray          # centroid, in the original frame
    radius: float               # scale that maps original -> normalised

    def __len__(self) -> int:
        return len(self.normals)


def _stone_axis(points: np.ndarray) -> np.ndarray:
    """Symmetry axis, pointing from culet toward table.

    A brilliant is wider than it is tall, so the axis is the direction of
    least variance. The sign comes from the cut's own asymmetry: the pavilion
    is about 43% of the diameter deep and the crown only 14.5% tall, so the
    culet end sits much further from the centroid than the table end.
    """
    centred = points - points.mean(axis=0)
    _values, vectors = np.linalg.eigh(centred.T @ centred)
    axis = vectors[:, 0]
    projection = centred @ axis
    if abs(projection.min()) < abs(projection.max()):
        axis = -axis
    return axis / np.linalg.norm(axis)


def local_basis(axis: np.ndarray) -> np.ndarray:
    """Rows [right, forward, axis] for a stone's own frame.

    Shared by `extract` and by the per-stone frames handed to the shader: if
    the two disagreed by so much as a roll about the axis, the baked planes
    would be rotated relative to the geometry they are supposed to bound.
    """
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(helper @ axis)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    right = np.cross(helper, axis)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(axis, right), axis])


def stone_frames(verts: np.ndarray, faces: np.ndarray) -> list[dict]:
    """Centre, axis basis and radius for every individual stone.

    One compiled shader serves the lot -- these are what tell it, per stone,
    where that stone sits and which way it is turned. It is what a scene
    graph would give for free if every stone were its own node.
    """
    out = []
    for body in components(verts, faces):
        points = np.asarray(body.vertices, dtype=np.float64)
        centre = points.mean(axis=0)
        axis = _stone_axis(points)
        radius = float(np.abs(points - centre).max())
        if radius <= 0:
            continue
        out.append({"centre": centre, "basis": local_basis(axis),
                    "radius": radius,
                    "verts": points, "faces": np.asarray(body.faces)})
    return out


def extract(verts: np.ndarray, faces: np.ndarray) -> StoneShape:
    """Collapse a tessellated stone into its distinct facet planes."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    centre = verts.mean(axis=0)
    axis = _stone_axis(verts)
    radius = float(np.abs(verts - centre).max())
    if radius <= 0:
        raise ValueError("degenerate stone")

    # Normalised frame: table along +Z, centred, radius 1. Every stone of the
    # same cut then shares one plane set, which is what lets a single compiled
    # shader serve all 191 stones on a pave piece.
    basis = local_basis(axis)                                # rows
    local = (verts - centre) @ basis.T / radius

    triangles = local[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    keep = areas > (areas.sum() * _MIN_AREA_FRACTION)
    if not keep.any():
        raise ValueError("stone has no usable faces")

    normals = cross[keep] / np.linalg.norm(cross[keep], axis=1, keepdims=True)
    points = triangles[keep][:, 0]

    # Force every normal outward. A cut stone is convex, so the outward
    # direction at a facet is simply away from the centroid. This is not
    # tidying: a Brep writes a separate render mesh per face and their
    # windings do not agree, so a third of the normals come out pointing INTO
    # the stone. Measured before this line, vertices sat up to 2.0 units --
    # two full stone radii -- on the wrong side of their own facet's plane,
    # and any ray traced against that set leaves immediately.
    flipped = np.einsum("ij,ij->i", normals, points) < 0.0
    normals[flipped] *= -1.0

    offsets = np.einsum("ij,ij->i", normals, points)
    weights = areas[keep]

    # Merge by (normal, offset), biggest facet first so the representative
    # point sits on real geometry rather than on a sliver at the rim.
    order = np.argsort(-weights)
    chosen_n: list[np.ndarray] = []
    chosen_d: list[float] = []
    for index in order:
        n, d = normals[index], offsets[index]
        for existing_n, existing_d in zip(chosen_n, chosen_d):
            if (float(n @ existing_n) > _NORMAL_TOLERANCE
                    and abs(d - existing_d) < _OFFSET_TOLERANCE):
                break
        else:
            chosen_n.append(n)
            chosen_d.append(float(d))

    plane_normals = np.asarray(chosen_n)
    plane_points = plane_normals * np.asarray(chosen_d)[:, None]
    return StoneShape(points=plane_points, normals=plane_normals,
                      axis=axis, centre=centre, radius=radius)


def components(verts: np.ndarray, faces: np.ndarray):
    """Split welded gem geometry into individual stones.

    The weld is not optional: a Brep writes a separate render mesh per face,
    so without it every FACET reads as its own body -- which is exactly how a
    catalogue full of proper brilliants first measured as 2-face plates.
    """
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh.split(only_watertight=False)


def representative(verts: np.ndarray, faces: np.ndarray) -> StoneShape:
    """The facet planes of the largest stone, to stand for the whole set.

    Every stone of one cut is the same shape at a different size and pose, so
    one plane set serves them all -- so the plane set is computed once per
    cut and reused across the catalogue.
    """
    bodies = components(verts, faces)
    if not bodies:
        raise ValueError("no stones found")
    biggest = max(bodies, key=lambda c: float(c.extents.max()))
    return extract(np.asarray(biggest.vertices), np.asarray(biggest.faces))


# -- when the CAD has no cut to read ------------------------------------------

# How much of a stone's surface the 60 largest planes must carry before its own
# geometry is trusted as a cut. A round brilliant has 57 facets, so on a real
# one this is nearly everything; measured on the catalogue:
#
#     sampleA_12 hero stone   60 planes carry 97% of the surface   121 planes
#     JCAD-000000140 hero     60 planes carry 37%                  365 planes
#
# The second is not a cut stone at all. Its wireframe is a flat table over a
# smoothly revolved cone -- a placeholder the designer dropped into the seat --
# and 364 of its 365 planes carry under 1% of the surface each. Tracing that
# gives a polished bowl with a band of stripes where the revolve is tessellated,
# which is exactly what it is. No shader can reflect a facet that the geometry
# does not have.
_CUT_PLANE_BUDGET = 60
_CUT_COVERAGE = 0.75


def cut_coverage(verts: np.ndarray, faces: np.ndarray) -> float:
    """Fraction of the surface carried by the largest `_CUT_PLANE_BUDGET`."""
    shape = extract(verts, faces)
    verts = np.asarray(verts, dtype=np.float64)
    triangles = verts[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = areas.sum()
    if total <= 0:
        return 0.0
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True),
                                 1e-12)
    centre = verts.mean(axis=0)
    outward = np.einsum("ij,ij->i", normals, triangles[:, 0] - centre) < 0.0
    normals[outward] *= -1.0

    nearest = (normals @ shape.normals.T).argmax(axis=1)
    per_plane = np.zeros(len(shape))
    np.add.at(per_plane, nearest, areas)
    biggest = np.sort(per_plane)[::-1][:_CUT_PLANE_BUDGET]
    return float(biggest.sum() / total)


def _brilliant_mesh() -> tuple[np.ndarray, np.ndarray]:
    """A 57-facet round brilliant at textbook proportions.

    Everything is in girdle-radius units. The numbers are the standard ones a
    cutter works to, not invented: table 57% of the girdle DIAMETER measured
    across the flats, crown 16.2% of the diameter tall, girdle 3% thick,
    pavilion 43.1% deep, star facets reaching 55% of the way from the table to
    the girdle and lower-girdle halves 77% of the way from the girdle to the
    culet.

    The star and lower-girdle points are SOLVED onto the bezel and pavilion
    planes rather than placed at a plausible height. Placing them cost an
    afternoon the first time: a kite whose four corners are not coplanar is
    not a facet, it is a fold, and a folded kite makes the body non-convex --
    table corners then measured 0.055 (5.5% of the radius) outside their own
    neighbouring plane. The tracer takes the nearest plane the ray heads
    toward as the exit point and that shortcut is only exact on a convex body,
    so the fold would have leaked rays out through the crown.
    """
    half = np.pi / 8
    table_apothem = 0.57
    table_radius = table_apothem / np.cos(half)
    crown_height = 0.324                      # 0.162 of the diameter
    girdle_half = 0.03                        # 0.03 of the diameter, halved
    pavilion_depth = 0.862                    # 0.431 of the diameter
    star_radius = table_apothem + 0.55 * (1.0 - table_apothem)
    lower_radius = 1.0 - 0.77
    culet_height = -girdle_half - pavilion_depth

    def plane_2d(first, second):
        """Outward (radial, axial) normal and offset of the facet through two
        points given as (radius, height) in a half-plane through the axis."""
        normal = np.array([second[1] - first[1], first[0] - second[0]])
        normal /= np.linalg.norm(normal)
        if normal[0] < 0.0:                   # outward is away from the axis
            normal = -normal
        return normal, float(normal @ np.asarray(first, float))

    def on_plane(normal, offset, radius):
        """Height at which the facet plane crosses a ray 22.5 degrees off its
        own azimuth -- where the star and lower-girdle points must sit."""
        return (offset - normal[0] * radius * np.cos(half)) / normal[1]

    bezel, bezel_d = plane_2d((table_radius, crown_height), (1.0, girdle_half))
    pavilion, pavilion_d = plane_2d((1.0, -girdle_half), (0.0, culet_height))
    star_height = on_plane(bezel, bezel_d, star_radius)
    lower_height = on_plane(pavilion, pavilion_d, lower_radius)

    def ring(count, radius, height, offset=0.0):
        angles = np.arange(count) * (2 * np.pi / count) + offset
        return np.column_stack([radius * np.cos(angles),
                                radius * np.sin(angles),
                                np.full(count, height)])

    points = np.vstack([ring(8, table_radius, crown_height),
                        ring(8, star_radius, star_height, half),
                        ring(16, 1.0, girdle_half),
                        ring(16, 1.0, -girdle_half),
                        ring(8, lower_radius, lower_height, half),
                        [[0.0, 0.0, culet_height]]])
    T, S, GT, GB, L, C = 0, 8, 16, 32, 48, 56

    triangles: list[list[int]] = []
    triangles += [[T, T + i, T + i + 1] for i in range(1, 7)]    # table
    for k in range(8):
        nxt, prv = (k + 1) % 8, (k + 7) % 8
        # Bezel kite: table vertex, its two star points, the girdle below it.
        triangles += [[T + k, S + k, GT + 2 * k],
                      [T + k, GT + 2 * k, S + prv]]
        # Star facet, between two table vertices.
        triangles += [[T + k, T + nxt, S + k]]
        # Upper girdle halves, either side of the star point.
        triangles += [[S + k, GT + 2 * k, GT + 2 * k + 1],
                      [S + k, GT + 2 * k + 1, GT + (2 * k + 2) % 16]]
        # Pavilion main, a kite from the girdle down to the culet.
        triangles += [[GB + 2 * k, L + k, C],
                      [GB + 2 * k, C, L + prv]]
        # Lower girdle halves.
        triangles += [[L + k, GB + 2 * k + 1, GB + 2 * k],
                      [L + k, GB + (2 * k + 2) % 16, GB + 2 * k + 1]]
    for j in range(16):                                          # girdle band
        nxt = (j + 1) % 16
        triangles += [[GT + j, GB + j, GB + nxt],
                      [GT + j, GB + nxt, GT + nxt]]
    return points, np.asarray(triangles, dtype=np.int64)


@lru_cache(maxsize=1)
def standard_brilliant() -> StoneShape:
    """The plane set of a textbook round brilliant, in the normalised frame.

    Built as geometry and then put through `extract`, so it is normalised,
    merged and outward-oriented by exactly the same code as a stone read from
    a file. Deriving the planes by hand would let the two drift apart.

    One shared brilliant is reused across the catalogue rather than trusting
    whatever mesh each listing happens to carry.
    """
    return extract(*_brilliant_mesh())


def plane_set(verts: np.ndarray, faces: np.ndarray) -> tuple[StoneShape, bool]:
    """Planes to trace this stone against, and whether they were substituted.

    A stone whose own geometry is a cut is traced against that cut. A stone
    whose geometry is a smooth placeholder keeps its measured position, axis
    and size -- those are real -- and borrows a standard brilliant's facets,
    because the alternative is rendering a polished pebble truthfully.
    """
    own = extract(verts, faces)
    if cut_coverage(verts, faces) >= _CUT_COVERAGE:
        return own, False
    ideal = standard_brilliant()
    return StoneShape(points=ideal.points, normals=ideal.normals,
                      axis=own.axis, centre=own.centre,
                      radius=own.radius), True
