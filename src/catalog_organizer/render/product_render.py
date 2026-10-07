"""Coloured product renders for the PDF catalogue.

Separate from `snapshotter/` on purpose. Those renders are a flat grey clay
shader built to feed the classifier — the code says so itself — and they are
tied to golden-file expectations. These are sales images.

Lighting comes from two real studio HDR environments (`assets/hdr/`), one
tuned for metal and one for stones, supplied by the user from the reference
they work to. That split is the whole reason this module renders in two
passes: a VTK renderer carries exactly one environment texture, so metal and
gems are drawn separately, each under its own environment, and composited by
depth. It costs a second render per view and it is the difference between
"a CAD screenshot" and "a product photo".

Two views per product — face-on and three-quarter — which is what the user
asked for and what a jewellery catalogue page conventionally shows.

Renders are cached under
`cache/renders/<file_id>/v<n>_<metal>_<stone>_<view>.png`. The version prefix
is part of the key so a change to the lighting pipeline invalidates old
images instead of silently serving them next to new ones.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyvista as pv

from catalog_organizer.core.paths import PROJECT_ROOT, cache_dir
from catalog_organizer.render import materials as mat
from catalog_organizer.render import facets
from catalog_organizer.render import gem_trace
from catalog_organizer.render import studio
from catalog_organizer.render.materials import MetalMaterial, StoneMaterial

VIEW_NAMES = ("front", "iso")

# Bump when the lighting or material pipeline changes so cached PNGs from the
# previous look are not reused. 5 = the studio rig, ACES tone mapping and the
# physical metal reflectances. 8 = gems are traced against their own facet
# planes (`render/gem_trace.py`), which changes every stone in the catalogue.
# 9 = the gem gain re-swept against that tracer and absorption given a density.
#
# `_LOOK_KEY` exists because relying on the bump alone failed once, quietly and
# expensively: the gem gain and the absorption density were retuned, every
# measurement confirmed the new look, and the app went on serving the frames it
# had already cached under the same name -- so the retune was reported as
# ineffective and the next hour went to re-checking numbers that were correct.
# The constants the look actually depends on are hashed into the filename here,
# which means forgetting the bump costs nothing.
RENDER_VERSION = 9


def _look_key() -> str:
    """Four hex digits standing for every tuned constant in the look.

    Anything here that moves gives every cached frame a new name, so a retune
    is visible in the app the moment it is made rather than the next time
    someone remembers to bump the version by hand.
    """
    import hashlib

    parts = (studio.GEM_LOOKUP_GAIN, studio.EXPOSURE_GEM,
             studio.EXPOSURE_METAL, studio.CURVE_CONTRAST,
             gem_trace.ABSORPTION_DENSITY, gem_trace.REFRACTION_INDEX,
             gem_trace.RAY_BOUNCES, gem_trace.CROWDED_RAY_BOUNCES,
             mat.POLISHED_ROUGHNESS, SUPERSAMPLE)
    text = "|".join(f"{value!r}" for value in parts)
    return hashlib.blake2s(text.encode(), digest_size=2).hexdigest()

# Each view is drawn at this multiple of the delivered size and filtered down.
# A prong or a bezel edge is a sub-pixel sliver of near-mirror metal, and the
# radiance either side of that edge differs by more than an order of magnitude;
# MSAA resolves the coverage but not that, so edges crawled with bright specks.
# Measured on a curved silhouette: 2x gave 89 distinct coverage levels, 3x gave
# 161, 4x gave 219 for 1.8x the pixels of 3x. 3 is where the curve flattens.
SUPERSAMPLE = 3

# How far the three-quarter view swings off the face-on axis. 0.5 / 0.38 puts
# it near a 35° tilt: enough to show thickness, prong height and how stones sit
# proud of the metal, without foreshortening the design itself.
_ISO_SWING = (0.50, 0.38)

# Ring framing. Swept at 45/55/65/75 degrees against four real rings — a
# signet, a patterned band, a twisted band and a stone-set piece — and 55 was
# the one that showed the top face, the band's shoulder AND some of the bore
# on every one of them. 45 gave up too much of the face to the hole; 65 and
# beyond closed the bore and the pieces stopped reading as rings.
_RING_TILT_DEGREES = 55.0
# Yaw off dead-centre, in units of the side vector. A symmetric head-on shot
# looks like a technical drawing. The front view was at 0.18 — near enough to
# head-on that a wide band hid its own top — and went to 0.40 in the same
# sweep; iso turns further the other way so the pair are genuinely two angles.
_RING_YAW = (0.40, -0.55)
# Rings get framed slightly wider than everything else. A ring is tall in the
# frame and its shoulders run out to the sides, so at the common zoom it
# crowds its own edges in a way a flat pendant does not.
_RING_ZOOM_FACTOR = 1.13

# Real product photography uses a long lens, not an orthographic camera. 20°
# is roughly an 85 mm equivalent: enough perspective that the near side of a
# band reads as nearer, without the barrel distortion a wide angle would add.
_FOV_DEGREES = 20.0

# Above this share of the frame's MIDDLE being empty, the chosen axis is
# looking straight through a hole and the ring composition takes over.
# Measured: halo and pavé rings 0-8 %, pendants 6-19 %, open-bore rings
# 92-100 %. 60 % sits in the empty gap between the groups.
_MAX_CENTRAL_HOLE = 0.60

BACKGROUND = (255, 255, 255)

HDR_DIR = PROJECT_ROOT / "assets" / "hdr"
METAL_HDR = "env-metal-6.hdr"
GEM_HDR = "env-gem-1.hdr"


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderResult:
    file_id: str
    metal_key: str
    stone_key: str
    views: dict[str, Path]
    had_stones: bool
    engine: str = "vtk"         # which engine drew it: "vtk" (this module) or "web" (webview.engine)


def renders_dir(file_id: str) -> Path:
    d = cache_dir() / "renders" / file_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_polydata(arrays: tuple[np.ndarray, np.ndarray]) -> pv.PolyData:
    verts, faces = arrays
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    padded = np.empty((faces.shape[0], 4), dtype=np.int64)
    padded[:, 0] = 3
    padded[:, 1:] = faces
    return pv.PolyData(verts, padded.flatten())


# ── environment ──────────────────────────────────────────────────────────────

def _hdr_is_equirectangular(reader) -> bool:
    """Reject a supplied HDR whose projection cannot be what VTK assumes.

    `SetEnvironmentTexture` reads the image as an equirectangular panorama, and
    an equirect is always 2:1. Both files shipped in `assets/hdr/` are 512x512,
    so neither was ever mapped onto the sphere correctly — a silent failure,
    because a wrongly-projected environment still renders, just wrongly. This
    is the guard that makes it loud.
    """
    width, height, _ = reader.GetOutput().GetDimensions()
    return height > 0 and abs(width / height - 2.0) < 0.02


@lru_cache(maxsize=4)
def _load_hdr(name: str) -> pv.Texture | None:
    """Read a Radiance .hdr into a VTK texture usable as a PBR environment.

    `pv.read` does not produce a texture VTK will accept here, so the reader is
    driven directly. `ColorModeToDirectScalars` matters: without it VTK maps
    the float radiance values through a lookup table and the highlights — the
    part that makes metal look polished — get clamped away.

    Returns None for a non-2:1 file. The procedural rig is a better environment
    than a misprojected photograph, and silently preferring the photograph is
    how the metal ended up flat.
    """
    path = HDR_DIR / name
    if not path.exists():
        return None
    try:
        from vtkmodules.vtkIOImage import vtkHDRReader

        reader = vtkHDRReader()
        reader.SetFileName(str(path))
        reader.Update()
        if not _hdr_is_equirectangular(reader):
            return None
        texture = pv.Texture()
        texture.SetInputDataObject(reader.GetOutput())
        texture.SetColorModeToDirectScalars()
        texture.MipmapOn()
        texture.InterpolateOn()
        return texture
    except Exception:
        return None


def environment_for(kind: str) -> tuple[pv.Texture, bool]:
    """(texture, came_from_a_file) for "metal" or "gem".

    The procedural studio rig is the default rather than a fallback — see
    `render/studio.py` for the measurements behind that. A supplied HDR wins
    only if it is a usable equirect, so dropping a proper 2:1 panorama into
    `assets/hdr/` still overrides the rig.
    """
    supplied = _load_hdr(METAL_HDR if kind == "metal" else GEM_HDR)
    if supplied is not None:
        return supplied, True
    return studio.environment("gem" if kind == "gem" else "metal"), False


def hdr_status() -> dict[str, dict[str, bool]]:
    """Whether each supplied HDR exists and whether it is actually usable.

    Reported separately on purpose: "the file is there" was the only thing
    checked before, and both files were there and both were being misprojected.
    """
    out: dict[str, dict[str, bool]] = {}
    for kind, name in (("metal", METAL_HDR), ("gem", GEM_HDR)):
        out[kind] = {
            "present": (HDR_DIR / name).exists(),
            "usable": _load_hdr(name) is not None,
        }
    return out


# ── camera ───────────────────────────────────────────────────────────────────

def facing_weights(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-axis Σ |face normal · axis| × face area — how much surface faces
    each direction.

    This is the signal that picks the catalogue view. Two simpler rules were
    tried against real files and both chose wrong:

      * "look down the thinnest axis" showed a halo ring edge-on as a thin C.
      * "look down the axis the piece is roundest about" (the rule the sprue
        code uses to find a finger axis) looked straight down the band and
        showed only its cross-section.

    Surface facing gets it right for the same shapes: a flat pendant's faces
    mostly point along its thin axis, and a chain link's faces point across its
    length rather than along it.
    """
    tri = verts[faces]
    # Cross product of two edges = 2 × area × unit normal, so no separate area
    # term or normalisation is needed.
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return np.abs(normals).sum(axis=0)


def _silhouette_occupancy(verts: np.ndarray, faces: np.ndarray, axis: int,
                          grid: int = 64) -> np.ndarray:
    """Boolean map of what the piece covers when viewed down `axis`.

    Every triangle marks the cells its 2-D bounding box touches. Sampling
    corners and centroids instead was tried first and is wrong for coarse
    geometry: a box built from twelve triangles put twenty sample points on a
    64x64 grid and measured as almost entirely empty, so a solid slab was
    read as a hole. Bounding boxes never under-fill, and a triangle cannot
    span a genuine hole because there is nothing there to tessellate.
    """
    plane = [i for i in (0, 1, 2) if i != axis]
    triangles = verts[faces][:, :, plane]
    low = triangles.reshape(-1, 2).min(axis=0)
    span = np.maximum(triangles.reshape(-1, 2).max(axis=0) - low, 1e-9)

    cells = np.clip(((triangles - low) / span * (grid - 1)), 0, grid - 1)
    lo = np.floor(cells.min(axis=1)).astype(int)
    hi = np.ceil(cells.max(axis=1)).astype(int)

    occupied = np.zeros((grid, grid), dtype=bool)
    # Fast path: a triangle inside a single cell (the overwhelming majority in
    # a dense mesh) is marked without a loop.
    single = (hi[:, 0] - lo[:, 0] <= 1) & (hi[:, 1] - lo[:, 1] <= 1)
    if single.any():
        occupied[lo[single, 0], lo[single, 1]] = True
    for x0, y0, x1, y1 in zip(lo[~single, 0], lo[~single, 1],
                              hi[~single, 0], hi[~single, 1]):
        occupied[x0:x1 + 1, y0:y1 + 1] = True
    return occupied


def central_hole_fraction(verts: np.ndarray, faces: np.ndarray, axis: int,
                          grid: int = 64, core: float = 0.22) -> float:
    """How empty the MIDDLE of the frame is, viewed down `axis`.

    This measures the failure directly instead of inferring it. Looking down a
    plain ring's bore puts a hole in the middle of the picture and nothing of
    the design reads — the "ürün ters geliyor" complaint. A halo or pavé face
    fills that same middle, and for those the down-the-bore view is right.

    Measured across real pieces, and the two groups do not overlap:

        halo / pavé-faced rings   0.0 %,  0.0 %,  8.4 %   → keep the view
        pendants (cross, winged)  6.2 %, 18.7 %           → keep the view
        rings with an open bore   92.0 %, 100.0 %         → recompose

    Two weaker tests were tried first and both mixed the groups up: overall
    silhouette fill (a winged pendant covers only 40 % of the frame, less than
    some rings) and a solid/hollow probe of the 3-D centre point (empty for a
    pendant whose middle happens to fall between two wings).
    """
    occupied = _silhouette_occupancy(verts, faces, axis, grid)
    centre = grid // 2
    radius = max(1, int(grid * core / 2))
    core_cells = occupied[centre - radius: centre + radius + 1,
                          centre - radius: centre + radius + 1]
    if core_cells.size == 0:
        return 0.0
    return float(1.0 - core_cells.mean())


def feature_azimuth(
    verts: np.ndarray, bore_axis: int, bounds
) -> np.ndarray | None:
    """Unit direction, perpendicular to the bore axis, aimed at the design.

    Knowing to view a ring from the side is only half the answer — you also
    have to know WHICH side. Snapping to a world axis picks one arbitrarily,
    and on a real user-supplied ring that landed 90° off: the twist and its
    stone sat at −90° while the camera looked from 0°, so the render showed
    the plain back of the band.

    A ring's shank is thin and even; its design (setting, twist, shoulders)
    is where the band grows. So: bin the vertices by angle around the bore,
    measure how far each sector reaches along the bore axis, and take the
    circular mean of whatever rises above the median. On that ring the band
    measures 2.8 mm across the design half and 0.9 mm across the plain half —
    an unambiguous signal — and the two twist strands peak either side of
    −90° with the crossing (where the stone sits) between them, which is
    exactly where the circular mean lands.

    Returns None when no sector stands out, i.e. a perfectly even band with
    no "front" to find — the caller then keeps its axis-aligned choice.
    """
    if len(verts) < 32:
        return None
    center = np.array([
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ])
    plane = [i for i in (0, 1, 2) if i != bore_axis]
    rel = verts - center
    angle = np.arctan2(rel[:, plane[1]], rel[:, plane[0]])
    height = np.abs(rel[:, bore_axis])

    sectors = 36
    idx = (((angle + np.pi) / (2 * np.pi)) * sectors).astype(int) % sectors
    peak = np.full(sectors, np.nan)
    for s in range(sectors):
        sel = idx == s
        if sel.sum() >= 5:
            peak[s] = height[sel].max()
    if np.isnan(peak).all():
        return None

    median = float(np.nanmedian(peak))
    weight = np.nan_to_num(peak - median, nan=0.0)
    weight[weight < 0] = 0.0
    if weight.sum() <= 1e-9:
        return None
    # A weak, evenly-spread signal means there is no real "front" — an even
    # band. Require the strongest sector to stand meaningfully above the
    # median before trusting the answer.
    if weight.max() < 0.15 * max(median, 1e-6):
        return None

    centres = (np.arange(sectors) + 0.5) / sectors * 2 * np.pi - np.pi
    # Circular mean: a design spanning several sectors (or two strands with a
    # gap between them) resolves to the middle rather than to one edge.
    x = float((weight * np.cos(centres)).sum())
    y = float((weight * np.sin(centres)).sum())
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return None

    direction = np.zeros(3)
    direction[plane[0]] = x
    direction[plane[1]] = y
    return direction / np.linalg.norm(direction)


def open_bore_axis(
    silhouette_arrays: tuple[np.ndarray, np.ndarray] | None,
) -> int | None:
    """Which axis you can see through, or None if the piece is not a ring.

    Asks about the PIECE, not about whichever axis the facing weights happened
    to pick. Testing only the chosen axis meant a ring whose facing landed on
    its band was never recognised as one: of four real rings, three measured a
    bore fraction of 1.00 on an axis the facing had not chosen and 0.00 on the
    axis it had, so they were composed as flat plates and their details were
    unreadable.

    Scanning all three axes is safe because the groups do not come close to
    touching. Measured per axis on six rings and six pendants:

        rings     every one has exactly one axis at 1.00
        pendants  the highest axis of any piece is 0.18

    This costs about a second on a million-face mesh, so the result is
    computed once and handed to `camera_axes` rather than recomputed there.
    """
    if silhouette_arrays is None or not len(silhouette_arrays[1]):
        return None
    holes = [central_hole_fraction(*silhouette_arrays, a) for a in (0, 1, 2)]
    return int(np.argmax(holes)) if max(holes) > _MAX_CENTRAL_HOLE else None


# Which pose a category is photographed in.
#
# The report was that products of one category arrive at different angles --
# "bazi urunler dik bazi urunler capraz bazi urunler yatay" -- and the cause
# was that the pose was chosen per PIECE, from whatever its geometry happened
# to suggest, rather than per category. Three separate things could each swing
# it: whether the piece had stones (they steered `facing_weights`, so a plain
# band and a set band picked different "design" directions), whether the bore
# test cleared its threshold (below it, a ring fell through to the generic
# path and was shot like a plate), and which of the two remaining axes came
# out longer (which decided "up", and so whether a piece stood or lay down).
#
# A catalogue is a comparison. Two rings side by side have to be shot the same
# way or the customer reads the difference in framing as a difference in the
# product, so the category picks the pose and the geometry only says where the
# piece's own axes are.
# Only a ring is shot down its own opening. A bracelet or a chain is laid out
# in a catalogue, not photographed through its hole, and the measurement agrees
# -- pose spread across the catalogue's own products, view direction / up
# direction, in each piece's own frame (0 = every product identical):
#
#                  as-was        bore pose      flat pose
#     bracelet   0.321 / 0.108  0.187 / 0.201  0.000 / 0.000
#     chain      0.242 / 0.041  0.114 / 0.363  0.001 / 0.003
#
# The bore pose actually made a bracelet's "up" WORSE than no rule at all,
# because the opening of a cuff is not the feature anyone looks at.
# Below this there is no hole worth looking through, so a piece labelled a
# ring is shot flat instead. `_MAX_CENTRAL_HOLE` (0.60) is the threshold for
# "is this a ring at all"; this one is far lower because the category has
# already answered that -- it only rejects pieces with no opening whatsoever.
_RING_POSE_MIN_HOLE = 0.25

_BORE_CATEGORIES = frozenset({"ring"})
_FLAT_CATEGORIES = frozenset({"pendant", "charm", "brooch", "earring",
                              "bracelet", "chain", "necklace"})


_UNSET = object()


def flat_pose(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(face normal, up, side) for a flat piece, from its OWN axes.

    Not the bounding box. The box is aligned to the FILE, and a designer who
    modelled a pendant lying at an angle then got a render taken at that
    angle -- which is half of the "bazi urunler dik bazi urunler capraz"
    report, and the half that no per-category rule could fix on its own,
    because the two pendants really were being shot along different
    directions. Principal axes belong to the SHAPE, so the same pendant saved
    in any orientation is photographed identically.

    Least spread is the face normal (a flat piece is thin one way), most
    spread is the long direction and becomes "up".

    The sign of "up" is decided by width: a pendant tapers toward its bail,
    so the narrower end is the end it hangs by and belongs at the top of the
    frame. Without this the axis is only defined up to sign and half the
    catalogue would hang upside down -- deterministically, but upside down.
    """
    centred = points - points.mean(axis=0)
    _values, vectors = np.linalg.eigh(centred.T @ centred)
    normal, side, up = (vectors[:, 0], vectors[:, 1], vectors[:, 2])

    along = centred @ up
    lo, hi = np.percentile(along, 15), np.percentile(along, 85)
    across = np.abs(centred @ side)
    top = across[along >= hi].mean() if (along >= hi).any() else 0.0
    bottom = across[along <= lo].mean() if (along <= lo).any() else 0.0
    if top > bottom:                      # narrow end is the bail end
        up = -up
    return normal, up, side


def camera_axes(
    bounds,
    mesh_arrays: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    avoid_bore: bool = False,
    silhouette_arrays: tuple[np.ndarray, np.ndarray] | None = None,
    known_bore: object = _UNSET,
    category: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(face-on direction, three-quarter direction, up vector).

    A fixed world-axis camera cannot work here — jewellery is modelled however
    the designer happened to orient it, and a hard-coded "front" caught real
    pendants edge-on and rendered them as a gold sliver.

    With geometry available the view axis comes from `facing_weights`. Without
    it the fallback is the thinnest axis. "Up" is the longer of the two
    remaining axes, so the piece stands the tall way in a portrait frame.

    `avoid_bore` guards a second real failure, found on a user-supplied ring:
    with no gem geometry to anchor the decision (an STL, or a polished piece
    with no stones), `facing_weights` on bare metal alone picked a halo ring's
    flat stone-cluster face correctly on four different real files — but on a
    plain twisted band it picked the axis running straight through the finger
    hole, because a band's rim simply has more surface area than its sides.
    The frame came out as an almost featureless circle; the actual design was
    only visible from the side. `avoid_bore` is passed True precisely when
    there is no gem signal to trust, and only THEN, if the chosen axis turns
    out to look straight through a hole (`central_hole_fraction` is high), swaps in
    the next-best axis — a lateral view — with "up" set to the discarded
    bore axis so the piece still stands upright in frame. When gems ARE
    driving the choice this is skipped entirely: it was validated correct
    even looking straight down a bore on four different real stone-set rings.
    """
    extents = np.array([
        bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4],
    ], dtype=np.float64)
    extents = np.maximum(extents, 1e-9)

    bore_corrected = False
    bore_axis = 0
    order: list[int] | None = None
    family = (category or "").strip().lower()
    silhouette_for_pose = silhouette_arrays or mesh_arrays

    if family in _FLAT_CATEGORIES:
        # Face-on, every time. The thinnest axis IS the face of a flat piece,
        # and reading it off the extents is the same rule for every product,
        # where `facing_weights` gave a different answer depending on what the
        # piece happened to carry. "Up" is the longest remaining axis, so a
        # pendant stands the tall way in a portrait frame whether or not its
        # designer built it lying down.
        if silhouette_for_pose is not None and len(silhouette_for_pose[0]):
            front, up, side = flat_pose(np.asarray(silhouette_for_pose[0],
                                                   dtype=float))
        else:
            view_axis = int(np.argmin(extents))
            others = [i for i in (0, 1, 2) if i != view_axis]
            others.sort(key=lambda i: extents[i], reverse=True)
            front = np.zeros(3); front[view_axis] = 1.0
            up = np.zeros(3); up[others[0]] = 1.0
            side = np.zeros(3); side[others[1]] = 1.0
        iso = front + _ISO_SWING[0] * side + _ISO_SWING[1] * up
        return front, iso / np.linalg.norm(iso), up

    if (family in _BORE_CATEGORIES and silhouette_for_pose is not None
            and len(silhouette_for_pose[1])):
        # A ring is always shot down its own bore, whether or not the hole
        # test clears the threshold that decides "is this a ring at all". The
        # threshold answers a classification question; here the category has
        # already answered it, so the widest hole simply wins.
        found = (known_bore if known_bore is not _UNSET else
                 open_bore_axis(silhouette_for_pose))
        if found is None:
            holes = [central_hole_fraction(*silhouette_for_pose, a)
                     for a in (0, 1, 2)]
            found = (int(np.argmax(holes))
                     if max(holes) > _RING_POSE_MIN_HOLE else None)
        if found is None:
            # Labelled a ring, but there is no hole to look through. The
            # category is a label and the geometry is the fact -- JCAD-000000001
            # is filed as "ring/other" with no inner diameter at all and a
            # 29 x 27 x 5.8 mm box, i.e. a flat plate, and forcing the ring
            # pose on it aimed the camera down whichever axis happened to have
            # the largest gap. Shoot it like the flat piece it is and let the
            # mis-classification show up in the catalogue as a plate.
            front, up, side = flat_pose(
                np.asarray(silhouette_for_pose[0], dtype=float))
            iso = front + _ISO_SWING[0] * side + _ISO_SWING[1] * up
            return front, iso / np.linalg.norm(iso), up
        bore_corrected = True
        bore_axis = int(found)
        # The design direction comes from the WHOLE piece, never from the
        # stones alone: a set band and a plain band are the same ring shape,
        # and letting two 1 mm stones define "up" is what made two rings from
        # one category arrive turned differently.
        mesh_arrays = silhouette_for_pose
        order = [int(i) for i in np.argsort(extents)[::-1]]
        view_axis = order[0]
    elif mesh_arrays is not None and len(mesh_arrays[1]):
        order = [int(i) for i in np.argsort(facing_weights(*mesh_arrays))[::-1]]
        view_axis = order[0]
        # The hole test runs on the WHOLE piece, not on whichever half chose
        # the axis: a halo's ring of stones has a gap in the middle all by
        # itself, and testing the stones alone reported a hole where the metal
        # underneath plainly fills the frame.
        silhouette = silhouette_arrays or mesh_arrays
        # NOT named `bore_axis`: that is a local below, and shadowing it here
        # made every piece look pre-classified as a ring.
        found = (open_bore_axis(silhouette) if known_bore is _UNSET
                 else known_bore)
        if avoid_bore and found is not None:
            bore_corrected = True
            bore_axis = int(found)
    else:
        view_axis = int(np.argmin(extents))

    def _unit(axis: int) -> np.ndarray:
        v = np.zeros(3)
        v[axis] = 1.0
        return v

    if bore_corrected:
        # The classic ring shot, matched against a real reference photograph:
        # the piece stands with its design at the top of the frame and the
        # camera sits about 45° off the bore, turned slightly to one side. You
        # see through the hole, the band has depth, and the setting reads.
        #
        # Two earlier attempts were rendered and rejected against that same
        # reference: straight down the bore (a flat featureless circle) and a
        # dead-level side view (a profile with the design edge-on).
        bore = _unit(bore_axis)
        design = feature_azimuth(mesh_arrays[0], bore_axis, bounds)
        if design is None:
            design = _unit(order[1])
        side = np.cross(design, bore)
        side_norm = np.linalg.norm(side)
        side = side / side_norm if side_norm > 1e-9 else _unit(order[2])

        tilt = np.radians(_RING_TILT_DEGREES)
        base = bore * np.cos(tilt) + design * np.sin(tilt)
        front = base + _RING_YAW[0] * side
        front /= np.linalg.norm(front)
        iso = base + _RING_YAW[1] * side
        iso /= np.linalg.norm(iso)
        # "up" is the design direction, which is what puts the setting at the
        # top of the frame instead of wherever the model happened to be built.
        return front, iso, design

    others = [i for i in (0, 1, 2) if i != view_axis]
    others.sort(key=lambda i: extents[i], reverse=True)
    up_axis, side_axis = others

    front = _unit(view_axis)
    up = _unit(up_axis)
    iso = front + _ISO_SWING[0] * _unit(side_axis) + _ISO_SWING[1] * up
    iso /= np.linalg.norm(iso)
    return front, iso, up


def _frame(plotter: pv.Plotter, bounds, direction: np.ndarray,
           up: np.ndarray, zoom: float) -> None:
    """Point the camera at the piece and frame it.

    Perspective, not orthographic. The reference product photographs this is
    matched against have visible perspective — the near side of a band reads
    as nearer — and an orthographic render of the same ring came out looking
    like a technical drawing beside them. `_FOV_DEGREES` keeps it to a long
    lens, so the effect is depth rather than distortion.

    `zoom` keeps its old meaning (the half-height of the framed view, as a
    fraction of the piece's largest dimension) so existing callers frame the
    same amount of the piece as before the switch.
    """
    center = np.array([
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    ])
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2],
                 bounds[5] - bounds[4], 1e-6)
    half_height = extent * zoom
    distance = half_height / np.tan(np.radians(_FOV_DEGREES) / 2.0)

    plotter.camera.view_angle = _FOV_DEGREES
    plotter.camera.position = tuple(center + np.asarray(direction) * distance)
    plotter.camera.focal_point = tuple(center)
    plotter.camera.up = tuple(up)


# ── passes ───────────────────────────────────────────────────────────────────

def _add_metal(plotter, mesh, metal: MetalMaterial) -> None:
    # `split_sharp_edges` is not cosmetic. Without it VTK averages the vertex
    # normals across EVERY edge, including the 90-degree ones of a cut-out or
    # a bezel wall. On a near-mirror surface an averaged normal across a
    # crease sweeps the reflection through a wide arc over one pixel, which
    # rendered patterned bands with serrated edges and horizontal banding in
    # their recesses. Measured on a real ring: total edge gradient energy fell
    # from 7513 to 5446 the moment the creases were kept.
    plotter.add_mesh(mesh, color=metal.color, pbr=True, metallic=metal.metallic,
                     roughness=metal.roughness, smooth_shading=True,
                     split_sharp_edges=True)


def _camera_position(bounds, direction, zoom: float) -> np.ndarray:
    """Where `_frame` will put the camera. Computed rather than read back so
    the per-stone uniforms can be set before the camera is placed."""
    centre = np.array([0.5 * (bounds[0] + bounds[1]),
                       0.5 * (bounds[2] + bounds[3]),
                       0.5 * (bounds[4] + bounds[5])])
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2],
                 bounds[5] - bounds[4], 1e-6)
    distance = (extent * zoom) / np.tan(np.radians(_FOV_DEGREES) / 2.0)
    return centre + np.asarray(direction, dtype=float) * distance


def _add_gem(plotter, mesh, stone: StoneMaterial, *, arrays=None, bounds=None,
             direction=None, up=None, zoom: float = 0.62,
             trace: bool = True) -> None:
    """Draw the stones, tracing each ray against their real facet planes.

    One actor per stone, one compiled shader for the lot: the facets are the
    same for every stone of a cut and are baked into the GLSL, while each
    stone's position and orientation arrive as uniforms. VTK caches programs
    by source, so a pave piece compiles once and draws 191 actors through it.
    See `render/gem_trace.py` for why an approximation could not do this.

    FLAT shading throughout: averaging normals across a brilliant blurs its
    facets into one smooth dome, and every flash a gem has comes from a facet
    boundary.
    """
    if arrays is None:
        arrays = (np.asarray(mesh.points),
                  mesh.faces.reshape(-1, 4)[:, 1:])
    if stone.opaque or not trace or direction is None:
        # Onyx has no interior to see into; the live orbit preview skips the
        # trace because it redraws on every mouse drag. Both fall back to a
        # plain polished surface.
        plotter.add_mesh(mesh, color=stone.color, pbr=True, metallic=1.0,
                         roughness=stone.roughness, smooth_shading=False)
        return

    try:
        frames = facets.stone_frames(*arrays)
        biggest = max(frames, key=lambda f: f["radius"])
        # `plane_set` hands back a standard brilliant's facets when the CAD
        # has no cut to read -- JCAD-000000140's stones, for instance, are a
        # flat table over a smoothly revolved cone, a placeholder dropped into
        # the seat. Only the INTERIOR is borrowed: the silhouette and the
        # surface normals stay the CAD's, because those are the ones that fit
        # the setting.
        shape, _borrowed = facets.plane_set(biggest["verts"],
                                            biggest["faces"])
    except Exception:
        plotter.add_mesh(mesh, color=stone.color, pbr=True, metallic=1.0,
                         roughness=stone.roughness, smooth_shading=False)
        return

    # Their viewer drops the bounce count on crowded pieces and so does this:
    # a pave stone is ten pixels across, nobody reads its interior, and every
    # one of them otherwise pays a full 121-plane trace per bounce per
    # channel.
    bounces = (gem_trace.CROWDED_RAY_BOUNCES
               if len(frames) >= gem_trace.CROWDED_STONE_COUNT
               else gem_trace.RAY_BOUNCES)

    code = gem_trace.build(shape, ior=stone.ior, dispersion=stone.dispersion,
                           bounces=bounces,
                           absorption=gem_trace.absorption_from(stone.color))
    rotation = gem_trace.view_basis(direction, up)
    camera = _camera_position(bounds, direction, zoom)
    lookup = studio.gem_lookup()

    for frame in frames:
        actor = plotter.add_mesh(
            _to_polydata((frame["verts"], frame["faces"])), color="#FFFFFF",
            pbr=True, metallic=1.0, roughness=stone.roughness,
            smooth_shading=False)
        actor.GetProperty().SetTexture("gemEnv", lookup)
        actor.GetShaderProperty().AddFragmentShaderReplacement(
            "//VTK::Light::Impl", True, code, False)
        basis = frame["basis"] @ rotation.T        # stone axes, in view space
        gem_trace.set_frame(
            actor, right=basis[0], up=basis[1], axis=basis[2],
            centre=rotation @ (frame["centre"] - camera),
            radius=frame["radius"])


def _render_pass(
    mesh: pv.PolyData,
    *,
    kind: str,
    arrays=None,
    backdrop: pv.PolyData | None = None,
    metal: MetalMaterial,
    stone: StoneMaterial,
    bounds,
    direction: np.ndarray,
    up: np.ndarray,
    zoom: float,
    resolution: int,
) -> np.ndarray:
    """One material under one rig. Returns RGBA on a transparent background.

    Transparent rather than drawn against the catalogue's page colour, because
    tone mapping runs over the whole frame: at the exposure that keeps gold
    saturated, a light background is dragged down to mid grey along with it.
    The page is composited underneath afterwards, unaffected.
    """
    env, _from_file = environment_for(kind)
    plotter = pv.Plotter(off_screen=True, window_size=(resolution, resolution),
                         lighting="none")
    plotter.set_environment_texture(env, is_srgb=False)

    if backdrop is not None:
        # Only its pixels behind the stones survive compositing, but it has to
        # be here: a gem drawn over an empty pass shows the page through
        # itself instead of the metal it is set into, which is what made the
        # first stones read as frosted glass.
        _add_metal(plotter, backdrop, metal)
    if kind == "metal":
        _add_metal(plotter, mesh, metal)
    else:
        _add_gem(plotter, mesh, stone, arrays=arrays, bounds=bounds,
                 direction=direction, up=up, zoom=zoom)

    _frame(plotter, bounds, direction, up, zoom)
    # Orient the rig AFTER framing — it is derived from the placed camera.
    studio.orient_environment_to_camera(plotter)
    studio.install_tone_mapping(
        plotter,
        studio.EXPOSURE_GEM if kind == "gem" else studio.EXPOSURE_METAL)

    plotter.show(auto_close=False)
    rgba = np.asarray(plotter.screenshot(transparent_background=True),
                      dtype=np.uint8)
    plotter.close()
    return rgba


# Flat key colours for the segmentation pass. Pure primaries so classification
# is an exact comparison, not a threshold.
_KEY_METAL = (255, 0, 0)
_KEY_GEM = (0, 255, 0)


def _render_mask(
    metal_mesh: pv.PolyData | None,
    gem_mesh: pv.PolyData | None,
    *,
    bounds,
    direction: np.ndarray,
    up: np.ndarray,
    zoom: float,
    resolution: int,
) -> np.ndarray:
    """Which pixel belongs to which material, with occlusion resolved.

    Compositing was first attempted from PyVista's depth image; on this VTK
    build `get_image_depth` came back as a single constant with no NaNs once
    SSAO and anti-aliasing were enabled, so every stone was composited away.
    This pass sidesteps the depth buffer entirely: both meshes are drawn
    together, unlit, in flat key colours, and the z-buffer sorts them exactly
    as it does in the real passes. Cheap — no PBR, no ambient occlusion, no
    anti-aliasing (hard edges are what a mask wants).
    """
    plotter = pv.Plotter(off_screen=True, window_size=(resolution, resolution),
                         lighting="none")
    # Black, not white. The mask is anti-aliased like everything else, so a
    # half-covered edge pixel is a blend of its key colour with the background.
    # Against black that blend stays on the key's own axis and `composite`
    # reads it correctly; against white it picked up the other key's channels
    # and the edge of every plain metal piece was scored as part gem.
    plotter.background_color = "black"
    if metal_mesh is not None:
        plotter.add_mesh(metal_mesh, color=_KEY_METAL, lighting=False)
    if gem_mesh is not None:
        plotter.add_mesh(gem_mesh, color=_KEY_GEM, lighting=False)
    _frame(plotter, bounds, direction, up, zoom)
    plotter.show(auto_close=False)
    mask = np.asarray(plotter.image, dtype=np.int16)
    plotter.close()
    return mask


def composite(
    metal_rgba: np.ndarray | None,
    gem_rgba: np.ndarray | None,
    mask: np.ndarray,
) -> np.ndarray:
    """Merge the two lit passes, keeping every pass's own anti-aliased edge.

    The mask decides *which material* a pixel is, never *whether* a pixel is
    covered. Coverage is the passes' own alpha, and that distinction is the
    whole point of this function.

    An earlier version classified with a hard threshold — a pixel counted as
    metal only if its mask colour sat within 90 of the key. A silhouette pixel
    at 65% coverage renders as (165, 0, 0), which is exactly 90 away, so every
    edge pixel below two-thirds coverage was deleted outright. The passes came
    out of the renderer with 89 distinct coverage levels along a curved
    silhouette and this function threw all of them away, leaving a 1-bit edge.
    That is what made round forms look stepped.
    """
    if gem_rgba is None:
        return metal_rgba
    if metal_rgba is None:
        return gem_rgba

    m = mask.astype(np.float32)
    red, green = m[..., 0], m[..., 1]
    total = red + green
    # Weight, not a verdict: 0 on pure metal, 1 on pure gem, and a smooth ramp
    # across the boundary between them because the mask itself is anti-aliased.
    weight = np.where(total > 8.0, green / np.maximum(total, 1e-6), 0.0)
    weight = weight[..., None]

    rgb = (metal_rgba[..., :3].astype(np.float32) * (1.0 - weight)
           + gem_rgba[..., :3].astype(np.float32) * weight)
    # The gem pass draws the metal behind the stones too, so it covers the
    # whole piece; the metal pass covers only the metal. Either may hold the
    # outer edge, so coverage is whichever saw more of it.
    alpha = np.maximum(metal_rgba[..., 3], gem_rgba[..., 3])

    out = np.empty(metal_rgba.shape, dtype=np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = alpha
    return out


def _camera_tag(direction: np.ndarray, up: np.ndarray, zoom: float) -> str:
    """Short, deterministic filename fragment for a camera setting.

    A hand-set angle from the manual-orbit dialog must not share a cache slot
    with the automatic guess (or with a different hand-set angle) — otherwise
    the second orbit-and-save a user does would keep serving the first one's
    picture. Folded into the target filename only when an override is active;
    the automatic path is untouched and keeps its plain `v3_metal_stone_view`
    name.
    """
    import hashlib
    payload = np.concatenate([direction, up, [zoom]]).round(4).tobytes()
    return hashlib.sha1(payload).hexdigest()[:10]


def render_product(
    metal_arrays: tuple[np.ndarray, np.ndarray] | None,
    gem_arrays: tuple[np.ndarray, np.ndarray] | None,
    *,
    file_id: str,
    metal_key: str,
    stone_key: str,
    resolution: int = 1400,
    zoom: float = 0.62,
    out_dir: Path | None = None,
    overwrite: bool = False,
    camera: dict[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    gems_guide_camera: bool = True,
    category: str | None = None,
) -> RenderResult:
    """Render face-on + three-quarter views in the chosen colours.

    `camera`, when given, is `{"front": (direction, up), "iso": (direction,
    up)}` — hand-set angles from the Render tab's manual-orbit dialog. A view
    missing from the dict keeps the automatic `camera_axes()` guess; this is
    how a product can have one customised angle and one automatic one.
    """
    if metal_arrays is None and gem_arrays is None:
        raise RenderError(f"{file_id}: çizilecek geometri yok")

    metal = mat.metal(metal_key)
    stone = mat.stone(stone_key)
    out_dir = out_dir or renders_dir(file_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    metal_mesh = _to_polydata(metal_arrays) if metal_arrays else None
    gem_mesh = _to_polydata(gem_arrays) if gem_arrays else None

    combined = [m for m in (metal_mesh, gem_mesh) if m is not None]
    bounds = combined[0].bounds
    for extra in combined[1:]:
        b = extra.bounds
        bounds = (min(bounds[0], b[0]), max(bounds[1], b[1]),
                  min(bounds[2], b[2]), max(bounds[3], b[3]),
                  min(bounds[4], b[4]), max(bounds[5], b[5]))

    # Choose the view from the STONES when the piece has any. Measured on six
    # real stone-set files: gem facing picked the setting face every time,
    # while metal facing was pulled onto a wide band's flat side walls and
    # rendered a halo ring as a plain gold cup with no stones in sight.
    #
    # `avoid_bore` only arms when there is no gem geometry to trust (an STL,
    # or a genuinely stone-free piece) — with gems driving the choice, aiming
    # straight down a ring's bore was validated correct on four real files.
    # Stones point the camera at the design — but only where the split
    # between stone and metal is a fact rather than a guess. A .3dm carries it
    # in its layers; an STL is inferred by a narrow heuristic, and on the
    # user's twisted band the two 1 mm stones it found dragged the camera onto
    # their own tiny facing direction and turned the render edge-on. Inferred
    # stones still get their colour, they just do not aim the camera.
    facing_source = (gem_arrays if (gem_arrays is not None and gems_guide_camera)
                     else metal_arrays) or gem_arrays
    if metal_arrays is not None and gem_arrays is not None:
        silhouette = (np.vstack([metal_arrays[0], gem_arrays[0]]),
                      np.vstack([metal_arrays[1],
                                 gem_arrays[1] + metal_arrays[0].shape[0]]))
    else:
        silhouette = metal_arrays or gem_arrays
    bore = open_bore_axis(silhouette)
    front_dir, iso_dir, up = camera_axes(
        bounds, facing_source, avoid_bore=True, silhouette_arrays=silhouette,
        known_bore=bore, category=category)
    directions = {"front": front_dir, "iso": iso_dir}
    ups = {"front": up, "iso": up}
    ring_zoom = zoom * (_RING_ZOOM_FACTOR if bore is not None else 1.0)
    zooms = {"front": ring_zoom, "iso": ring_zoom}

    if camera:
        for view, (d, u) in camera.items():
            if view not in directions:
                continue
            direction = np.asarray(d, dtype=float)
            norm = np.linalg.norm(direction)
            directions[view] = direction / norm if norm else direction
            up_vec = np.asarray(u, dtype=float)
            up_norm = np.linalg.norm(up_vec)
            ups[view] = up_vec / up_norm if up_norm else up_vec

    targets: dict[str, Path] = {}
    for view in VIEW_NAMES:
        if camera and view in camera:
            tag = _camera_tag(directions[view], ups[view], zooms[view])
            targets[view] = out_dir / (
                f"v{RENDER_VERSION}{_look_key()}_{metal_key}_{stone_key}"
                f"_{view}_c{tag}.png")
        else:
            targets[view] = out_dir / (
                f"v{RENDER_VERSION}{_look_key()}_{metal_key}_{stone_key}"
                f"_{view}.png")
    if not overwrite and all(p.exists() for p in targets.values()):
        return RenderResult(file_id, metal_key, stone_key, targets,
                            gem_arrays is not None)

    from PIL import Image

    # Everything is drawn oversized and filtered down at the end — see
    # SUPERSAMPLE. The mask has to be drawn at the same size or it will not
    # line up with the passes it is cutting.
    big = resolution * SUPERSAMPLE
    page = studio.gradient_background(big, big)

    for view, direction in directions.items():
        view_up = ups[view]
        view_zoom = zooms[view]
        common = dict(metal=metal, stone=stone, bounds=bounds,
                      direction=direction, up=view_up, zoom=view_zoom,
                      resolution=big)
        metal_rgba = gem_rgba = None
        if metal_mesh is not None:
            metal_rgba = _render_pass(metal_mesh, kind="metal", **common)
        if gem_mesh is not None:
            gem_rgba = _render_pass(gem_mesh, kind="gem", arrays=gem_arrays,
                                    backdrop=metal_mesh, **common)
        mask = _render_mask(metal_mesh, gem_mesh, bounds=bounds,
                            direction=direction, up=view_up, zoom=view_zoom,
                            resolution=big)
        merged = composite(metal_rgba, gem_rgba, mask)
        # Floor, then the shadow over it, then the piece — the order a real
        # surface would build the picture in.
        grounded = page * studio.contact_shadow(merged[..., 3])
        grounded = studio.composite_over(
            studio.floor_reflection(merged), grounded).astype(float)
        flat = studio.composite_over(merged, grounded)
        Image.fromarray(studio.downsample(flat, SUPERSAMPLE)).save(
            targets[view])

    return RenderResult(file_id, metal_key, stone_key, targets,
                        gem_arrays is not None)


def load_source_arrays(
    source_path: Path,
) -> tuple[tuple[np.ndarray, np.ndarray] | None,
           tuple[np.ndarray, np.ndarray] | None]:
    """(metal_arrays, gem_arrays) for a .3dm or .stl file.

    Shared by `load_and_render()` and the Render tab's manual-orbit dialog —
    both need the identical geometry the real renderer would use, not a
    simplified stand-in, or the angle a user picks by hand wouldn't match
    what actually gets rendered.

    STL carries no layer information, so it always comes back metal-only —
    there is no way to tell a stone from the band. Stated rather than
    guessed: colouring an STL's stones is not possible from the file alone.
    """
    suffix = source_path.suffix.lower()
    if suffix == ".3dm":
        from catalog_organizer.snapshotter.threedm import load_3dm_split_arrays
        return load_3dm_split_arrays(source_path)
    if suffix == ".stl":
        import trimesh
        mesh = trimesh.load(source_path, force="mesh")
        return _split_stl_roles(mesh)
    raise RenderError(f"desteklenmeyen dosya tipi: {source_path.suffix}")


# Debris: a component must be BOTH nearly faceless and vanishingly small.
# On the user's ring the 83 scraps left by the converter are 2-face slivers
# under 0.15 mm, while every meaningful component (down to a 1.2 mm stone) is
# far above both limits.
_STL_DEBRIS_MAX_FACES = 8
_STL_DEBRIS_MAX_DIAGONAL_FRACTION = 0.005

# The guard that makes role-splitting safe. An STL exported from a finished
# render model is ONE body plus a few extras; a CAD assembly exported to STL
# is thousands of separate solids with no dominant one. Measured across eight
# real catalogue pieces the largest component held 2.0-23.1 % of the faces; on
# the user's exported ring it held 94.5 %. 80 % sits in that gap, so the rule
# below cannot fire on an assembly.
_STL_SINGLE_BODY_FRACTION = 0.80

# Flatness (thinnest extent ÷ longest) separating engraving from solids. On
# the ring: brand engraving and the "14K" hallmark measure 0.01-0.17, the two
# stones 0.58 and 0.62.
_STL_DECAL_FLATNESS = 0.35


def _split_stl_roles(
    mesh,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
    """(metal, gems) from an unlabelled STL — conservatively.

    An STL carries no layers, so roles have to be inferred, and inference here
    is dangerous: colouring a prong as a ruby is worse than showing no stone
    at all. So the split only runs on the ONE shape of file where it was
    measured to work, and otherwise everything is metal.

    Evidence, all measured rather than assumed:

      * A general "small chunky component = stone" rule was tried against four
        real stone-set pieces whose true split is known from their .3dm
        layers. It mislabelled 452, 1479 and 2170 metal components as stones.
        Rejected.
      * A general "drop flat components" rule deleted 19-69 % of the geometry
        of six real pieces. Rejected.
      * Both work on a single-body export: the guard above keeps them there.

    On the user's ring — the only file with an independent ground truth, its
    own glTF material assignment — this recovers all seven components
    correctly: two stones kept as gems, brand engraving and hallmark dropped,
    body kept as metal. That is one file. It is narrow on purpose.
    """
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:
        return (verts, faces), None
    if len(parts) < 2:
        return (verts, faces), None

    parts = sorted(parts, key=lambda p: len(p.faces), reverse=True)
    body = parts[0]
    total = len(faces)
    if len(body.faces) < _STL_SINGLE_BODY_FRACTION * total:
        # A CAD assembly. No dependable way to tell a stone from a prong here.
        return (verts, faces), None

    max_diagonal = (float(np.linalg.norm(body.extents))
                    * _STL_DEBRIS_MAX_DIAGONAL_FRACTION)
    metal_parts = [body]
    gem_parts = []
    for part in parts[1:]:
        tiny = float(np.linalg.norm(part.extents)) <= max_diagonal
        if len(part.faces) <= _STL_DEBRIS_MAX_FACES and tiny:
            continue                                    # converter debris
        extents = np.sort(part.extents)
        flatness = extents[0] / max(extents[2], 1e-9)
        if flatness < _STL_DECAL_FLATNESS:
            continue                                    # engraving, hallmark
        gem_parts.append(part)

    def _merge(group):
        if not group:
            return None
        out_v, out_f, offset = [], [], 0
        for part in group:
            part_v = np.asarray(part.vertices)
            out_v.append(part_v)
            out_f.append(np.asarray(part.faces) + offset)
            offset += part_v.shape[0]
        return np.vstack(out_v), np.vstack(out_f)

    return _merge(metal_parts), _merge(gem_parts)


def load_and_render(
    source_path: Path,
    *,
    file_id: str,
    metal_key: str,
    stone_key: str,
    **kwargs,
) -> RenderResult:
    """Load a .3dm or .stl and render it.

    Pass `category` (the record's `main_category`) so every product of one
    category is photographed in the same pose -- see `camera_axes`.
    """
    metal_arrays, gem_arrays = load_source_arrays(source_path)
    kwargs.setdefault("gems_guide_camera",
                      source_path.suffix.lower() != ".stl")
    return render_product(metal_arrays, gem_arrays, file_id=file_id,
                          metal_key=metal_key, stone_key=stone_key, **kwargs)
