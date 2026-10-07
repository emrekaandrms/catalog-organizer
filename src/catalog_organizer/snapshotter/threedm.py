from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import rhino3dm as r3

from catalog_organizer.core.paths import snapshots_dir, thumbnails_dir
from catalog_organizer.snapshotter.base import render_views


def _mesh_to_arrays(mesh: r3.Mesh) -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [(v.X, v.Y, v.Z) for v in mesh.Vertices],
        dtype=np.float64,
    )
    faces: list[tuple[int, int, int]] = []
    for f in mesh.Faces:
        # f is (a, b, c, d). Triangle when c == d.
        a, b, c, d = f[0], f[1], f[2], f[3]
        faces.append((a, b, c))
        if d != c:
            faces.append((a, c, d))
    return verts, np.array(faces, dtype=np.int64) if faces else np.zeros((0, 3), dtype=np.int64)


def _bbox_to_mesh(bbox: r3.BoundingBox) -> tuple[np.ndarray, np.ndarray]:
    """Last-resort fallback for objects with no extractable render mesh."""
    mn, mx = bbox.Min, bbox.Max
    verts = np.array([
        (mn.X, mn.Y, mn.Z), (mx.X, mn.Y, mn.Z),
        (mx.X, mx.Y, mn.Z), (mn.X, mx.Y, mn.Z),
        (mn.X, mn.Y, mx.Z), (mx.X, mn.Y, mx.Z),
        (mx.X, mx.Y, mx.Z), (mn.X, mx.Y, mx.Z),
    ], dtype=np.float64)
    faces = np.array([
        (0, 1, 2), (0, 2, 3),    # bottom
        (4, 5, 6), (4, 6, 7),    # top
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ], dtype=np.int64)
    return verts, faces


_RENDER_MESH_TYPES = (
    r3.MeshType.Render,
    r3.MeshType.Default,
    r3.MeshType.Preview,
    r3.MeshType.Any,
)


# MatrixGold layer naming convention (verified against user samples,
# 2026-05-19): every non-metal object lives on a layer whose name contains
# one of these substrings (case-insensitive). When loading the mesh purely
# for *metal volume* calculation, we drop these layers so stones, cutter
# booleans, sizing rings, lights, and design curves don't get folded into
# the castable-metal mass. Snapshot rendering keeps everything so the VLM
# can still see stones for classification.
_NON_METAL_LAYER_PATTERNS = (
    "gem",
    "stone",
    "cutter",
    "cutting",
    "finger size",
    "creation curve",
    "light",
)


def _is_non_metal_layer_name(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in _NON_METAL_LAYER_PATTERNS)


_GEM_LAYER_PATTERNS = ("gem", "stone")


def _is_gem_layer_name(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in _GEM_LAYER_PATTERNS)


# Mirrors cad/stone_extraction.py's _MAX_GEM_TO_METAL_GAP_MM /
# _MAX_STONE_TO_PIECE_AXIS_FRACTION (same evidence: a generic placeholder
# "Gem" object left in the file by MatrixGold/the designer, floating free
# of the metal and/or grossly oversized in some axis vs the piece itself —
# proven via cross-file duplicate-value evidence on sampleB_15_21.3dm /
# sampleA_43.3dm, see the accuracy-remediation notes, item 1). That module
# already filters these out of the stone COUNT/carat estimate. This
# duplicates the same two thresholds (not imported — stone_extraction.py
# imports FROM this module for its own metal point cloud, so importing
# back would be circular) to ALSO drop them from the visual snapshot
# render: a 2026-07-24 real-catalog pilot showed the exact same proxy
# object rendering as a huge faceted blob dead-center of the piece,
# visually obscuring the actual design (a letter pendant became
# unrecognisable) and very likely degrading VLM classification too. Keep
# both copies in sync if the thresholds ever change.
_PROXY_GEM_MAX_METAL_GAP_MM = 2.0
_PROXY_GEM_MAX_AXIS_FRACTION = 0.90


def _find_proxy_gem_ids(
    f3: r3.File3dm,
    gem_layer_idx: set[int],
    metal_verts: list[np.ndarray],
) -> set[str]:
    """Return the string Attributes.Id of every gem-layer object that looks
    like a floating/oversized placeholder rather than a real, positioned
    stone (see threshold docstring above). Returns an empty set if there's
    no metal geometry to compare against (better to keep a maybe-bogus
    stone visible than to silently drop real gems over an unrelated
    metal-load failure)."""
    if not metal_verts:
        return set()
    metal_cloud = np.vstack(metal_verts)
    if metal_cloud.shape[0] == 0:
        return set()
    from scipy.spatial import cKDTree  # noqa: PLC0415
    metal_tree = cKDTree(metal_cloud)
    piece_extents = metal_cloud.max(axis=0) - metal_cloud.min(axis=0)

    dropped: set[str] = set()
    for obj in f3.Objects:
        if obj.Attributes.LayerIndex not in gem_layer_idx:
            continue
        geo = obj.Geometry
        try:
            bb = geo.GetBoundingBox()
        except Exception:
            continue
        mn, mx = bb.Min, bb.Max
        extents = np.array([mx.X - mn.X, mx.Y - mn.Y, mx.Z - mn.Z])
        if np.min(extents) <= 0:
            continue
        centroid = np.array([(mn.X + mx.X) / 2.0, (mn.Y + mx.Y) / 2.0, (mn.Z + mx.Z) / 2.0])

        gap, _ = metal_tree.query(centroid)
        if float(gap) > _PROXY_GEM_MAX_METAL_GAP_MM:
            dropped.add(str(obj.Attributes.Id))
            continue

        too_big = bool(np.any(
            (piece_extents > 0) & (extents > _PROXY_GEM_MAX_AXIS_FRACTION * piece_extents)
        ))
        if too_big:
            dropped.add(str(obj.Attributes.Id))

    return dropped


# Cutter layers specifically. MatrixGold puts boolean-subtract shapes
# (stone seats, drill holes, etc.) on a "Cutting Objects" layer. They've
# not been applied to the metal body yet, so to compute the *post-cast*
# metal weight we need to subtract their volume from the metal mesh.
_CUTTER_LAYER_PATTERNS = ("cutter", "cutting")


def _is_cutter_layer_name(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in _CUTTER_LAYER_PATTERNS)


# Empirically, only ~half of a stone-seat cutter actually sits inside the
# metal body — the pointed tip extends past the inner surface. Multiplying
# the raw cutter mesh volume by this factor approximates the real removed
# material without paying for a full boolean intersection. Calibrated
# against the user's workshop scale (target: 0.40-0.90 g loss per ring).
_CUTTER_INSIDE_METAL_FRACTION = 0.50

# Physical cap: cutters carve stone seats out of the surface — they can
# never consume most of the body. Workshop-calibrated rings lose 15-20 %
# of gross volume to seats; 30 % is a generous ceiling. Without this cap
# a halo ring with a 11 mm centre stone (huge cutter cylinder) produced a
# cutter estimate LARGER than the metal itself → net volume 0 → no weight
# (observed on samples/sampleA_43.3dm, 2026-06-08).
MAX_CUTTER_FRACTION_OF_GROSS = 0.30


def capped_cutter_volume_mm3(path, gross_volume_mm3: float) -> float:
    """Volume to subtract from `gross_volume_mm3` for stone-seat cutters.

    Primary path: exact Rhino mesh-boolean intersection (`cad.rhino_engine`)
    against the user's licensed Rhino install — a real measurement, not a
    guess. Falls back to the legacy 0.50-fraction heuristic, capped at
    `MAX_CUTTER_FRACTION_OF_GROSS`, only when Rhino can't be booted on this
    machine. See the development log (2026-06-29c): the heuristic's constant was proven
    wrong (3 independent boolean engines converged on ~1/17th of what it
    guessed on `sampleB_15_21.3dm`) — it happened to hit scale truth on the
    calibration set only because a separate, unrelated over-inclusion bug
    in the Creation-Curves reinstatement rule was adding back a similar
    amount of extra volume elsewhere.

    The exact path is never capped — a real boolean measurement cannot
    over-subtract by construction, unlike the heuristic it replaces.
    """
    if gross_volume_mm3 <= 0:
        return 0.0

    from catalog_organizer.cad import rhino_engine  # noqa: PLC0415
    try:
        # Against the SAME welded body the gross volume was taken from, so
        # the two measurements cannot disagree about what the metal is.
        exact = rhino_engine.exact_cutter_volume_mm3(
            path, metal_arrays=metal_union_mesh(Path(path)))
    except Exception:
        exact = None
    if exact is not None:
        return min(exact, gross_volume_mm3)  # sanity floor only, no % cap

    raw = estimate_cutter_volume_mm3(path)
    return min(raw, MAX_CUTTER_FRACTION_OF_GROSS * gross_volume_mm3)


def estimate_cutter_volume_mm3(path, fraction_inside_metal: float = _CUTTER_INSIDE_METAL_FRACTION) -> float:
    """Sum the |volume| of every object on a 'Cutter / Cutting Objects' layer,
    scaled by `fraction_inside_metal` (default 0.50).

    Why the factor: the cutter Brep is a complete cylinder/cone shape that
    *crosses* the metal surface — only the part inside the body is removed
    by the boolean. Naïvely subtracting the whole cutter volume
    over-estimates by ~2× on a ring's stone seats (verified on
    samples/sampleA_14/_18/_42). A full boolean intersection would be
    correct but costs seconds per file on million-triangle MatrixGold
    meshes; the constant fraction is calibrated against the user's
    workshop-measured 0.40-0.90 g per-ring loss.

    Returns 0.0 if there are no cutter-labelled layers or no extractable
    geometry — caller should treat that as "no cutters to apply".
    """
    from pathlib import Path as _P  # local import keeps Path optional here
    f3 = r3.File3dm.Read(str(_P(path)))
    if f3 is None:
        return 0.0
    cutter_layer_ids = {
        i for i, lay in enumerate(f3.Layers)
        if _is_cutter_layer_name(lay.Name) and getattr(lay, "Visible", True)
    }
    if not cutter_layer_ids:
        return 0.0

    import numpy as np  # noqa: PLC0415
    import trimesh  # noqa: PLC0415

    total = 0.0
    for obj in f3.Objects:
        if obj.Attributes.LayerIndex not in cutter_layer_ids:
            continue
        geo = obj.Geometry
        # Cutters are usually Brep cylinders/cones — convert to a mesh and
        # take the signed volume. For InstanceReferences we approximate
        # with the bounding-box volume (still a useful upper bound).
        try:
            if isinstance(geo, r3.Mesh):
                v, f = _mesh_to_arrays(geo)
                if v.shape[0] == 0:
                    continue
                tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
                vol = float(abs(tm.volume)) if tm.is_winding_consistent else 0.0
            elif isinstance(geo, r3.Brep):
                v, f = _brep_to_arrays(geo)
                if v.shape[0] == 0:
                    continue
                tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
                vol = float(abs(tm.volume)) if tm.is_winding_consistent else 0.0
            else:
                bb = geo.GetBoundingBox()
                mn, mx = bb.Min, bb.Max
                vol = float(
                    max(0.0, mx.X - mn.X)
                    * max(0.0, mx.Y - mn.Y)
                    * max(0.0, mx.Z - mn.Z)
                )
        except Exception:
            vol = 0.0
        total += vol
    return total * float(fraction_inside_metal)


def _brep_to_arrays(brep: r3.Brep) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every BrepFace's cached render mesh into (verts, faces).

    Rhino/MatrixGold writes .3dm files with display-mesh caches per face;
    those are what the Rhino viewport draws. Without these we'd be left
    with the bounding box, which destroys VLM accuracy on jewelry.
    """
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    offset = 0
    for face in brep.Faces:
        mesh = None
        for mt in _RENDER_MESH_TYPES:
            try:
                m = face.GetMesh(mt)
            except Exception:
                m = None
            if m is not None and len(m.Vertices) > 0 and len(m.Faces) > 0:
                mesh = m
                break
        if mesh is None:
            continue
        v, f = _mesh_to_arrays(mesh)
        if v.shape[0] == 0 or f.shape[0] == 0:
            continue
        all_verts.append(v)
        all_faces.append(f + offset)
        offset += v.shape[0]

    if not all_verts:
        # Truly no render mesh anywhere — keep bbox as a last resort.
        return _bbox_to_mesh(brep.GetBoundingBox())
    return np.vstack(all_verts), np.vstack(all_faces)


def load_3dm_as_trimesh_arrays(
    path: Path,
    *,
    only_metal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a .3dm into merged (verts, faces) arrays.

    By default returns every visible object on every visible layer — this
    is what the snapshot renderer needs so the VLM can see stones and the
    overall composition. Pass ``only_metal=True`` for the metal-volume
    pipeline: layers whose names match `_NON_METAL_LAYER_PATTERNS`
    (gem / cutter / cutting objects / finger sizes / creation curves /
    lights) are excluded, leaving only the castable metal body. Heads
    (stone-setting prongs) are intentionally kept because they are
    physically part of the metal piece.

    Empirical impact on user samples (2026-05-19): on stone-set 3DM rings,
    metal volume drops ~30-60 % once gems + cutters are filtered out,
    bringing computed weight in line with scale values.

    For the full-composition render (``only_metal=False``), individual
    gem-layer objects that look like a floating/oversized placeholder
    proxy (not a real, positioned stone) are also dropped — see
    `_find_proxy_gem_ids` docstring. Real stones render normally.
    """
    f3 = r3.File3dm.Read(str(path))
    if f3 is None:
        raise ValueError(f"rhino3dm could not open: {path}")

    hidden_layers: set[int] = set()
    skipped_by_name: set[int] = set()
    metal_layer_idx: set[int] = set()
    gem_layer_idx: set[int] = set()
    for i, layer in enumerate(f3.Layers):
        if not getattr(layer, "Visible", True):
            hidden_layers.add(i)
            continue
        non_metal = _is_non_metal_layer_name(layer.Name)
        if only_metal and non_metal:
            skipped_by_name.add(i)
        if not non_metal:
            metal_layer_idx.add(i)
        if _is_gem_layer_name(layer.Name):
            gem_layer_idx.add(i)

    proxy_gem_ids: set[str] = set()
    if not only_metal and gem_layer_idx:
        metal_verts_for_proxy_check: list[np.ndarray] = []
        for obj in f3.Objects:
            if obj.Attributes.LayerIndex not in metal_layer_idx:
                continue
            geo = obj.Geometry
            if isinstance(geo, r3.Mesh):
                v, _f = _mesh_to_arrays(geo)
            elif isinstance(geo, r3.Brep):
                v, _f = _brep_to_arrays(geo)
            else:
                continue
            if v.shape[0]:
                metal_verts_for_proxy_check.append(v)
        proxy_gem_ids = _find_proxy_gem_ids(f3, gem_layer_idx, metal_verts_for_proxy_check)

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    offset = 0
    for obj in f3.Objects:
        idx = obj.Attributes.LayerIndex
        if idx in hidden_layers or idx in skipped_by_name:
            continue
        if idx in gem_layer_idx and str(obj.Attributes.Id) in proxy_gem_ids:
            continue
        geo = obj.Geometry
        if isinstance(geo, r3.Mesh):
            v, f = _mesh_to_arrays(geo)
        elif isinstance(geo, r3.Brep):
            v, f = _brep_to_arrays(geo)
        else:
            continue
        if v.shape[0] == 0 or f.shape[0] == 0:
            continue
        all_verts.append(v)
        all_faces.append(f + offset)
        offset += v.shape[0]

    if not all_verts:
        raise ValueError(
            f"No usable geometry in {path}"
            + (f" after dropping {len(skipped_by_name)} non-metal layer(s)"
               if only_metal else "")
        )
    return np.vstack(all_verts), np.vstack(all_faces)


# "Creation Curves" reinstatement: a closed solid on a creation layer whose
# 90th-percentile vertex distance to the standard metal exceeds this many mm
# occupies space no other metal covers → it IS the metal (mis-layered).
# Ground-truthed on samples/ (2026-06-11): construction solids hug the real
# surface (p90 ≤ 0.33 mm); real mis-layered bodies sit clear of it
# (p90 ≥ 0.42 mm). 0.37 splits the observed gap.
_CREATION_UNIQUE_SPACE_P90_MM = 0.37

# Second gate: reject sweep profiles and offset rails parked on a creation
# layer. It USED to sit at 6.0 mm³, and the note explaining that number said
# including the smaller solids pushed sampleC_14 to +7.3 % against its scale
# weight while excluding them landed it at −1.7 %.
#
# That was the gate standing in for a different error. The gross volume at the
# time summed overlapping parts, and on that file the overlap is 173 mm³ —
# so throwing away 143 mm³ of real creation-layer metal brought the total back
# to roughly the right place for entirely the wrong reason. With the union in
# (see `_union_volume_mm3`) the compensation is no longer wanted, and the
# gate's own cost shows up. Against the workshop scale, holding everything
# else fixed:
#
#     gate      sampleC_14 (15.60 g)      sampleB_15_21 (1.28 g)
#     6.0 mm³        13.50 g  −13.5 %            1.26 g  −1.2 %
#     2.0 mm³        14.24 g   −8.7 %            1.26 g  −1.2 %
#     0.5 mm³        14.46 g   −7.3 %            1.26 g  −1.2 %
#     0.0 mm³        14.47 g   −7.2 %            1.26 g  −1.2 %
#
# 0.5 keeps genuine curve junk out (below it nothing more is recovered) and
# gives back everything the 6.0 gate was discarding. It does NOT close
# sampleC_14: that file is still 7 % light, which is a SEPARATE shortfall
# the old over-count was hiding, and it is recorded as open rather than tuned
# away — no constant here can be honestly chosen to fix one file at the cost
# of the physics on every other one.
_CREATION_MIN_SOLID_MM3 = 0.5

# A bbox-fill-fraction gate ("does this solid nearly fill its own bounding
# box, like a plate, vs a thin rail") was tried here 2026-06-29c after the
# exact-cutter-boolean fix exposed a ~20 mm³ excess in sampleB_15_21.3dm's
# Creation-Curves reinstatement (two ~0.15 fill objects were real junk).
# REVERTED: re-validated against all 6 golden files and found sampleC_14's
# genuine base plates (confirmed real metal, needed for its −1.7 % accuracy)
# have an EVEN LOWER fill fraction (0.047-0.095) than sampleB's junk objects
# (0.148-0.166) — a large flat plate shaped like the pendant's silhouette
# (not a rectangle) naturally has low bbox-fill. The signal inverts between
# files and cannot be a single global threshold. Left as a known, unsolved
# gap rather than shipping a rule proven to regress a different file — see
# the development log (2026-06-29c).


def detect_note_object_indices(
    object_arrays: list[tuple[int, np.ndarray]],
    *,
    gap_mm: float = 1.0,
    max_cluster_fraction: float = 0.40,
) -> set[int]:
    """Identify engraved-text / annotation objects to drop from a .3dm.

    Designers park notes in the drawing — the size ("13 boy"), the customer
    name, revision marks — as little extruded-text Breps sitting BESIDE the
    piece. They are not metal, but they inflate both the bounding box and the
    computed weight. `sampleA_12.3dm` carries a "customer-monogram" monogram below the ring.

    Detection (ground-truthed on the user's files, 2026-06-12):
      1. Cluster the candidate objects into spatial blobs by single-linkage
         on actual surface proximity (`gap_mm`). The piece is one blob; the
         text letters form their own blob (they sit ~2 mm clear of the body).
      2. The main blob = the largest by total |volume|. Its axis-aligned
         bounding box is the *piece envelope*.
      3. A non-main blob is a NOTE when it lies mostly OUTSIDE that envelope
         (a note is placed beside the piece). Stone seats / melee settings,
         which are also separate small blobs, sit INSIDE the envelope and are
         therefore kept.

    `object_arrays` is a list of `(global_index, vertices)` for every object
    on a metal layer. Returns the set of global indices to drop.

    Validated: sampleA_12 drops the customer-monogram text + 4 stray melee (20.6 mm³);
    sampleA_14 (cross with 30 detached pavé seats) drops nothing; every
    test4 file drops nothing (weights unchanged).
    """
    from scipy.sparse import csr_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    n = len(object_arrays)
    if n < 3:
        return set()

    samples = [v[:: max(1, len(v) // 60)] for _, v in object_arrays]
    owner = np.concatenate([[i] * len(s) for i, s in enumerate(samples)])
    cloud = np.vstack(samples)
    pairs = cKDTree(cloud).query_pairs(gap_mm, output_type="ndarray")
    # No early return on empty pairs: with no links every object is its own
    # singleton cluster, which is exactly what we want for a single-body
    # piece sitting next to separate note objects.
    if len(pairs):
        oi, oj = owner[pairs[:, 0]], owner[pairs[:, 1]]
        m = oi != oj
        rows = np.concatenate([oi[m], oj[m]])
        cols = np.concatenate([oj[m], oi[m]])
    else:
        rows = np.empty(0, dtype=int)
        cols = np.empty(0, dtype=int)
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)

    from collections import defaultdict  # noqa: PLC0415
    members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        members[labels[i]].append(i)

    def _bbox_diag(idxs: list[int]) -> float:
        bv = np.vstack([object_arrays[i][1] for i in idxs])
        return float(np.linalg.norm(bv.max(0) - bv.min(0)))

    # Main blob = the one with the largest spatial extent (the piece). Using
    # bbox diagonal, not vertex count, so a high-res note can't masquerade as
    # the body.
    main = max(members, key=lambda L: _bbox_diag(members[L]))
    main_v = np.vstack([object_arrays[i][1] for i in members[main]])
    mn, mx = main_v.min(0), main_v.max(0)
    main_diag = float(np.linalg.norm(mx - mn))
    margin = 0.02 * main_diag

    note_idx: set[int] = set()
    for label, members_i in members.items():
        if label == main:
            continue
        # A note is much smaller than the piece (density-independent test).
        if _bbox_diag(members_i) >= max_cluster_fraction * main_diag:
            continue  # large detached blob = a real part (e.g. earring half)
        blob_v = np.vstack([object_arrays[i][1] for i in members_i])
        outside = ((blob_v < mn - margin) | (blob_v > mx + margin)).any(axis=1).mean()
        if outside > 0.5:
            for i in members_i:
                note_idx.add(object_arrays[i][0])
    return note_idx


# A cast piece weighs the volume of the UNION of its parts, not the sum.
#
# Jewellery CAD overlaps constantly: a prong sits inside the head it grows
# from, a bezel wall runs into the shank, a bail passes through the loop it
# hangs on. Summing per-object volumes counts every one of those overlaps
# twice, and the piece is reported heavier than it can ever be cast. Measured
# on the workshop files, sum over union:
#
#     sampleA_187   +0.7 %        sampleA_42   +18.7 %
#     sampleA_12     +5.3 %        sampleA_43   +19.5 %
#     sampleA_35     +6.0 %        sampleA_112 +20.8 %
#     sampleA_14    +15.2 %        sampleA_18   +24.7 %
#
# -- which is the "gramlar %20-25 yuksek cikiyor" report, in one number per
# file, and it is systematic: it has the same sign on every file and gets
# worse the more separate components a design carries. The pieces at the top
# of that list are the plain ones; the ones at the bottom are dense with
# prongs and bezels.
def _union_volume_mm3(solids: list) -> float | None:
    """Volume of the boolean union, or None when the union cannot be taken."""
    import trimesh  # noqa: PLC0415

    if not solids:
        return 0.0
    if len(solids) == 1:
        return float(abs(solids[0].volume))
    try:
        merged = trimesh.boolean.union(solids, engine="manifold")
    except Exception:
        return None
    if merged is None or merged.is_empty:
        return None
    return float(abs(merged.volume))


def _as_solid(v, f):
    """Weld a Brep's per-face render mesh into a closed solid, or None.

    Rhino writes a separate mesh per Brep face, so nothing is watertight as
    it arrives -- vertices along a shared edge are duplicated and the
    windings between faces disagree. Welding fixes both for most objects
    (68 of 104 on sampleA_12; filling the remaining pinholes takes it to 97).
    What is left over is genuinely open geometry whose enclosed volume is a
    guess either way, and it keeps the per-object treatment it has always
    had rather than being forced into a union it would corrupt.
    """
    import trimesh  # noqa: PLC0415

    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    try:
        mesh.fix_normals()
    except Exception:
        return None
    if mesh.is_watertight and abs(mesh.volume) > 0:
        return mesh
    try:
        trimesh.repair.fill_holes(mesh)
        mesh.fix_normals()
    except Exception:
        return None
    if mesh.is_watertight and abs(mesh.volume) > 0:
        return mesh
    return None


def metal_volume_mm3(path: Path) -> tuple[float, str]:
    """Metal volume of a .3dm, computed *per object* with a hybrid rule.

    Why not one signed volume over the merged mesh: workshop files mix two
    failure modes that point in opposite directions (ground-truthed against
    the user's scale on samples/, 2026-06-11):

      * Objects whose sub-shells have inverted winding — their contributions
        CANCEL inside a single signed integral. `sampleB_15_21.3dm` measured
        69 mm³ merged vs 123 mm³ real (scale: 1.28 g).
      * Brep objects arrive as per-face open shells. Splitting those into
        connected components and summing |volume| per component counts the
        cone-to-origin garbage of every open patch. `sampleC_13.3dm`
        exploded to 17,000 mm³ that way (real: ~998 mm³).

    Hybrid rule, per object:
      1. If every connected component of the object's mesh is watertight →
         sum the components' |signed volume| (immune to flipped shells).
      2. Otherwise (open face-soup, i.e. Brep skins) → |signed volume| of
         the whole object's mesh (faces share one consistent orientation
         from the Brep surface normals, so the integral encloses correctly).

    Additionally, "Creation Curves" layers are re-examined: legacy files
    sometimes park real metal there (`sampleB_15_21.3dm` keeps the entire
    base plates of a foot pendant on it — 74 of 124 mm³!). A creation
    object is counted as metal when it is a closed solid that occupies
    *unique space*: its p90 vertex distance to the standard metal exceeds
    `_CREATION_UNIQUE_SPACE_P90_MM`. Construction solids (sweep rails,
    profile copies) hug the metal surface and fail that test.

    Returns (volume_mm3, confidence) where confidence is "medium" (these
    are render meshes, not analytic solids). Raises ValueError when no
    metal geometry exists.
    """
    import trimesh  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    f3 = r3.File3dm.Read(str(path))
    if f3 is None:
        raise ValueError(f"rhino3dm could not open: {path}")

    excluded: set[int] = set()
    creation: set[int] = set()
    for i, layer in enumerate(f3.Layers):
        name = layer.Name or ""
        if not getattr(layer, "Visible", True):
            excluded.add(i)
        elif "creation" in name.lower():
            creation.add(i)
        elif _is_non_metal_layer_name(name):
            excluded.add(i)

    def _object_arrays(obj):
        geo = obj.Geometry
        if isinstance(geo, r3.Mesh):
            return _mesh_to_arrays(geo)
        if isinstance(geo, r3.Brep):
            return _brep_to_arrays(geo)
        return None, None

    def _object_volume(mesh: "trimesh.Trimesh") -> float:
        try:
            comps = mesh.split(only_watertight=False)
        except Exception:
            comps = []
        if comps is not None and len(comps) > 0 and all(c.is_watertight for c in comps):
            return float(sum(abs(c.volume) for c in comps))
        return float(abs(mesh.volume))

    # ── Note / engraved-text detection (drop before measuring) ──────────
    std_objects = []   # (global_index, v, f)
    for gi, obj in enumerate(f3.Objects):
        idx = obj.Attributes.LayerIndex
        if idx in excluded or idx in creation:
            continue
        v, f = _object_arrays(obj)
        if v is None or v.shape[0] == 0 or f.shape[0] == 0:
            continue
        std_objects.append((gi, v, f))
    note_indices = detect_note_object_indices(
        [(gi, v) for gi, v, _ in std_objects]
    )

    # ── Pass 1: standard metal layers ───────────────────────────────────
    # Solids go into one union; anything that will not weld shut keeps the
    # per-object treatment described above. See `_union_volume_mm3` for the
    # measurement that put the union here.
    solids: list = []
    leftover = 0.0
    n_objects = 0
    metal_vertex_clouds: list[np.ndarray] = []
    for gi, v, f in std_objects:
        if gi in note_indices:
            continue
        n_objects += 1
        metal_vertex_clouds.append(v)
        solid = _as_solid(v, f)
        if solid is not None:
            solids.append(solid)
        else:
            leftover += _object_volume(
                trimesh.Trimesh(vertices=v, faces=f, process=False))

    # ── Pass 2: creation-curve solids in unique space ───────────────────
    # Whether this reinstatement rule actually adds anything is a real,
    # unresolved judgment call — see the accuracy-remediation notes, item 2.
    # Two independent geometric signals (bbox-fill-fraction, object naming)
    # were tried 2026-06-30 to separate real mis-layered metal from
    # construction-curve junk within the reinstated set; both failed to
    # generalise (a rule tuned on sampleB_15_21.3dm inverted on sampleC_14's
    # confirmed-real plates). Rather than ship an unvalidated classifier,
    # any file where this pass actually contributes volume is flagged with
    # confidence="low" so downstream reporting/CSV surfaces the uncertainty
    # instead of presenting a point estimate as if it were exact.
    creation_contributed = False
    if creation and metal_vertex_clouds:
        tree = cKDTree(np.vstack(metal_vertex_clouds))
        for obj in f3.Objects:
            if obj.Attributes.LayerIndex not in creation:
                continue
            v, f = _object_arrays(obj)
            if v is None or v.shape[0] == 0 or f.shape[0] == 0:
                continue
            mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
            vol = float(abs(mesh.volume))
            if vol < _CREATION_MIN_SOLID_MM3:
                continue  # curves, degenerate surfaces, sweep-profile solids
            sample = v[:: max(1, len(v) // 800)]
            dists, _ = tree.query(sample)
            if float(np.percentile(dists, 90)) > _CREATION_UNIQUE_SPACE_P90_MM:
                n_objects += 1
                creation_contributed = True
                solid = _as_solid(v, f)
                if solid is not None:
                    solids.append(solid)
                else:
                    leftover += _object_volume(mesh)

    if n_objects == 0:
        raise ValueError(f"No metal geometry in {path}")

    union = _union_volume_mm3(solids)
    if union is None:
        # The union engine refused this file. Fall back to the old sum rather
        # than to nothing -- it is the number this function returned for a
        # year, it is merely too high, and a weight that is too high beats no
        # weight at all on a screen the user prices from.
        total = sum(abs(m.volume) for m in solids) + leftover
    else:
        total = union + leftover
    confidence = "low" if creation_contributed else "medium"
    return total, confidence



def metal_union_mesh(path: Path):
    """The metal body as welded, non-overlapping solids, or None.

    Returns a list of `(vertices, faces)` -- the connected components of the
    union, which together are the whole piece and individually cannot overlap.

    Exists so the cutter measurement can be taken against the same body the
    weight is taken from. Intersecting each seat cutter against each metal
    part and summing removes the overlap TWICE wherever two parts overlap
    inside a seat -- the mirror image of the error the union fixed on the
    gross side, and pointing the other way, so leaving it in place would
    have turned an over-estimate into a patchy under-estimate.
    """
    import trimesh  # noqa: PLC0415

    f3 = r3.File3dm.Read(str(path))
    if f3 is None:
        return None
    excluded: set[int] = set()
    for i, layer in enumerate(f3.Layers):
        name = layer.Name or ""
        if not getattr(layer, "Visible", True):
            excluded.add(i)
        elif "creation" in name.lower() or _is_non_metal_layer_name(name):
            excluded.add(i)

    solids = []
    for obj in f3.Objects:
        if obj.Attributes.LayerIndex in excluded:
            continue
        geo = obj.Geometry
        if isinstance(geo, r3.Mesh):
            v, f = _mesh_to_arrays(geo)
        elif isinstance(geo, r3.Brep):
            v, f = _brep_to_arrays(geo)
        else:
            continue
        if v is None or v.shape[0] == 0 or f.shape[0] == 0:
            continue
        solid = _as_solid(v, f)
        if solid is not None:
            solids.append(solid)

    if not solids:
        return None
    if len(solids) == 1:
        merged = solids[0]
    else:
        try:
            merged = trimesh.boolean.union(solids, engine="manifold")
        except Exception:
            return None
    if merged is None or merged.is_empty:
        return None

    # Split into connected components before handing them over. They are the
    # same body, but as separate bodies they cannot overlap each other, so the
    # caller's per-part bounding-box cull works again -- and that cull is what
    # keeps the measurement quick. Against one 112k-triangle union mesh every
    # seat cutter had to be intersected with the whole piece and sampleA_42
    # went from 21 s to 60 s; split, the cull is back and so is the speed.
    try:
        parts = merged.split(only_watertight=False)
    except Exception:
        parts = []
    if len(parts) > 1:
        return [(np.asarray(c.vertices), np.asarray(c.faces)) for c in parts]
    return [(np.asarray(merged.vertices), np.asarray(merged.faces))]

def metal_bbox_extents_mm(path: Path) -> tuple[float, float, float] | None:
    """Axis-aligned bounding-box extents of the metal body, notes excluded.

    Used for .3dm dimensions so engraved-text monograms placed beside the
    piece (see `detect_note_object_indices`) don't inflate the size. Returns
    `(width, height, depth)` sorted as the raw X/Y/Z extents, or None when no
    metal geometry is present.
    """
    f3 = r3.File3dm.Read(str(path))
    if f3 is None:
        return None
    excluded: set[int] = set()
    creation: set[int] = set()
    for i, layer in enumerate(f3.Layers):
        name = layer.Name or ""
        if not getattr(layer, "Visible", True):
            excluded.add(i)
        elif "creation" in name.lower():
            creation.add(i)
        elif _is_non_metal_layer_name(name):
            excluded.add(i)

    std = []
    for gi, obj in enumerate(f3.Objects):
        idx = obj.Attributes.LayerIndex
        if idx in excluded or idx in creation:
            continue
        geo = obj.Geometry
        if isinstance(geo, r3.Mesh):
            v, f = _mesh_to_arrays(geo)
        elif isinstance(geo, r3.Brep):
            v, f = _brep_to_arrays(geo)
        else:
            continue
        if v is None or v.shape[0] == 0:
            continue
        std.append((gi, v))
    if not std:
        return None
    notes = detect_note_object_indices(std)
    kept = [v for gi, v in std if gi not in notes]
    if not kept:
        kept = [v for _, v in std]
    allv = np.vstack(kept)
    w, h, d = (allv.max(0) - allv.min(0)).tolist()
    return float(w), float(h), float(d)


def _load_3dm_as_polydata(path: Path) -> pv.PolyData:
    verts, faces = load_3dm_as_trimesh_arrays(path)
    n = faces.shape[0]
    pv_faces = np.empty((n, 4), dtype=np.int64)
    pv_faces[:, 0] = 3
    pv_faces[:, 1:] = faces
    return pv.PolyData(verts, pv_faces.flatten())


def snapshot_3dm(
    path: Path,
    file_id: str,
    resolution: int = 1024,
    thumbnail_size: int = 256,
) -> dict[str, Path]:
    polydata = _load_3dm_as_polydata(path)
    out_dir = snapshots_dir(file_id)
    thumb_path = thumbnails_dir() / f"{file_id}.webp"

    pngs = render_views(
        polydata,
        out_dir=out_dir,
        resolution=resolution,
        thumbnail_path=thumb_path,
        thumbnail_size=thumbnail_size,
    )
    pngs["thumbnail"] = thumb_path
    return pngs


# ── Block (InstanceReference) resolution ─────────────────────────────────────
#
# MatrixGold places gems as *block instances*, not as free Breps: in a real
# 68-file sample every stone-set piece stored 60+ stones as InstanceReference
# objects pointing at one shared definition, with only the untransformed
# definition master sitting loose on the Gem layer. Neither
# `load_3dm_as_trimesh_arrays` nor the volume pipeline resolves those, so a
# catalogue render asked to colour "the stones" would have found none.
#
# Resolving them is required for the PDF catalogue's stone colour to mean
# anything. It is deliberately NOT wired into the classification snapshot path
# in the same change — that path has golden-file expectations attached.

_MAX_INSTANCE_DEPTH = 4


def _xform_to_matrix(xform) -> np.ndarray:
    return np.array([
        [xform.M00, xform.M01, xform.M02, xform.M03],
        [xform.M10, xform.M11, xform.M12, xform.M13],
        [xform.M20, xform.M21, xform.M22, xform.M23],
        [xform.M30, xform.M31, xform.M32, xform.M33],
    ], dtype=np.float64)


def _apply_xform(verts: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return verts @ matrix[:3, :3].T + matrix[:3, 3]


def _geometry_arrays(geo) -> tuple[np.ndarray, np.ndarray] | None:
    if isinstance(geo, r3.Mesh):
        return _mesh_to_arrays(geo)
    if isinstance(geo, r3.Brep):
        return _brep_to_arrays(geo)
    return None


def _instance_definition_members(f3) -> tuple[dict[str, list[str]], set[str]]:
    """(definition id → member object ids, all member ids).

    The member ids double as a skip-list: definition masters also appear in
    `f3.Objects`, parked at the definition origin. Drawing them directly puts
    an extra, mis-placed copy of every stone in the render — this is the
    "giant stone blob in the middle of the product" the proxy-gem filter was
    built to hide.
    """
    by_def: dict[str, list[str]] = {}
    members: set[str] = set()
    for idef in f3.InstanceDefinitions:
        ids = [str(i) for i in idef.GetObjectIds()]
        by_def[str(idef.Id)] = ids
        members.update(ids)
    return by_def, members


def _resolve_instance(
    geo,
    objects_by_id: dict,
    members_by_def: dict[str, list[str]],
    depth: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expand an InstanceReference into concrete, transformed (verts, faces).

    Recurses so a block containing blocks still resolves; `_MAX_INSTANCE_DEPTH`
    stops a self-referencing definition from looping forever.
    """
    if depth > _MAX_INSTANCE_DEPTH:
        return []
    matrix = _xform_to_matrix(geo.Xform)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for member_id in members_by_def.get(str(geo.ParentIdefId), []):
        member = objects_by_id.get(member_id)
        if member is None:
            continue
        member_geo = member.Geometry
        if isinstance(member_geo, r3.InstanceReference):
            nested = _resolve_instance(
                member_geo, objects_by_id, members_by_def, depth + 1)
            out.extend((_apply_xform(v, matrix), f) for v, f in nested)
            continue
        got = _geometry_arrays(member_geo)
        if got is None or got[0].shape[0] == 0 or got[1].shape[0] == 0:
            continue
        out.append((_apply_xform(got[0], matrix), got[1]))
    return out


def load_3dm_split_arrays(
    path: Path,
) -> tuple[tuple[np.ndarray, np.ndarray] | None,
           tuple[np.ndarray, np.ndarray] | None]:
    """Load a .3dm as TWO meshes: (metal, gems).

    `load_3dm_as_trimesh_arrays` merges everything into one body because the
    VLM only ever needed a single silhouette. A marketing render needs the
    split — metal takes a gold/silver PBR material, stones take a coloured gem
    material, and one mesh cannot carry two materials.

    Layer classification matches the merged loader (hidden layers out,
    `_is_non_metal_layer_name` out) so a render can never show something the
    analysis pipeline classified as junk: cutters, creation curves, finger
    sizes and light gizmos land in neither half.

    Unlike the merged loader this one resolves block instances, which is where
    the real stones actually live. Either half may be None — a polished piece
    has no gems.
    """
    f3 = r3.File3dm.Read(str(path))
    if f3 is None:
        raise ValueError(f"rhino3dm could not open: {path}")

    hidden_layers: set[int] = set()
    metal_layer_idx: set[int] = set()
    gem_layer_idx: set[int] = set()
    for i, layer in enumerate(f3.Layers):
        if not getattr(layer, "Visible", True):
            hidden_layers.add(i)
            continue
        if _is_gem_layer_name(layer.Name):
            gem_layer_idx.add(i)
        elif not _is_non_metal_layer_name(layer.Name):
            metal_layer_idx.add(i)

    members_by_def, member_ids = _instance_definition_members(f3)
    objects_by_id = {str(o.Attributes.Id): o for o in f3.Objects}

    metal: list[tuple[np.ndarray, np.ndarray]] = []
    gems: list[tuple[np.ndarray, np.ndarray]] = []

    for obj in f3.Objects:
        idx = obj.Attributes.LayerIndex
        if idx in hidden_layers:
            continue
        if str(obj.Attributes.Id) in member_ids:
            continue  # definition master, drawn only via its references
        if idx in gem_layer_idx:
            bucket = gems
        elif idx in metal_layer_idx:
            bucket = metal
        else:
            continue

        geo = obj.Geometry
        if isinstance(geo, r3.InstanceReference):
            bucket.extend(
                _resolve_instance(geo, objects_by_id, members_by_def))
            continue
        got = _geometry_arrays(geo)
        if got is None or got[0].shape[0] == 0 or got[1].shape[0] == 0:
            continue
        bucket.append(got)

    # Drop engraved notes — the size stamp ("17 boy cilalı"), the customer
    # name, a hallmark — that designers park beside the piece on a metal
    # layer. The volume pipeline already filters these; the render did not,
    # and the first real catalogue page came out with the text rendered
    # larger than the product, which had been squashed into a corner because
    # the text inflated the bounding box the camera frames on.
    if len(metal) >= 3:
        try:
            notes = detect_note_object_indices(
                [(i, v) for i, (v, _f) in enumerate(metal)])
        except Exception:
            notes = set()
        if notes and len(notes) < len(metal):
            metal = [part for i, part in enumerate(metal) if i not in notes]

    def _merge(parts: list[tuple[np.ndarray, np.ndarray]]):
        if not parts:
            return None
        verts: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        offset = 0
        for v, f in parts:
            verts.append(v)
            faces.append(f + offset)
            offset += v.shape[0]
        return np.vstack(verts), np.vstack(faces)

    return _merge(metal), _merge(gems)
