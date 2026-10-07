"""Exact cutter-subtraction via Rhino.Inside — the same Rhino engine (and
RhinoCommon API) Gemvision Matrix uses under the hood, running inside this
process against the user's own licensed Rhino install (5/6/7/8).

Why this module exists: the legacy heuristic in `snapshotter/threedm.py`
(`estimate_cutter_volume_mm3`) multiplied raw cutter-mesh volume by a
calibrated constant (0.50) because a full boolean was assumed too slow/
fragile. That assumption was wrong on both counts:

  * Slow: real per-cutter mesh-boolean-intersection against only the metal
    parts whose bounding box overlaps runs in a few seconds even on a
    150+-object file (see the development log (2026-06-29c)).
  * Fragile: it IS fragile if you try to union hundreds of separate Brep/
    mesh parts into one solid first (Rhino's own NURBS boolean union failed
    outright on a 163-part file) — but per-cutter, per-overlapping-part
    intersection with Rhino's MESH boolean (not Brep boolean) is robust.
    Cross-validated against two other independent boolean engines (Rhino
    Brep boolean, manifold3d) on `samples/sampleB_15_21.3dm`: all three converged
    on ~1 mm³ real removal where the old heuristic guessed ~17 mm³.

Root-cause finding (the development log (2026-06-29c)): the 0.50 fraction was NOT actually
measuring cutter overlap — it happened to land close to scale truth on the
calibration files because a SEPARATE bug (over-inclusion in the "Creation
Curves" reinstatement heuristic) was adding back a similar amount of extra
volume elsewhere. Two wrongs cancelling is why accuracy was inconsistent
across products. This module removes one of the two wrongs with a real
measurement; the Creation Curves heuristic is untouched (no evidence it's
wrong — see docstring in threedm.py) and remains a to-do for a following
session with more forensic time.

Public API:
    is_available() -> bool           # can we boot a licensed Rhino here?
    exact_cutter_volume_mm3(path) -> float | None   # None = Rhino unavailable
"""
from __future__ import annotations

from pathlib import Path

_BOOTED = False
_BOOT_FAILED = False


def _boot() -> bool:
    """Idempotent rhinoinside.load(). Returns True if a licensed Rhino is
    now loaded in this process (or already was). Rhino can only be booted
    ONCE per process — safe to call this repeatedly."""
    global _BOOTED, _BOOT_FAILED
    if _BOOTED:
        return True
    if _BOOT_FAILED:
        return False
    try:
        import rhinoinside  # noqa: PLC0415
        rhinoinside.load()
        _BOOTED = True
        return True
    except Exception:
        _BOOT_FAILED = True
        return False


def is_available() -> bool:
    """Whether exact Rhino-based measurement can run on this machine.
    False when Rhino isn't installed/licensed — callers should fall back
    to the legacy heuristic in that case."""
    return _boot()


# A cutter's bounding-box diagonal above this is treated as a structural
# boolean (drilled hole, sizing adjustment), not a gem seat — mirrors the
# size sanity check used elsewhere in the stone pipeline.
_MAX_PLAUSIBLE_CUTTER_DIAG_MM = 20.0


def exact_cutter_volume_mm3(path: str | Path,
                            metal_arrays=None) -> float | None:
    """Exact volume removed from the metal body by every 'Cutting Objects'
    / 'Cutter' layer object, via real Rhino mesh-boolean intersection.

    Uses the SAME rhino3dm-extracted vertex/face arrays the already-
    calibrated gross-volume pipeline uses (imported from threedm.py), so
    this measurement is geometrically consistent with `metal_volume_mm3`.

    Returns None if Rhino can't be booted on this machine (caller should
    fall back to the heuristic). Returns 0.0 if Rhino is available but the
    file has no cutter layers.
    """
    if not _boot():
        return None

    import rhino3dm as r3  # noqa: PLC0415
    import Rhino  # noqa: PLC0415
    from System.Collections.Generic import List  # noqa: PLC0415
    from catalog_organizer.snapshotter.threedm import (  # noqa: PLC0415
        _brep_to_arrays, _is_non_metal_layer_name, _mesh_to_arrays,
    )

    path = Path(path)
    f3 = r3.File3dm.Read(str(path))
    if f3 is None:
        return 0.0
    layers = {i: (lay.Name or "") for i, lay in enumerate(f3.Layers)}

    def _rhino_mesh(v, f) -> "Rhino.Geometry.Mesh":
        m = Rhino.Geometry.Mesh()
        for p in v:
            m.Vertices.Add(float(p[0]), float(p[1]), float(p[2]))
        for tri in f:
            m.Faces.AddFace(int(tri[0]), int(tri[1]), int(tri[2]))
        m.Normals.ComputeNormals()
        m.Compact()
        return m

    metal_meshes: list = []
    cutter_meshes: list = []
    # One welded union stands in for the metal parts when the caller has it.
    # Intersecting a seat cutter against each part SEPARATELY and summing
    # removes the shared volume once per part, so anywhere two parts overlap
    # inside a seat the removal is counted twice -- the same double count the
    # gross volume had, pointing the other way. Measured on the workshop
    # files the cutters are 1.6-14.4% of the gross, so this is not a rounding
    # detail: left alone it would have turned the fixed over-estimate into a
    # patchy under-estimate on exactly the dense, many-pronged pieces that
    # were furthest out to begin with.
    if metal_arrays is not None:
        for mv, mf in metal_arrays:
            if mv is not None and len(mv) and mf is not None and len(mf):
                metal_meshes.append(_rhino_mesh(mv, mf))
    for obj in f3.Objects:
        ln = layers.get(obj.Attributes.LayerIndex, "")
        low = ln.lower()
        g = obj.Geometry
        if isinstance(g, r3.Mesh):
            v, f = _mesh_to_arrays(g)
        elif isinstance(g, r3.Brep):
            v, f = _brep_to_arrays(g)
        else:
            continue
        if v is None or len(v) == 0 or len(f) == 0:
            continue
        if "cutter" in low or "cutting" in low:
            diag = float(((v.max(axis=0) - v.min(axis=0)) ** 2).sum() ** 0.5)
            if diag > _MAX_PLAUSIBLE_CUTTER_DIAG_MM:
                continue  # structural boolean, not a gem seat
            cutter_meshes.append(_rhino_mesh(v, f))
        elif (metal_arrays is None and not _is_non_metal_layer_name(ln)
                and "creation" not in low):
            metal_meshes.append(_rhino_mesh(v, f))

    if not metal_meshes or not cutter_meshes:
        return 0.0

    removed = 0.0
    for cut in cutter_meshes:
        cb = cut.GetBoundingBox(True)
        for mpart in metal_meshes:
            bb = mpart.GetBoundingBox(True)
            if (cb.Min.X > bb.Max.X or cb.Max.X < bb.Min.X or
                    cb.Min.Y > bb.Max.Y or cb.Max.Y < bb.Min.Y or
                    cb.Min.Z > bb.Max.Z or cb.Max.Z < bb.Min.Z):
                continue  # no bbox overlap — can't intersect
            try:
                la = List[Rhino.Geometry.Mesh](); la.Add(mpart)
                lb = List[Rhino.Geometry.Mesh](); lb.Add(cut)
                result = Rhino.Geometry.Mesh.CreateBooleanIntersection(la, lb)
            except Exception:
                continue
            if not result:
                continue
            for piece in result:
                mp = Rhino.Geometry.VolumeMassProperties.Compute(piece)
                if mp:
                    removed += abs(mp.Volume)
    return removed
