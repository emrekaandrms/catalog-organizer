"""Geometric stone extraction from .3dm files.

MatrixGold places stones on layers whose names contain "Gem" (or sometimes
"Stone"). The geometry is almost always an `InstanceReference` to a
library block such as "Diamond_Round 01 03" — a single mesh definition
re-used for every stone of that size. Each instance's bounding box is in
world space (transform already baked in), so we can read the cut size
directly.

The shape (round / oval / marquise / etc.) is taken from the block name
when available (e.g. "Diamond_Round" → round, "Diamond_Marquise" →
marquise). When the block name is missing or generic, we fall back to a
bbox-aspect-ratio heuristic.

Stones of identical (shape, x, y) are grouped — the typical pavé ring has
40 round stones at exactly 1.30 mm, listing them individually would
flood the CSV. Each group becomes one `StoneEntry` with `quantity = N`.

Outputs `StoneSummary` directly so the pipeline can drop it into the
`CatalogRecord` without further processing.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import rhino3dm as r3

from catalog_organizer.core.schemas import StoneEntry, StoneSummary
from catalog_organizer.weight.stones import StoneWeightTable


# Layers whose names match these substrings are inspected for stone
# instances. Case-insensitive.
_GEM_LAYER_PATTERNS = ("gem", "stone")


# Block-name substrings → cut shape. Order matters only when names overlap
# ("Diamond_Round 01 03" matches "round" first which is what we want).
_BLOCK_NAME_TO_SHAPE: tuple[tuple[str, str], ...] = (
    ("round",    "round"),
    ("oval",     "oval"),
    ("marquise", "marquise"),
    ("baguette", "baguette"),
    ("princess", "princess"),
    ("pear",     "pear"),
    ("emerald",  "emerald"),
)


def _is_gem_layer(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in _GEM_LAYER_PATTERNS)


def _shape_from_block_name(name: str | None) -> str | None:
    if not name:
        return None
    n = name.lower()
    for pat, shape in _BLOCK_NAME_TO_SHAPE:
        if pat in n:
            return shape
    return None


def _shape_from_bbox(extents: tuple[float, float, float]) -> str:
    """Last-resort cut inference from bbox aspect.

    Bbox alone cannot distinguish round from princess (both ~1:1) or
    oval from emerald (both wider than tall) — these fall back to the
    aspect-only guess and are reported with a lower confidence by the
    caller.
    """
    x, y, _z = sorted(extents, reverse=True)   # x>=y>=z
    if y <= 0:
        return "unknown"
    aspect = x / y
    if aspect < 1.15:
        return "round"
    if aspect < 1.7:
        return "oval"
    if aspect < 3.0:
        return "marquise"
    return "baguette"


def _normalised_extents_mm(geom) -> tuple[float, float, float] | None:
    """Return (x, y, z) bbox extents in mm; None if the geometry can't be
    bounded (e.g. degenerate)."""
    try:
        bb = geom.GetBoundingBox()
    except Exception:
        return None
    mn, mx = bb.Min, bb.Max
    extents = (float(mx.X - mn.X), float(mx.Y - mn.Y), float(mx.Z - mn.Z))
    if min(extents) <= 0:
        return None
    return extents


def _world_centroid_mm(geom) -> tuple[float, float, float] | None:
    """World-space bbox-centre of a gem-layer object, for the floating-
    proxy sanity check below."""
    try:
        bb = geom.GetBoundingBox()
    except Exception:
        return None
    mn, mx = bb.Min, bb.Max
    return ((mn.X + mx.X) / 2.0, (mn.Y + mx.Y) / 2.0, (mn.Z + mx.Z) / 2.0)


# A "Gem" (singular, generic) layer sometimes holds a leftover proxy/
# placeholder object instead of a real positioned stone — verified on
# sampleB_15_21.3dm and sampleA_43.3dm, where a non-InstanceReference Brep on
# a plain "Gem" layer reported an IDENTICAL bogus 11.05 mm / 8.548 ct
# "centre stone" on two unrelated files (sampleB is a 2.6 mm-thick pendant;
# an 11 mm stone cannot physically fit). Real stones sit flush with the
# metal; a proxy floats free. Same threshold validated in the (now removed)
# render engine's `_filter_floating_gems`. See the accuracy-remediation notes
# item 1 (2026-06-29/30).
_MAX_GEM_TO_METAL_GAP_MM = 2.0


def _metal_point_cloud(path: Path):
    """Load the castable-metal vertex cloud for the floating-proxy distance
    check. Returns None if metal geometry can't be loaded (caller then
    skips the position check — better to keep a maybe-bogus stone than to
    crash extraction over an unrelated file-read issue)."""
    try:
        from catalog_organizer.snapshotter.threedm import (  # noqa: PLC0415
            load_3dm_as_trimesh_arrays,
        )
        verts, _faces = load_3dm_as_trimesh_arrays(path, only_metal=True)
        return verts
    except Exception:
        return None


# Second, independent sanity check: a stone cannot physically be thicker/
# wider/taller than the piece it's set into, AXIS BY AXIS. Catches proxies
# that DO touch the metal (distance check alone misses them) but are
# grossly oversized in at least one dimension — verified on sampleB_15_21.3dm:
# the bogus "Gem" Brep sits 1.03 mm from metal (passes the distance check)
# and its single-largest axis extent (11.07 mm) is smaller than the piece's
# overall max extent (19.38 mm) — so a "biggest dimension vs biggest
# dimension" comparison ALSO misses it. Its Z-extent (6.65 mm) is 2.5x the
# piece's own Z-extent (2.60 mm) though: a per-axis comparison is the only
# one that catches "too thick for this flat pendant."
_MAX_STONE_TO_PIECE_AXIS_FRACTION = 0.90


def extract_stone_summary_from_3dm(
    path: Path,
    weight_table: StoneWeightTable | None = None,
) -> StoneSummary | None:
    """Walk the .3dm file, gather every object on a gem-named layer, and
    return a populated `StoneSummary`. Returns `None` if the file has no
    gem-layer geometry (so the caller can fall back to a polished
    placeholder).

    The center stone is identified as the **single** largest stone
    (quantity-1 group with the highest per-stone carat). All other groups
    become side stones. If no group has quantity 1 — typical of pavé-only
    pieces — `center_stone` is `None` and every group is reported as side.
    """
    f3 = r3.File3dm.Read(str(path))
    if f3 is None:
        return None

    gem_layers = {
        i: lay.Name for i, lay in enumerate(f3.Layers)
        if _is_gem_layer(lay.Name) and getattr(lay, "Visible", True)
    }
    if not gem_layers:
        return None

    # Map block UUID → block name so we can read shape from "Diamond_Round" etc.
    block_name_by_id: dict[str, str] = {
        str(d.Id): d.Name for d in f3.InstanceDefinitions
    }

    # Floating-proxy filter: a real stone sits flush with the metal; a
    # leftover placeholder object floats free (see _MAX_GEM_TO_METAL_GAP_MM
    # docstring above). Build the metal point cloud once, lazily.
    metal_cloud = _metal_point_cloud(path)
    metal_tree = None
    piece_extents = None
    if metal_cloud is not None and len(metal_cloud) > 0:
        from scipy.spatial import cKDTree  # noqa: PLC0415
        metal_tree = cKDTree(metal_cloud)
        piece_extents = metal_cloud.max(axis=0) - metal_cloud.min(axis=0)

    # Group identical (shape, rounded x, rounded y) so a pavé of 40 identical
    # stones folds into one StoneEntry with quantity=40.
    groups: dict[tuple[str, float, float], list] = defaultdict(list)
    dropped_floating = 0
    for obj in f3.Objects:
        idx = obj.Attributes.LayerIndex
        if idx not in gem_layers:
            continue
        geo = obj.Geometry
        extents = _normalised_extents_mm(geo)
        if extents is None:
            continue

        if metal_tree is not None:
            centroid = _world_centroid_mm(geo)
            if centroid is not None:
                gap, _ = metal_tree.query(centroid)
                if float(gap) > _MAX_GEM_TO_METAL_GAP_MM:
                    dropped_floating += 1
                    continue  # floating proxy/placeholder, not a real stone

        if piece_extents is not None:
            too_big = any(
                pe > 0 and se > _MAX_STONE_TO_PIECE_AXIS_FRACTION * pe
                for se, pe in zip(extents, piece_extents)
            )
            if too_big:
                dropped_floating += 1
                continue  # thicker/wider/taller than the piece in some axis

        x, y, _z = sorted(extents, reverse=True)

        # Shape lookup: block name first (most reliable), bbox second.
        shape = None
        block_name = None
        if isinstance(geo, r3.InstanceReference):
            block_name = block_name_by_id.get(str(geo.ParentIdefId))
            shape = _shape_from_block_name(block_name)
        if not shape:
            shape = _shape_from_bbox(extents)

        # Round to 0.05 mm so near-identical instances coalesce despite
        # floating-point noise from the world-space transform.
        key = (shape, round(x * 20) / 20.0, round(y * 20) / 20.0)
        groups[key].append((extents, block_name))

    if not groups:
        return None

    wt = weight_table or StoneWeightTable()
    entries: list[StoneEntry] = []
    for (shape, x, y), members in groups.items():
        qty = len(members)
        if shape == "round":
            size_mm = f"{x:.2f}"
            carat_each = wt.estimate_carat("round", x)
        else:
            size_mm = f"{x:.2f}x{y:.2f}"
            carat_each = wt.estimate_carat(shape, (x, y))
        # Block-named shape → higher confidence; bbox-only guess → lower.
        confidence = 0.90 if any(m[1] for m in members) else 0.65
        entries.append(StoneEntry(
            shape=shape if shape in (
                "round", "oval", "baguette", "princess",
                "pear", "marquise", "emerald",
            ) else "unknown",
            size_mm=size_mm,
            quantity=qty,
            estimated_carat_each=carat_each,
            estimated_total_carat=carat_each * qty,
            source="geometry",
            confidence=confidence,
        ))

    # A solitaire's centre stone is normally a quantity-1 group, larger
    # than any pavé. Pick the single biggest such group.
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
        total_estimated_carat=total,
    )
