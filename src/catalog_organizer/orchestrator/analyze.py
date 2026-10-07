"""VLM-free geometric analysis of a single CAD file.

Everything the workshop actually needs — metal weight, stone sizes,
dimensions, sprue detection, sprue weight — is computable from geometry
alone. The VLM contributes only category/tags, and its instability
(timeouts, truncated JSON) must never block these numbers.

`analyze_file(path)` runs the full geometric stack:

    mesh load (metal-only layers for .3dm)
      → bbox + volume (winding-consistent signed volume)
      → cutter-volume subtraction (.3dm stone seats)
      → sprue detection (slab-profile) + volume subtraction
      → metal weights (8 alloys) for net metal volume
      → stone extraction (.3dm gem layers / .stl seat slicing)
      → ring bore (largest passable cylinder, stone-components stripped)
      → final product weight (metal + stones) per alloy

and returns an `AnalysisResult` that renders to console text or a CSV row.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import trimesh

from catalog_organizer.cad.measurements import (
    estimate_volume,
    measure_bbox,
    measure_ring,
)
from catalog_organizer.cad.sprue import detect_sprue
from catalog_organizer.cad.stone_extraction import extract_stone_summary_from_3dm
from catalog_organizer.core.config import load_metal_densities
from catalog_organizer.core.schemas import MetalWeights, SpruInfo, StoneSummary
from catalog_organizer.weight.metals import compute_metal_weights

_GRAMS_PER_CARAT = 0.2


@dataclass
class AnalysisResult:
    source_path: str
    file_extension: str

    # Dimensions
    bbox_width_mm: float = 0.0
    bbox_height_mm: float = 0.0
    bbox_depth_mm: float = 0.0

    # Ring bore (None when no valid 8-40 mm bore exists → not a ring)
    ring_inner_diameter_mm: float | None = None
    ring_band_width_mm: float | None = None

    # Volumes (mm³)
    gross_volume_mm3: float = 0.0       # metal mesh as loaded
    cutter_volume_mm3: float = 0.0      # .3dm stone-seat cutters (subtracted)
    sprue_volume_mm3: float = 0.0       # detected sprue (subtracted)
    net_metal_volume_mm3: float = 0.0   # what actually gets cast & kept
    volume_confidence: str = "unavailable"

    # Weights
    metal_weights: MetalWeights | None = None          # net metal, per alloy
    sprue_weights: dict[str, float] = field(default_factory=dict)  # per alloy
    stone_weight_g: float = 0.0
    product_weights: dict[str, float] = field(default_factory=dict)  # metal+stones

    # Stones
    stone_summary: StoneSummary | None = None

    # Sprue
    sprue: SpruInfo | None = None

    elapsed_s: float = 0.0
    error: str | None = None


def analyze_file(path: Path, densities: dict | None = None) -> AnalysisResult:
    """Run the full geometric stack on one .3dm / .stl file. Never raises —
    failures land in `result.error` so batch callers keep moving."""
    path = Path(path)
    ext = path.suffix.lower()
    result = AnalysisResult(source_path=str(path), file_extension=ext)
    t0 = time.monotonic()
    try:
        _analyze_into(result, path, ext, densities or load_metal_densities())
    except Exception as exc:  # noqa: BLE001 — diagnostic surface, not control flow
        result.error = f"{type(exc).__name__}: {exc}"
    result.elapsed_s = time.monotonic() - t0
    return result


def _analyze_into(result: AnalysisResult, path: Path, ext: str, densities: dict) -> None:
    from catalog_organizer.orchestrator.pipeline import _load_mesh  # noqa: PLC0415

    if ext not in (".3dm", ".stl"):
        raise ValueError(f"unsupported extension: {ext}")

    mesh: trimesh.Trimesh = _load_mesh(path, ext, only_metal=True)

    # ── Dimensions + gross volume ────────────────────────────────────────
    bbox = measure_bbox(mesh)
    result.bbox_width_mm = bbox.width
    result.bbox_height_mm = bbox.height
    result.bbox_depth_mm = bbox.depth

    if ext == ".3dm":
        # Per-object hybrid volume (handles flipped sub-shells, open Brep
        # skins, and mis-layered metal on Creation Curves). The merged
        # signed integral is wrong for legacy workshop files — see
        # threedm.metal_volume_mm3 docstring + the development log (2026-06-11).
        from catalog_organizer.snapshotter.threedm import (  # noqa: PLC0415
            metal_bbox_extents_mm, metal_volume_mm3,
        )
        vol_mm3, vol_conf = metal_volume_mm3(path)
        # Dimensions from the note-excluded metal body (engraved monograms
        # placed beside the piece must not inflate the size). the development log (2026-06-12).
        ext_mm = metal_bbox_extents_mm(path)
        if ext_mm is not None:
            result.bbox_width_mm, result.bbox_height_mm, result.bbox_depth_mm = ext_mm
    else:
        vol_mm3, vol_conf = estimate_volume(mesh)
    result.gross_volume_mm3 = vol_mm3
    result.volume_confidence = vol_conf

    # ── Cutter subtraction (.3dm only, capped at 30 % of gross) ─────────
    if ext == ".3dm":
        from catalog_organizer.snapshotter.threedm import (  # noqa: PLC0415
            capped_cutter_volume_mm3,
        )
        try:
            result.cutter_volume_mm3 = capped_cutter_volume_mm3(path, vol_mm3)
        except Exception:
            result.cutter_volume_mm3 = 0.0

    # ── Sprue (geometric only — no VLM in this path) ────────────────────
    # Sprues exist only in casting/print exports (.stl). A .3dm is the
    # design file: running the detector there produced a false positive
    # (sampleC_13.3dm, −155 mm³ of metal that was never a sprue).
    cut_overlap = 0.0
    if ext == ".stl":
        sprue = detect_sprue(vlm_says_sprue=False, vlm_confidence=0.0, mesh=mesh)
        result.sprue = sprue
        if sprue.detected and sprue.estimated_volume_mm3:
            result.sprue_volume_mm3 = float(sprue.estimated_volume_mm3)
            cut_overlap = float(sprue.cut_overlap_mm3 or 0.0)

    # ── Net metal volume → per-alloy weights ────────────────────────────
    net = max(0.0, vol_mm3 - result.cutter_volume_mm3
              - result.sprue_volume_mm3 - cut_overlap)
    result.net_metal_volume_mm3 = net
    if net > 0:
        result.metal_weights = compute_metal_weights(net, densities)
    if result.sprue_volume_mm3 > 0:
        sprue_w = compute_metal_weights(result.sprue_volume_mm3, densities)
        result.sprue_weights = {
            k: round(v, 3) for k, v in sprue_w.model_dump().items()
        }

    # ── Stones ───────────────────────────────────────────────────────────
    stones: StoneSummary | None = None
    if ext == ".3dm":
        stones = extract_stone_summary_from_3dm(path)
    else:
        # On a casting tree of many identical small pieces (chain links on
        # runner bars), the link through-holes look exactly like stone
        # seats to the slice detector — 5 MM KOLYE.stl reported 42 phantom
        # stones. Mass-produced trees don't carry set stones in the wax,
        # so skip seat detection there. Trees of a FEW pieces (e.g. 3
        # rings on rails) keep it.
        skip_seats = False
        if result.sprue is not None and result.sprue.detected:
            try:
                n_pieces = len(mesh.split(only_watertight=False))
                skip_seats = n_pieces >= 10
            except Exception:
                skip_seats = False
        if not skip_seats:
            try:
                from catalog_organizer.cad.stl_stone_seats import (  # noqa: PLC0415
                    build_stone_summary_for_stl,
                )
                from catalog_organizer.weight.stones import StoneWeightTable  # noqa: PLC0415
                stones = build_stone_summary_for_stl(mesh, StoneWeightTable())
            except Exception:
                stones = None
    result.stone_summary = stones
    if stones is not None:
        result.stone_weight_g = round(stones.total_estimated_carat * _GRAMS_PER_CARAT, 3)

    # ── Final product weight = net metal + stones, per alloy ────────────
    if result.metal_weights is not None:
        result.product_weights = {
            k: round(v + result.stone_weight_g, 3)
            for k, v in result.metal_weights.model_dump().items()
        }

    # ── Ring bore (reported only when physically plausible) ─────────────
    try:
        ring = measure_ring(mesh)
        if 8.0 <= ring.inner_diameter_mm <= 40.0:
            result.ring_inner_diameter_mm = ring.inner_diameter_mm
            result.ring_band_width_mm = ring.band_width_mm
    except Exception:
        pass


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_text(r: AnalysisResult) -> str:
    """Human-readable report (console / diagnostics log)."""
    lines: list[str] = []
    name = Path(r.source_path).name
    lines.append("=" * 64)
    lines.append(f"FILE: {name}   ({r.elapsed_s:.1f}s)")
    lines.append("=" * 64)
    if r.error:
        lines.append(f"ERROR: {r.error}")
        return "\n".join(lines)

    lines.append(f"Dimensions (mm):   {r.bbox_width_mm:.1f} x {r.bbox_height_mm:.1f} x {r.bbox_depth_mm:.1f}")
    if r.ring_inner_diameter_mm is not None:
        lines.append(f"Ring bore:         {r.ring_inner_diameter_mm:.2f} mm inner diameter"
                     f"   (band width {r.ring_band_width_mm:.2f} mm)")
    lines.append("")
    lines.append(f"Gross metal volume: {r.gross_volume_mm3:>9.1f} mm3   [{r.volume_confidence}]")
    if r.cutter_volume_mm3 > 0:
        lines.append(f"  - cutters:        {r.cutter_volume_mm3:>9.1f} mm3")
    if r.sprue_volume_mm3 > 0:
        lines.append(f"  - sprue:          {r.sprue_volume_mm3:>9.1f} mm3")
    lines.append(f"Net metal volume:   {r.net_metal_volume_mm3:>9.1f} mm3")
    lines.append("")

    if r.sprue is not None and r.sprue.detected:
        lines.append(f"SPRUE: detected (source={r.sprue.source}, conf={r.sprue.confidence:.2f})")
        if r.sprue_weights:
            sw = r.sprue_weights
            lines.append(f"  sprue weight: {sw.get('silver_925_g', 0):.2f} g silver"
                         f" / {sw.get('gold_14k_yellow_g', 0):.2f} g 14K-Y"
                         f" / {sw.get('gold_18k_yellow_g', 0):.2f} g 18K-Y")
    else:
        lines.append("SPRUE: none detected")
    lines.append("")

    if r.metal_weights is not None:
        w = r.metal_weights
        lines.append("METAL WEIGHTS (net, g):")
        lines.append(f"  925 Silver:  {w.silver_925_g:8.2f}     Platinum:   {w.platinum_g:8.2f}")
        lines.append(f"  10K Y/W:     {w.gold_10k_yellow_g:8.2f} / {w.gold_10k_white_g:.2f}")
        lines.append(f"  14K Y/W:     {w.gold_14k_yellow_g:8.2f} / {w.gold_14k_white_g:.2f}")
        lines.append(f"  18K Y/W:     {w.gold_18k_yellow_g:8.2f} / {w.gold_18k_white_g:.2f}")
    else:
        lines.append("METAL WEIGHTS: unavailable (no usable volume)")
    lines.append("")

    s = r.stone_summary
    if s is not None and (s.center_stone or s.side_stones):
        total_qty = (1 if s.center_stone else 0) + sum(e.quantity for e in s.side_stones)
        lines.append(f"STONES: {total_qty} pcs, {s.total_estimated_carat:.3f} ct total"
                     f"  ({r.stone_weight_g:.3f} g)")
        if s.center_stone:
            c = s.center_stone
            lines.append(f"  center: {c.shape} {c.size_mm} mm  {c.estimated_total_carat:.3f} ct")
        for e in s.side_stones:
            lines.append(f"  side:   {e.shape} {e.size_mm} mm  x{e.quantity}"
                         f"  {e.estimated_total_carat:.3f} ct")
    else:
        lines.append("STONES: none detected")
    lines.append("")

    if r.product_weights:
        pw = r.product_weights
        lines.append("FINAL PRODUCT WEIGHT (metal + stones, g):")
        lines.append(f"  925 Silver:  {pw.get('silver_925_g', 0):8.2f}     Platinum:   {pw.get('platinum_g', 0):8.2f}")
        lines.append(f"  14K Y/W:     {pw.get('gold_14k_yellow_g', 0):8.2f} / {pw.get('gold_14k_white_g', 0):.2f}")
        lines.append(f"  18K Y/W:     {pw.get('gold_18k_yellow_g', 0):8.2f} / {pw.get('gold_18k_white_g', 0):.2f}")
    return "\n".join(lines)


def to_flat_dict(r: AnalysisResult) -> dict:
    """Flatten for CSV / JSON export (golden tests use this too)."""
    d: dict = {
        "file": Path(r.source_path).name,
        "extension": r.file_extension,
        "error": r.error or "",
        "bbox_width_mm": round(r.bbox_width_mm, 2),
        "bbox_height_mm": round(r.bbox_height_mm, 2),
        "bbox_depth_mm": round(r.bbox_depth_mm, 2),
        "ring_inner_diameter_mm": round(r.ring_inner_diameter_mm, 2) if r.ring_inner_diameter_mm else "",
        "ring_band_width_mm": round(r.ring_band_width_mm, 2) if r.ring_band_width_mm else "",
        "gross_volume_mm3": round(r.gross_volume_mm3, 1),
        "cutter_volume_mm3": round(r.cutter_volume_mm3, 1),
        "sprue_detected": bool(r.sprue.detected) if r.sprue else False,
        "sprue_volume_mm3": round(r.sprue_volume_mm3, 1),
        "net_metal_volume_mm3": round(r.net_metal_volume_mm3, 1),
        "volume_confidence": r.volume_confidence,
        "stone_count": 0,
        "stone_total_carat": 0.0,
        "stone_weight_g": r.stone_weight_g,
        "stone_detail": "",
    }
    s = r.stone_summary
    if s is not None:
        d["stone_count"] = (1 if s.center_stone else 0) + sum(e.quantity for e in s.side_stones)
        d["stone_total_carat"] = round(s.total_estimated_carat, 3)
        parts = []
        if s.center_stone:
            c = s.center_stone
            parts.append(f"center:{c.shape}:{c.size_mm}:qty=1:{c.estimated_total_carat:.3f}ct")
        for e in s.side_stones:
            parts.append(f"{e.shape}:{e.size_mm}:qty={e.quantity}:{e.estimated_total_carat:.3f}ct")
        d["stone_detail"] = "|".join(parts)
    if r.metal_weights is not None:
        for k, v in r.metal_weights.model_dump().items():
            d[f"metal_{k}"] = round(v, 3)
    for k, v in r.sprue_weights.items():
        d[f"sprue_{k}"] = v
    for k, v in r.product_weights.items():
        d[f"product_{k}"] = v
    return d
