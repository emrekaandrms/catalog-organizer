"""Single-file pipeline: ManifestEntry → CatalogRecord.

Composes snapshot → VLM → tag validation → measurement → scale check →
weight estimation → sprue → final record. Dependencies are injected so
tests can swap the VLM client for a deterministic stub.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import trimesh

from catalog_organizer.cad.measurements import (
    estimate_volume,
    measure_bbox,
    measure_bracelet,
    measure_earring,
    measure_pendant,
    measure_ring,
)
from catalog_organizer.cad.scale_check import check_scale
from catalog_organizer.cad.sprue import detect_sprue
from catalog_organizer.cad.stone_extraction import extract_stone_summary_from_3dm
from catalog_organizer.catalog.tag_validator import TagValidator
from catalog_organizer.core.schemas import (
    CatalogRecord,
    ManifestEntry,
    Measurements,
    StoneSummary,
    VLMResult,
)
from catalog_organizer.snapshotter.stl import snapshot_stl
from catalog_organizer.snapshotter.threedm import (
    estimate_cutter_volume_mm3,
    load_3dm_as_trimesh_arrays,
    snapshot_3dm,
)
from catalog_organizer.weight.metals import compute_metal_weights


class VLMClientProtocol(Protocol):
    def classify(
        self,
        snapshot_paths: list[Path],
        second_pass: dict | None = None,
    ) -> VLMResult: ...


@dataclass
class PipelineDeps:
    vlm_client: VLMClientProtocol
    tag_validator: TagValidator
    metal_densities: dict
    scale_thresholds: dict
    snapshot_resolution: int = 1024
    thumbnail_size: int = 256


@dataclass
class PreparedItem:
    """Output of the snapshot/measure phase, input to the VLM phase.

    Holds the trimesh.Trimesh so the VLM worker can re-run category-
    specific measurements once the category is known. The bbox + volume
    are pre-computed because they don't depend on the VLM's answer.
    """
    entry: ManifestEntry
    snapshot_paths: list[Path]
    mesh: trimesh.Trimesh
    bbox_measurements: Measurements   # bbox + volume only (no category fields)
    vol_conf: str                     # "high" / "low" / "unavailable"
    geometry_source: str              # "trimesh" / "rhino"


def _load_mesh(
    path: Path,
    extension: str,
    *,
    only_metal: bool = False,
) -> trimesh.Trimesh:
    """Load a CAD file as a trimesh.

    `only_metal=True` is honoured for `.3dm` only — it asks the rhino3dm
    loader to drop layers conventionally holding stones / cutters / sizing
    rings / curves / lights so the returned mesh contains only the
    castable metal body. For STL files there's no layer structure, so the
    flag is a no-op (STL workflow doesn't embed stones in the mesh anyway).
    """
    if extension == ".stl":
        mesh = trimesh.load(path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"STL did not produce a trimesh: {path}")
        return mesh
    if extension == ".3dm":
        verts, faces = load_3dm_as_trimesh_arrays(path, only_metal=only_metal)
        return trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    raise ValueError(f"Unsupported extension: {extension}")


def _take_snapshots(
    path: Path,
    file_id: str,
    extension: str,
    resolution: int,
    thumbnail_size: int,
) -> dict[str, Path]:
    if extension == ".stl":
        return snapshot_stl(path, file_id, resolution=resolution, thumbnail_size=thumbnail_size)
    return snapshot_3dm(path, file_id, resolution=resolution, thumbnail_size=thumbnail_size)


def _build_measurements(
    mesh: trimesh.Trimesh,
    main_category: str,
    geometry_source: str,
) -> tuple[Measurements, str]:
    """Build Measurements + return the volume-confidence label.

    Returns (measurements, vol_confidence ∈ {"high", "low", "unavailable"}).
    """
    bbox = measure_bbox(mesh)
    vol_mm3, vol_conf = estimate_volume(mesh)
    kwargs: dict[str, Any] = dict(
        bbox_width_mm=bbox.width,
        bbox_height_mm=bbox.height,
        bbox_depth_mm=bbox.depth,
        volume_mm3=vol_mm3,
        geometry_source=geometry_source,
    )

    if main_category in ("ring", "wedding_band"):
        r = measure_ring(mesh)
        kwargs.update(
            ring_inner_diameter_mm=r.inner_diameter_mm,
            ring_inner_circumference_mm=r.inner_circumference_mm,
            ring_size_eu=r.size_eu,
            ring_band_width_mm=r.band_width_mm,
            ring_top_width_mm=r.top_width_mm,
            ring_top_height_mm=r.top_height_mm,
        )
    elif main_category == "earring":
        e = measure_earring(mesh)
        kwargs.update(
            earring_total_width_mm=e.total_width_mm,
            earring_total_height_mm=e.total_height_mm,
            earring_depth_mm=e.depth_mm,
            earring_pair_or_single=e.pair_or_single,
        )
    elif main_category == "bracelet":
        b = measure_bracelet(mesh)
        kwargs.update(
            bracelet_inner_diameter_x_mm=b.inner_diameter_x_mm,
            bracelet_inner_diameter_y_mm=b.inner_diameter_y_mm,
            bracelet_inner_circumference_mm=b.inner_circumference_mm,
            bracelet_width_mm=b.width_mm,
        )
    elif main_category == "pendant":
        p = measure_pendant(mesh)
        kwargs.update(
            pendant_width_mm=p.width_mm,
            pendant_height_mm=p.height_mm,
            pendant_depth_mm=p.depth_mm,
        )

    return Measurements(**kwargs), vol_conf


# Physical sanity bounds. A measurement outside these ranges is almost
# certainly garbage produced by the wrong algorithm running on the wrong
# geometry (typically: VLM mis-classifies a flat pendant as a ring, and
# the PCA-based ring measurer returns ~0 mm inner diameter on a flat plate).
# Drop the offending fields so the catalog is not polluted with nonsense,
# and force review so a human can inspect the snapshots.
# Conservative physical bounds.
# - The smallest commercial ring (US size 0) has ~12.4 mm inner diameter.
#   We use 8 mm as the floor to leave room for kids' / toe rings while
#   still rejecting the sub-millimeter values produced by the PCA
#   ring measurer when fed a flat pendant.
# - The largest practical ring inner diameter is ~26 mm; 40 mm is a
#   generous ceiling that still excludes bracelets and bangles.
_PHYSICAL_BOUNDS = {
    "ring":     {"inner_diameter_mm": (8.0, 40.0)},
    "bracelet": {"inner_diameter_mm": (40.0, 110.0)},
}


def _sanitize_measurements(
    m: Measurements,
    main_category: str,
) -> tuple[Measurements, bool, str | None]:
    """Null out category-specific fields whose values are physically impossible.

    Returns (sanitized_measurements, was_sanitized, reason).
    """
    bounds = _PHYSICAL_BOUNDS.get(main_category)
    if not bounds:
        return m, False, None

    if main_category == "ring":
        lo, hi = bounds["inner_diameter_mm"]
        inner = m.ring_inner_diameter_mm
        if inner is not None and (inner < lo or inner > hi):
            data = m.model_dump()
            for key in (
                "ring_inner_diameter_mm",
                "ring_inner_circumference_mm",
                "ring_size_eu",
                "ring_band_width_mm",
                "ring_top_width_mm",
                "ring_top_height_mm",
            ):
                data[key] = None
            return (
                Measurements(**data),
                True,
                f"ring inner diameter {inner:.1f}mm outside [{lo}, {hi}]; "
                f"measurements zeroed (likely mis-classified flat geometry)",
            )

    if main_category == "bracelet":
        lo, hi = bounds["inner_diameter_mm"]
        inner = m.bracelet_inner_diameter_x_mm
        if inner is not None and (inner < lo or inner > hi):
            data = m.model_dump()
            for key in (
                "bracelet_inner_diameter_x_mm",
                "bracelet_inner_diameter_y_mm",
                "bracelet_inner_circumference_mm",
                "bracelet_opening_gap_mm",
                "bracelet_width_mm",
            ):
                data[key] = None
            return (
                Measurements(**data),
                True,
                f"bracelet inner diameter {inner:.1f}mm outside [{lo}, {hi}]; "
                f"measurements zeroed",
            )

    return m, False, None


def _empty_stone_summary(polished_status: str) -> StoneSummary:
    return StoneSummary(
        status=polished_status,
        center_stone=None,
        side_stones=[],
        total_estimated_carat=0.0,
    )


def _build_stone_summary(
    source_path: Path,
    file_extension: str,
    vlm: VLMResult,
    mesh=None,
) -> StoneSummary:
    """Decide what `StoneSummary` to attach to the catalog record.

    * .3dm — gem-layer objects give per-stone bbox sizes and the block-name
      suffix yields the cut shape. Most reliable; first preference.
    * .stl — stones are stripped before STL export, but the seats are
      still cut into the metal. `stl_stone_seats.build_stone_summary_for_stl`
      slices the mesh and measures the open seats; diameter → CZ carat.
    * Anything else (or geometric extractors return None) — fall through to
      the VLM's yes/no + visible count without sizes.
    """
    from catalog_organizer.weight.stones import StoneWeightTable  # noqa: PLC0415

    polished = vlm.polished_status

    # ── .stl: geometric seat detection ─────────────────────────────────────
    if file_extension == ".stl" and mesh is not None:
        try:
            from catalog_organizer.cad.stl_stone_seats import (  # noqa: PLC0415
                build_stone_summary_for_stl,
            )
            summary = build_stone_summary_for_stl(mesh, StoneWeightTable())
        except Exception:
            summary = None
        if summary is not None:
            return summary
        # No seats found → fall through to VLM signal below

    # ── .3dm: geometric extractor on the file ──────────────────────────────
    if file_extension == ".3dm":
        summary = extract_stone_summary_from_3dm(source_path)
        if summary is not None:
            return summary

    # ── VLM fallback (count only, no sizes) ────────────────────────────────
    if vlm.stone_presence != "stone":
        return _empty_stone_summary(polished)
    n = max(0, int(vlm.estimated_visible_stone_count))
    if n == 0:
        return StoneSummary(
            status="stone", center_stone=None, side_stones=[],
            total_estimated_carat=0.0,
        )
    from catalog_organizer.core.schemas import StoneEntry  # noqa: PLC0415
    placeholder = StoneEntry(
        shape="unknown",
        size_mm="unknown",
        quantity=n,
        estimated_carat_each=0.0,
        estimated_total_carat=0.0,
        source="vlm",
        confidence=float(vlm.tag_confidence or 0.5),
    )
    return StoneSummary(
        status="stone",
        center_stone=None,
        side_stones=[placeholder],
        total_estimated_carat=0.0,
    )


def prepare_snapshots(
    entry: ManifestEntry,
    deps: PipelineDeps,
) -> PreparedItem:
    """CPU + VTK work for one file. Single-threaded (VTK constraint).

    Loads the mesh, renders the 4 snapshot PNGs, and computes the
    category-agnostic measurements (bbox + volume). The returned
    `PreparedItem` is fed to `finalize_from_vlm()` after the VLM call.
    """
    path = Path(entry.source_path)
    snaps = _take_snapshots(
        path, entry.file_id, entry.file_extension,
        resolution=deps.snapshot_resolution,
        thumbnail_size=deps.thumbnail_size,
    )
    snapshot_paths = [snaps[v] for v in ("front", "side", "top", "iso")]

    # For .3dm we load a *metal-only* mesh (layers tagged as gem / cutter /
    # sizing / curves / lights are dropped) and use it for every downstream
    # measurement: bbox, volume, ring/earring/bracelet dimensions. The
    # full-geometry version is only needed for the snapshot rendering path,
    # which has already happened above and reads the file directly.
    # For STL the only_metal flag is a no-op (no layer structure).
    mesh = _load_mesh(path, entry.file_extension, only_metal=True)
    geometry_source = "trimesh" if entry.file_extension == ".stl" else "rhino"

    # Bbox + volume don't depend on the VLM's category answer.
    from catalog_organizer.cad.measurements import measure_bbox as _mb  # noqa: PLC0415
    bbox = _mb(mesh)
    if entry.file_extension == ".3dm":
        # Per-object hybrid volume — merged signed integral is wrong for
        # legacy files (flipped sub-shells cancel; mis-layered metal on
        # Creation Curves). See threedm.metal_volume_mm3 / the development log (2026-06-11).
        from catalog_organizer.snapshotter.threedm import (  # noqa: PLC0415
            metal_bbox_extents_mm, metal_volume_mm3,
        )
        vol_mm3, vol_conf = metal_volume_mm3(path)
        # Note-excluded dimensions (engraved monograms beside the piece).
        _ext = metal_bbox_extents_mm(path)
        if _ext is not None:
            from catalog_organizer.cad.measurements import BBox as _BBox  # noqa: PLC0415
            bbox = _BBox(width=_ext[0], height=_ext[1], depth=_ext[2],
                         volume_mm3=bbox.volume_mm3)
    else:
        vol_mm3, vol_conf = estimate_volume(mesh)

    # For .3dm: subtract the cutter volume (stone seats, drilled holes that
    # haven't been boolean-applied yet) so the reported volume matches the
    # post-cast metal mass. The workshop measured a 0.40-0.90 g per-ring
    # loss after cutters open; this step closes that gap.
    if entry.file_extension == ".3dm":
        from catalog_organizer.snapshotter.threedm import (  # noqa: PLC0415
            capped_cutter_volume_mm3,
        )
        try:
            cutter_vol = capped_cutter_volume_mm3(path, vol_mm3)
        except Exception:
            cutter_vol = 0.0
        if cutter_vol > 0:
            vol_mm3 = max(0.0, vol_mm3 - cutter_vol)

    bbox_measurements = Measurements(
        bbox_width_mm=bbox.width,
        bbox_height_mm=bbox.height,
        bbox_depth_mm=bbox.depth,
        volume_mm3=vol_mm3,
        geometry_source=geometry_source,
    )

    return PreparedItem(
        entry=entry,
        snapshot_paths=snapshot_paths,
        mesh=mesh,
        bbox_measurements=bbox_measurements,
        vol_conf=vol_conf,
        geometry_source=geometry_source,
    )


def finalize_from_vlm(
    prepared: PreparedItem,
    vlm: VLMResult,
    deps: PipelineDeps,
) -> CatalogRecord:
    """Build the final CatalogRecord from a PreparedItem + VLM answer.

    Safe to run on any worker thread (no VTK or shared mesh state).
    """
    entry = prepared.entry

    validated_tags = deps.tag_validator.validate(
        vlm.controlled_tags,
        main_category=vlm.main_category,
        stone_presence=vlm.stone_presence,
        estimated_visible_stone_count=vlm.estimated_visible_stone_count,
    )

    # Re-build measurements WITH category context now that we know it.
    # Then overwrite bbox/volume with the pre-computed values so we
    # don't pay for estimate_volume() twice.
    measurements, _ = _build_measurements(
        prepared.mesh, vlm.main_category, prepared.geometry_source,
    )
    data = measurements.model_dump()
    data["bbox_width_mm"]  = prepared.bbox_measurements.bbox_width_mm
    data["bbox_height_mm"] = prepared.bbox_measurements.bbox_height_mm
    data["bbox_depth_mm"]  = prepared.bbox_measurements.bbox_depth_mm
    data["volume_mm3"]     = prepared.bbox_measurements.volume_mm3
    measurements = Measurements(**data)

    measurements, was_sanitized, sanitize_reason = _sanitize_measurements(
        measurements, vlm.main_category,
    )

    scale_warn, scale_reason = check_scale(measurements, vlm.main_category, deps.scale_thresholds)

    # Sprues exist only in casting/print exports (.stl) — a .3dm is the
    # design file, and the slab detector false-positived on one
    # (sampleC_13.3dm). For .3dm we force "no sprue" regardless of VLM.
    if entry.file_extension == ".3dm":
        sprue = detect_sprue(vlm_says_sprue=False, vlm_confidence=0.0, mesh=None)
    else:
        sprue = detect_sprue(
            vlm.visible_sprue_or_casting_stem,
            vlm.sprue_confidence,
            mesh=prepared.mesh,
        )

    # Subtract sprue volume from the metal mass calculation when we have a
    # quantitative estimate. For casting trees the fused feeder joints
    # (cut_overlap_mm3) leave with the scrap too, so they are subtracted
    # from the piece — but NOT added to the reported sprue weight.
    metal_volume_mm3 = measurements.volume_mm3
    if sprue.detected and sprue.estimated_volume_mm3 and sprue.estimated_volume_mm3 > 0:
        metal_volume_mm3 = max(
            0.0,
            measurements.volume_mm3
            - sprue.estimated_volume_mm3
            - float(sprue.cut_overlap_mm3 or 0.0),
        )

    metal_weights = (
        compute_metal_weights(metal_volume_mm3, deps.metal_densities)
        if metal_volume_mm3 > 0
        else None
    )
    weight_source = (
        "stl_volume_density" if entry.file_extension == ".stl" and metal_volume_mm3 > 0
        else "rhino_volume_density" if entry.file_extension == ".3dm" and metal_volume_mm3 > 0
        else "unavailable"
    )
    # vol_conf coming from estimate_volume: high / medium / low / unavailable
    # We surface it directly. The earlier code downgraded "high" → "medium"
    # because at that time `mesh.volume` was treated as unverified; now we
    # know it matches MatrixGold within ~0.5 % on real jewelry STLs.
    weight_confidence = (
        "high"   if prepared.vol_conf == "high"
        else "medium" if prepared.vol_conf == "medium"
        else "low" if prepared.vol_conf == "low"
        else "unavailable"
    )

    # weight_confidence="low" means metal_volume_mm3() had to reinstate
    # "Creation Curves" solids whose real-vs-construction-junk status can't
    # be resolved geometrically (the accuracy-remediation notes, item 2, 2026-
    # 06-30) — the reported weight is a real, defensible number but rests
    # on an unverifiable judgment call, so route it to manual review rather
    # than presenting it with the same confidence as a clean measurement.
    low_volume_confidence = weight_confidence == "low"
    needs_review = (vlm.needs_manual_review or scale_warn or was_sanitized
                    or low_volume_confidence)
    review_reason = (
        sanitize_reason if was_sanitized
        else scale_reason if scale_warn
        else "creation_curve_volume_ambiguous" if low_volume_confidence
        else ("vlm_low_confidence" if vlm.needs_manual_review else None)
    )

    # Sellability (the scale/sales design notes §2, 2026-07-24): computed
    # ONLY from design_complete — brand deliberately does NOT factor in
    # (professional design workshop drawing commissioned/brand-referenced
    # pieces as a paid service, not counterfeit retail; brand stays
    # catalog/search metadata only, see schemas.py CatalogRecord docstring
    # comment). No separate VLM "is this sellable" opinion either — too
    # fuzzy a judgment to trust unmeasured.
    if vlm.design_complete_confidence < 0.80:
        sellability = "needs_review"
        sellability_reason = "design_complete_uncertain"
    elif not vlm.design_complete:
        sellability = "not_sellable"
        sellability_reason = "design_incomplete"
    else:
        sellability = "sellable"
        sellability_reason = None

    return CatalogRecord(
        file_id=entry.file_id,
        source_path=entry.source_path,
        file_extension=entry.file_extension,
        sha256=entry.sha256,
        main_category=vlm.main_category,
        main_category_confidence=vlm.main_category_confidence,
        subcategory=vlm.subcategory,
        subcategory_confidence=vlm.subcategory_confidence,
        controlled_tags=validated_tags,
        tag_confidence=vlm.tag_confidence,
        polished_status=vlm.polished_status,
        brand=None if vlm.brand_guess == "unknown" else vlm.brand_guess,
        brand_confidence=vlm.brand_confidence,
        rich_description=vlm.rich_description,
        design_complete=vlm.design_complete,
        design_complete_confidence=vlm.design_complete_confidence,
        incompleteness_notes=vlm.incompleteness_notes,
        sellability=sellability,
        sellability_reason=sellability_reason,
        measurements=measurements,
        metal_weights=metal_weights,
        weight_source=weight_source,
        weight_confidence=weight_confidence,
        stone_summary=_build_stone_summary(
            Path(entry.source_path), entry.file_extension, vlm,
            mesh=prepared.mesh,
        ),
        sprue=sprue,
        scale_warning=scale_warn,
        scale_warning_reason=review_reason if scale_warn else None,
        snapshot_paths=[str(p) for p in prepared.snapshot_paths],
        state="needs_review" if needs_review else "final",
        needs_manual_review=needs_review,
        review_reason=review_reason,
        manual_correction_applied=False,
        processing_log_ref=str(uuid.uuid4()),
        processed_at=datetime.now(tz=timezone.utc),
    )


def process_one(
    entry: ManifestEntry,
    deps: PipelineDeps,
) -> CatalogRecord:
    """Convenience wrapper: prepare + classify + finalize, single-threaded.

    Used by tests and by `BatchRunner(parallel_workers=1)` for backwards
    compatibility. New parallel execution path calls the two halves
    directly from different worker threads — see `orchestrator.batch`.
    """
    prepared = prepare_snapshots(entry, deps)
    vlm: VLMResult = deps.vlm_client.classify(prepared.snapshot_paths)
    return finalize_from_vlm(prepared, vlm, deps)
