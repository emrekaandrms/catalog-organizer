from __future__ import annotations

from catalog_organizer.core.schemas import Measurements


def check_scale(
    measurements: Measurements,
    main_category: str,
    thresholds: dict,
) -> tuple[bool, str | None]:
    """
    Apply §C.6 scale thresholds.

    Returns (scale_warning, reason). Item is NOT rejected on warning —
    it still flows through but lands in the review queue.
    """
    cat_t = thresholds.get(main_category)
    if not cat_t:
        return False, None

    max_bbox = max(
        measurements.bbox_width_mm,
        measurements.bbox_height_mm,
        measurements.bbox_depth_mm,
    )

    lo = float(cat_t.get("min_bbox_mm", 0))
    hi = float(cat_t.get("max_bbox_mm", 1e9))
    if max_bbox < lo:
        return True, f"bbox max {max_bbox:.1f}mm below {main_category} min {lo}"
    if max_bbox > hi:
        return True, f"bbox max {max_bbox:.1f}mm above {main_category} max {hi}"

    if main_category == "ring" and measurements.ring_inner_diameter_mm is not None:
        inner = measurements.ring_inner_diameter_mm
        ilo = float(cat_t.get("min_inner_mm", 0))
        ihi = float(cat_t.get("max_inner_mm", 1e9))
        if inner < ilo:
            return True, f"ring inner diameter {inner:.1f}mm below {ilo}"
        if inner > ihi:
            return True, f"ring inner diameter {inner:.1f}mm above {ihi}"

    return False, None
