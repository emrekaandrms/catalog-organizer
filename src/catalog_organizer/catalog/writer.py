from __future__ import annotations

import csv
import threading
from pathlib import Path

import orjson

from catalog_organizer.core.paths import data_dir
from catalog_organizer.core.schemas import CatalogRecord

_FLUSH_EVERY = 20


# Column order is the *export contract*. Grouped logically so a human opening
# the CSV in Excel can read left-to-right: identity → classification →
# measurements → weights → stones → sprue → meta. Adding a new column means
# extending the list; never reorder existing columns (downstream consumers
# may pin by index).
_CSV_COLUMNS = [
    # ── Identity ────────────────────────────────────────────────────────────
    "file_id",
    "source_path",
    "file_extension",
    "sha256",

    # ── Classification ──────────────────────────────────────────────────────
    "main_category",
    "main_category_confidence",
    "subcategory",
    "subcategory_confidence",
    "controlled_tags",
    "tag_confidence",
    "polished_status",

    # ── Measurements: core (every record) ───────────────────────────────────
    "bbox_width_mm",
    "bbox_height_mm",
    "bbox_depth_mm",
    "volume_mm3",
    "geometry_source",

    # ── Measurements: ring (filled only for ring/wedding_band) ──────────────
    "ring_inner_diameter_mm",
    "ring_inner_circumference_mm",
    "ring_size_eu",
    "ring_band_width_mm",
    "ring_top_width_mm",
    "ring_top_height_mm",

    # ── Measurements: earring ───────────────────────────────────────────────
    "earring_total_height_mm",
    "earring_total_width_mm",
    "earring_depth_mm",
    "earring_pin_detected",
    "earring_pair_or_single",
    "hoop_outer_diameter_mm",
    "hoop_inner_diameter_mm",

    # ── Measurements: bracelet ──────────────────────────────────────────────
    "bracelet_inner_diameter_x_mm",
    "bracelet_inner_diameter_y_mm",
    "bracelet_inner_circumference_mm",
    "bracelet_opening_gap_mm",
    "bracelet_width_mm",

    # ── Measurements: pendant ───────────────────────────────────────────────
    "pendant_height_mm",
    "pendant_width_mm",
    "pendant_depth_mm",
    "bail_detected",
    "bail_inner_width_mm",
    "bail_inner_height_mm",

    # ── Metal weights (volume × density per metal). Yellow vs white gold
    # are split because the alloy bases differ enough to shift mass by ~5%.
    "silver_925_g",
    "gold_10k_yellow_g",
    "gold_10k_white_g",
    "gold_14k_yellow_g",
    "gold_14k_white_g",
    "gold_18k_yellow_g",
    "gold_18k_white_g",
    "platinum_g",
    "weight_source",
    "weight_confidence",

    # ── Final product weights = metal mass + total stone mass.
    # `total_stone_weight_g` is just `total_estimated_carat × 0.2`
    # (definition of metric carat). Each `product_*_g` adds it to the
    # matching metal column so a customer-facing line ("a piece weighs
    # X g in 18k white gold") doesn't require manual addition.
    "total_stone_weight_g",
    "product_silver_925_g",
    "product_gold_10k_yellow_g",
    "product_gold_10k_white_g",
    "product_gold_14k_yellow_g",
    "product_gold_14k_white_g",
    "product_gold_18k_yellow_g",
    "product_gold_18k_white_g",
    "product_platinum_g",

    # ── Stones ──────────────────────────────────────────────────────────────
    "stone_status",
    "center_stone_shape",
    "center_stone_size_mm",
    "center_stone_quantity",
    "center_stone_estimated_carat",
    "center_stone_total_carat",
    "center_stone_confidence",
    "side_stones_count",
    "side_stones_total_carat",
    "side_stones_detail",     # pipe-separated: "shape:size:qty:carat | ..."
    "total_estimated_carat",

    # ── Sprue ───────────────────────────────────────────────────────────────
    # The metal columns above are already NET of the sprue (pipeline.py
    # subtracts it before computing mass), so these give the other half of
    # the picture: what the runner itself weighs. On real wedding bands the
    # bore support bar is 4-14 % of the piece — too much to leave as a
    # volume the user has to convert by hand, which is why the per-metal
    # columns are back (they were dropped on 2026-05-xx as derivable).
    # A sprue is always cast in the same metal as the piece it feeds, so
    # read only the column matching the metal you are quoting.
    "sprue_detected",
    "sprue_source",
    "sprue_confidence",
    "sprue_estimated_volume_mm3",
    "sprue_silver_925_g",
    "sprue_gold_10k_yellow_g",
    "sprue_gold_10k_white_g",
    "sprue_gold_14k_yellow_g",
    "sprue_gold_14k_white_g",
    "sprue_gold_18k_yellow_g",
    "sprue_gold_18k_white_g",
    "sprue_platinum_g",

    # ── Meta / review state ─────────────────────────────────────────────────
    "scale_warning",
    "scale_warning_reason",
    "state",
    "needs_manual_review",
    "review_reason",
    "manual_correction_applied",
    "processing_log_ref",
    "snapshot_paths",
    "processed_at",
    "schema_version",
]


# ── Formatting helpers ──────────────────────────────────────────────────────
# Excel parses empty cells as blanks, "0" as zero, and "0.000" as a float.
# We use "" for None (blank cell, easy to filter) and fixed-precision floats
# so column widths stay readable.

def _f(v: float | int | None, digits: int = 3) -> str:
    if v is None:
        return ""
    return f"{float(v):.{digits}f}"


def _i(v: int | None) -> str:
    return "" if v is None else str(int(v))


def _b(v: bool | None) -> str:
    if v is None:
        return ""
    return "1" if v else "0"


def _s(v: str | None) -> str:
    return "" if v is None else str(v)


def _sprue_weights(sp) -> dict[str, float] | None:
    """Mass of the sprue itself, per metal.

    Returns None when there is nothing to weigh (no sprue, or a VLM-only
    detection that carries no volume estimate) so the CSV shows blanks
    rather than a misleading 0.000.
    """
    if not sp.detected or not sp.estimated_volume_mm3:
        return None
    try:
        from catalog_organizer.core.config import load_metal_densities  # noqa: PLC0415
        from catalog_organizer.weight.metals import compute_metal_weights  # noqa: PLC0415
        return compute_metal_weights(
            float(sp.estimated_volume_mm3), load_metal_densities(),
        ).model_dump()
    except Exception:
        # Density table unreadable — the rest of the row is still valid.
        return None


def _record_to_csv_row(rec: CatalogRecord) -> dict[str, str]:
    m = rec.measurements
    mw = rec.metal_weights
    ss = rec.stone_summary
    cs = ss.center_stone
    sp = rec.sprue
    spw = _sprue_weights(sp)

    return {
        # Identity
        "file_id":        rec.file_id,
        "source_path":    rec.source_path,
        "file_extension": rec.file_extension,
        "sha256":         rec.sha256,

        # Classification
        "main_category":             rec.main_category,
        "main_category_confidence":  _f(rec.main_category_confidence),
        "subcategory":               rec.subcategory,
        "subcategory_confidence":    _f(rec.subcategory_confidence),
        "controlled_tags":           "|".join(rec.controlled_tags),
        "tag_confidence":            _f(rec.tag_confidence),
        "polished_status":           rec.polished_status,

        # Measurements: core
        "bbox_width_mm":   _f(m.bbox_width_mm),
        "bbox_height_mm":  _f(m.bbox_height_mm),
        "bbox_depth_mm":   _f(m.bbox_depth_mm),
        "volume_mm3":      _f(m.volume_mm3, digits=1),
        "geometry_source": m.geometry_source,

        # Measurements: ring
        "ring_inner_diameter_mm":      _f(m.ring_inner_diameter_mm),
        "ring_inner_circumference_mm": _f(m.ring_inner_circumference_mm),
        "ring_size_eu":                _f(m.ring_size_eu, digits=2),
        "ring_band_width_mm":          _f(m.ring_band_width_mm),
        "ring_top_width_mm":           _f(m.ring_top_width_mm),
        "ring_top_height_mm":          _f(m.ring_top_height_mm),

        # Measurements: earring
        "earring_total_height_mm": _f(m.earring_total_height_mm),
        "earring_total_width_mm":  _f(m.earring_total_width_mm),
        "earring_depth_mm":        _f(m.earring_depth_mm),
        "earring_pin_detected":    _b(m.earring_pin_detected),
        "earring_pair_or_single":  _s(m.earring_pair_or_single),
        "hoop_outer_diameter_mm":  _f(m.hoop_outer_diameter_mm),
        "hoop_inner_diameter_mm":  _f(m.hoop_inner_diameter_mm),

        # Measurements: bracelet
        "bracelet_inner_diameter_x_mm":   _f(m.bracelet_inner_diameter_x_mm),
        "bracelet_inner_diameter_y_mm":   _f(m.bracelet_inner_diameter_y_mm),
        "bracelet_inner_circumference_mm": _f(m.bracelet_inner_circumference_mm),
        "bracelet_opening_gap_mm":        _f(m.bracelet_opening_gap_mm),
        "bracelet_width_mm":              _f(m.bracelet_width_mm),

        # Measurements: pendant
        "pendant_height_mm":    _f(m.pendant_height_mm),
        "pendant_width_mm":     _f(m.pendant_width_mm),
        "pendant_depth_mm":     _f(m.pendant_depth_mm),
        "bail_detected":        _b(m.bail_detected),
        "bail_inner_width_mm":  _f(m.bail_inner_width_mm),
        "bail_inner_height_mm": _f(m.bail_inner_height_mm),

        # Metal weights — empty cells when weight_source is "unavailable"
        "silver_925_g":        _f(mw.silver_925_g)       if mw else "",
        "gold_10k_yellow_g":   _f(mw.gold_10k_yellow_g)  if mw else "",
        "gold_10k_white_g":    _f(mw.gold_10k_white_g)   if mw else "",
        "gold_14k_yellow_g":   _f(mw.gold_14k_yellow_g)  if mw else "",
        "gold_14k_white_g":    _f(mw.gold_14k_white_g)   if mw else "",
        "gold_18k_yellow_g":   _f(mw.gold_18k_yellow_g)  if mw else "",
        "gold_18k_white_g":    _f(mw.gold_18k_white_g)   if mw else "",
        "platinum_g":          _f(mw.platinum_g)         if mw else "",
        "weight_source":       rec.weight_source,
        "weight_confidence":   rec.weight_confidence,

        # Stones-to-grams + final product weight per metal. 1 ct = 0.2 g
        # (definition of metric carat), so total_stone_g = total_carat × 0.2.
        # Each product_*_g column adds the stones onto the matching metal
        # mass so the user can read the final piece weight directly.
        "total_stone_weight_g":      _f(ss.total_estimated_carat * 0.2, digits=3),
        "product_silver_925_g":      _f((mw.silver_925_g + ss.total_estimated_carat * 0.2) if mw else None, digits=3),
        "product_gold_10k_yellow_g": _f((mw.gold_10k_yellow_g + ss.total_estimated_carat * 0.2) if mw else None, digits=3),
        "product_gold_10k_white_g":  _f((mw.gold_10k_white_g  + ss.total_estimated_carat * 0.2) if mw else None, digits=3),
        "product_gold_14k_yellow_g": _f((mw.gold_14k_yellow_g + ss.total_estimated_carat * 0.2) if mw else None, digits=3),
        "product_gold_14k_white_g":  _f((mw.gold_14k_white_g  + ss.total_estimated_carat * 0.2) if mw else None, digits=3),
        "product_gold_18k_yellow_g": _f((mw.gold_18k_yellow_g + ss.total_estimated_carat * 0.2) if mw else None, digits=3),
        "product_gold_18k_white_g":  _f((mw.gold_18k_white_g  + ss.total_estimated_carat * 0.2) if mw else None, digits=3),
        "product_platinum_g":        _f((mw.platinum_g        + ss.total_estimated_carat * 0.2) if mw else None, digits=3),

        # Stones
        "stone_status":                 ss.status,
        "center_stone_shape":           _s(cs.shape if cs else None),
        "center_stone_size_mm":         _s(cs.size_mm if cs else None),
        "center_stone_quantity":        _i(cs.quantity if cs else None),
        "center_stone_estimated_carat": _f(cs.estimated_carat_each if cs else None),
        "center_stone_total_carat":     _f(cs.estimated_total_carat if cs else None),
        "center_stone_confidence":      _f(cs.confidence if cs else None),
        "side_stones_count":            str(sum(s.quantity for s in ss.side_stones)),
        "side_stones_total_carat":      _f(sum(s.estimated_total_carat for s in ss.side_stones)),
        "side_stones_detail":           " | ".join(
            f"{s.shape}:{s.size_mm}:qty={s.quantity}:{s.estimated_total_carat:.3f}ct"
            for s in ss.side_stones
        ),
        "total_estimated_carat":        _f(ss.total_estimated_carat),

        # Sprue (single metal: same as the piece, so no per-metal columns)
        "sprue_detected":             _b(sp.detected),
        "sprue_source":               sp.source,
        "sprue_confidence":           _f(sp.confidence),
        "sprue_estimated_volume_mm3": _f(sp.estimated_volume_mm3, digits=1),
        "sprue_silver_925_g":         _f(spw["silver_925_g"] if spw else None, digits=3),
        "sprue_gold_10k_yellow_g":    _f(spw["gold_10k_yellow_g"] if spw else None, digits=3),
        "sprue_gold_10k_white_g":     _f(spw["gold_10k_white_g"] if spw else None, digits=3),
        "sprue_gold_14k_yellow_g":    _f(spw["gold_14k_yellow_g"] if spw else None, digits=3),
        "sprue_gold_14k_white_g":     _f(spw["gold_14k_white_g"] if spw else None, digits=3),
        "sprue_gold_18k_yellow_g":    _f(spw["gold_18k_yellow_g"] if spw else None, digits=3),
        "sprue_gold_18k_white_g":     _f(spw["gold_18k_white_g"] if spw else None, digits=3),
        "sprue_platinum_g":           _f(spw["platinum_g"] if spw else None, digits=3),

        # Meta
        "scale_warning":             _b(rec.scale_warning),
        "scale_warning_reason":      _s(rec.scale_warning_reason),
        "state":                     rec.state,
        "needs_manual_review":       _b(rec.needs_manual_review),
        "review_reason":             _s(rec.review_reason),
        "manual_correction_applied": _b(rec.manual_correction_applied),
        "processing_log_ref":        rec.processing_log_ref,
        "snapshot_paths":            "|".join(rec.snapshot_paths),
        "processed_at":              rec.processed_at.isoformat(),
        "schema_version":            str(rec.schema_version),
    }


class CatalogWriter:
    """
    Append-only writer for catalog_master.jsonl with periodic CSV export.

    - JSONL is append-only on every write, fsynced every _FLUSH_EVERY records.
    - CSV is rewritten in full on each flush (idempotent snapshot).
    """

    def __init__(
        self,
        jsonl_path: Path | None = None,
        csv_path: Path | None = None,
        flush_every: int = _FLUSH_EVERY,
    ) -> None:
        d = data_dir()
        self._jsonl_path = jsonl_path or (d / "catalog_master.jsonl")
        self._csv_path = csv_path or (d / "catalog_export.csv")
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._jsonl_path.open("ab")
        self._lock = threading.Lock()
        self._unflushed = 0
        self._flush_every = flush_every

    def append(self, record: CatalogRecord) -> None:
        line = orjson.dumps(record.model_dump(mode="json")) + b"\n"
        with self._lock:
            self._fh.write(line)
            self._unflushed += 1
            if self._unflushed >= self._flush_every:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            self._fh.close()

    def _flush_locked(self) -> None:
        import os
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._unflushed = 0
        self._rewrite_csv_locked()

    def _rewrite_csv_locked(self) -> None:
        records = list(read_all_records(self._jsonl_path))
        with self._csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            for rec in records:
                writer.writerow(_record_to_csv_row(rec))


def read_all_records(path: Path) -> list[CatalogRecord]:
    """Read every record from catalog_master.jsonl. Skips malformed lines."""
    if not path.exists():
        return []
    out: list[CatalogRecord] = []
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(CatalogRecord.model_validate(orjson.loads(line)))
            except Exception:
                continue
    return out
