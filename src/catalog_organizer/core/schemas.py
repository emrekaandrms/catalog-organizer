from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator


class ManifestEntry(BaseModel):
    file_id: str
    source_path: str
    file_extension: Literal[".3dm", ".stl"]
    file_size_bytes: int
    created_at: datetime | None
    modified_at: datetime | None
    sha256: str
    duplicate_of: str | None
    state: Literal["new", "missing"]
    processing_batch: str
    scan_timestamp: datetime


class Measurements(BaseModel):
    bbox_width_mm: float
    bbox_height_mm: float
    bbox_depth_mm: float
    volume_mm3: float
    geometry_source: Literal["rhino", "trimesh", "matrixgold", "unknown"]

    ring_inner_diameter_mm: float | None = None
    ring_inner_circumference_mm: float | None = None
    ring_size_eu: float | None = None
    ring_band_width_mm: float | None = None
    ring_top_width_mm: float | None = None
    ring_top_height_mm: float | None = None

    earring_total_height_mm: float | None = None
    earring_total_width_mm: float | None = None
    earring_depth_mm: float | None = None
    earring_pin_detected: bool | None = None
    earring_pair_or_single: Literal["pair", "single", "unclear"] | None = None
    hoop_outer_diameter_mm: float | None = None
    hoop_inner_diameter_mm: float | None = None

    bracelet_inner_diameter_x_mm: float | None = None
    bracelet_inner_diameter_y_mm: float | None = None
    bracelet_inner_circumference_mm: float | None = None
    bracelet_opening_gap_mm: float | None = None
    bracelet_width_mm: float | None = None

    pendant_height_mm: float | None = None
    pendant_width_mm: float | None = None
    pendant_depth_mm: float | None = None
    bail_detected: bool | None = None
    bail_inner_width_mm: float | None = None
    bail_inner_height_mm: float | None = None


class MetalWeights(BaseModel):
    """Mass in grams for each metal/alloy variant we track. Yellow vs white
    gold are split because the alloying base differs (Cu/Ag for yellow,
    Pd/Ni for white) and the resulting density gap is ~5-6%. Reporting one
    value for "10k gold" would mis-report by ~0.5g on a small ring.

    Default = 0.0 keeps old JSONL records (which only had the unified
    gold_10k_g / gold_14k_g / gold_18k_g fields) loadable — those rows
    will have the yellow columns populated by the migration validator and
    white columns left as 0.0 until reprocessed.
    """
    silver_925_g: float = 0.0
    gold_10k_yellow_g: float = 0.0
    gold_10k_white_g: float = 0.0
    gold_14k_yellow_g: float = 0.0
    gold_14k_white_g: float = 0.0
    gold_18k_yellow_g: float = 0.0
    gold_18k_white_g: float = 0.0
    platinum_g: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_gold_keys(cls, data):
        """Map the pre-split keys (gold_10k_g, gold_14k_g, gold_18k_g) to
        the yellow variant. Original defaults were yellow gold densities so
        this is the correct semantic equivalent for legacy records."""
        if isinstance(data, dict):
            if "gold_10k_g" in data and "gold_10k_yellow_g" not in data:
                data["gold_10k_yellow_g"] = data.pop("gold_10k_g")
            if "gold_14k_g" in data and "gold_14k_yellow_g" not in data:
                data["gold_14k_yellow_g"] = data.pop("gold_14k_g")
            if "gold_18k_g" in data and "gold_18k_yellow_g" not in data:
                data["gold_18k_yellow_g"] = data.pop("gold_18k_g")
        return data


class StoneEntry(BaseModel):
    shape: Literal["round", "oval", "baguette", "princess",
                   "pear", "marquise", "emerald", "unknown"]
    size_mm: str
    quantity: int
    estimated_carat_each: float
    estimated_total_carat: float
    source: Literal["matrixgold", "geometry", "vlm", "unknown"]
    confidence: float


class StoneSummary(BaseModel):
    status: Literal["stone", "polished", "unclear"]
    center_stone: StoneEntry | None
    side_stones: list[StoneEntry]
    total_estimated_carat: float


class SpruInfo(BaseModel):
    detected: bool
    source: Literal["vlm", "geometry", "vlm_geometry_combined", "none"]
    confidence: float
    estimated_volume_mm3: float | None
    # Casting-tree rails only: volume of the fused feeder joints where the
    # pieces overlap the runner bars. The cut happens at the piece surface,
    # so this sliver leaves with the sprue scrap — the weight pipeline
    # subtracts (estimated_volume_mm3 + cut_overlap_mm3) from the metal,
    # but the *reported* sprue weight uses estimated_volume_mm3 alone
    # (matches the workshop's scrap scale within ±2 %).
    cut_overlap_mm3: float = 0.0
    # NOTE: no per-metal weight field. The sprue is always the same metal as
    # the piece it gates (it's part of the same casting pour). The CSV
    # exports a single `sprue_mass_g` computed against whichever metal the
    # user is interested in — see `compute_sprue_mass_g` in pipeline.


class CatalogRecord(BaseModel):
    file_id: str
    source_path: str
    file_extension: Literal[".3dm", ".stl"]
    sha256: str

    main_category: str
    main_category_confidence: float
    subcategory: str
    subcategory_confidence: float
    controlled_tags: list[str]
    tag_confidence: float
    polished_status: Literal["stone", "polished", "unclear"]

    # Brand + rich free-text description (the catalogue-search design notes, 2026-07-23).
    # Defaults keep pre-existing JSONL rows loadable without reprocessing —
    # brand is "unknown" confidence 0.0 and description empty until a file
    # is (re)run through the enriched VLM prompt.
    brand: str | None = None
    brand_confidence: float = 0.0
    rich_description: str = ""

    # Design-consistency + sale-readiness (the scale/sales design notes §1/§2,
    # 2026-07-24). design_complete/confidence/notes come straight from the
    # VLM (same single call). sellability is NOT a VLM opinion — it's
    # computed in pipeline.py from design_complete alone; brand deliberately
    # does not factor in (this is a professional design workshop drawing
    # commissioned/brand-referenced pieces as a paid drafting service, not
    # a counterfeit-retail scenario, so brand is catalog/search metadata
    # only, never a sellability gate — corrected 2026-07-24 after an
    # earlier draft of the plan wrongly proposed the opposite).
    design_complete: bool = True
    design_complete_confidence: float = 0.0
    incompleteness_notes: str = ""

    sellability: Literal["sellable", "not_sellable", "needs_review"] = "needs_review"
    sellability_reason: str | None = None
    sales_channel: Literal["stl_digital", "physical_cast_render", "undecided"] = "undecided"

    measurements: Measurements
    metal_weights: MetalWeights | None
    weight_source: Literal["matrixgold_report", "rhino_volume_density",
                           "stl_volume_density", "unavailable"]
    weight_confidence: Literal["high", "medium", "low", "unavailable"]

    stone_summary: StoneSummary
    sprue: SpruInfo

    scale_warning: bool
    scale_warning_reason: str | None

    snapshot_paths: list[str]

    state: Literal["snapshot_done", "vlm_done", "measured", "weighted",
                   "final", "needs_review", "failed"]
    needs_manual_review: bool
    review_reason: str | None
    manual_correction_applied: bool
    processing_log_ref: str
    processed_at: datetime
    schema_version: Literal[1] = 1

    @field_validator("controlled_tags")
    @classmethod
    def at_least_five_tags(cls, v: list[str]) -> list[str]:
        if len(v) < 5:
            raise ValueError(f"controlled_tags must have >= 5 entries, got {len(v)}")
        return v


class AuditEvent(BaseModel):
    event_id: str
    timestamp: datetime
    file_id: str | None
    phase: Literal["scan", "snapshot", "vlm", "measure",
                   "weight", "stone", "sprue", "review", "move"]
    level: Literal["info", "warn", "error"]
    code: str
    message: str
    extra: dict[str, Any] = {}


class Correction(BaseModel):
    correction_id: str
    file_id: str
    timestamp: datetime
    user: str = "local"
    before: dict[str, Any]
    after: dict[str, Any]
    note: str | None


class VLMResult(BaseModel):
    """Validated VLM response. Matches the JSON shape in vlm_prompt_templates.yaml.

    Non-core fields are optional with safe defaults because real-world Ollama
    responses occasionally truncate the JSON object and omit them entirely.
    `main_category` and `controlled_tags` are required — we refuse to proceed
    without at least a label. Whenever a field falls back to its default we
    set `needs_manual_review=True` via the model-level validator so a human
    eye flags the patched record.
    """
    main_category: str
    main_category_confidence: float = 0.0
    subcategory: str = "other"
    subcategory_confidence: float = 0.0
    controlled_tags: list[str]
    tag_confidence: float = 0.0
    stone_presence: Literal["stone", "no_stone", "unclear"] = "unclear"
    estimated_visible_stone_count: int = 0
    polished_status: Literal["stone", "polished", "unclear"] = "unclear"
    visible_sprue_or_casting_stem: bool = False
    sprue_confidence: float = 0.0
    brand_guess: str = "unknown"
    brand_confidence: float = 0.0
    rich_description: str = ""

    # Design-consistency check (the scale/sales design notes §1, 2026-07-24):
    # "tamamlanmış" means internally consistent, not print/production-ready
    # (that concept doesn't apply to .3dm). Concrete examples from the user:
    # a sprue/casting stub sitting alone (not a product), prongs present but
    # no stone seat ever cut behind them, a decorative part not actually
    # seated against the body it should attach to (visible gap).
    design_complete: bool = True
    design_complete_confidence: float = 0.0
    incompleteness_notes: str = ""

    uncertainty_notes: list[str] = []
    needs_second_pass: bool = False
    needs_manual_review: bool = False

    @field_validator("polished_status", mode="before")
    @classmethod
    def _normalize_polished_status_alias(cls, v):
        """The VLM sometimes emits "polished_model" — a controlled_tags
        production tag (see tag_dictionary.yaml) — in this field instead of
        the enum value "polished"; the two concepts are semantically close
        and the model conflates them. Observed on ~9% of a real 65-file
        pilot batch (2026-07-24, PilotBatch folder), each occurrence
        raising vlm.schema_violation and discarding the ENTIRE otherwise-
        valid response (category/tags/brand/description). Normalize the
        known alias instead of hard-failing on it.
        """
        if v == "polished_model":
            return "polished"
        return v

    @model_validator(mode="after")
    def _flag_low_confidence(self) -> "VLMResult":
        """If any core confidence is at its 0.0 default, the model probably
        didn't emit it — surface for human review."""
        if (self.main_category_confidence == 0.0 or
                self.tag_confidence == 0.0 or
                self.subcategory_confidence == 0.0 or
                self.design_complete_confidence == 0.0):
            self.needs_manual_review = True
        return self
