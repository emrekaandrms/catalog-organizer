from __future__ import annotations

from catalog_organizer.weight.metals import compute_metal_weights
from catalog_organizer.weight.stones import StoneWeightTable

_DENSITIES = {
    "silver_925":      {"density_g_cm3": 10.36},
    "gold_10k_yellow": {"density_g_cm3": 11.57},
    "gold_10k_white":  {"density_g_cm3": 11.01},
    "gold_14k_yellow": {"density_g_cm3": 13.07},
    "gold_14k_white":  {"density_g_cm3": 12.56},
    "gold_18k_yellow": {"density_g_cm3": 15.58},
    "gold_18k_white":  {"density_g_cm3": 14.66},
    "platinum_950":    {"density_g_cm3": 20.13},
}


def test_metal_weights_at_1_cm3():
    # 1000 mm³ = 1 cm³ → mass equals density in grams.
    w = compute_metal_weights(1000.0, _DENSITIES)
    assert abs(w.silver_925_g - 10.36) < 1e-6
    assert abs(w.gold_18k_yellow_g - 15.58) < 1e-6
    assert abs(w.gold_18k_white_g - 14.66) < 1e-6
    assert abs(w.platinum_g - 20.13) < 1e-6


def test_metal_weights_yellow_vs_white_diverge():
    """Yellow and white gold of the same karat must yield different masses
    (verified against the user's workshop calibration, 2026-05-19)."""
    w = compute_metal_weights(1000.0, _DENSITIES)
    # White 18k is ~6% lighter than yellow 18k at the same volume
    assert w.gold_18k_white_g < w.gold_18k_yellow_g
    assert abs((w.gold_18k_yellow_g - w.gold_18k_white_g) / w.gold_18k_yellow_g - 0.059) < 0.005


def test_metal_weights_zero_volume():
    w = compute_metal_weights(0.0, _DENSITIES)
    assert w.silver_925_g == 0.0
    assert w.gold_10k_yellow_g == 0.0
    assert w.gold_10k_white_g == 0.0
    assert w.platinum_g == 0.0


def test_metal_weights_legacy_jsonl_migrates_gold_keys():
    """Old records written before the yellow/white split (pre-2026-05-19)
    used a single `gold_10k_g` key. The model_validator must remap those
    to the yellow variant on load so the historical catalog still parses."""
    from catalog_organizer.core.schemas import MetalWeights
    legacy = {
        "silver_925_g": 4.41,
        "gold_10k_g":   4.67,
        "gold_14k_g":   5.32,
        "gold_18k_g":   6.62,
        "platinum_g":   8.5,
    }
    mw = MetalWeights.model_validate(legacy)
    assert mw.silver_925_g == 4.41
    assert mw.gold_10k_yellow_g == 4.67
    assert mw.gold_14k_yellow_g == 5.32
    assert mw.gold_18k_yellow_g == 6.62
    # White columns absent in legacy → default 0.0
    assert mw.gold_10k_white_g == 0.0
    assert mw.gold_18k_white_g == 0.0


def test_cz_round_table_known_values():
    """CZ table is the workshop default. Values come directly from
    cubic_zirconia_cz_weight_table_CORRECTED_micro_values.xlsx — pinning a
    few endpoints catches accidental table swaps."""
    table = StoneWeightTable()   # default = CZ
    # 5 mm CZ round = 0.79 ct (vs 0.50 for diamond)
    assert abs(table.round_carat(5.0) - 0.79) < 1e-3
    # 2 mm CZ round = 0.06 ct
    assert abs(table.round_carat(2.0) - 0.06) < 1e-3
    # 1.25 mm CZ round = 0.013 ct (anchor row)
    assert abs(table.round_carat(1.25) - 0.013) < 1e-4


def test_cz_estimate_returns_carat_and_grams():
    """`estimate` returns (carat, grams). Grams comes straight from the
    table when present; otherwise it's carat × 0.2 (standard conversion)."""
    table = StoneWeightTable()
    ct, g = table.estimate("round", 5.0)
    # 1 ct = 0.2 g
    assert abs(g - ct * 0.2) < 1e-3
    # Sanity: at 5 mm, CZ row says 0.158 g per stone
    assert abs(g - 0.158) < 0.005


def test_cz_non_round_uses_pair_table():
    """Non-round CZ shapes look up the LONGxSHORT row directly from the
    per-shape CSV. 5x3 oval = 0.35 ct in the table."""
    table = StoneWeightTable()
    ct, g = table.estimate("oval", (5.0, 3.0))
    # The table's nearest row should be ~0.35 ct
    assert 0.2 < ct < 0.5, f"oval 5x3 ct={ct}"


def test_diamond_fallback_still_works():
    """material='diamond' must still drive the legacy round.csv + shape
    factors path — kept for any consumer that needs Diamond carat."""
    table = StoneWeightTable(material="diamond")
    # 5 mm diamond round = 0.500 carat (legacy table)
    assert abs(table.round_carat(5.0) - 0.500) < 1e-6
    # Non-round shape factor path: oval 6×4 mm
    table = StoneWeightTable(
        material="diamond",
        shape_factors={"oval": 1.05},
    )
    ct, _ = table.estimate("oval", (6.0, 4.0))
    expected = 0.250 * 1.05 * 1.5   # round_carat(4)=0.250
    assert abs(ct - expected) < 1e-6
