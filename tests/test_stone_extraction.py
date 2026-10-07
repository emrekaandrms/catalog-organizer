"""Unit tests for stone extraction helpers.

The end-to-end extractor (`extract_stone_summary_from_3dm`) is covered
via the manual smoke-test on real samples/*.3dm samples — they're
gigabytes of MatrixGold output and shouldn't go into the test fixtures.
This module pins the small, pure helpers that drive the inference.
"""
from __future__ import annotations

from catalog_organizer.cad.stone_extraction import (
    _is_gem_layer,
    _shape_from_block_name,
    _shape_from_bbox,
)


def test_is_gem_layer_matches_matrixgold_names():
    """Only gem/stone layers should be tagged; cutters / curves / lights
    are non-metal but NOT gems and must not be picked up here."""
    for name in ("Gem", "Gem 01", "Gem 02", "Gem 03", "Gem 04",
                 "stone seats", "GEM PLACEHOLDER"):
        assert _is_gem_layer(name), f"expected gem: {name!r}"
    for name in ("Metal 01", "Cutting Objects", "Heads",
                 "Creation Curves", "Lights", "User Layer 01"):
        assert not _is_gem_layer(name), f"not a gem layer: {name!r}"


def test_shape_from_block_name_reads_matrixgold_library_names():
    """MatrixGold ships with named diamond blocks ('Diamond_Round',
    'Diamond_Oval 01' …). Reading shape from the block name is far more
    reliable than guessing from bbox aspect."""
    assert _shape_from_block_name("Diamond_Round") == "round"
    assert _shape_from_block_name("Diamond_Round 01 03") == "round"
    assert _shape_from_block_name("Diamond_Oval 5x3") == "oval"
    assert _shape_from_block_name("Diamond_Marquise") == "marquise"
    assert _shape_from_block_name("Diamond_Baguette") == "baguette"
    assert _shape_from_block_name("Diamond_Princess") == "princess"
    assert _shape_from_block_name("Diamond_Pear") == "pear"
    assert _shape_from_block_name("Diamond_Emerald") == "emerald"

    # Custom / unknown name → no shape inference
    assert _shape_from_block_name("MyCustomGem") is None
    assert _shape_from_block_name(None) is None
    assert _shape_from_block_name("") is None


def test_shape_from_bbox_aspect_ratio_fallback():
    """When block name is missing we fall back to bbox aspect.
    The ratio thresholds aren't precise but they distinguish the four
    coarse buckets (round / oval / marquise / baguette)."""
    # Round-ish: 4.0 × 4.0 × 2.4
    assert _shape_from_bbox((4.0, 4.0, 2.4)) == "round"
    # Oval: 5 × 3 — aspect 1.67 → falls into oval bucket
    assert _shape_from_bbox((5.0, 3.0, 1.8)) == "oval"
    # Marquise: 8 × 3 — aspect 2.67
    assert _shape_from_bbox((8.0, 3.0, 1.8)) == "marquise"
    # Baguette: 6 × 1.5 — aspect 4.0
    assert _shape_from_bbox((6.0, 1.5, 1.0)) == "baguette"


def test_shape_from_bbox_handles_zero_extent():
    """A degenerate flat polygon (one extent = 0) must not divide-by-zero."""
    assert _shape_from_bbox((5.0, 0.0, 0.0)) == "unknown"
