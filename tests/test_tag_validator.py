from __future__ import annotations

from catalog_organizer.catalog.tag_validator import TagValidator

# Minimal dictionary + fallback table to make the tests deterministic
# and independent of the project's tag_dictionary.yaml content.
_DICT = [
    "ring", "band", "closed_form", "polished", "round_form", "geometric",
    "earring", "pair", "drop_structure",
    "single_piece", "stone", "center_stone", "unisex",
]
_FB = {
    "ring":     ["ring", "band", "closed_form", "polished", "round_form"],
    "earring":  ["earring", "pair", "drop_structure", "closed_form", "polished"],
    "unknown":  ["single_piece", "closed_form", "polished", "geometric", "unisex"],
}


def make_validator() -> TagValidator:
    return TagValidator(dictionary=_DICT, fallbacks=_FB)


def test_keeps_valid_tags_at_or_above_threshold():
    v = make_validator()
    out = v.validate(
        ["ring", "band", "polished", "round_form", "closed_form"],
        main_category="ring",
    )
    assert out == ["ring", "band", "polished", "round_form", "closed_form"]


def test_strips_invalid_tags_and_pads_from_fallback():
    v = make_validator()
    out = v.validate(
        ["ring", "totally_made_up_tag", "polished", "another_fake"],
        main_category="ring",
    )
    assert out[:2] == ["ring", "polished"]
    assert len(out) == 5
    assert all(t in _DICT for t in out)


def test_empty_tags_filled_entirely_from_fallback():
    v = make_validator()
    out = v.validate([], main_category="ring")
    assert out == ["ring", "band", "closed_form", "polished", "round_form"]


def test_stone_replaces_last_fallback_when_count_gt_1():
    v = make_validator()
    out = v.validate(
        [],
        main_category="ring",
        stone_presence="stone",
        estimated_visible_stone_count=5,
    )
    assert out[-1] == "stone"
    assert len(out) == 5


def test_center_stone_replaces_last_fallback_when_count_eq_1():
    v = make_validator()
    out = v.validate(
        [],
        main_category="ring",
        stone_presence="stone",
        estimated_visible_stone_count=1,
    )
    assert out[-1] == "center_stone"
    assert len(out) == 5
