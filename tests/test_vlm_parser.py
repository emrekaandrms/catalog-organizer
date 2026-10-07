from __future__ import annotations

import pytest

from catalog_organizer.vlm.parser import VLMParseError, parse_vlm_response
from catalog_organizer.vlm.prompt import build_messages

_GOOD = """{
  "main_category": "ring",
  "main_category_confidence": 0.93,
  "subcategory": "solitaire",
  "subcategory_confidence": 0.88,
  "controlled_tags": ["ring", "band", "polished", "round_form", "single_piece"],
  "tag_confidence": 0.85,
  "stone_presence": "stone",
  "estimated_visible_stone_count": 1,
  "polished_status": "polished",
  "visible_sprue_or_casting_stem": false,
  "sprue_confidence": 0.92,
  "uncertainty_notes": [],
  "needs_second_pass": false,
  "needs_manual_review": false
}"""

_GOOD_WITH_FENCE = "```json\n" + _GOOD + "\n```"

_GOOD_WITH_PROSE = "Here is the classification:\n" + _GOOD + "\nEnd."

_BAD_JSON = "{ this is not: valid json"

_BAD_SCHEMA = """{
  "main_category": "ring",
  "main_category_confidence": 0.93,
  "subcategory": "solitaire",
  "subcategory_confidence": 0.88,
  "controlled_tags": [],
  "tag_confidence": 0.85,
  "stone_presence": "BOGUS_VALUE",
  "estimated_visible_stone_count": 1,
  "polished_status": "polished",
  "visible_sprue_or_casting_stem": false,
  "sprue_confidence": 0.92
}"""


def test_parses_clean_json():
    result = parse_vlm_response(_GOOD)
    assert result.main_category == "ring"
    assert result.subcategory == "solitaire"
    assert result.controlled_tags == ["ring", "band", "polished", "round_form", "single_piece"]


def test_parses_with_code_fence():
    result = parse_vlm_response(_GOOD_WITH_FENCE)
    assert result.main_category == "ring"


def test_parses_with_surrounding_prose():
    result = parse_vlm_response(_GOOD_WITH_PROSE)
    assert result.main_category == "ring"


def test_invalid_json_raises():
    with pytest.raises(VLMParseError) as exc_info:
        parse_vlm_response(_BAD_JSON)
    assert exc_info.value.code == "vlm.invalid_json"


def test_schema_violation_raises():
    with pytest.raises(VLMParseError) as exc_info:
        parse_vlm_response(_BAD_SCHEMA)
    assert exc_info.value.code == "vlm.schema_violation"


_PARTIAL = """{
  "main_category": "ring",
  "subcategory": "band",
  "controlled_tags": ["ring", "band", "polished", "round_form", "single_piece"],
  "needs_manual_review": false
}"""


def test_partial_response_uses_defaults_and_flags_for_review():
    """Regression: real-world Ollama runs return JSON with 8 required fields
    omitted ~13% of the time. The parser must accept this and route to review.
    """
    result = parse_vlm_response(_PARTIAL)
    assert result.main_category == "ring"
    assert result.subcategory == "band"
    # Confidences defaulted to 0.0
    assert result.main_category_confidence == 0.0
    assert result.tag_confidence == 0.0
    # Defaults for stone / polished / sprue
    assert result.stone_presence == "unclear"
    assert result.polished_status == "unclear"
    assert result.visible_sprue_or_casting_stem is False
    # The validator must overrule the model's "no review needed" claim
    # because we patched in defaults.
    assert result.needs_manual_review is True


def test_missing_main_category_still_raises():
    """We refuse to proceed without at least a label — only non-core fields default."""
    bad = '{"controlled_tags": ["ring"]}'
    with pytest.raises(VLMParseError) as exc_info:
        parse_vlm_response(bad)
    assert exc_info.value.code == "vlm.schema_violation"


def test_parses_response_with_thinking_block():
    """qwen3.5 / qwen3-vl emit <think>...</think> before the JSON when the
    server doesn't honour think:false. The parser must strip the block
    rather than choke on the curly braces inside."""
    raw = (
        "<think>\n"
        "Let me analyse the four views. The object looks circular with a band.\n"
        "It has {weird brace} inside the thinking that would confuse a naive regex.\n"
        "I'll classify as ring.\n"
        "</think>\n\n"
        + _GOOD
    )
    result = parse_vlm_response(raw)
    assert result.main_category == "ring"
    assert result.subcategory == "solitaire"


def test_build_messages_does_not_treat_template_json_as_format_placeholders(tmp_path):
    """Regression: the user_template embeds a literal JSON shape `{ "main_category": ... }`.
    Python's `str.format()` would parse those braces and raise KeyError on the inner key
    names. build_messages must use plain replacement instead.
    """
    # A 1-byte file is enough; build_messages only base64-encodes paths.
    fake_png = tmp_path / "front.png"
    fake_png.write_bytes(b"\x89PNG")
    msgs = build_messages([fake_png])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    # The literal JSON shape from the template must survive into the user message.
    user_text = msgs[1]["content"]
    assert '"main_category"' in user_text
    assert '"controlled_tags"' in user_text
