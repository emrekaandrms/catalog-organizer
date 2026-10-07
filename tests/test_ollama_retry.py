"""Empty-response retry test for OllamaClient (regression for D.13c).

Real-world Ollama occasionally returns `{"message": {"content": ""}}` as a
transient hiccup. The previous code raised vlm.invalid_json on the first try.
The retry loop now treats an empty body the same way it treats a connection
error: retry once before bubbling up.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from catalog_organizer.vlm.ollama_client import OllamaClient
from catalog_organizer.vlm.parser import VLMParseError


_GOOD_BODY = """{
  "main_category": "ring",
  "main_category_confidence": 0.91,
  "subcategory": "band",
  "subcategory_confidence": 0.85,
  "controlled_tags": ["ring", "band", "polished", "round_form", "single_piece"],
  "tag_confidence": 0.88,
  "stone_presence": "no_stone",
  "estimated_visible_stone_count": 0,
  "polished_status": "polished",
  "visible_sprue_or_casting_stem": false,
  "sprue_confidence": 0.9,
  "uncertainty_notes": [],
  "needs_second_pass": false,
  "needs_manual_review": false
}"""


def _mock_response(content: str):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json = MagicMock(return_value={"message": {"content": content}})
    return r


def test_empty_response_retries_once_and_succeeds(tmp_path):
    """First call returns empty content, second returns a valid response."""
    fake_png = tmp_path / "f.png"
    fake_png.write_bytes(b"\x89PNG")

    client = OllamaClient(model="qwen3.5:9b-q4_K_M", timeout_s=5)
    with patch(
        "catalog_organizer.vlm.ollama_client.requests.post",
        side_effect=[_mock_response(""), _mock_response(_GOOD_BODY)],
    ) as mock_post:
        result = client.classify([fake_png])
    assert mock_post.call_count == 2
    assert result.main_category == "ring"


def test_two_empty_responses_raise_parse_error(tmp_path):
    """Both attempts return empty → bubble up VLMParseError (vlm.invalid_json)."""
    fake_png = tmp_path / "f.png"
    fake_png.write_bytes(b"\x89PNG")

    client = OllamaClient(model="qwen3.5:9b-q4_K_M", timeout_s=5)
    with patch(
        "catalog_organizer.vlm.ollama_client.requests.post",
        side_effect=[_mock_response(""), _mock_response("")],
    ):
        with pytest.raises(VLMParseError) as exc_info:
            client.classify([fake_png])
    assert exc_info.value.code == "vlm.invalid_json"
