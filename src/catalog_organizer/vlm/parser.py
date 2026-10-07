from __future__ import annotations

import json
import re

from pydantic import ValidationError

from catalog_organizer.core.schemas import VLMResult


class VLMParseError(ValueError):
    def __init__(self, code: str, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.code = code      # "vlm.invalid_json" or "vlm.schema_violation"
        self.raw = raw


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
# qwen3.5 / qwen3-vl ship with `thinking` capability — when not suppressed by
# the API `think:false` flag, they emit a <think>...</think> block before the
# answer. The block can be huge; we strip it before searching for JSON.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> str:
    """Strip thinking blocks + code fences and return the largest {...} substring."""
    text = _THINK_BLOCK_RE.sub("", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise VLMParseError("vlm.invalid_json", "No JSON object found in response", raw=text)
    return match.group(0)


def parse_vlm_response(raw_text: str) -> VLMResult:
    """Parse VLM raw text → VLMResult. Raises VLMParseError with code on failure."""
    json_str = _extract_json_object(raw_text)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise VLMParseError("vlm.invalid_json", f"JSON decode failed: {exc}", raw=raw_text) from exc

    try:
        return VLMResult.model_validate(data)
    except ValidationError as exc:
        raise VLMParseError("vlm.schema_violation", str(exc), raw=raw_text) from exc
