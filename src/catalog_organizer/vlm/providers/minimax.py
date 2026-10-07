"""MiniMax Chat Completions provider with vision (MiniMax-VL-01, MiniMax-M1).

MiniMax exposes an OpenAI-compatible Chat Completions endpoint at
``https://api.minimax.io/v1/text/chatcompletion_v2`` (China) and
``https://api.minimaxi.com/v1/text/chatcompletion_v2`` (international).
Default here is the international endpoint; users in CN should swap via
Settings.

Authentication is Bearer API key. Like OpenAI, images are sent as
content parts — MiniMax accepts either an HTTP URL or a base64 data
URI under the same ``image_url`` shape. We use data URIs so no separate
upload round-trip is needed.

The response wrapping is OpenAI-compatible (``choices[0].message.content``)
so the parser pipeline is unchanged.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from catalog_organizer.vlm.cost import actual_call_cost_usd
from catalog_organizer.vlm.parser import VLMParseError, parse_vlm_response
from catalog_organizer.vlm.prompt import build_messages


class VLMHTTPError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "vlm.http_failed"


@dataclass
class MiniMaxProvider:
    """MiniMax vision client.

    `group_id` is required by some MiniMax billing flows; the v2 endpoint
    can usually deduce it from the API key, but we keep the field so a
    customer with multi-tenant access can switch projects without
    changing keys.
    """

    api_key: str
    model: str = "MiniMax-VL-01"
    base_url: str = "https://api.minimaxi.com/v1"
    timeout_s: float = 60.0
    temperature: float = 0.1
    max_output_tokens: int = 800
    group_id: str | None = None

    last_timings: dict | None = field(default=None, init=False)

    # ── VLMProvider conformance ─────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        return f"MiniMax ({self.model})"

    def estimate_cost_usd(self, n_images: int = 4) -> float:
        from catalog_organizer.vlm.cost import estimate_call_cost_usd
        return estimate_call_cost_usd("minimax", self.model, n_images=n_images)

    # ── Primary call ────────────────────────────────────────────────────────

    @property
    def chat_url(self) -> str:
        # v2 endpoint speaks the OpenAI-compatible JSON shape.
        return f"{self.base_url.rstrip('/')}/text/chatcompletion_v2"

    def classify(
        self,
        snapshot_paths: list[Path],
        second_pass: dict | None = None,
    ):
        if not self.api_key:
            raise VLMHTTPError("MiniMax API key not configured (Settings → VLM Provider).")

        ollama_messages = build_messages(snapshot_paths, second_pass=second_pass)
        mm_messages = _convert_to_minimax_messages(ollama_messages, snapshot_paths)

        image_bytes = sum(p.stat().st_size for p in snapshot_paths if p.exists())
        payload: dict = {
            "model": self.model,
            "messages": mm_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            # MiniMax accepts response_format like OpenAI; we still also rely
            # on the universal parser to handle missing JSON envelopes.
            "response_format": {"type": "json_object"},
        }
        if self.group_id:
            payload["group_id"] = self.group_id
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        attempts: list[dict] = []
        responses: list[dict] = []
        last_exc: Exception | None = None
        for attempt_idx in range(2):
            t0 = time.monotonic()
            try:
                resp = requests.post(
                    self.chat_url, headers=headers, json=payload,
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
                # MiniMax sometimes nests the OpenAI-compatible payload under
                # `data` for the legacy endpoint; v2 is flat. Be permissive.
                content = _extract_minimax_content(data)
                wall_ms = (time.monotonic() - t0) * 1000.0
                responses.append(data)
                if not content:
                    attempts.append({"i": attempt_idx, "wall_ms": wall_ms,
                                     "outcome": "empty_response"})
                    last_exc = VLMParseError("vlm.invalid_json", "Empty content")
                    continue
                attempts.append({"i": attempt_idx, "wall_ms": wall_ms,
                                 "outcome": "ok"})
                self.last_timings = _extract_timings(
                    self.model, data, image_bytes, attempts,
                )
                result = parse_vlm_response(content)
                _dump_call_log(
                    self.model, snapshot_paths, mm_messages, payload,
                    attempts, responses, image_bytes, error=None,
                )
                return result
            except requests.Timeout as exc:
                wall_ms = (time.monotonic() - t0) * 1000.0
                attempts.append({"i": attempt_idx, "wall_ms": wall_ms,
                                 "outcome": "timeout"})
                responses.append({"error": f"timeout: {exc}"})
                last_exc = exc
                continue
            except requests.ConnectionError as exc:
                wall_ms = (time.monotonic() - t0) * 1000.0
                attempts.append({"i": attempt_idx, "wall_ms": wall_ms,
                                 "outcome": "connection_error"})
                responses.append({"error": f"connection_error: {exc}"})
                last_exc = exc
                continue
            except requests.HTTPError as exc:
                wall_ms = (time.monotonic() - t0) * 1000.0
                attempts.append({"i": attempt_idx, "wall_ms": wall_ms,
                                 "outcome": "http_error"})
                body = ""
                try:
                    body = exc.response.text[:500]
                except Exception:
                    pass
                responses.append({"error": f"http_error: {exc}", "body": body})
                self.last_timings = _extract_timings(
                    self.model, {}, image_bytes, attempts,
                )
                _dump_call_log(
                    self.model, snapshot_paths, mm_messages, payload,
                    attempts, responses, image_bytes, error=str(exc),
                )
                raise VLMHTTPError(f"MiniMax HTTP error: {exc} body={body}") from exc

        self.last_timings = _extract_timings(self.model, {}, image_bytes, attempts)
        _dump_call_log(
            self.model, snapshot_paths, mm_messages, payload,
            attempts, responses, image_bytes, error=str(last_exc),
        )
        if isinstance(last_exc, VLMParseError):
            raise last_exc
        raise VLMHTTPError(f"MiniMax transport failure after retry: {last_exc}")


# ── helpers ────────────────────────────────────────────────────────────────

def _convert_to_minimax_messages(
    ollama_messages: list[dict],
    snapshot_paths: list[Path],
) -> list[dict]:
    """MiniMax's v2 schema mirrors OpenAI's content-parts. Same translation
    as the OpenAI provider, kept independent so a divergence in MiniMax's
    schema (it changes more often than OpenAI's) can be patched here
    without affecting the OpenAI path."""
    out: list[dict] = []
    for m in ollama_messages:
        if m["role"] == "system":
            out.append({"role": "system", "content": m["content"]})
            continue
        parts: list[dict] = [{"type": "text", "text": m["content"]}]
        for p in snapshot_paths:
            if not p.exists():
                continue
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        out.append({"role": m["role"], "content": parts})
    return out


def _extract_minimax_content(data: dict) -> str:
    """MiniMax response shape: {choices: [{message: {content: ...}}]}."""
    try:
        return (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
        )
    except (AttributeError, IndexError, TypeError):
        return ""


def _extract_timings(model: str, data: dict, image_bytes: int,
                     attempts: list[dict]) -> dict:
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    last_attempt = attempts[-1] if attempts else {"wall_ms": 0.0}
    cost = actual_call_cost_usd("minimax", model, prompt_tokens, completion_tokens)
    return {
        "provider":          "minimax",
        "model":             model,
        "total_ms":          float(last_attempt.get("wall_ms", 0.0)),
        "load_ms":           0.0,
        "prompt_eval_ms":    0.0,
        "eval_ms":           0.0,
        "prompt_eval_count": prompt_tokens,
        "eval_count":        completion_tokens,
        "image_bytes":       int(image_bytes),
        "attempts":          list(attempts),
        "cost_usd":          cost,
    }


def _strip_images_from_messages(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        m2 = dict(m)
        if isinstance(m2.get("content"), list):
            new_parts = []
            for part in m2["content"]:
                if part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    new_parts.append({
                        "type": "image_url",
                        "image_url": f"<base64 omitted, {len(url)} chars>",
                    })
                else:
                    new_parts.append(part)
            m2["content"] = new_parts
        out.append(m2)
    return out


def _dump_call_log(
    model: str,
    snapshot_paths: list[Path],
    messages: list[dict],
    payload: dict,
    attempts: list[dict],
    responses: list[dict],
    image_bytes: int,
    error: str | None,
) -> None:
    try:
        from catalog_organizer.core.paths import vlm_raw_dir  # noqa: PLC0415
        file_id = snapshot_paths[0].parent.name if snapshot_paths else "unknown"
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        out_path = vlm_raw_dir() / f"{ts}_{file_id}_minimax.json"
        log_payload = dict(payload)
        log_payload["messages"] = _strip_images_from_messages(messages)
        record = {
            "provider":      "minimax",
            "timestamp_utc": ts,
            "file_id":       file_id,
            "snapshot_paths": [str(p) for p in snapshot_paths],
            "image_bytes":   image_bytes,
            "model":         model,
            "messages":      log_payload["messages"],
            "options":       {k: v for k, v in payload.items()
                              if k not in ("messages",)},
            "attempts":      attempts,
            "responses":     responses,
            "error":         error,
        }
        out_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass
