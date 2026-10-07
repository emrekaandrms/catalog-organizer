"""OpenAI-compatible Chat Completions provider with vision.

Despite the module name this class backs EVERY OpenAI-protocol vision
endpoint the app supports — OpenAI itself (gpt-4o), GLM/Zhipu, and the
generic "OpenAI-compatible" option users point at OpenRouter, DeepSeek,
Together, a self-hosted vLLM, or anything else speaking the same wire
format. Only `base_url`, credentials, `provider_name` and
`display_vendor` differ; see `vlm/providers/__init__.py` for the
concrete wiring.


Uses the public ``POST /v1/chat/completions`` endpoint with Bearer-token
auth. Images are sent as ``image_url`` content parts with base64 data
URIs — no separate upload step. The OpenAI response is parsed with the
same `parse_vlm_response` used for Ollama, including the `<think>` /
fenced-JSON tolerance, so prompt/parsing improvements stay shared.

Per-call audit JSONs land in `cache/vlm_raw/` alongside Ollama dumps
with `provider=openai` so post-hoc analysis can filter by source.
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
    """Same exception type the Ollama client raises — pipeline catches one."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "vlm.http_failed"


@dataclass
class OpenAIProvider:
    """Vision-capable Chat Completions client for the OpenAI API.

    Authentication is API-key only (no organisation header by default —
    pass `org_id` if your account requires one). The endpoint is hardened
    against the same failure modes as the Ollama client: connection
    error, timeout, HTTP error, empty content.
    """

    api_key: str
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    timeout_s: float = 60.0
    temperature: float = 0.1
    max_output_tokens: int = 800        # cap so a runaway response can't bill
    org_id: str | None = None           # OpenAI org header, optional

    # Which provider key this instance represents. The wire protocol is
    # identical for every OpenAI-compatible endpoint (GLM/Zhipu,
    # OpenRouter, DeepSeek, Together, a local vLLM, …), so they all reuse
    # this class — only the base_url, credentials and this label differ.
    # It drives the pricing lookup, `cache/vlm_raw/` dump filenames and
    # the `provider` field inside them, so cost reporting and post-hoc
    # analysis stay correctly attributed per backend.
    provider_name: str = "openai"
    display_vendor: str = "OpenAI"

    # `response_format={"type":"json_object"}` is an OpenAI extension.
    # Most compatible gateways support it, but some reject the whole
    # request with a 400 when they don't. The response parser already
    # tolerates prose, code fences and <think> blocks, so JSON mode is an
    # optimisation rather than a requirement — turn it off for endpoints
    # that choke on it.
    use_json_mode: bool = True

    last_timings: dict | None = field(default=None, init=False)

    # ── VLMProvider conformance ─────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        return f"{self.display_vendor} ({self.model})"

    def estimate_cost_usd(self, n_images: int = 4) -> float:
        from catalog_organizer.vlm.cost import estimate_call_cost_usd
        return estimate_call_cost_usd(self.provider_name, self.model, n_images=n_images)

    # ── Primary call ────────────────────────────────────────────────────────

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def classify(
        self,
        snapshot_paths: list[Path],
        second_pass: dict | None = None,
    ):
        if not self.api_key:
            raise VLMHTTPError(
                f"{self.display_vendor} API key not configured "
                "(Settings → VLM Provider)."
            )

        # Reuse the universal message builder — same system / user / taxonomy
        # / tag dictionary text that drives Ollama, repackaged below.
        ollama_messages = build_messages(snapshot_paths, second_pass=second_pass)
        openai_messages = _convert_to_openai_messages(ollama_messages, snapshot_paths)

        image_bytes = sum(p.stat().st_size for p in snapshot_paths if p.exists())
        payload = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.org_id:
            headers["OpenAI-Organization"] = self.org_id

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
                content = (
                    data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                )
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
                    self.provider_name, self.model, data, image_bytes, attempts,
                )
                result = parse_vlm_response(content)
                _dump_call_log(
                    self.provider_name, self.model, snapshot_paths,
                    openai_messages, payload,
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
                    self.provider_name, self.model, {}, image_bytes, attempts,
                )
                _dump_call_log(
                    self.provider_name, self.model, snapshot_paths,
                    openai_messages, payload,
                    attempts, responses, image_bytes, error=str(exc),
                )
                raise VLMHTTPError(
                    f"{self.display_vendor} HTTP error: {exc} body={body}"
                ) from exc

        self.last_timings = _extract_timings(
            self.provider_name, self.model, {}, image_bytes, attempts,
        )
        _dump_call_log(
            self.provider_name, self.model, snapshot_paths,
            openai_messages, payload,
            attempts, responses, image_bytes, error=str(last_exc),
        )
        if isinstance(last_exc, VLMParseError):
            raise last_exc
        raise VLMHTTPError(
            f"{self.display_vendor} transport failure after retry: {last_exc}"
        )


# ── helpers ────────────────────────────────────────────────────────────────

def _convert_to_openai_messages(
    ollama_messages: list[dict],
    snapshot_paths: list[Path],
) -> list[dict]:
    """Translate the Ollama-style messages produced by `build_messages`
    into the OpenAI content-parts shape. Ollama puts images as a flat
    `images: [b64, ...]`; OpenAI expects each image as a typed part:
    ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``.
    """
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
                "image_url": {"url": f"data:image/png;base64,{b64}",
                              "detail": "high"},
            })
        out.append({"role": m["role"], "content": parts})
    return out


def _extract_timings(provider: str, model: str, data: dict, image_bytes: int,
                     attempts: list[dict]) -> dict:
    """Mirror Ollama's `last_timings` shape so the rest of the app
    (Diagnostics breakdown, status bar cost widget) stays provider-
    agnostic. Per-phase durations don't exist on OpenAI; we surface the
    wall time of the successful attempt as `total_ms` and the token
    counts from `usage`."""
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    last_attempt = attempts[-1] if attempts else {"wall_ms": 0.0}
    cost = actual_call_cost_usd(provider, model, prompt_tokens, completion_tokens)
    return {
        "provider":          provider,
        "model":             model,
        "total_ms":          float(last_attempt.get("wall_ms", 0.0)),
        "load_ms":           0.0,    # not applicable
        "prompt_eval_ms":    0.0,    # OpenAI doesn't expose phase split
        "eval_ms":           0.0,
        "prompt_eval_count": prompt_tokens,
        "eval_count":        completion_tokens,
        "image_bytes":       int(image_bytes),
        "attempts":          list(attempts),
        "cost_usd":          cost,
    }


def _strip_images_from_messages(messages: list[dict]) -> list[dict]:
    """Replace base64 image data with byte counts before dumping to disk."""
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
    provider: str,
    model: str,
    snapshot_paths: list[Path],
    messages: list[dict],
    payload: dict,
    attempts: list[dict],
    responses: list[dict],
    image_bytes: int,
    error: str | None,
) -> None:
    """Write the same audit JSON the Ollama client writes, tagged with the
    concrete backend (`openai`, `glm`, `openai_compatible`, …) so the dump
    folder stays filterable per provider."""
    try:
        from catalog_organizer.core.paths import vlm_raw_dir  # noqa: PLC0415

        file_id = (
            snapshot_paths[0].parent.name
            if snapshot_paths else "unknown"
        )
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        out_path = vlm_raw_dir() / f"{ts}_{file_id}_{provider}.json"

        log_payload = dict(payload)
        log_payload["messages"] = _strip_images_from_messages(messages)

        record = {
            "provider":      provider,
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
