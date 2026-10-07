from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from catalog_organizer.vlm.parser import VLMParseError, parse_vlm_response
from catalog_organizer.vlm.prompt import build_messages


class VLMHTTPError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "vlm.http_failed"


@dataclass
class OllamaClient:
    host: str = "127.0.0.1"
    port: int = 11434
    model: str = "qwen3.5:9b-q4_K_M"
    timeout_s: float = 60.0
    temperature: float = 0.1
    num_ctx: int = 8192
    keep_alive: str = "10m"

    def __post_init__(self) -> None:
        # Populated after each classify() — None until the first call.
        # Keys: load_ms, prompt_eval_ms, eval_ms, total_ms,
        #       prompt_eval_count, eval_count, image_bytes
        self.last_timings: dict | None = None

    @property
    def chat_url(self) -> str:
        return f"http://{self.host}:{self.port}/api/chat"

    # ── VLMProvider conformance ─────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        host_hint = self.host if self.host not in ("127.0.0.1", "localhost") else "local"
        return f"Ollama ({self.model} · {host_hint})"

    def estimate_cost_usd(self, n_images: int = 4) -> float:
        # Local compute — no monetary cost. We don't try to charge users
        # for electricity or GPU depreciation; the status bar will show
        # $0.00 alongside the model identifier so the UI stays consistent.
        return 0.0

    def classify(
        self,
        snapshot_paths: list[Path],
        second_pass: dict | None = None,
    ):
        """
        Submit messages to Ollama and return a parsed VLMResult.
        One automatic retry on transport error.
        Raises VLMHTTPError on transport failure, VLMParseError on bad output.

        Side effects:
          - populates `self.last_timings` with Ollama's per-phase durations
            and per-attempt outcomes
          - writes a full audit record to `cache/vlm_raw/{ts}_{file_id}.json`
            for every call (successful or failed), containing the prompt
            text, options, all attempts, and the raw response. Snapshot
            paths are recorded but NOT base64 image data (too large to dump
            on every call).
        """
        messages = build_messages(snapshot_paths, second_pass=second_pass)
        # Sum of image bytes is useful context for prompt_eval cost.
        image_bytes = sum(p.stat().st_size for p in snapshot_paths if p.exists())
        # NOTE: do NOT set `format=json` here.
        # qwen3.5 and qwen3-vl have `thinking` capability — they prepend a
        # <think>...</think> reasoning block before the answer. `format=json`
        # imposes grammar-constrained generation ("output must start with `{`"),
        # which is irreconcilable with the thinking-first behaviour: the model
        # spins on GPU at 100% trying to find a grammar-valid first token and
        # never produces output, causing read timeouts. We let the model speak
        # freely and rely on the parser to extract the JSON object from the
        # tail of the response (it already strips code fences and prose).
        #
        # `think: false` is sent so the model skips the thinking block when
        # supported (Ollama ≥ 0.3 with thinking-capable models). On older
        # servers / models that don't recognise it, Ollama silently ignores
        # the flag.
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": 800,   # JSON output is ~400 tokens; cap with headroom
            },
        }

        attempts: list[dict] = []
        responses: list[dict] = []   # Ollama JSON body per attempt (or {"error": ...})
        last_exc: Exception | None = None
        for attempt_idx in range(2):
            t0 = time.monotonic()
            try:
                resp = requests.post(self.chat_url, json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                wall_ms = (time.monotonic() - t0) * 1000.0
                responses.append(data)
                if not content:
                    attempts.append({"i": attempt_idx, "wall_ms": wall_ms, "outcome": "empty_response"})
                    last_exc = VLMParseError("vlm.invalid_json", "Empty response content")
                    continue
                attempts.append({"i": attempt_idx, "wall_ms": wall_ms, "outcome": "ok"})
                self.last_timings = _extract_timings(data, image_bytes, attempts)
                result = parse_vlm_response(content)
                _dump_call_log(
                    snapshot_paths, messages, payload, attempts, responses,
                    image_bytes, error=None,
                )
                return result
            except requests.Timeout as exc:
                wall_ms = (time.monotonic() - t0) * 1000.0
                attempts.append({"i": attempt_idx, "wall_ms": wall_ms, "outcome": "timeout"})
                responses.append({"error": f"timeout: {exc}"})
                last_exc = exc
                continue
            except requests.ConnectionError as exc:
                wall_ms = (time.monotonic() - t0) * 1000.0
                attempts.append({"i": attempt_idx, "wall_ms": wall_ms, "outcome": "connection_error"})
                responses.append({"error": f"connection_error: {exc}"})
                last_exc = exc
                continue
            except requests.HTTPError as exc:
                wall_ms = (time.monotonic() - t0) * 1000.0
                attempts.append({"i": attempt_idx, "wall_ms": wall_ms, "outcome": "http_error"})
                responses.append({"error": f"http_error: {exc}"})
                self.last_timings = _extract_timings({}, image_bytes, attempts)
                _dump_call_log(
                    snapshot_paths, messages, payload, attempts, responses,
                    image_bytes, error=str(exc),
                )
                raise VLMHTTPError(f"Ollama HTTP error: {exc}") from exc

        # Both attempts exhausted — record final state before raising.
        self.last_timings = _extract_timings({}, image_bytes, attempts)
        _dump_call_log(
            snapshot_paths, messages, payload, attempts, responses,
            image_bytes, error=str(last_exc),
        )
        if isinstance(last_exc, VLMParseError):
            raise last_exc
        raise VLMHTTPError(f"Ollama transport failure after retry: {last_exc}")


def _strip_images_from_messages(messages: list[dict]) -> list[dict]:
    """Copy messages but replace `images` (base64 strings) with byte counts —
    base64 PNGs would balloon the audit JSON to MBs per call."""
    out: list[dict] = []
    for m in messages:
        m2 = dict(m)
        if "images" in m2 and isinstance(m2["images"], list):
            m2["images"] = [f"<base64 omitted, {len(s)} chars>" for s in m2["images"]]
        out.append(m2)
    return out


def _dump_call_log(
    snapshot_paths: list[Path],
    messages: list[dict],
    payload: dict,
    attempts: list[dict],
    responses: list[dict],
    image_bytes: int,
    error: str | None,
) -> None:
    """Write a full audit record for one VLM call. Never raises — logging
    must not break the VLM path."""
    try:
        # Lazy import to avoid a cycle (paths -> core/config -> ...).
        from catalog_organizer.core.paths import vlm_raw_dir  # noqa: PLC0415

        file_id = (
            snapshot_paths[0].parent.name
            if snapshot_paths
            else "unknown"
        )
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        out_path = vlm_raw_dir() / f"{ts}_{file_id}.json"

        # Stripped-down payload: keep options/model/format, drop base64 images.
        log_payload = dict(payload)
        log_payload["messages"] = _strip_images_from_messages(messages)

        record = {
            "timestamp_utc": ts,
            "file_id":       file_id,
            "snapshot_paths": [str(p) for p in snapshot_paths],
            "image_bytes":   image_bytes,
            "model":         payload.get("model"),
            "options":       payload.get("options"),
            "keep_alive":    payload.get("keep_alive"),
            "format":        payload.get("format"),
            "messages":      log_payload["messages"],
            "attempts":      attempts,
            "responses":     responses,
            "error":         error,
        }
        out_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception:
        # Logging is best-effort — never block the VLM call.
        pass


def _extract_timings(data: dict, image_bytes: int, attempts: list[dict]) -> dict:
    """Convert Ollama /api/chat response fields (nanoseconds) into ms,
    and attach the per-attempt outcome list so callers can show *why*
    wall-clock time was higher than Ollama's own total_duration."""
    ns_to_ms = lambda v: float(v) / 1e6 if isinstance(v, (int, float)) else 0.0
    return {
        "total_ms":         ns_to_ms(data.get("total_duration")),
        "load_ms":          ns_to_ms(data.get("load_duration")),
        "prompt_eval_ms":   ns_to_ms(data.get("prompt_eval_duration")),
        "eval_ms":          ns_to_ms(data.get("eval_duration")),
        "prompt_eval_count": int(data.get("prompt_eval_count") or 0),
        "eval_count":        int(data.get("eval_count") or 0),
        "image_bytes":       int(image_bytes),
        "attempts":          list(attempts),
    }
