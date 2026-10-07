"""Discover which models a running Ollama server actually has pulled.

The Settings UI uses this to populate a model dropdown instead of making
the user type an exact tag by hand (a typo there produces a confusing
404 mid-batch rather than an up-front error).

Why the vision filter matters: this pipeline sends 4 PNG snapshots per
file. A text-only model accepts the request and answers *plausibly* from
the prompt text alone — it never sees the jewelry. The result is a
catalog full of confident, wholly invented classifications. Filtering to
vision-capable models makes that failure impossible to select by
accident.

`GET /api/tags` reports per-model `capabilities` (verified against the
user's Ollama 2026-07-28: `["vision", "completion", "tools", "thinking"]`
for qwen3.5:9b-q4_K_M). Older Ollama builds omit the field entirely — in
that case `OllamaModel.vision` is None, meaning "unknown", and callers
must NOT filter it out. Silently hiding every model because the server
is old would look like "Ollama is empty".
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

_DEFAULT_TIMEOUT_S = 5.0


class OllamaUnreachable(RuntimeError):
    """The Ollama server could not be reached or gave an unusable reply.

    Carries the URL we tried so the UI can show something actionable
    ("is Ollama running on this host/port?") rather than a bare stack
    trace.
    """

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"Cannot reach Ollama at {url}: {reason}")
        self.url = url
        self.reason = reason


@dataclass(frozen=True)
class OllamaModel:
    name: str
    size_bytes: int = 0
    capabilities: tuple[str, ...] = ()
    families: tuple[str, ...] = ()
    # True when the server reported capabilities and listed "vision";
    # False when it reported capabilities without it; None when the
    # server didn't report capabilities at all (older Ollama).
    capabilities_reported: bool = True

    @property
    def vision(self) -> bool | None:
        """Tri-state on purpose — see module docstring. None means the
        server didn't tell us, which is NOT the same as "no vision"."""
        if not self.capabilities_reported:
            return None
        return "vision" in self.capabilities

    @property
    def size_label(self) -> str:
        if self.size_bytes <= 0:
            return ""
        return f"{self.size_bytes / 1_000_000_000:.1f} GB"

    def display_label(self) -> str:
        """What the dropdown shows: tag plus size, e.g.
        'qwen3.5:9b-q4_K_M  ·  6.6 GB'."""
        return f"{self.name}  ·  {self.size_label}" if self.size_label else self.name


def tags_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/api/tags"


def list_ollama_models(
    host: str = "127.0.0.1",
    port: int = 11434,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> list[OllamaModel]:
    """Return every model pulled on the Ollama server at host:port.

    Sorted by name. Raises `OllamaUnreachable` for any connection,
    timeout, HTTP or payload problem — callers show that message rather
    than presenting an empty dropdown that looks like "no models".
    """
    url = tags_url(host, port)
    try:
        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
    except requests.Timeout as exc:
        raise OllamaUnreachable(url, f"timed out after {timeout_s:g}s") from exc
    except requests.ConnectionError as exc:
        raise OllamaUnreachable(url, "connection refused (is Ollama running?)") from exc
    except requests.HTTPError as exc:
        raise OllamaUnreachable(url, f"HTTP {exc.response.status_code}") from exc
    except ValueError as exc:                       # includes JSONDecodeError
        raise OllamaUnreachable(url, "server did not return JSON") from exc
    except requests.RequestException as exc:
        raise OllamaUnreachable(url, str(exc)) from exc

    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise OllamaUnreachable(url, "reply had no 'models' list")

    out: list[OllamaModel] = []
    for entry in raw_models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("model")
        if not name:
            continue
        raw_caps = entry.get("capabilities")
        details = entry.get("details") or {}
        out.append(OllamaModel(
            name=str(name),
            size_bytes=int(entry.get("size") or 0),
            capabilities=tuple(str(c) for c in raw_caps) if isinstance(raw_caps, list) else (),
            families=tuple(str(f) for f in (details.get("families") or [])),
            capabilities_reported=isinstance(raw_caps, list),
        ))
    return sorted(out, key=lambda m: m.name.lower())


def filter_vision_models(models: list[OllamaModel]) -> list[OllamaModel]:
    """Keep models that can actually see the snapshots.

    Models whose capabilities the server never reported (`vision is None`)
    are KEPT — we can't prove they lack vision, and dropping them on an
    older Ollama would empty the list entirely.
    """
    return [m for m in models if m.vision is not False]
