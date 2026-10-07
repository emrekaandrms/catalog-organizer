"""Live model discovery for the Settings dropdown (vlm/ollama_models.py)."""
from __future__ import annotations

import pytest
import requests

from catalog_organizer.vlm import ollama_models as om


class _FakeResponse:
    def __init__(self, payload, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _patch_get(monkeypatch, result):
    def fake_get(url, timeout=None):
        if isinstance(result, Exception):
            raise result
        return _FakeResponse(result)
    monkeypatch.setattr(om.requests, "get", fake_get)


_REAL_TAGS_PAYLOAD = {
    "models": [
        {
            "name": "qwen3.5:9b-q4_K_M",
            "size": 6594474711,
            "capabilities": ["vision", "completion", "tools", "thinking"],
            "details": {"families": ["qwen35"]},
        },
        {
            "name": "llama3:8b",
            "size": 4700000000,
            "capabilities": ["completion", "tools"],
            "details": {"families": ["llama"]},
        },
    ]
}


def test_parses_real_tags_payload(monkeypatch):
    """Shape pinned against the user's actual Ollama /api/tags reply
    (2026-07-28), which does include per-model `capabilities`."""
    _patch_get(monkeypatch, _REAL_TAGS_PAYLOAD)
    models = om.list_ollama_models()
    assert [m.name for m in models] == ["llama3:8b", "qwen3.5:9b-q4_K_M"]  # sorted
    qwen = models[1]
    assert qwen.vision is True
    assert qwen.size_label == "6.6 GB"
    assert qwen.display_label() == "qwen3.5:9b-q4_K_M  ·  6.6 GB"


def test_text_only_model_reports_vision_false(monkeypatch):
    _patch_get(monkeypatch, _REAL_TAGS_PAYLOAD)
    llama = om.list_ollama_models()[0]
    assert llama.name == "llama3:8b"
    assert llama.vision is False


def test_vision_is_none_when_server_omits_capabilities(monkeypatch):
    """Older Ollama builds don't report `capabilities`. That must read as
    "unknown", not "no vision" — see filter test below."""
    _patch_get(monkeypatch, {"models": [{"name": "old:latest", "size": 100}]})
    model = om.list_ollama_models()[0]
    assert model.capabilities_reported is False
    assert model.vision is None


def test_filter_keeps_unknown_capability_models(monkeypatch):
    """Regression guard: filtering must not empty the dropdown on an older
    Ollama that reports no capabilities — that would look to the user like
    "no models installed" when in fact every model is usable."""
    _patch_get(monkeypatch, {
        "models": [
            {"name": "unknown:latest", "size": 1},                       # no caps key
            {"name": "seer:latest", "size": 1, "capabilities": ["vision"]},
            {"name": "blind:latest", "size": 1, "capabilities": ["completion"]},
        ]
    })
    kept = {m.name for m in om.filter_vision_models(om.list_ollama_models())}
    assert kept == {"unknown:latest", "seer:latest"}   # only proven text-only dropped


@pytest.mark.parametrize("failure,expected_fragment", [
    (requests.ConnectionError("refused"), "connection refused"),
    (requests.Timeout("slow"), "timed out"),
])
def test_transport_failures_raise_ollama_unreachable(monkeypatch, failure, expected_fragment):
    _patch_get(monkeypatch, failure)
    with pytest.raises(om.OllamaUnreachable) as exc:
        om.list_ollama_models(host="10.0.0.5", port=1234)
    assert "10.0.0.5:1234" in str(exc.value)
    assert expected_fragment in str(exc.value)


def test_non_json_reply_raises_rather_than_returning_empty(monkeypatch):
    """An empty dropdown and a broken server must be distinguishable —
    returning [] here would render as "no models installed"."""
    _patch_get(monkeypatch, ValueError("not json"))
    with pytest.raises(om.OllamaUnreachable):
        om.list_ollama_models()


def test_reply_without_models_list_raises(monkeypatch):
    _patch_get(monkeypatch, {"unexpected": "shape"})
    with pytest.raises(om.OllamaUnreachable):
        om.list_ollama_models()


def test_empty_but_valid_server_returns_empty_list(monkeypatch):
    """A reachable Ollama with nothing pulled is NOT an error."""
    _patch_get(monkeypatch, {"models": []})
    assert om.list_ollama_models() == []
