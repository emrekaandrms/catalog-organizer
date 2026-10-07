"""OpenAI-compatible provider registry (GLM + generic endpoint).

These backends all reuse OpenAIProvider over the same wire protocol, so
the risk isn't the HTTP call — it's mis-attribution: a GLM run costing
money under the "openai" label, or a half-configured custom endpoint
silently falling back to OpenAI's servers with the user's key.
"""
from __future__ import annotations

import pytest

from catalog_organizer.vlm import providers as P
from catalog_organizer.vlm.cost import list_selectable_models


@pytest.fixture
def with_key(monkeypatch):
    """Pretend a key is stored for every provider."""
    monkeypatch.setattr(P, "get_api_key", lambda provider: "test-key-123")


def test_supported_providers_includes_new_backends():
    assert set(P.SUPPORTED_PROVIDERS) >= {
        "ollama", "openai", "minimax", "glm", "openai_compatible",
    }


def test_glm_builds_with_its_own_endpoint_and_label(with_key):
    client = P.create_vlm_provider({"provider": "glm"})
    assert client.provider_name == "glm"
    assert client.base_url == "https://api.z.ai/api/paas/v4"
    assert client.model == "glm-4.5v"
    assert "GLM" in client.display_name


def test_glm_cost_is_not_attributed_to_openai(with_key):
    """Regression guard: OpenAIProvider used to hard-code "openai" for
    pricing and for the cache/vlm_raw dump filename. Reusing the class for
    GLM without parameterising that would bill GLM traffic at gpt-4o
    rates in the status bar and mislabel every audit dump."""
    client = P.create_vlm_provider({"provider": "glm"})
    # GLM is deliberately unpriced (rates vary by region/plan), so the
    # estimate must be 0.0 — NOT gpt-4o's non-zero rate.
    assert client.estimate_cost_usd() == 0.0
    openai_client = P.create_vlm_provider({"provider": "openai"})
    assert openai_client.estimate_cost_usd() > 0.0


def test_openai_compatible_requires_explicit_base_url(with_key):
    """No default endpoint on purpose: silently defaulting to OpenAI would
    send the user's images and key to the wrong company."""
    with pytest.raises(P.ProviderConfigurationError) as exc:
        P.create_vlm_provider({"provider": "openai_compatible"})
    assert "Base URL" in str(exc.value)


def test_openai_compatible_accepts_any_endpoint(with_key):
    client = P.create_vlm_provider({
        "provider": "openai_compatible",
        "openai_compatible": {
            "base_url": "http://localhost:8000/v1",
            "model": "qwen2-vl-7b",
        },
    })
    assert client.base_url == "http://localhost:8000/v1"
    assert client.model == "qwen2-vl-7b"
    assert client.provider_name == "openai_compatible"


def test_openai_compatible_defaults_json_mode_off(with_key):
    """`response_format` is an OpenAI extension; some gateways reject the
    whole request over it. The parser tolerates prose anyway, so the safe
    default for an unknown endpoint is off."""
    client = P.create_vlm_provider({
        "provider": "openai_compatible",
        "openai_compatible": {"base_url": "http://x/v1", "model": "m"},
    })
    assert client.use_json_mode is False
    assert P.create_vlm_provider({"provider": "glm"}).use_json_mode is True


def test_json_mode_is_user_overridable(with_key):
    client = P.create_vlm_provider({
        "provider": "openai_compatible",
        "openai_compatible": {
            "base_url": "http://x/v1", "model": "m", "use_json_mode": True,
        },
    })
    assert client.use_json_mode is True


@pytest.mark.parametrize("provider", ["glm", "openai_compatible"])
def test_missing_api_key_fails_loudly(monkeypatch, provider):
    monkeypatch.setattr(P, "get_api_key", lambda p: None)
    with pytest.raises(P.ProviderConfigurationError) as exc:
        P.create_vlm_provider({"provider": provider})
    assert "API key" in str(exc.value)


def test_unknown_provider_lists_the_supported_ones():
    with pytest.raises(P.ProviderConfigurationError) as exc:
        P.create_vlm_provider({"provider": "definitely-not-real"})
    for name in ("ollama", "glm", "openai_compatible"):
        assert name in str(exc.value)


def test_glm_models_selectable_even_though_unpriced():
    """list_priced_models() would return [] for GLM and leave the Settings
    dropdown empty; list_selectable_models() must still offer the ids."""
    from catalog_organizer.vlm.cost import list_priced_models
    assert list_priced_models("glm") == []
    assert "glm-4.5v" in list_selectable_models("glm")
    # OpenAI keeps working through the same call.
    assert "gpt-4o" in list_selectable_models("openai")
