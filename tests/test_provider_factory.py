"""`vlm.providers.create_vlm_provider` — selects the right backend."""
from __future__ import annotations

import pytest


@pytest.fixture
def fake_keyring(monkeypatch):
    """Same in-memory keyring as test_secrets, scoped here for isolation."""
    import keyring
    store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(keyring, "get_password",
                        lambda s, u: store.get((s, u)))
    monkeypatch.setattr(keyring, "set_password",
                        lambda s, u, p: store.__setitem__((s, u), p))
    monkeypatch.setattr(keyring, "delete_password",
                        lambda s, u: store.pop((s, u), None))
    return store


def test_factory_returns_ollama_provider_for_default_settings(fake_keyring):
    from catalog_organizer.vlm.providers import create_vlm_provider
    from catalog_organizer.vlm.ollama_client import OllamaClient
    client = create_vlm_provider({
        "provider": "ollama",
        "ollama":   {"host": "192.168.1.5", "port": 11434, "model": "abc:1b"},
    })
    assert isinstance(client, OllamaClient)
    assert client.host  == "192.168.1.5"
    assert client.model == "abc:1b"


def test_factory_legacy_flat_ollama_keys_still_work(fake_keyring):
    """Old configs (pre-2026-05-21) had host/port/model at the top of the
    `vlm` block without any `provider` field. Factory must treat that as
    Ollama and use those flat keys."""
    from catalog_organizer.vlm.providers import create_vlm_provider
    client = create_vlm_provider({
        "host":      "10.0.0.4",
        "port":      11434,
        "model":     "qwen3.5:9b-q4_K_M",
        "timeout_s": 120,
    })
    assert client.host  == "10.0.0.4"
    assert client.model == "qwen3.5:9b-q4_K_M"


def test_factory_openai_requires_api_key(fake_keyring):
    """Selecting OpenAI without a keyring entry must fail fast with a
    helpful message — never silently fall through to an unconfigured
    client that would hit the API as 'anonymous'."""
    from catalog_organizer.vlm.providers import (
        ProviderConfigurationError,
        create_vlm_provider,
    )
    with pytest.raises(ProviderConfigurationError) as exc:
        create_vlm_provider({"provider": "openai", "openai": {"model": "gpt-4o"}})
    assert "API key" in str(exc.value)


def test_factory_openai_with_keyring_key_returns_provider(fake_keyring):
    from catalog_organizer.core.secrets import set_api_key
    from catalog_organizer.vlm.providers import create_vlm_provider
    from catalog_organizer.vlm.providers.openai import OpenAIProvider

    set_api_key("openai", "sk-real-key")
    client = create_vlm_provider({
        "provider": "openai",
        "openai":   {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
    })
    assert isinstance(client, OpenAIProvider)
    assert client.model    == "gpt-4o"
    assert client.api_key  == "sk-real-key"


def test_factory_minimax_with_keyring_key(fake_keyring):
    from catalog_organizer.core.secrets import set_api_key
    from catalog_organizer.vlm.providers import create_vlm_provider
    from catalog_organizer.vlm.providers.minimax import MiniMaxProvider

    set_api_key("minimax", "mm-key-x")
    client = create_vlm_provider({
        "provider": "minimax",
        "minimax":  {"model": "MiniMax-VL-01", "group_id": "G1"},
    })
    assert isinstance(client, MiniMaxProvider)
    assert client.group_id == "G1"


def test_factory_rejects_unknown_provider(fake_keyring):
    from catalog_organizer.vlm.providers import (
        ProviderConfigurationError,
        create_vlm_provider,
    )
    with pytest.raises(ProviderConfigurationError):
        create_vlm_provider({"provider": "bogus"})
