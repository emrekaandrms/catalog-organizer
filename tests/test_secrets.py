"""`core.secrets` — keyring round-trip with an in-memory backend.

We replace the OS keyring with a `keyring.backends.fake.Keyring` (or a
hand-rolled stub if the test environment doesn't bundle the fake backend)
so the tests don't pollute the developer's real credential manager.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def fake_keyring(monkeypatch):
    """Swap in a dict-backed keyring for the duration of one test."""
    import keyring
    store: dict[tuple[str, str], str] = {}

    class _FakeBackend:
        priority = 1
        def get_password(self, service, username):
            return store.get((service, username))
        def set_password(self, service, username, password):
            store[(service, username)] = password
        def delete_password(self, service, username):
            store.pop((service, username), None)

    monkeypatch.setattr(keyring, "get_keyring", lambda: _FakeBackend())
    monkeypatch.setattr(keyring, "get_password",
                        lambda s, u: store.get((s, u)))
    monkeypatch.setattr(keyring, "set_password",
                        lambda s, u, p: store.__setitem__((s, u), p))
    monkeypatch.setattr(keyring, "delete_password",
                        lambda s, u: store.pop((s, u), None))
    return store


def test_set_and_get_api_key_round_trip(fake_keyring):
    from catalog_organizer.core.secrets import get_api_key, set_api_key

    set_api_key("openai", "sk-test-1234")
    assert get_api_key("openai") == "sk-test-1234"


def test_keys_are_scoped_per_provider(fake_keyring):
    """openai and minimax keys must not collide — each lives under its own
    keyring username."""
    from catalog_organizer.core.secrets import get_api_key, set_api_key

    set_api_key("openai",  "sk-openai-1")
    set_api_key("minimax", "mm-minimax-9")
    assert get_api_key("openai")  == "sk-openai-1"
    assert get_api_key("minimax") == "mm-minimax-9"


def test_set_empty_string_deletes_the_key(fake_keyring):
    from catalog_organizer.core.secrets import get_api_key, set_api_key

    set_api_key("openai", "sk-temp")
    assert get_api_key("openai") == "sk-temp"
    set_api_key("openai", "")
    assert get_api_key("openai") is None


def test_delete_is_idempotent_on_missing_provider(fake_keyring):
    """Delete on a never-stored provider must be a no-op, not raise.
    The Settings UI calls this whenever the user clicks 'Forget key'
    even if no key was ever saved."""
    from catalog_organizer.core.secrets import delete_api_key, get_api_key
    delete_api_key("openai")
    assert get_api_key("openai") is None
