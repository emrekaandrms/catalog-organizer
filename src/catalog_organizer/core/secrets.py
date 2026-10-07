"""Cross-platform secret storage for VLM API keys.

Backed by the `keyring` package, which uses Windows Credential Manager on
Windows, the Keychain on macOS, and the Secret Service on Linux. We never
write API keys to YAML or JSON files — anything in the project tree could
be checked into git or shared by accident.

All entries live under the service name ``CatalogOrganizer``, keyed by
``vlm/{provider}`` (e.g. ``vlm/openai``, ``vlm/minimax``). When the user
swaps providers in Settings, only the new provider's key is read — the
others remain stored but dormant.
"""
from __future__ import annotations

from typing import Final

try:
    import keyring                       # type: ignore[import-not-found]
    import keyring.errors as _kerr       # type: ignore[import-not-found]
    _KEYRING_AVAILABLE = True
except ImportError:
    keyring = None                       # type: ignore[assignment]
    _kerr = None                         # type: ignore[assignment]
    _KEYRING_AVAILABLE = False


_SERVICE: Final[str] = "CatalogOrganizer"
_KEY_PREFIX: Final[str] = "vlm/"


class SecretsBackendUnavailable(RuntimeError):
    """Raised when `keyring` isn't importable or no OS backend works.
    Callers should surface this to the user — there is no safe automatic
    fallback to plain-text storage."""

    def __init__(self) -> None:
        super().__init__(
            "OS credential manager not available (install the `keyring` "
            "package and verify your platform's credential backend)."
        )


def _username(provider: str) -> str:
    """Per-provider keyring username so multiple keys can co-exist under
    the same service entry."""
    return _KEY_PREFIX + provider.lower().strip()


def get_api_key(provider: str) -> str | None:
    """Return the stored API key for `provider` (lowercase, e.g. 'openai'),
    or None if no key has been saved or no backend is available.

    Reads are always graceful: a missing keyring backend on this machine
    means "we don't have a key", not a crash. This lets Settings UI build
    safely even on a fresh system without keyring installed; the banner
    `is_backend_available()` already warns the user separately.
    """
    if not _KEYRING_AVAILABLE:
        return None
    try:
        return keyring.get_password(_SERVICE, _username(provider))
    except Exception:
        # NoKeyringError on Linux without Secret Service, KeyringError
        # on Windows when the vault hasn't been initialised, etc.
        return None


def set_api_key(provider: str, api_key: str) -> None:
    """Persist `api_key` for `provider`. Overwrites silently if a key
    already exists. An empty string deletes the entry (use `delete_api_key`
    if you want to be explicit).

    Unlike `get_api_key`, this DOES raise on a missing backend — you
    can't fake a save and there's no safe local fallback that preserves
    the secret guarantee.
    """
    if not _KEYRING_AVAILABLE:
        raise SecretsBackendUnavailable()
    if not api_key:
        delete_api_key(provider)
        return
    try:
        keyring.set_password(_SERVICE, _username(provider), api_key)
    except Exception as exc:
        raise SecretsBackendUnavailable() from exc


def delete_api_key(provider: str) -> None:
    """Remove the stored API key for `provider`. Silently no-ops if the
    entry doesn't exist or no backend is available — Settings UI calls
    this whenever the user clicks 'Forget key' and it should never crash."""
    if not _KEYRING_AVAILABLE:
        return
    try:
        keyring.delete_password(_SERVICE, _username(provider))
    except Exception:
        # PasswordDeleteError on macOS/Linux; KeyringError on Win when no entry.
        pass


def is_backend_available() -> bool:
    """Quick probe so the Settings UI can warn the user up front."""
    return _KEYRING_AVAILABLE
