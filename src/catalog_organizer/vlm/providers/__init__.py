"""Provider factory.

Call sites (pipeline, Process panel, Diagnostics, app bootstrap) build a
provider with::

    from catalog_organizer.vlm.providers import create_vlm_provider
    vlm = create_vlm_provider(load_pipeline_settings().get("vlm", {}))

Settings schema (pipeline_settings.yaml) — flat keys at the top of the
``vlm`` block are *shared* (provider name, parallel workers, timeout),
per-provider blocks below configure each backend:

    vlm:
      provider: ollama          # see SUPPORTED_PROVIDERS
      parallel_workers: 2
      timeout_s: 300
      temperature: 0.1
      ollama:
        host: 127.0.0.1
        port: 11434
        model: qwen3.5:9b-q4_K_M
        num_ctx: 8192
        keep_alive: "10m"
      openai:
        model: gpt-4o
        base_url: https://api.openai.com/v1
      minimax:
        model: MiniMax-VL-01
        base_url: https://api.minimaxi.com/v1
        group_id: ""
      glm:
        model: glm-4.5v
        base_url: https://api.z.ai/api/paas/v4
      openai_compatible:        # OpenRouter / DeepSeek / vLLM / anything
        model: ""
        base_url: ""            # required — no default on purpose
        use_json_mode: false

API keys for every cloud provider come from the OS credential manager —
never from the YAML file. Use Settings → VLM Provider to enter them.

Backward compatibility: if a record has no `provider` field and uses the
flat Ollama keys (host/port/model at the top of `vlm:`), we treat it as
the legacy Ollama config so existing installs keep working without
touching their config file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catalog_organizer.core.secrets import get_api_key


class ProviderConfigurationError(RuntimeError):
    """Raised when the requested provider exists but is mis-configured
    (missing API key, unknown model, etc.). Settings UI surfaces the
    message; pipeline aborts the batch with a clear reason."""


@dataclass(frozen=True)
class _CompatSpec:
    """Defaults for a backend that speaks the OpenAI Chat Completions
    protocol. All of these share `OpenAIProvider` — only the endpoint,
    labelling and JSON-mode support differ."""
    vendor: str          # human label ("GLM (Zhipu AI)")
    base_url: str        # default endpoint; "" = user must supply one
    model: str           # default model id
    json_mode: bool      # send response_format={"type":"json_object"}?


# Registry of OpenAI-compatible backends beyond `openai` itself.
#
# `openai_compatible` is the deliberate escape hatch: rather than adding
# a hard-coded entry for every gateway that appears (OpenRouter, DeepSeek,
# Together, Groq, a self-hosted vLLM…), the user points this one at any
# endpoint speaking the same protocol. It ships with no default base_url
# precisely so a half-configured selection fails loudly instead of
# silently calling OpenAI.
_OPENAI_COMPATIBLE: dict[str, _CompatSpec] = {
    "glm": _CompatSpec(
        vendor="GLM (Zhipu AI)",
        # International endpoint. Users on the mainland platform should
        # switch this to https://open.bigmodel.cn/api/paas/v4 in Settings.
        base_url="https://api.z.ai/api/paas/v4",
        model="glm-4.5v",
        json_mode=True,
    ),
    "opencode": _CompatSpec(
        # OpenCode Zen — a gateway that fronts GPT, Claude and Gemini behind
        # one key. Verified against https://opencode.ai/docs/zen/ (2026-09):
        # OpenAI-compatible /chat/completions and /models under this base,
        # Bearer auth, key issued at https://opencode.ai/auth.
        #
        # The default is a Claude vision model because most of Zen's catalogue
        # is coding-focused and this pipeline is useless without image input.
        # The dropdown is editable and Test Connection lists what the key can
        # actually reach, so a different pick is one field away.
        vendor="OpenCode Zen",
        base_url="https://opencode.ai/zen/v1",
        model="claude-sonnet-5",
        json_mode=True,
    ),
    "openai_compatible": _CompatSpec(
        vendor="OpenAI-compatible endpoint",
        base_url="",
        model="",
        # Unknown gateway: assume no response_format support so a working
        # call isn't rejected outright. The parser tolerates prose and
        # fenced JSON anyway, and the user can switch it on in Settings.
        json_mode=False,
    ),
}

SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "ollama", "openai", "minimax", *sorted(_OPENAI_COMPATIBLE),
)


def create_vlm_provider(settings: dict[str, Any]):
    """Build a `VLMProvider` from the `vlm` block of pipeline_settings.

    Reads:
      * `provider`            — which backend to instantiate
      * `timeout_s`, `temperature` — shared knobs (provider can override
        from its own block if it wants)
      * `<provider>.*`        — per-backend config
    """
    provider_name = (settings.get("provider") or _infer_legacy_provider(settings)).lower()
    shared_timeout = float(settings.get("timeout_s", 300))
    shared_temp = float(settings.get("temperature", 0.1))

    if provider_name == "ollama":
        block = settings.get("ollama") or _legacy_ollama_block(settings)
        from catalog_organizer.vlm.ollama_client import OllamaClient
        return OllamaClient(
            host=block.get("host", "127.0.0.1"),
            port=int(block.get("port", 11434)),
            model=block.get("model", "qwen3.5:9b-q4_K_M"),
            timeout_s=float(block.get("timeout_s", shared_timeout)),
            temperature=float(block.get("temperature", shared_temp)),
            num_ctx=int(block.get("num_ctx", 8192)),
            keep_alive=str(block.get("keep_alive", "10m")),
        )

    if provider_name == "openai":
        block = settings.get("openai", {})
        api_key = get_api_key("openai")
        if not api_key:
            raise ProviderConfigurationError(
                "OpenAI provider selected but no API key found in keyring. "
                "Open Settings → VLM Provider and paste your sk-... key."
            )
        from catalog_organizer.vlm.providers.openai import OpenAIProvider
        return OpenAIProvider(
            api_key=api_key,
            model=block.get("model", "gpt-4o"),
            base_url=block.get("base_url", "https://api.openai.com/v1"),
            timeout_s=float(block.get("timeout_s", shared_timeout)),
            temperature=float(block.get("temperature", shared_temp)),
            max_output_tokens=int(block.get("max_output_tokens", 800)),
            org_id=block.get("org_id") or None,
        )

    if provider_name == "minimax":
        block = settings.get("minimax", {})
        api_key = get_api_key("minimax")
        if not api_key:
            raise ProviderConfigurationError(
                "MiniMax provider selected but no API key found in keyring. "
                "Open Settings → VLM Provider and paste your key."
            )
        from catalog_organizer.vlm.providers.minimax import MiniMaxProvider
        return MiniMaxProvider(
            api_key=api_key,
            model=block.get("model", "MiniMax-VL-01"),
            base_url=block.get("base_url", "https://api.minimaxi.com/v1"),
            timeout_s=float(block.get("timeout_s", shared_timeout)),
            temperature=float(block.get("temperature", shared_temp)),
            max_output_tokens=int(block.get("max_output_tokens", 800)),
            group_id=block.get("group_id") or None,
        )

    if provider_name in _OPENAI_COMPATIBLE:
        spec = _OPENAI_COMPATIBLE[provider_name]
        block = settings.get(provider_name, {})
        api_key = get_api_key(provider_name)
        if not api_key:
            raise ProviderConfigurationError(
                f"{spec.vendor} provider selected but no API key found in "
                "keyring. Open Settings → VLM Provider and paste your key."
            )
        base_url = (block.get("base_url") or spec.base_url).strip()
        if not base_url:
            raise ProviderConfigurationError(
                f"{spec.vendor} needs a Base URL. Open Settings → VLM "
                "Provider and enter the endpoint (…/v1)."
            )
        from catalog_organizer.vlm.providers.openai import OpenAIProvider
        return OpenAIProvider(
            api_key=api_key,
            model=block.get("model") or spec.model,
            base_url=base_url,
            timeout_s=float(block.get("timeout_s", shared_timeout)),
            temperature=float(block.get("temperature", shared_temp)),
            max_output_tokens=int(block.get("max_output_tokens", 800)),
            provider_name=provider_name,
            display_vendor=spec.vendor,
            use_json_mode=bool(block.get("use_json_mode", spec.json_mode)),
        )

    raise ProviderConfigurationError(
        f"Unknown VLM provider: {provider_name!r}. "
        f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def _infer_legacy_provider(settings: dict[str, Any]) -> str:
    """Old configs (pre-2026-05-21) had Ollama keys flat in the `vlm`
    block and no `provider` field. Treat any such record as Ollama so
    nobody has to edit YAML before re-launching."""
    return "ollama"


def _legacy_ollama_block(settings: dict[str, Any]) -> dict[str, Any]:
    """Re-package the flat legacy Ollama keys into the nested shape the
    new factory expects."""
    return {
        "host":        settings.get("host", "127.0.0.1"),
        "port":        settings.get("port", 11434),
        "model":       settings.get("model", "qwen3.5:9b-q4_K_M"),
        "timeout_s":   settings.get("timeout_s", 300),
        "temperature": settings.get("temperature", 0.1),
        "num_ctx":     settings.get("num_ctx", 8192),
        "keep_alive":  settings.get("keep_alive", "10m"),
    }
