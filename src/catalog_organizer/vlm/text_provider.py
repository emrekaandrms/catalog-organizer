"""Metin-tabanlı (görselsiz) LLM sağlayıcıları — ilan metni üretimi için.

Neden ayrı: `VLMProvider` protokolü görsel sınıflandırma etrafında kurulu
(`classify(snapshot_paths, ...)`). İlan metni üretiminde görsel yok — görsel
anlayış zaten analiz aşamasında `rich_description`'a damıtıldı. Resmi ikinci
kez göndermek maliyeti ve süreyi iki katına çıkarır.

Metin-tabanlı olmasının ikinci faydası: bulut sağlayıcıları (GLM, OpenCode)
bu işte hızlı ve ucuz, üstelik GPU'yu meşgul etmiyor — analiz sürerken ilan
metni üretilebiliyor.

Sağlayıcı seçimi ve API anahtarları mevcut altyapıdan geliyor: aynı
`pipeline_settings.yaml` bloğu, aynı keyring girdileri. Yeni bir ayar ekranı
ya da ikinci bir anahtar yok.
"""
from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

import requests

from catalog_organizer.core.secrets import get_api_key
from catalog_organizer.vlm.parser import _extract_json_object
from catalog_organizer.vlm.providers import (
    _OPENAI_COMPATIBLE, ProviderConfigurationError,
)


class TextProviderError(RuntimeError):
    """Ağ hatası ya da modelden JSON çıkaramama."""


@runtime_checkable
class TextProvider(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Modelden JSON nesnesi iste ve ayrıştırılmış hâlde döndür."""
        ...

    @property
    def display_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...


def _parse_json_or_raise(raw: str, who: str) -> dict[str, Any]:
    """Modelin cevabından JSON nesnesini çıkar.

    Yerel modeller cevabı düz metinle çevreliyor, ```json çitiyle sarıyor ya
    da <think> bloğu ekliyor. `_extract_json_object` bu üçünü de tolere
    ediyor — VLM hattında zaten kanıtlanmış davranış.
    """
    try:
        payload = json.loads(_extract_json_object(raw))
    except Exception as exc:
        preview = raw[:300].replace("\n", " ")
        raise TextProviderError(
            f"{who} JSON döndürmedi. Cevabın başı: {preview!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise TextProviderError(f"{who} JSON nesnesi değil {type(payload).__name__} döndürdü")
    return payload


class OllamaTextProvider:
    """Yerel Ollama. `format=json` ile sunucu tarafında JSON zorlanıyor."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 11434,
        model: str = "qwen3.5:9b-q4_K_M",
        timeout_s: float = 300.0,
        temperature: float = 0.3,
        num_ctx: int = 8192,
        keep_alive: str = "10m",
    ) -> None:
        self._url = f"http://{host}:{port}/api/chat"
        self._model = model
        self._timeout = timeout_s
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive

    @property
    def display_name(self) -> str:
        return f"Ollama ({self._model})"

    @property
    def model_name(self) -> str:
        return self._model

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        body = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": self._temperature,
                "num_ctx": self._num_ctx,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            resp = requests.post(self._url, json=body, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise TextProviderError(f"Ollama'ya ulaşılamadı: {exc}") from exc
        content = (data.get("message") or {}).get("content", "")
        return _parse_json_or_raise(content, self.display_name)


class OpenAICompatTextProvider:
    """OpenAI Chat Completions protokolü konuşan her uç nokta —
    OpenAI, GLM, OpenCode Zen, OpenRouter, self-hosted vLLM…"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        vendor: str,
        timeout_s: float = 120.0,
        temperature: float = 0.4,
        max_output_tokens: int = 1600,
        use_json_mode: bool = True,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._vendor = vendor
        self._timeout = timeout_s
        self._temperature = temperature
        self._max_tokens = max_output_tokens
        self._json_mode = use_json_mode

    @property
    def display_name(self) -> str:
        return f"{self._vendor} ({self._model})"

    @property
    def model_name(self) -> str:
        return self._model

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise TextProviderError(f"{self._vendor} çağrısı başarısız: {exc}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise TextProviderError(
                f"{self._vendor} beklenmedik cevap şekli döndürdü"
            ) from exc
        return _parse_json_or_raise(content, self.display_name)


def create_text_provider(settings: dict[str, Any]) -> TextProvider:
    """`pipeline_settings.yaml`'ın `vlm` bloğundan metin sağlayıcı kurar.

    Aynı `provider` seçimi ve aynı keyring anahtarları kullanılır — kullanıcı
    ilan metni için ayrı bir sağlayıcı yapılandırmaz.
    """
    name = (settings.get("provider") or "ollama").lower()
    timeout = float(settings.get("timeout_s", 300))
    # Sınıflandırmadaki 0.1 burada fazla kısıtlayıcı: pazarlama metninde biraz
    # çeşitlilik isteniyor, 100 üründe aynı cümlelerin tekrarlanması istenmiyor.
    temperature = float(settings.get("listing_temperature", 0.4))

    if name == "ollama":
        block = settings.get("ollama", {})
        return OllamaTextProvider(
            host=block.get("host", "127.0.0.1"),
            port=int(block.get("port", 11434)),
            model=block.get("model", "qwen3.5:9b-q4_K_M"),
            timeout_s=float(block.get("timeout_s", timeout)),
            temperature=temperature,
            num_ctx=int(block.get("num_ctx", 8192)),
            keep_alive=str(block.get("keep_alive", "10m")),
        )

    if name == "openai":
        api_key = get_api_key("openai")
        if not api_key:
            raise ProviderConfigurationError(
                "OpenAI seçili ama keyring'de anahtar yok. "
                "Ayarlar → VLM Provider'dan girin."
            )
        block = settings.get("openai", {})
        return OpenAICompatTextProvider(
            api_key=api_key,
            model=block.get("model", "gpt-4o"),
            base_url=block.get("base_url", "https://api.openai.com/v1"),
            vendor="OpenAI",
            timeout_s=float(block.get("timeout_s", timeout)),
            temperature=temperature,
            max_output_tokens=int(block.get("listing_max_tokens", 1600)),
            use_json_mode=True,
        )

    if name in _OPENAI_COMPATIBLE:
        spec = _OPENAI_COMPATIBLE[name]
        block = settings.get(name, {})
        api_key = get_api_key(name)
        if not api_key:
            raise ProviderConfigurationError(
                f"{spec.vendor} seçili ama keyring'de anahtar yok. "
                "Ayarlar → VLM Provider'dan girin."
            )
        base_url = (block.get("base_url") or spec.base_url).strip()
        if not base_url:
            raise ProviderConfigurationError(
                f"{spec.vendor} için Base URL gerekli (…/v1)."
            )
        return OpenAICompatTextProvider(
            api_key=api_key,
            model=block.get("model") or spec.model,
            base_url=base_url,
            vendor=spec.vendor,
            timeout_s=float(block.get("timeout_s", timeout)),
            temperature=temperature,
            max_output_tokens=int(block.get("listing_max_tokens", 1600)),
            use_json_mode=bool(block.get("use_json_mode", spec.json_mode)),
        )

    # MiniMax metin uç noktası bu projede hiç denenmedi; sessizce yanlış bir
    # istek göndermektense açıkça reddediyoruz.
    raise ProviderConfigurationError(
        f"'{name}' ilan metni üretimi için desteklenmiyor. "
        "Desteklenenler: ollama, openai, glm, opencode, openai_compatible."
    )
