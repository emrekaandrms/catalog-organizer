"""Per-provider pricing data + helpers to estimate USD cost for a VLM call.

Prices below are *list* USD per 1M tokens at the time of writing (Q2 2026).
Providers update them frequently — when a discrepancy is reported, edit
this table. The Settings UI shows the active rate so users can audit.

Image-token math (OpenAI / vision models):
  * One 1024×1024 image at "high detail" is billed as ~765 input tokens
    on gpt-4o (85 base + 4 tiles × 170). Lower-detail mode bills ~85.
  * The pipeline sends 4 images per file → ~3060 image tokens.
  * Text prompt (system + taxonomy + tag dictionary + JSON shape) ≈ 1500
    tokens. Total ≈ 4500 input + ~500 output.

These numbers feed `estimate_call_cost_usd()` so the status bar and
end-of-run dialog can show a faithful estimate.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _ModelPricing:
    input_usd_per_1m: float
    output_usd_per_1m: float
    # Token cost per 1024×1024 image at "high detail". Some providers
    # charge images flatly (one rate per image regardless of size).
    image_tokens_per_image: int = 0
    # Flat image fee in USD, used when image_tokens_per_image == 0.
    image_usd_each: float = 0.0


# Provider → model → pricing
_PRICING: dict[str, dict[str, _ModelPricing]] = {
    "openai": {
        # https://openai.com/api/pricing/  (snapshot Q2 2026)
        "gpt-4o":       _ModelPricing(2.50, 10.00, image_tokens_per_image=765),
        "gpt-4o-mini":  _ModelPricing(0.15,  0.60, image_tokens_per_image=2833),  # mini bills more image tokens
        "gpt-4-turbo":  _ModelPricing(10.00, 30.00, image_tokens_per_image=765),
    },
    "minimax": {
        # https://www.minimax.io/platform/pricing  (snapshot Q2 2026)
        # MiniMax does NOT publish a vision-specific list; we use its flagship
        # text rate as a placeholder. Settings UI lets users override.
        "MiniMax-VL-01":   _ModelPricing(0.20, 1.10, image_tokens_per_image=512),
        "MiniMax-M1":      _ModelPricing(0.20, 1.10, image_tokens_per_image=512),
        "MiniMax-Text-01": _ModelPricing(0.20, 1.10, image_tokens_per_image=0),
    },
    # Ollama runs locally — no monetary cost, just compute.
    "ollama": {},
    # GLM / Zhipu and generic OpenAI-compatible endpoints are deliberately
    # left unpriced: their rates vary by region, plan and (for a custom
    # gateway) by whoever operates it. `estimate_call_cost_usd` returns
    # 0.0 for anything absent here and the Settings UI renders
    # "(not in pricing table)" — an honest blank rather than a number that
    # could be wrong by an order of magnitude on a 90k-file run.
    "glm": {},
    "opencode": {},
    "openai_compatible": {},
}


# Model ids offered in the Settings dropdown, independent of whether we
# have pricing for them. Kept separate from `_PRICING` on purpose: a model
# we can list is not necessarily a model whose cost we can vouch for.
# Every dropdown in the UI is editable, so this is a convenience list, not
# a whitelist — a newer model id typed by hand still works.
_KNOWN_MODELS: dict[str, tuple[str, ...]] = {
    "glm": ("glm-4.5v", "glm-4v-plus", "glm-4v"),
    # OpenCode Zen fronts several vendors. Only vision-capable families are
    # listed: this pipeline sends 4 images per file, and a text-only model
    # answers confidently from the prompt alone without ever seeing the
    # jewellery. Zen's catalogue moves quickly, so the combo stays editable
    # and Test Connection reports what the key can actually reach.
    "opencode": (
        "claude-sonnet-5", "claude-opus-5", "claude-haiku-4.5",
        "gpt-5.5", "gpt-5.4", "gpt-5.4-mini",
        "gemini-3.5-flash", "gemini-3.1-pro",
    ),
}


# Estimated text prompt + output token counts per classify() call.
# Calibrated against actual `prompt_eval_count` / `eval_count` returned by
# Ollama on the project's real prompt.
_TEXT_PROMPT_TOKENS_DEFAULT = 1500
_OUTPUT_TOKENS_DEFAULT = 500


def estimate_call_cost_usd(
    provider: str,
    model: str,
    n_images: int = 4,
    prompt_tokens: int = _TEXT_PROMPT_TOKENS_DEFAULT,
    output_tokens: int = _OUTPUT_TOKENS_DEFAULT,
) -> float:
    """Predicted USD for a single classify() call.

    Returns 0.0 for any provider that isn't in the pricing table — that
    includes Ollama and any custom provider the user wires in later.
    """
    pricing = _PRICING.get(provider.lower(), {}).get(model)
    if pricing is None:
        return 0.0
    image_tokens = n_images * pricing.image_tokens_per_image
    input_tokens = prompt_tokens + image_tokens
    text_cost = (
        input_tokens / 1_000_000 * pricing.input_usd_per_1m
        + output_tokens / 1_000_000 * pricing.output_usd_per_1m
    )
    flat_image_cost = n_images * pricing.image_usd_each
    return float(text_cost + flat_image_cost)


def actual_call_cost_usd(
    provider: str,
    model: str,
    prompt_tokens: int,
    output_tokens: int,
) -> float:
    """Cost computed from the *actual* token counts returned by the
    provider after a call completes — used to populate the live status
    bar and the run-end summary. Falls back to 0.0 for unknown
    provider/model combos."""
    pricing = _PRICING.get(provider.lower(), {}).get(model)
    if pricing is None:
        return 0.0
    return float(
        prompt_tokens / 1_000_000 * pricing.input_usd_per_1m
        + output_tokens / 1_000_000 * pricing.output_usd_per_1m
    )


def list_priced_models(provider: str) -> list[str]:
    """Models we have pricing data for under `provider`."""
    return sorted(_PRICING.get(provider.lower(), {}).keys())


def list_selectable_models(provider: str) -> list[str]:
    """Model ids to offer in the Settings dropdown for `provider`.

    Union of the priced models and the known-but-unpriced ones, so a
    backend we can't cost (GLM today) still gets a usable picker instead
    of an empty box. The dropdown stays editable either way — this is a
    convenience list, never a whitelist.
    """
    key = provider.lower()
    return sorted(set(_PRICING.get(key, {})) | set(_KNOWN_MODELS.get(key, ())))


def get_pricing(provider: str, model: str) -> _ModelPricing | None:
    """Raw pricing record for display ('$2.50/1M in, $10/1M out')."""
    return _PRICING.get(provider.lower(), {}).get(model)
