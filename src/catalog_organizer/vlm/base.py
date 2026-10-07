"""Shared types for VLM providers.

The pipeline only knows about `VLMProvider` — a thin protocol every
provider (Ollama, OpenAI, MiniMax, …) implements. Adding a new provider
means writing a class with two methods (`classify`, `estimate_cost_usd`)
and one attribute (`last_timings`), then wiring it into the factory in
`vlm.providers.__init__`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from catalog_organizer.core.schemas import VLMResult


@runtime_checkable
class VLMProvider(Protocol):
    """Contract every VLM provider must satisfy.

    `classify` is the only mandatory request. `last_timings` carries per-
    call diagnostic data (provider-specific keys are fine; `total_ms`
    should always exist). `estimate_cost_usd` lets the UI show the
    expected dollar cost before a batch starts.
    """

    last_timings: dict | None

    def classify(
        self,
        snapshot_paths: list[Path],
        second_pass: dict | None = None,
    ) -> VLMResult:
        """Submit the snapshot images + classification prompt to the
        underlying model and return a parsed `VLMResult`. Raises
        `VLMHTTPError` on transport failure and `VLMParseError` on a bad
        model response (definitions in `vlm.parser`)."""
        ...

    def estimate_cost_usd(self, n_images: int = 4) -> float:
        """Estimated USD cost for one classify() call. Returns 0.0 for
        local providers (Ollama) — they pay in compute, not dollars."""
        ...

    @property
    def display_name(self) -> str:
        """Short label for status bar / settings, e.g. ``OpenAI (gpt-4o)``."""
        ...
