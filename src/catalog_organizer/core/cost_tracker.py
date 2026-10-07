"""Session-wide cost & usage tracker.

Each VLM call (regardless of provider) feeds its `last_timings` dict in
here via `record_call(...)`. The status bar reads the current session
total to display "$X.XX  · N calls" live, and the Process panel pops a
summary dialog at the end of a batch.

This is a Qt-aware singleton — `tracker_singleton()` returns the shared
instance and exposes a `changed` signal so widgets can react. Headless
(test) code can read the totals via `snapshot()` without Qt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal


@dataclass
class CostSample:
    """One data point per successful classify() call."""
    provider: str
    model: str
    prompt_tokens: int
    output_tokens: int
    cost_usd: float
    wall_ms: float


@dataclass
class CostSnapshot:
    """Immutable view of the current totals — safe to ship across threads."""
    total_calls: int = 0
    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_wall_ms: float = 0.0
    samples: list[CostSample] = field(default_factory=list)
    provider: str = ""
    model: str = ""

    @property
    def avg_cost_usd(self) -> float:
        return (self.total_cost_usd / self.total_calls) if self.total_calls else 0.0


class CostTracker(QObject):
    """Qt-aware accumulator. Emits `changed` after every record/reset so
    the status bar can re-render without polling."""

    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._snap = CostSnapshot()

    # ── recording ──────────────────────────────────────────────────────────

    def record_call(self, timings: dict | None) -> None:
        """Add one classify() outcome. Accepts the `last_timings` dict from
        any provider — silently ignores None / malformed input so a buggy
        provider can never crash the UI."""
        if not isinstance(timings, dict):
            return
        sample = CostSample(
            provider=str(timings.get("provider", "")),
            model=str(timings.get("model", "")),
            prompt_tokens=int(timings.get("prompt_eval_count") or 0),
            output_tokens=int(timings.get("eval_count") or 0),
            cost_usd=float(timings.get("cost_usd") or 0.0),
            wall_ms=float(timings.get("total_ms") or 0.0),
        )
        s = self._snap
        s.total_calls += 1
        s.total_cost_usd += sample.cost_usd
        s.total_prompt_tokens += sample.prompt_tokens
        s.total_output_tokens += sample.output_tokens
        s.total_wall_ms += sample.wall_ms
        # `provider/model` track the most recent — useful for status bar text.
        s.provider = sample.provider or s.provider
        s.model    = sample.model    or s.model
        s.samples.append(sample)
        self.changed.emit()

    def reset(self) -> None:
        """Zero the totals. Use at the start of a fresh batch so per-run
        cost dialogs are accurate."""
        self._snap = CostSnapshot()
        self.changed.emit()

    # ── reading ────────────────────────────────────────────────────────────

    def snapshot(self) -> CostSnapshot:
        return self._snap


# ── module-level singleton ──────────────────────────────────────────────────

_INSTANCE: Optional[CostTracker] = None


def tracker_singleton() -> CostTracker:
    """Return the process-wide CostTracker. Lazy so importing the module
    inside a test (where Qt may not be initialised) is cheap."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = CostTracker()
    return _INSTANCE
