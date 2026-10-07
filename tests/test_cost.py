"""Cost calculations + session-tracker accumulation."""
from __future__ import annotations


def test_ollama_cost_is_zero():
    from catalog_organizer.vlm.cost import estimate_call_cost_usd
    assert estimate_call_cost_usd("ollama", "anything") == 0.0


def test_openai_gpt_4o_estimate_in_sane_range():
    """gpt-4o vision: 4 × 765 image tokens + 1500 prompt + 500 output at
    $2.50/1M in + $10/1M out → ~$0.0162 per call. Allow tolerance for
    table updates but pin order-of-magnitude."""
    from catalog_organizer.vlm.cost import estimate_call_cost_usd
    cost = estimate_call_cost_usd("openai", "gpt-4o", n_images=4)
    assert 0.005 < cost < 0.05, f"got ${cost}"


def test_unknown_model_returns_zero():
    """A model not in the pricing table must not raise — UI shows 'n/a'."""
    from catalog_organizer.vlm.cost import estimate_call_cost_usd
    assert estimate_call_cost_usd("openai", "future-model-v99") == 0.0


def test_actual_cost_uses_returned_token_counts():
    """`actual_call_cost_usd` plugs the provider-reported token counts
    into the same per-1M math — this is what the live status bar shows."""
    from catalog_organizer.vlm.cost import actual_call_cost_usd
    # gpt-4o: 4000 input × $2.50/1M = $0.010, 500 output × $10/1M = $0.005
    cost = actual_call_cost_usd("openai", "gpt-4o", 4000, 500)
    assert abs(cost - 0.015) < 0.001


def test_tracker_accumulates_and_resets():
    from catalog_organizer.core.cost_tracker import CostTracker

    t = CostTracker()
    assert t.snapshot().total_calls == 0
    t.record_call({
        "provider": "openai", "model": "gpt-4o",
        "prompt_eval_count": 4000, "eval_count": 500,
        "cost_usd": 0.015, "total_ms": 800.0,
    })
    t.record_call({
        "provider": "openai", "model": "gpt-4o",
        "prompt_eval_count": 4200, "eval_count": 600,
        "cost_usd": 0.018, "total_ms": 900.0,
    })
    s = t.snapshot()
    assert s.total_calls == 2
    assert abs(s.total_cost_usd - 0.033) < 1e-9
    assert s.total_prompt_tokens == 8200
    assert s.total_output_tokens == 1100

    t.reset()
    assert t.snapshot().total_calls == 0
    assert t.snapshot().total_cost_usd == 0.0


def test_tracker_ignores_garbage_input():
    """Defensive: any provider buggy enough to set last_timings to None
    or a non-dict must not crash the tracker (the GUI thread can't
    afford to die mid-batch)."""
    from catalog_organizer.core.cost_tracker import CostTracker
    t = CostTracker()
    t.record_call(None)
    t.record_call("not a dict")        # type: ignore[arg-type]
    t.record_call({})                  # missing keys but still a dict — counts
    s = t.snapshot()
    # Only the empty dict counts; None / string ignored.
    assert s.total_calls == 1
    assert s.total_cost_usd == 0.0
