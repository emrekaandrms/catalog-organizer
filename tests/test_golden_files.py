"""Golden-file regression tests on real workshop CAD files.

The expected values live in `test_case/expected.json`, captured with:

    py -3.12 -m catalog_organizer.app analyze FILE1 FILE2 --golden test_case/expected.json

Each entry maps an input path → the flat analysis dict. This test re-runs
`analyze_file` on every input that still exists on disk and asserts the
production-critical numbers stay inside tight tolerances. If a referenced
file is missing (e.g. on another machine) it is skipped, never failed —
golden tests pin behaviour on the author's workstation without breaking CI.

Tolerances (chosen against workshop-scale ground truthing, see the development log):
  dimensions ±0.1 mm · ring bore ±0.5 mm · volumes ±2 % · weights ±2 %
  stone count exact · total carat ±5 % · sprue detected exact · sprue vol ±10 %
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent.parent / "test_case" / "expected.json"


def _load_cases() -> list[tuple[str, dict]]:
    if not GOLDEN.exists():
        return []
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    return [(p, exp) for p, exp in data.items()]


_CASES = _load_cases()


def _rel(actual: float, expected: float, tol: float) -> bool:
    if expected == 0:
        return abs(actual) < 1e-9 or abs(actual) < tol
    return abs(actual - expected) / abs(expected) <= tol


@pytest.mark.skipif(not _CASES, reason="test_case/expected.json not present")
@pytest.mark.parametrize("path,expected", _CASES, ids=[Path(p).name for p, _ in _CASES])
def test_golden_file_analysis(path: str, expected: dict):
    src = Path(path)
    if not src.exists():
        pytest.skip(f"input file not on this machine: {src}")

    from catalog_organizer.orchestrator.analyze import analyze_file, to_flat_dict

    actual = to_flat_dict(analyze_file(src))

    assert actual["error"] == "", f"analysis errored: {actual['error']}"

    # Dimensions ±0.1 mm
    for key in ("bbox_width_mm", "bbox_height_mm", "bbox_depth_mm"):
        assert abs(actual[key] - expected[key]) <= 0.1, (
            f"{key}: {actual[key]} vs expected {expected[key]}"
        )

    # Ring bore ±0.5 mm (and presence must match)
    exp_bore = expected.get("ring_inner_diameter_mm", "")
    act_bore = actual.get("ring_inner_diameter_mm", "")
    if exp_bore == "":
        assert act_bore == "", f"unexpected ring bore appeared: {act_bore}"
    else:
        assert act_bore != "", "ring bore disappeared"
        assert abs(float(act_bore) - float(exp_bore)) <= 0.5

    # Volumes ±2 %
    for key in ("gross_volume_mm3", "net_metal_volume_mm3"):
        assert _rel(actual[key], expected[key], 0.02), (
            f"{key}: {actual[key]} vs expected {expected[key]} (>2%)"
        )

    # Metal weights ±2 % (every alloy present in golden)
    for key, exp_val in expected.items():
        if key.startswith("metal_") and isinstance(exp_val, (int, float)):
            assert key in actual, f"{key} missing from analysis output"
            assert _rel(actual[key], exp_val, 0.02), (
                f"{key}: {actual[key]} vs expected {exp_val} (>2%)"
            )

    # Stones: count exact, carat ±5 %
    assert actual["stone_count"] == expected["stone_count"], (
        f"stone_count: {actual['stone_count']} vs {expected['stone_count']}"
    )
    assert _rel(actual["stone_total_carat"], expected["stone_total_carat"], 0.05)

    # Sprue: detection exact, volume ±10 %
    assert actual["sprue_detected"] == expected["sprue_detected"]
    if expected["sprue_detected"]:
        assert _rel(actual["sprue_volume_mm3"], expected["sprue_volume_mm3"], 0.10)
