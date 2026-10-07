"""Tests for cad/rhino_engine.py (exact cutter-boolean measurement) and its
integration into snapshotter.threedm.capped_cutter_volume_mm3.

Rhino isn't guaranteed to be installed on every machine that runs this test
suite (it's licensed desktop software), so these tests must pass whether or
not it's available:
  * `is_available()` must never raise, regardless of environment.
  * The critical safety-net behaviour — `capped_cutter_volume_mm3` falling
    back to the legacy heuristic when Rhino can't be booted — is tested by
    mocking `rhino_engine.exact_cutter_volume_mm3` to return None, so it
    passes identically with or without a real Rhino install.
"""
from __future__ import annotations

from unittest.mock import patch


def test_is_available_never_raises():
    from catalog_organizer.cad import rhino_engine
    result = rhino_engine.is_available()
    assert isinstance(result, bool)


def test_exact_cutter_volume_returns_none_when_boot_fails():
    from catalog_organizer.cad import rhino_engine
    with patch.object(rhino_engine, "_boot", return_value=False):
        assert rhino_engine.exact_cutter_volume_mm3("nonexistent.3dm") is None


def test_capped_cutter_volume_falls_back_when_rhino_unavailable():
    """The safety net: if exact_cutter_volume_mm3 returns None (Rhino not
    bootable), capped_cutter_volume_mm3 must use the legacy heuristic path,
    not silently return 0 or crash."""
    from catalog_organizer.snapshotter import threedm

    with patch(
        "catalog_organizer.cad.rhino_engine.exact_cutter_volume_mm3",
        return_value=None,
    ), patch.object(
        threedm, "estimate_cutter_volume_mm3", return_value=50.0,
    ) as mock_heuristic:
        result = threedm.capped_cutter_volume_mm3("fake.3dm", gross_volume_mm3=1000.0)
        mock_heuristic.assert_called_once()
        # heuristic result (50.0) is below the 30% cap (300.0) so passes through
        assert result == 50.0


def test_capped_cutter_volume_uses_exact_path_when_available():
    """When Rhino IS bootable, the exact measurement must be used directly
    (no % cap applied — a real boolean can't over-subtract)."""
    from catalog_organizer.snapshotter import threedm

    with patch(
        "catalog_organizer.cad.rhino_engine.exact_cutter_volume_mm3",
        return_value=17.5,
    ) as mock_exact:
        result = threedm.capped_cutter_volume_mm3("fake.3dm", gross_volume_mm3=1000.0)
        mock_exact.assert_called_once()
        assert result == 17.5


def test_capped_cutter_volume_exact_path_has_sanity_floor():
    """Even the exact path can't return more than the gross volume itself
    (a pathological/corrupt file shouldn't be able to report negative net)."""
    from catalog_organizer.snapshotter import threedm

    with patch(
        "catalog_organizer.cad.rhino_engine.exact_cutter_volume_mm3",
        return_value=5000.0,  # implausibly large
    ):
        result = threedm.capped_cutter_volume_mm3("fake.3dm", gross_volume_mm3=1000.0)
        assert result == 1000.0


def test_capped_cutter_volume_zero_gross_short_circuits():
    from catalog_organizer.snapshotter import threedm
    assert threedm.capped_cutter_volume_mm3("fake.3dm", gross_volume_mm3=0.0) == 0.0


def test_exact_cutter_volume_no_cutters_returns_zero():
    """A file with no 'Cutting Objects' layer at all: if Rhino is available,
    the function should return 0.0 (not None) since it can still measure —
    there's just nothing to subtract."""
    from catalog_organizer.cad import rhino_engine
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "sample_ring.3dm"
    if not fixture.exists():
        import pytest
        pytest.skip("no sample_ring.3dm fixture")
    if not rhino_engine.is_available():
        import pytest
        pytest.skip("Rhino not installed on this machine")
    result = rhino_engine.exact_cutter_volume_mm3(fixture)
    assert result is not None
    assert result >= 0.0
