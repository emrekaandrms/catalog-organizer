"""Measurement sanity-guard tests for _sanitize_measurements (D.13c).

When the VLM mis-classifies a flat pendant as a ring, the PCA-based ring
algorithm returns a near-zero inner diameter. We zero out the ring_* fields
rather than letting that nonsense into the catalog.
"""
from __future__ import annotations

from catalog_organizer.core.schemas import Measurements
from catalog_organizer.orchestrator.pipeline import _sanitize_measurements


def _make_ring_measurements(inner_diameter_mm: float) -> Measurements:
    return Measurements(
        bbox_width_mm=29.1, bbox_height_mm=27.3, bbox_depth_mm=5.8,
        volume_mm3=2359.0, geometry_source="rhino",
        ring_inner_diameter_mm=inner_diameter_mm,
        ring_inner_circumference_mm=inner_diameter_mm * 3.14159,
        ring_band_width_mm=2.0,
    )


def test_sanitizer_passes_through_valid_ring():
    m = _make_ring_measurements(inner_diameter_mm=16.6)
    out, was_sanitized, reason = _sanitize_measurements(m, "ring")
    assert was_sanitized is False
    assert reason is None
    assert out.ring_inner_diameter_mm == 16.6


def test_sanitizer_zeros_degenerate_tiny_inner_diameter():
    """A flat pendant mis-classified as a ring returned 0.9mm — clearly bogus."""
    m = _make_ring_measurements(inner_diameter_mm=0.9)
    out, was_sanitized, reason = _sanitize_measurements(m, "ring")
    assert was_sanitized is True
    assert "0.9mm" in reason
    assert out.ring_inner_diameter_mm is None
    assert out.ring_inner_circumference_mm is None
    assert out.ring_band_width_mm is None
    # bbox + volume should survive untouched
    assert out.bbox_width_mm == 29.1
    assert out.volume_mm3 == 2359.0


def test_sanitizer_zeros_small_but_not_zero_inner_diameter():
    """5 mm is physically impossible for any real ring — drop it."""
    m = _make_ring_measurements(inner_diameter_mm=5.0)
    out, was_sanitized, _ = _sanitize_measurements(m, "ring")
    assert was_sanitized is True
    assert out.ring_inner_diameter_mm is None


def test_sanitizer_keeps_small_but_plausible_kids_ring():
    """A small but plausible 10 mm ring (kids / toe) must survive."""
    m = _make_ring_measurements(inner_diameter_mm=10.0)
    out, was_sanitized, _ = _sanitize_measurements(m, "ring")
    assert was_sanitized is False
    assert out.ring_inner_diameter_mm == 10.0


def test_sanitizer_zeros_oversized_inner_diameter():
    """Inner diameter > 40mm is not a finger ring; null fields."""
    m = _make_ring_measurements(inner_diameter_mm=120.0)
    out, was_sanitized, reason = _sanitize_measurements(m, "ring")
    assert was_sanitized is True
    assert out.ring_inner_diameter_mm is None


def test_sanitizer_skips_non_ring_categories():
    """Pendant measurements aren't bounded by these rules."""
    m = Measurements(
        bbox_width_mm=20.0, bbox_height_mm=15.0, bbox_depth_mm=3.0,
        volume_mm3=321.0, geometry_source="rhino",
    )
    out, was_sanitized, reason = _sanitize_measurements(m, "pendant")
    assert was_sanitized is False
    assert out is m
