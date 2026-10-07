from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from catalog_organizer.snapshotter import stl as stl_mod

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch):
    """Redirect cache_dir to a temp folder so tests don't pollute real cache."""
    def fake_cache_dir() -> Path:
        d = tmp_path / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr("catalog_organizer.core.paths.cache_dir", fake_cache_dir)
    monkeypatch.setattr("catalog_organizer.snapshotter.stl.snapshots_dir",
                        lambda file_id: fake_cache_dir() / "snapshots" / file_id)
    monkeypatch.setattr("catalog_organizer.snapshotter.stl.thumbnails_dir",
                        lambda: fake_cache_dir() / "thumbnails")
    return tmp_path


def test_snapshot_stl_produces_4_pngs_and_thumbnail(isolated_cache: Path):
    results = stl_mod.snapshot_stl(
        FIXTURES / "sample_ring.stl",
        file_id="JCAD-TEST-0001",
        resolution=256,
        thumbnail_size=128,
    )

    for view in ("front", "side", "top", "iso"):
        assert view in results
        assert results[view].exists(), f"missing {view}"

    thumb = results["thumbnail"]
    assert thumb.exists()
    assert thumb.suffix == ".webp"


def test_snapshot_images_are_not_blank(isolated_cache: Path):
    results = stl_mod.snapshot_stl(
        FIXTURES / "sample_ring.stl",
        file_id="JCAD-TEST-0002",
        resolution=256,
        thumbnail_size=128,
    )
    img = Image.open(results["iso"]).convert("RGB")
    pixels = list(img.getdata())
    # A non-blank render contains at least some non-white pixels (the mesh).
    non_white = sum(1 for r, g, b in pixels if (r, g, b) != (255, 255, 255))
    assert non_white > 100, "snapshot appears blank (no non-white pixels)"
