from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

# The render engine is the web viewer in an embedded Chromium. Qt wants that announced before the
# first QApplication exists (pytest-qt creates it lazily), exactly as app.py does at startup.
try:
    from PyQt6.QtCore import QCoreApplication, Qt

    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    import PyQt6.QtWebEngineWidgets  # noqa: F401
except ImportError:                       # no PyQt6-WebEngine: the engine tests skip themselves
    pass

from catalog_organizer.core.schemas import (
    CatalogRecord,
    Measurements,
    MetalWeights,
    SpruInfo,
    StoneSummary,
)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    return project_root / "config"


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


def make_catalog_record(
    file_id: str = "JCAD-000000001",
    main_category: str = "ring",
    tags: list[str] | None = None,
    **overrides,
) -> CatalogRecord:
    """Build a minimal valid CatalogRecord for tests."""
    now = datetime.now(tz=timezone.utc)
    base: dict = dict(
        file_id=file_id,
        source_path=f"D:\\archive\\{file_id}.stl",
        file_extension=".stl",
        sha256="a" * 64,
        main_category=main_category,
        main_category_confidence=0.95,
        subcategory="solitaire",
        subcategory_confidence=0.90,
        controlled_tags=tags or ["ring", "band", "polished", "round_form", "single_piece"],
        tag_confidence=0.85,
        polished_status="polished",
        measurements=Measurements(
            bbox_width_mm=20.0,
            bbox_height_mm=22.0,
            bbox_depth_mm=8.0,
            volume_mm3=3500.0,
            geometry_source="trimesh",
        ),
        metal_weights=MetalWeights(
            silver_925_g=36.3,
            gold_10k_yellow_g=40.5,
            gold_10k_white_g=38.5,
            gold_14k_yellow_g=45.7,
            gold_14k_white_g=43.9,
            gold_18k_yellow_g=54.5,
            gold_18k_white_g=51.3,
            platinum_g=70.5,
        ),
        weight_source="stl_volume_density",
        weight_confidence="medium",
        stone_summary=StoneSummary(
            status="polished",
            center_stone=None,
            side_stones=[],
            total_estimated_carat=0.0,
        ),
        sprue=SpruInfo(
            detected=False,
            source="none",
            confidence=0.95,
            estimated_volume_mm3=None,
        ),
        scale_warning=False,
        scale_warning_reason=None,
        snapshot_paths=[],
        state="final",
        needs_manual_review=False,
        review_reason=None,
        manual_correction_applied=False,
        processing_log_ref="evt-001",
        processed_at=now,
    )
    base.update(overrides)
    return CatalogRecord(**base)


@pytest.fixture
def record_factory():
    return make_catalog_record


def pytest_collection_modifyitems(config, items):
    """Tests marked `gpu` draw in an embedded Chromium and need WebGL2; CI runners without a GPU set
    CATALOG_ORGANIZER_NO_GPU=1 to skip them."""
    import os

    if os.environ.get("CATALOG_ORGANIZER_NO_GPU") != "1":
        return
    skip = pytest.mark.skip(reason="CATALOG_ORGANIZER_NO_GPU=1: no GPU on this machine")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_chromium_without_gpu(monkeypatch):
    """With CATALOG_ORGANIZER_NO_GPU=1 (CI runners) no test may start the embedded Chromium: on a
    machine without a GPU it takes the whole interpreter down. The Render tab then shows its
    "PyQt6-WebEngine needed" label instead of a player, and a test that tries anyway fails loudly
    here, on every machine, instead of crashing only on CI."""
    import os

    if os.environ.get("CATALOG_ORGANIZER_NO_GPU") != "1":
        return
    try:
        from catalog_organizer.gui.panels import render_panel
        from catalog_organizer.webview import service
    except ImportError:
        return
    monkeypatch.setattr(render_panel, "_create_player", lambda parent: None)

    def refuse(*args, **kwargs):
        raise AssertionError("this test would start the embedded Chromium; mark it @pytest.mark.gpu")

    monkeypatch.setattr(service.WebRenderService, "_ensure_view", refuse)
