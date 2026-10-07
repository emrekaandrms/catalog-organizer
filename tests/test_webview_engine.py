"""The in-app render engine: scene cache, the hidden Chromium, and the engine choice.

The Chromium tests draw for real (the engine IS a WebGL page), so they need the GPU path the app
needs; they skip only where PyQt6-WebEngine is not installed at all.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from catalog_organizer.webview import scene_cache
from catalog_organizer.webview.scene_cache import scene_key

sys.path.insert(0, str(Path(__file__).parent))
from test_webview import _piece  # noqa: E402  (the same slab-with-a-brilliant the export tests use)

BG = (234, 232, 226)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Scenes and renders go to a throwaway cache, never the user's."""
    monkeypatch.setattr(scene_cache, "cache_dir", lambda: tmp_path)
    import catalog_organizer.render.product_render as pr
    monkeypatch.setattr(pr, "cache_dir", lambda: tmp_path)
    return tmp_path


# ------------------------------------------------------------------ the scene cache

def test_a_scene_is_built_once_and_then_reused(cache):
    first = scene_cache.ensure_scene(None, "S1", category="pendant", arrays=_piece())
    again = scene_cache.ensure_scene(None, "S1", category="pendant", arrays=_piece())
    assert not first.from_cache and again.from_cache
    assert first.key == again.key and first.glb.exists()


def test_a_scene_is_rebuilt_when_asked_or_when_the_category_changes(cache):
    scene_cache.ensure_scene(None, "S2", category="pendant", arrays=_piece())
    forced = scene_cache.ensure_scene(None, "S2", category="pendant", arrays=_piece(), force=True)
    other = scene_cache.ensure_scene(None, "S2", category="ring", arrays=_piece())
    assert not forced.from_cache
    assert not other.from_cache, "the pose depends on the category, so the cached scene is stale"


def test_the_key_follows_the_source_file(tmp_path):
    from catalog_organizer.webview.look import STUDIO_LOOK
    source = tmp_path / "a.stl"
    source.write_bytes(b"one")
    before = scene_key(source, "ring", STUDIO_LOOK, 120_000)
    assert before == scene_key(source, "ring", STUDIO_LOOK, 120_000)
    source.write_bytes(b"two!")                         # the CAD file was saved again
    assert scene_key(source, "ring", STUDIO_LOOK, 120_000) != before
    assert scene_key(source, "pendant", STUDIO_LOOK, 120_000) != scene_key(
        source, "ring", STUDIO_LOOK, 120_000)


def test_a_scene_carries_both_catalogue_views_in_the_viewers_frame(cache):
    scene = scene_cache.ensure_scene(None, "S3", category="pendant", arrays=_piece())
    assert set(scene.views) == {"front", "iso"}
    assert scene.stones == 1
    assert np.allclose(scene.rotation @ scene.rotation.T, np.eye(3), atol=1e-6), "a pure rotation"
    # the iso view IS the canonical camera: +Z, looking at the origin
    assert np.allclose(scene.views["iso"]["dir"], [0.0, 0.0, 1.0], atol=1e-5)
    # a hand-set angle in the file's frame maps through the same rotation the scene was built with
    assert np.allclose(scene.canonical(scene.rotation.T @ [0.0, 0.6, 0.8]), [0.0, 0.6, 0.8], atol=1e-5)   # the scene stores 6 digits


# ------------------------------------------------------------------ which engine draws

def test_scripts_and_tests_without_a_running_chromium_use_the_vtk_engine(monkeypatch):
    import catalog_organizer.render.engine as chooser
    import catalog_organizer.render.product_render as pr
    from catalog_organizer.webview import service

    monkeypatch.setattr(service, "current", lambda: None)
    monkeypatch.setattr(pr, "load_and_render", lambda *a, **k: "vtk drew it")
    assert chooser.load_and_render(Path("x.3dm"), file_id="F", metal_key="m", stone_key="s") == "vtk drew it"


def test_the_app_uses_the_web_engine_and_the_environment_can_force_vtk(monkeypatch):
    import catalog_organizer.render.engine as chooser
    import catalog_organizer.render.product_render as pr
    from catalog_organizer.webview import engine as web_engine
    from catalog_organizer.webview import service

    monkeypatch.setattr(service, "current", lambda: object())
    monkeypatch.setattr(web_engine, "render_views", lambda *a, **k: "web drew it")
    monkeypatch.setattr(pr, "load_and_render", lambda *a, **k: "vtk drew it")

    def call():
        return chooser.load_and_render(Path("x.3dm"), file_id="F", metal_key="m",
                                       stone_key="s", resolution=500, category="ring")

    assert call() == "web drew it"
    monkeypatch.setenv("CATALOG_RENDER_ENGINE", "vtk")
    assert call() == "vtk drew it"


# ------------------------------------------------------------------ the hidden Chromium

pytest.importorskip("PyQt6.QtWebEngineWidgets")


@pytest.fixture
def engine(qtbot, cache, monkeypatch):
    """The real service and a scene builder that uses the synthetic piece instead of a CAD file."""
    from catalog_organizer.webview import engine as web_engine
    from catalog_organizer.webview import service

    service.install()
    real = scene_cache.ensure_scene
    monkeypatch.setattr(scene_cache, "ensure_scene",
                        lambda source, file_id, **kw: real(None, file_id, arrays=_piece(), **kw))
    yield web_engine
    service.uninstall()


def _render(qtbot, engine_module, **kwargs):
    """Called from a worker thread, as the Render tab and the PDF catalogue call it."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: engine_module.render_views(Path("x.3dm"), **kwargs))
        qtbot.waitUntil(future.done, timeout=90_000)
        return future.result()


def _pixels(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB")).astype(float)


@pytest.mark.gpu
def test_the_engine_draws_both_views_as_catalogue_pictures(qtbot, engine, cache):
    result = _render(qtbot, engine, file_id="E1", metal_key="yellow_gold", stone_key="white",
                     resolution=400, category="pendant")
    assert result.engine == "web" and result.had_stones
    assert set(result.views) == {"front", "iso"}
    for path in result.views.values():
        px = _pixels(path)
        assert px.shape[:2] == (400, 400)
        assert tuple(px[2, 2]) == BG, "the flat catalogue background"
        piece = np.abs(px - np.array(BG)).sum(axis=2) > 30
        assert 0.02 < piece.mean() < 0.9, f"the piece should fill part of the frame ({piece.mean():.3f})"


@pytest.mark.gpu
def test_the_metal_colour_reaches_the_picture(qtbot, engine, cache):
    gold = _render(qtbot, engine, file_id="E2", metal_key="yellow_gold", stone_key="white", resolution=300)
    rose = _render(qtbot, engine, file_id="E2", metal_key="rose_gold", stone_key="white", resolution=300)
    a, b = _pixels(gold.views["iso"]), _pixels(rose.views["iso"])
    piece = np.abs(a - np.array(BG)).sum(axis=2) > 30
    assert piece.any()
    assert (a[piece].mean(0) - b[piece].mean(0))[2] < -4, "rose gold has more blue than yellow gold"


@pytest.mark.gpu
def test_a_picture_is_drawn_once_and_then_read_from_the_cache(qtbot, engine, cache, monkeypatch):
    from catalog_organizer.webview import service

    first = _render(qtbot, engine, file_id="E3", metal_key="yellow_gold", stone_key="white", resolution=256)
    calls = []
    monkeypatch.setattr(service.WebRenderService, "capture",
                        lambda self, *a, **k: calls.append(1) or {})
    again = _render(qtbot, engine, file_id="E3", metal_key="yellow_gold", stone_key="white", resolution=256)
    assert calls == [] and again.views == first.views
    assert first.views["iso"].name.endswith("_256_iso.png"), "a different size is a different file"


@pytest.mark.gpu
def test_a_hand_set_angle_changes_the_picture_and_gets_its_own_file(qtbot, engine, cache):
    auto = _render(qtbot, engine, file_id="E4", metal_key="yellow_gold", stone_key="white", resolution=300)
    hand = _render(qtbot, engine, file_id="E4", metal_key="yellow_gold", stone_key="white", resolution=300,
                   camera={"front": ((1.0, 0.0, 0.2), (0.0, 0.0, 1.0))})
    assert hand.views["front"] != auto.views["front"], "must not share a cache slot with the automatic one"
    assert hand.views["iso"] == auto.views["iso"], "the other view keeps the automatic angle"
    assert not np.array_equal(_pixels(hand.views["front"]), _pixels(auto.views["front"]))
