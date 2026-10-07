from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import orjson
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import QWidget  # noqa: E402

from catalog_organizer.db.connection import Database  # noqa: E402
from catalog_organizer.db.sync import sync_from_jsonl  # noqa: E402
from catalog_organizer.render import materials as mat  # noqa: E402
from catalog_organizer.webview.scene_cache import Scene  # noqa: E402


class FakePlayer(QWidget):
    """Stands in for the web player: records what the panel asks of it. No Chromium, no GPU."""

    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calls: list[tuple] = []
        self.state = {"dir": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0]}

    def load(self, url, then=None):
        self.calls.append(("load", url))

    def clear(self):
        self.calls.append(("clear",))

    def set_metal(self, key):
        self.calls.append(("metal", key))

    def set_stone(self, key):
        self.calls.append(("stone", key))

    def go_to_view(self, name):
        self.calls.append(("view", name))

    def set_camera(self, direction, up):
        self.calls.append(("camera", tuple(direction), tuple(up)))

    def camera_state(self, callback):
        callback(self.state)

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]


class FakeService:
    def viewer_url(self, scene, *, hosted=True):
        return f"http://127.0.0.1:1/viewer.html?p={scene.file_id}&k={scene.key}&host=1"


def _scene(file_id, stones=1, rotation=None):
    return Scene(file_id=file_id, glb=Path("x.glb"), key="k1",
                 rotation=np.eye(3) if rotation is None else rotation,
                 stones=stones, from_cache=True,
                 views={"front": {"dir": [0.0, 0.5, 0.8660254], "up": [0.0, 1.0, 0.0]},
                        "iso": {"dir": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0]}})


@pytest.fixture
def env(tmp_path: Path, monkeypatch, record_factory):
    """A panel backed by real DB records, a stub scene builder and a stub player.

    Preparing a scene costs seconds and the player needs a GPU context; what these tests are
    about is the panel's own logic: which product, which colours, which view, and whether what
    the player shows agrees with the drop-downs.
    """
    prepared: list[tuple[str, str | None]] = []
    players: list[FakePlayer] = []
    scenes: dict = {}                      # per-test overrides: file_id -> Scene

    asked: list[dict] = []                 # every scene request, with the stone-placement flag

    def fake_ensure_scene(source, file_id, *, category=None, force=False, **kw):
        time.sleep(0.12)                   # long enough for a second pick to arrive mid-flight
        prepared.append((file_id, category))
        asked.append({"file_id": file_id, "place_stones": kw.get("place_stones", False)})
        return scenes.get(file_id) or _scene(file_id)

    fake_ensure_scene.asked = asked

    import catalog_organizer.gui.panels.render_panel as rp
    import catalog_organizer.webview.scene_cache as sc
    import catalog_organizer.webview.service as service

    def make_player(parent):
        player = FakePlayer(parent)
        players.append(player)
        return player

    monkeypatch.setattr(sc, "ensure_scene", fake_ensure_scene)
    monkeypatch.setattr(rp, "_create_player", make_player)
    monkeypatch.setattr(service, "current", lambda: FakeService())

    def _make(records):
        jsonl = tmp_path / "m.jsonl"
        with jsonl.open("wb") as fh:
            for rec in records:
                fh.write(orjson.dumps(rec.model_dump(mode="json")) + b"\n")
        db = Database(tmp_path / "c.db")
        sync_from_jsonl(db.conn, jsonl)
        return db, prepared, players, scenes

    return _make


def _records(record_factory, tmp_path: Path, n: int = 3):
    """Records whose source_path actually exists — the panel refuses to open a product whose
    CAD file has moved, and says so instead."""
    out = []
    for i in range(n):
        src = tmp_path / f"p{i}.3dm"
        src.write_bytes(b"not really a 3dm")
        out.append(record_factory(file_id=f"JCAD-{i:09d}",
                                  source_path=str(src),
                                  main_category="ring" if i else "pendant"))
    return out


def _settle(qtbot, panel, timeout=5000):
    qtbot.waitUntil(lambda: panel._worker is None and not panel._pending, timeout=timeout)


def _badges(panel):
    return [panel._badges.itemAt(i).widget().text()
            for i in range(panel._badges.count())
            if panel._badges.itemAt(i).widget() is not None]


def _panel(qtbot, db):
    from catalog_organizer.gui.panels.render_panel import RenderPanel
    panel = RenderPanel(db)
    qtbot.addWidget(panel)
    return panel


def test_panel_lists_catalogue_products(qtbot, env, record_factory, tmp_path):
    db, *_ = env(_records(record_factory, tmp_path))
    panel = _panel(qtbot, db)
    assert panel.product_ids() == ["JCAD-000000000", "JCAD-000000001", "JCAD-000000002"]
    db.close()


def test_filter_narrows_the_list(qtbot, env, record_factory, tmp_path):
    db, *_ = env(_records(record_factory, tmp_path))
    panel = _panel(qtbot, db)
    panel._filter.setText("pendant")
    panel.reload_products()
    assert panel.product_ids() == ["JCAD-000000000"]
    db.close()


def test_every_material_is_offered(qtbot, env, record_factory, tmp_path):
    db, *_ = env(_records(record_factory, tmp_path))
    panel = _panel(qtbot, db)
    assert panel._metal.count() == len(mat.METALS)
    assert panel._stone.count() == len(mat.STONES)
    assert panel.current_colours() == (mat.DEFAULT_METAL, mat.DEFAULT_STONE)
    db.close()


def test_selecting_a_product_opens_it_in_the_player(qtbot, env, record_factory, tmp_path):
    db, prepared, players, _ = env(_records(record_factory, tmp_path))
    panel = _panel(qtbot, db)
    panel.select_product("JCAD-000000001")
    _settle(qtbot, panel)
    # prepared with the record's category, so the pose follows the catalogue's category rule
    assert prepared[-1] == ("JCAD-000000001", "ring")
    assert len(players) == 2, "KARŞIDAN and ÇAPRAZ: one live player each"
    for player in players:
        url = player.of("load")[-1][1]
        assert "p=JCAD-000000001" in url and "host=1" in url
        assert f"metal={mat.DEFAULT_METAL}" in url and f"stone={mat.DEFAULT_STONE}" in url
    assert panel.save_button.isEnabled()
    assert all(b.isEnabled() for b in panel.save_angle_buttons.values())
    db.close()


def test_changing_a_colour_updates_the_player_without_rebuilding_anything(
    qtbot, env, record_factory, tmp_path
):
    db, prepared, players, _ = env(_records(record_factory, tmp_path))
    panel = _panel(qtbot, db)
    panel.select_product("JCAD-000000000")
    _settle(qtbot, panel)
    built = len(prepared)
    panel._set_current(panel._stone, "red")
    panel._set_current(panel._metal, "rose_gold")
    for player in players:
        assert player.of("stone")[-1] == ("stone", "red")
        assert player.of("metal")[-1] == ("metal", "rose_gold")
        assert len(player.of("load")) == 1, "…nor reload the player"
    assert len(prepared) == built, "a colour change must not prepare the scene again"
    db.close()


def test_rapid_product_picks_end_on_the_last_one(qtbot, env, record_factory, tmp_path):
    """The second pick arrives while the first is still preparing. It must neither be dropped
    (the player would show one product while the list claims another) nor start a second thread
    alongside the first."""
    db, prepared, players, _ = env(_records(record_factory, tmp_path))
    panel = _panel(qtbot, db)
    panel.select_product("JCAD-000000000")
    panel.select_product("JCAD-000000001")
    _settle(qtbot, panel)
    assert prepared[-1][0] == "JCAD-000000001"
    assert all("p=JCAD-000000001" in p.of("load")[-1][1] for p in players)
    assert panel.current_record().file_id == "JCAD-000000001"
    db.close()


def test_missing_source_file_is_reported_not_opened(qtbot, env, record_factory, tmp_path):
    rec = record_factory(file_id="GONE", source_path=str(tmp_path / "yok.3dm"))
    db, prepared, players, _ = env([rec])
    panel = _panel(qtbot, db)
    panel.select_product("GONE")
    assert prepared == []
    assert "bulunamadı" in panel._status.text()
    assert not panel.save_button.isEnabled()
    db.close()


def test_a_scene_failure_surfaces_in_the_status_line(
    qtbot, env, record_factory, tmp_path, monkeypatch
):
    db, _prepared, players, _ = env(_records(record_factory, tmp_path, n=1))

    import catalog_organizer.webview.scene_cache as sc

    def boom(*a, **k):
        raise ValueError("mesh bozuk")

    monkeypatch.setattr(sc, "ensure_scene", boom)
    panel = _panel(qtbot, db)
    panel.select_product("JCAD-000000000")
    _settle(qtbot, panel)
    assert "mesh bozuk" in panel._status.text()
    assert not panel.save_button.isEnabled()
    db.close()


def test_warns_when_analysis_says_stones_but_the_scene_has_none(
    qtbot, env, record_factory, tmp_path
):
    """The stone colour silently doing nothing is worse than saying so."""
    from catalog_organizer.core.schemas import StoneSummary

    src = tmp_path / "s.3dm"
    src.write_bytes(b"x")
    rec = record_factory(
        file_id="STONELESS", source_path=str(src),
        stone_summary=StoneSummary(status="stone", center_stone=None,
                                   side_stones=[], total_estimated_carat=0.0))
    db, _prepared, _players, scenes = env([rec])
    scenes["STONELESS"] = _scene("STONELESS", stones=0)
    panel = _panel(qtbot, db)
    panel.select_product("STONELESS")
    _settle(qtbot, panel)
    assert any("taş geometrisi yok" in t for t in _badges(panel)), _badges(panel)
    db.close()


# ---------------------------------------------------------------- görünümler ve açı

def test_both_catalogue_views_are_shown_side_by_side(qtbot, env, record_factory, tmp_path):
    """The catalogue page prints KARŞIDAN and ÇAPRAZ next to each other; so does this tab, each as
    its own live player on the matching view."""
    db, _prepared, players, _ = env(_records(record_factory, tmp_path, n=1))
    panel = _panel(qtbot, db)
    panel.select_product("JCAD-000000000")
    _settle(qtbot, panel)
    front, iso = players
    assert front.of("view")[-1] == ("view", "front")
    assert iso.of("view")[-1] == ("view", "iso")
    db.close()


def test_reset_button_disabled_until_an_override_exists(qtbot, env, record_factory, tmp_path):
    db, *_ = env(_records(record_factory, tmp_path, n=1))
    panel = _panel(qtbot, db)
    panel.select_product("JCAD-000000000")
    _settle(qtbot, panel)
    assert not panel.reset_angle_button.isEnabled()
    db.close()


def test_a_saved_angle_is_what_the_player_shows_for_that_view(qtbot, env, record_factory, tmp_path):
    """The point of the feature: a saved angle must reach the viewer, not sit in the database."""
    from catalog_organizer.db.camera_overrides import set_camera_override

    db, _prepared, players, _ = env(_records(record_factory, tmp_path, n=1))
    set_camera_override(db.conn, "JCAD-000000000", "iso", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    panel = _panel(qtbot, db)
    panel.select_product("JCAD-000000000")
    _settle(qtbot, panel)
    front, iso = players
    assert iso.of("camera")[-1] == ("camera", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert not iso.of("view"), "the saved angle replaces the automatic one for that view"
    assert front.of("view")[-1] == ("view", "front") and not front.of("camera")
    assert panel.reset_angle_button.isEnabled()
    assert any("elle ayarlı" in t for t in _badges(panel)), _badges(panel)
    db.close()


def test_saving_the_current_angle_stores_it_in_the_files_own_frame(
    qtbot, env, record_factory, tmp_path
):
    """The player orbits in the viewer's camera frame; the catalogue (and the database) work in the
    file's frame. The conversion is the transpose of the scene's rotation: with a quarter turn
    about Z, camera-frame +X is the file's -Y."""
    from catalog_organizer.db.camera_overrides import get_camera_overrides

    quarter_turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # file -> camera
    db, _prepared, players, scenes = env(_records(record_factory, tmp_path, n=1))
    scenes["JCAD-000000000"] = _scene("JCAD-000000000", rotation=quarter_turn)
    panel = _panel(qtbot, db)
    panel.select_product("JCAD-000000000")
    _settle(qtbot, panel)

    players[0].state = {"dir": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0]}    # the KARŞIDAN player
    players[1].state = {"dir": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0]}    # the ÇAPRAZ one: untouched
    panel._save_angle("front")

    overrides = get_camera_overrides(db.conn, "JCAD-000000000")
    assert set(overrides) == {"front"}, "saving one pane must not touch the other view"
    saved = overrides["front"]
    assert np.allclose(saved.direction, quarter_turn.T @ [1.0, 0.0, 0.0])
    assert np.allclose(saved.direction, [0.0, -1.0, 0.0])
    assert np.allclose(saved.up, [0.0, 0.0, 1.0])
    assert panel.reset_angle_button.isEnabled()
    assert any("elle ayarlı" in t for t in _badges(panel)), _badges(panel)
    db.close()


def test_reset_clears_the_override_and_returns_to_the_automatic_view(
    qtbot, env, record_factory, tmp_path
):
    from catalog_organizer.db.camera_overrides import get_camera_overrides, set_camera_override

    db, _prepared, players, _ = env(_records(record_factory, tmp_path, n=1))
    set_camera_override(db.conn, "JCAD-000000000", "iso", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    panel = _panel(qtbot, db)
    panel.select_product("JCAD-000000000")
    _settle(qtbot, panel)
    assert panel.reset_angle_button.isEnabled()

    panel._on_reset_angles()

    assert get_camera_overrides(db.conn, "JCAD-000000000") == {}
    assert not panel.reset_angle_button.isEnabled()
    assert players[1].of("view")[-1] == ("view", "iso")
    db.close()


# ---------------------------------------------------------------- PNG

def test_png_save_writes_both_views_in_the_chosen_colours(
    qtbot, env, record_factory, tmp_path, monkeypatch
):
    from PyQt6.QtGui import QColor, QImage
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    import catalog_organizer.render.engine as engine
    from catalog_organizer.render.product_render import RenderResult

    db, *_ = env(_records(record_factory, tmp_path, n=1))
    seen: dict = {}
    pngs = {}
    for name in ("front", "iso"):
        path = tmp_path / f"{name}.png"
        image = QImage(8, 8, QImage.Format.Format_RGB32)
        image.fill(QColor("#C49732"))
        image.save(str(path))
        pngs[name] = path

    def fake_render(source, *, file_id, metal_key, stone_key, **kwargs):
        seen.update(file_id=file_id, metal=metal_key, stone=stone_key, size=kwargs["resolution"],
                    category=kwargs["category"])
        return RenderResult(file_id, metal_key, stone_key, pngs, True, "web")

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(engine, "load_and_render", fake_render)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(out))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)

    panel = _panel(qtbot, db)
    panel.select_product("JCAD-000000000")
    _settle(qtbot, panel)
    panel._set_current(panel._metal, "rose_gold")
    panel._set_current(panel._stone, "blue")
    panel._quality.setCurrentText("Hızlı (700 px)")
    panel.save_button.click()
    qtbot.waitUntil(lambda: panel._save_worker is None, timeout=5000)

    assert seen == {"file_id": "JCAD-000000000", "metal": "rose_gold", "stone": "blue",
                    "size": 700, "category": "pendant"}
    assert sorted(p.name for p in out.iterdir()) == [
        "JCAD-000000000_rose_gold_blue_front.png", "JCAD-000000000_rose_gold_blue_iso.png"]
    db.close()


def test_stones_placed_in_seats_are_labelled_as_such(qtbot, env, record_factory, tmp_path):
    """A stone read from the file and a stone the program put in a detected seat are not the same
    claim; the user must be able to tell which picture they are looking at."""
    db, _prepared, _players, scenes = env(_records(record_factory, tmp_path, n=2))
    scenes["JCAD-000000000"] = Scene(file_id="JCAD-000000000", glb=Path("x.glb"), key="k1",
                                     rotation=np.eye(3), stones=7, from_cache=True,
                                     views=_scene("x").views, placed=True)
    panel = _panel(qtbot, db)
    panel.select_product("JCAD-000000000")
    _settle(qtbot, panel)
    assert any("7 taş yuvalara yerleştirildi" in t for t in _badges(panel)), _badges(panel)
    panel.select_product("JCAD-000000001")
    _settle(qtbot, panel)
    assert any(t == "taş uygulandı" for t in _badges(panel)), _badges(panel)
    db.close()


# ---------------------------------------------------------------- taşları yuvalara yerleştir

def _stl_records(record_factory, tmp_path):
    """A product recorded as stone-set and one recorded as stoneless, both STL; plus a .3dm."""
    from catalog_organizer.core.schemas import StoneSummary

    def summary(status):
        return StoneSummary(status=status, center_stone=None, side_stones=[], total_estimated_carat=0.0)

    out = []
    for fid, ext, status in (("WITH", ".stl", "stone"), ("WITHOUT", ".stl", "polished"), ("PLAIN3DM", ".3dm", "stone")):
        src = tmp_path / f"{fid}{ext}"
        src.write_bytes(b"x")
        out.append(record_factory(file_id=fid, source_path=str(src), main_category="ring",
                                  stone_summary=summary(status)))
    return out


def test_stones_are_placed_only_for_a_product_the_record_calls_stone_set(qtbot, env, record_factory, tmp_path):
    """Regression: a cocktail ring and a tennis bracelet recorded as stoneless were given stones."""
    import catalog_organizer.webview.scene_cache as sc

    db, *_ = env(_stl_records(record_factory, tmp_path))
    panel = _panel(qtbot, db)
    for fid, want_box, want_asked in (("WITH", True, True), ("WITHOUT", False, False)):
        panel.select_product(fid)
        _settle(qtbot, panel)
        assert panel.place_box.isChecked() is want_box
        assert panel.place_box.isEnabled(), "an STL: the switch is the user's"
        assert sc.ensure_scene.asked[-1] == {"file_id": fid, "place_stones": want_asked}
    db.close()


def test_a_3dm_has_no_such_switch(qtbot, env, record_factory, tmp_path):
    """A .3dm carries its stones in its own layer: nothing to place."""
    import catalog_organizer.webview.scene_cache as sc

    db, *_ = env(_stl_records(record_factory, tmp_path))
    panel = _panel(qtbot, db)
    panel.select_product("PLAIN3DM")
    _settle(qtbot, panel)
    assert not panel.place_box.isEnabled() and not panel.place_box.isChecked()
    assert sc.ensure_scene.asked[-1]["place_stones"] is False
    db.close()


def test_the_users_switch_is_saved_and_rebuilds_the_scene(qtbot, env, record_factory, tmp_path):
    from catalog_organizer.db.render_options import get_place_stones
    import catalog_organizer.webview.scene_cache as sc

    db, _prepared, players, _ = env(_stl_records(record_factory, tmp_path))
    panel = _panel(qtbot, db)
    panel.select_product("WITHOUT")
    _settle(qtbot, panel)
    before = len(sc.ensure_scene.asked)

    panel.place_box.setChecked(True)                      # the user says: this one has stones
    _settle(qtbot, panel)
    assert get_place_stones(db.conn, "WITHOUT") is True
    assert len(sc.ensure_scene.asked) == before + 1
    assert sc.ensure_scene.asked[-1] == {"file_id": "WITHOUT", "place_stones": True}
    assert len(players[0].of("load")) == 2, "the players show the rebuilt scene"

    # and it is still there when the product is opened again later
    panel.select_product("WITH")
    _settle(qtbot, panel)
    panel.select_product("WITHOUT")
    _settle(qtbot, panel)
    assert panel.place_box.isChecked()
    assert sc.ensure_scene.asked[-1]["place_stones"] is True
    db.close()


def test_png_save_uses_the_same_stone_decision(qtbot, env, record_factory, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    import catalog_organizer.render.engine as engine
    from catalog_organizer.render.product_render import RenderResult

    db, *_ = env(_stl_records(record_factory, tmp_path))
    seen = {}

    def fake_render(source, *, file_id, metal_key, stone_key, **kw):
        seen["place_stones"] = kw["place_stones"]
        return RenderResult(file_id, metal_key, stone_key, {}, False, "web")

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(engine, "load_and_render", fake_render)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(out))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    panel = _panel(qtbot, db)
    panel.select_product("WITH")
    _settle(qtbot, panel)
    panel.save_button.click()
    qtbot.waitUntil(lambda: panel._save_worker is None, timeout=5000)
    assert seen["place_stones"] is True
    db.close()
