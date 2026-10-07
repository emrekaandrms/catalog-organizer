"""GUI tests — theme, main window, sidebar navigation, settings."""
from __future__ import annotations

from pathlib import Path

import pytest

from catalog_organizer.catalog.index import CatalogIndex
from catalog_organizer.gui.theme import apply_dark_theme, _build_qss

# ---------------------------------------------------------------------------
# pytest-qt sets up a QApplication automatically via the `qapp` fixture.
# All widget tests use `qtbot` to ensure proper Qt lifecycle management.
# ---------------------------------------------------------------------------


def test_theme_builds_qss_without_error():
    """_build_qss must produce a non-empty string for any accent colour."""
    css = _build_qss("#ff0000")
    assert len(css) > 200
    assert "#ff0000" in css


def test_apply_dark_theme_runs(qapp):
    """apply_dark_theme must not raise, and must wire the palette to the
    theme tokens.

    Asserted against `theme.BG_BASE` rather than a literal hex: pinning the
    colour meant a deliberate palette change failed here as if it were a
    regression, which says nothing about correctness.
    """
    from PyQt6.QtGui import QColor

    from catalog_organizer.gui import theme

    apply_dark_theme(qapp, accent="#1e88e5")
    pal = qapp.palette()
    assert pal.window().color() == QColor(theme.BG_BASE)
    assert pal.highlight().color() == QColor("#1e88e5")   # accent honoured


def test_main_window_panel_count(qtbot, tmp_path, monkeypatch):
    """MainWindow registers exactly one sidebar item and stack page per _NAV entry."""
    _patch_paths(monkeypatch, tmp_path)
    from catalog_organizer.gui.main_window import MainWindow, _NAV
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._sidebar.count() == len(_NAV)
    assert win._stack.count() == len(_NAV)


def test_sidebar_navigation_updates_stack(qtbot, tmp_path, monkeypatch):
    """Clicking a sidebar row must switch the stack to the matching page."""
    _patch_paths(monkeypatch, tmp_path)
    from catalog_organizer.gui.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    # Process is row 1, Diagnostics is row 2.
    win._sidebar.setCurrentRow(1)
    assert win._stack.currentIndex() == 1
    win._sidebar.setCurrentRow(2)
    assert win._stack.currentIndex() == 2


def test_settings_panel_save_roundtrip(qtbot, tmp_path, monkeypatch):
    """SettingsPanel must write edited Ollama model back to
    pipeline_settings.yaml under the new provider-block schema."""
    import yaml

    _patch_paths(monkeypatch, tmp_path)
    cfg_dir = tmp_path / "config"   # already created by _patch_paths
    seed = {
        "vlm": {
            "provider": "ollama",
            "timeout_s": 60,
            "ollama":   {"host": "127.0.0.1", "port": 11434, "model": "old-model"},
        }
    }
    (cfg_dir / "pipeline_settings.yaml").write_text(yaml.dump(seed), encoding="utf-8")

    # Stub discovery so the form doesn't reach a real Ollama during tests.
    from catalog_organizer.vlm import ollama_models as om
    monkeypatch.setattr(om, "list_ollama_models",
                        lambda host, port, timeout_s=5.0: _fake_models())

    from catalog_organizer.gui.panels.settings import SettingsPanel
    panel = SettingsPanel()
    qtbot.addWidget(panel)
    # The Ollama sub-form lives under the provider stack; edit its model
    # field directly so the test exercises the actual UI path. The model
    # field is an editable QComboBox (live server list) as of 2026-07-28 —
    # a hand-typed tag that isn't installed must still round-trip.
    panel._vlm_tab._ollama_form._model.setCurrentText("new-model")
    panel._vlm_tab.save()

    saved = yaml.safe_load((cfg_dir / "pipeline_settings.yaml").read_text())
    assert saved["vlm"]["provider"]        == "ollama"
    assert saved["vlm"]["ollama"]["model"] == "new-model"


def test_settings_panel_parallel_workers_round_trip(qtbot, tmp_path, monkeypatch):
    """SettingsPanel must round-trip vlm.parallel_workers through YAML."""
    import yaml

    _patch_paths(monkeypatch, tmp_path)
    cfg_dir = tmp_path / "config"
    # Seed pipeline_settings with parallel_workers=1
    (cfg_dir / "pipeline_settings.yaml").write_text(
        yaml.dump({
            "vlm": {"parallel_workers": 1},
            "snapshot": {"resolution": 512, "thumbnail_size": 256},
        }),
        encoding="utf-8",
    )

    from catalog_organizer.gui.panels.settings import SettingsPanel
    panel = SettingsPanel()
    qtbot.addWidget(panel)

    # Bump to 3 via the new spinbox
    panel._pipeline_tab._parallel.setValue(3)
    panel._pipeline_tab.save()

    saved = yaml.safe_load((cfg_dir / "pipeline_settings.yaml").read_text())
    assert saved["vlm"]["parallel_workers"] == 3


def test_search_panel_free_text_filters_results(qtbot, tmp_path, monkeypatch, record_factory):
    """SearchPanel's free-text box must filter the visible result list
    against controlled_tags + rich_description (the catalogue-search design notes §5)."""
    _patch_paths(monkeypatch, tmp_path)
    from catalog_organizer.catalog.writer import CatalogWriter
    from catalog_organizer.db.connection import Database
    from catalog_organizer.db.sync import sync_from_jsonl
    from catalog_organizer.gui.panels.search_panel import SearchPanel

    jsonl = tmp_path / "catalog_master.jsonl"
    writer = CatalogWriter(jsonl_path=jsonl, csv_path=tmp_path / "export.csv", flush_every=1)
    writer.append(record_factory(file_id="A", rich_description="taşlı çiçek yüzük"))
    writer.append(record_factory(file_id="B", rich_description="cilalı düz yüzük"))
    writer.close()

    db = Database(tmp_path / "catalog.db")
    sync_from_jsonl(db.conn, jsonl)
    panel = SearchPanel(db)
    qtbot.addWidget(panel)

    assert panel._list.count() == 2

    panel._txt.setText("çiçek")
    panel._run_search()
    assert panel._list.count() == 1
    db.close()


def test_search_panel_refresh_button_picks_up_records_written_after_panel_opened(
    qtbot, tmp_path, monkeypatch, record_factory,
):
    """Regression (2026-07-24): CatalogIndex is loaded once at startup and
    no panel auto-refreshes it, so records appended by another process
    (e.g. the `run-pilot` CLI) while the GUI is already open were
    invisible — a real pilot batch produced 68 catalog records that a
    already-open Search tab showed none of. The refresh button must call
    index.reload() and pick up new records without restarting the app."""
    _patch_paths(monkeypatch, tmp_path)
    from catalog_organizer.catalog.writer import CatalogWriter
    from catalog_organizer.db.connection import Database
    from catalog_organizer.db.sync import sync_from_jsonl
    from catalog_organizer.gui.panels.search_panel import SearchPanel

    jsonl = tmp_path / "catalog_master.jsonl"
    writer = CatalogWriter(jsonl_path=jsonl, csv_path=tmp_path / "export.csv", flush_every=1)
    writer.append(record_factory(file_id="A"))
    writer.close()

    db = Database(tmp_path / "catalog.db")
    sync_from_jsonl(db.conn, jsonl)
    panel = SearchPanel(db)
    qtbot.addWidget(panel)
    assert panel._list.count() == 1

    # Simulate an external process (the CLI pilot runner) appending more
    # records to the SAME jsonl file while the panel is already open.
    writer2 = CatalogWriter(jsonl_path=jsonl, csv_path=tmp_path / "export.csv", flush_every=1)
    writer2.append(record_factory(file_id="B", brand="cartier"))
    writer2.append(record_factory(file_id="C"))
    writer2.close()

    # Without a re-sync the panel still shows the 1 record the DB knows about.
    assert panel._list.count() == 1

    panel.reload_from_disk(jsonl)
    assert panel._list.count() == 3
    assert panel._cb_brand.findText("cartier") >= 0
    db.close()


# ---------------------------------------------------------------------------
# VLM provider settings — live Ollama model dropdown
# ---------------------------------------------------------------------------

def _fake_models():
    from catalog_organizer.vlm.ollama_models import OllamaModel
    return [
        OllamaModel(name="qwen3.5:9b-q4_K_M", size_bytes=6_594_474_711,
                    capabilities=("vision", "completion")),
        OllamaModel(name="llama3:8b", size_bytes=4_700_000_000,
                    capabilities=("completion",)),          # text-only
    ]


def _ollama_form(qtbot, monkeypatch, tmp_path, models=None, error=None):
    """Build an _OllamaForm with the network stubbed out, and deliver a
    model list through the real background-worker signal path."""
    _patch_paths(monkeypatch, tmp_path)
    from catalog_organizer.vlm import ollama_models as om

    def fake_list(host, port, timeout_s=5.0):
        if error is not None:
            raise om.OllamaUnreachable(f"http://{host}:{port}/api/tags", error)
        return list(models or [])

    monkeypatch.setattr(om, "list_ollama_models", fake_list)

    from catalog_organizer.gui.panels.settings import _OllamaForm
    form = _OllamaForm()
    qtbot.addWidget(form)
    qtbot.waitUntil(lambda: not form._status.text().startswith("Connecting"),
                    timeout=5000)
    return form


def test_ollama_form_populates_dropdown_from_server(qtbot, tmp_path, monkeypatch):
    """The model field is a live dropdown, not free text — the whole point
    is that the user picks a tag that provably exists on the server."""
    form = _ollama_form(qtbot, monkeypatch, tmp_path, models=_fake_models())
    assert form._model.count() >= 1
    assert "Connected" in form._status.text()


def test_ollama_form_hides_text_only_models_by_default(qtbot, tmp_path, monkeypatch):
    """A text-only model can't see the snapshots and would invent
    classifications from the prompt alone. It must not be selectable by
    accident."""
    form = _ollama_form(qtbot, monkeypatch, tmp_path, models=_fake_models())
    shown = {form._model.itemData(i) for i in range(form._model.count())}
    assert "qwen3.5:9b-q4_K_M" in shown
    assert "llama3:8b" not in shown

    form._vision_only.setChecked(False)          # user opts in explicitly
    shown_all = {form._model.itemData(i) for i in range(form._model.count())}
    assert "llama3:8b" in shown_all


def test_ollama_form_saves_bare_tag_not_decorated_label(qtbot, tmp_path, monkeypatch):
    """Regression guard for the highest-consequence bug in this feature:
    dropdown entries read 'qwen3.5:9b-q4_K_M  ·  6.6 GB'. Persisting that
    display string as the model id would make every subsequent VLM call
    404 on a tag that doesn't exist."""
    form = _ollama_form(qtbot, monkeypatch, tmp_path, models=_fake_models())
    idx = form._model.findData("qwen3.5:9b-q4_K_M")
    assert idx >= 0
    form._model.setCurrentIndex(idx)

    assert "GB" in form._model.currentText()         # label really is decorated
    assert form.get_model() == "qwen3.5:9b-q4_K_M"   # …but the saved value isn't
    assert form.collect()["model"] == "qwen3.5:9b-q4_K_M"


def test_ollama_form_keeps_hand_typed_model(qtbot, tmp_path, monkeypatch):
    """The combo stays editable so a model that isn't pulled yet can still
    be configured. Typed text must win over the previously selected item's
    userData."""
    form = _ollama_form(qtbot, monkeypatch, tmp_path, models=_fake_models())
    form._model.setCurrentText("some-future-model:latest")
    assert form.get_model() == "some-future-model:latest"


def test_ollama_form_warns_when_configured_model_absent(qtbot, tmp_path, monkeypatch):
    """Server reachable but the configured tag isn't pulled — the user
    needs to see that before starting a batch, not as a mid-run 404."""
    form = _ollama_form(qtbot, monkeypatch, tmp_path, models=[])
    assert "not pulled" in form._status.text()
    # The selection is preserved rather than silently switched.
    assert form.get_model() == "qwen3.5:9b-q4_K_M"


def test_ollama_form_reports_unreachable_server(qtbot, tmp_path, monkeypatch):
    form = _ollama_form(qtbot, monkeypatch, tmp_path,
                        error="connection refused (is Ollama running?)")
    assert "✗" in form._status.text()
    assert "connection refused" in form._status.text()


def test_vlm_tab_exposes_all_providers_with_matching_stack(qtbot, tmp_path, monkeypatch):
    """_on_provider_change maps the combo index straight onto the stack
    index, so a mismatch would show the wrong form for the chosen
    provider."""
    _patch_paths(monkeypatch, tmp_path)
    from catalog_organizer.gui.panels.settings import _VlmTab
    tab = _VlmTab()
    qtbot.addWidget(tab)
    assert tab._provider.count() == tab._stack.count()
    for i in range(tab._provider.count()):
        assert tab._provider.itemData(i) == tab._stack.widget(i).PROVIDER


def test_vlm_tab_save_round_trips_every_provider_block(qtbot, tmp_path, monkeypatch):
    """Switching providers must not wipe another provider's saved config,
    and the Ollama block must carry the bare tag."""
    import yaml
    _patch_paths(monkeypatch, tmp_path)
    from catalog_organizer.gui.panels.settings import _VlmTab
    tab = _VlmTab()
    qtbot.addWidget(tab)
    tab._provider.setCurrentIndex(tab._provider.findData("glm"))
    tab.save()

    saved = yaml.safe_load((tmp_path / "config" / "pipeline_settings.yaml").read_text())
    vlm = saved["vlm"]
    assert vlm["provider"] == "glm"
    for key in ("ollama", "openai", "minimax", "glm", "openai_compatible"):
        assert key in vlm, f"{key} block was dropped on save"
    assert "·" not in vlm["ollama"]["model"]
    assert vlm["glm"]["base_url"].startswith("https://")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_paths(monkeypatch, tmp_path: Path) -> None:
    """Redirect config_dir and data_dir to isolated tmp dirs."""
    cfg = tmp_path / "config"
    dat = tmp_path / "data"
    cfg.mkdir(exist_ok=True)
    dat.mkdir(exist_ok=True)

    import yaml
    (cfg / "app_settings.yaml").write_text(
        yaml.dump({"paths": {}, "theme": {"accent_color": "#1e88e5"}}),
        encoding="utf-8",
    )
    (cfg / "pipeline_settings.yaml").write_text(
        yaml.dump({"vlm": {}, "snapshot": {}}), encoding="utf-8",
    )
    # SearchPanel (added 2026-07-24) reads categories.yaml at construction
    # time to populate its category/subcategory filter dropdowns.
    (cfg / "categories.yaml").write_text(
        yaml.dump({"categories": {"ring": {"subcategories": ["solitaire", "other"]}}}),
        encoding="utf-8",
    )

    cfg_fn = lambda: cfg  # noqa: E731
    dat_fn = lambda: dat  # noqa: E731

    monkeypatch.setattr("catalog_organizer.core.paths.config_dir", cfg_fn)
    monkeypatch.setattr("catalog_organizer.core.paths.data_dir",   dat_fn)
    monkeypatch.setattr("catalog_organizer.core.paths.cache_dir",  lambda: tmp_path / "cache")
    monkeypatch.setattr("catalog_organizer.catalog.index.data_dir", dat_fn)
    # settings.py imports config_dir directly — patch its module-level name too
    monkeypatch.setattr("catalog_organizer.gui.panels.settings.config_dir", cfg_fn)
