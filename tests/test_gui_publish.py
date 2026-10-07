from __future__ import annotations

import csv
from pathlib import Path

import orjson
import pytest

from catalog_organizer.db.connection import Database
from catalog_organizer.db.listings import get_listing, list_exports
from catalog_organizer.db.products import get_product
from catalog_organizer.db.selections import (
    add_items, create_selection, list_selections, selection_file_ids,
)
from catalog_organizer.db.sync import sync_from_jsonl

pytest.importorskip("PyQt6")

WOO_PAYLOAD = {
    "title": "Kalp Kolye Ucu",
    "short_description": "Zarif kalp kolye ucu.",
    "description": "<p>Zarif kalp kolye ucu.</p>",
    "tags": ["kalp", "kolye ucu"],
    "categories": ["Kolye", "Kolye Ucu"],
    "slug": "",
    "meta_description": "Zarif kalp kolye ucu.",
    "sustainability_note": "Siparişe özel dökülür.",
}


class FakeProvider:
    display_name = "Fake (test-model)"
    model_name = "test-model"

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        return dict(WOO_PAYLOAD)


def _write_jsonl(path: Path, records) -> None:
    with path.open("wb") as fh:
        for rec in records:
            fh.write(orjson.dumps(rec.model_dump(mode="json")) + b"\n")


@pytest.fixture
def env(tmp_path: Path):
    def _make(records):
        jsonl = tmp_path / "m.jsonl"
        _write_jsonl(jsonl, records)
        db = Database(tmp_path / "c.db")
        sync_from_jsonl(db.conn, jsonl)
        return db, jsonl
    return _make


# ---------------------------------------------------------------- Search

def test_search_panel_lists_from_db(qtbot, env, record_factory):
    from catalog_organizer.gui.panels.search_panel import SearchPanel
    db, _ = env([record_factory(file_id="A"), record_factory(file_id="B")])
    panel = SearchPanel(db)
    qtbot.addWidget(panel)
    assert panel.result_file_ids() == ["A", "B"]
    db.close()


def test_search_panel_category_filter(qtbot, env, record_factory):
    from catalog_organizer.gui.panels.search_panel import SearchPanel
    db, _ = env([record_factory(file_id="A", main_category="ring"),
                 record_factory(file_id="B", main_category="earring")])
    panel = SearchPanel(db)
    qtbot.addWidget(panel)
    panel.set_category("ring")
    panel.run_search()
    assert panel.result_file_ids() == ["A"]
    db.close()


def test_search_panel_reload_picks_up_new_rows(qtbot, env, record_factory):
    """The 'search tab shows nothing' bug: the index was loaded once and
    never refreshed. Reload must re-sync from disk."""
    from catalog_organizer.gui.panels.search_panel import SearchPanel
    db, jsonl = env([record_factory(file_id="A")])
    panel = SearchPanel(db)
    qtbot.addWidget(panel)
    assert panel.result_file_ids() == ["A"]

    with jsonl.open("ab") as fh:
        fh.write(orjson.dumps(
            record_factory(file_id="B").model_dump(mode="json")) + b"\n")
    panel.reload_from_disk(jsonl)
    assert panel.result_file_ids() == ["A", "B"]
    db.close()


def test_search_panel_turkish_free_text(qtbot, env, record_factory):
    from catalog_organizer.gui.panels.search_panel import SearchPanel
    db, _ = env([record_factory(file_id="A", rich_description="İnce zincir")])
    panel = SearchPanel(db)
    qtbot.addWidget(panel)
    panel.set_free_text("İNCE")
    panel.run_search()
    assert panel.result_file_ids() == ["A"]
    db.close()


def test_tick_and_untick(qtbot, env, record_factory):
    from catalog_organizer.gui.panels.search_panel import SearchPanel
    db, _ = env([record_factory(file_id="A"), record_factory(file_id="B")])
    panel = SearchPanel(db)
    qtbot.addWidget(panel)
    panel.set_checked("A", True)
    assert panel.checked_file_ids() == ["A"]
    panel.set_checked("A", False)
    assert panel.checked_file_ids() == []
    db.close()


def test_check_all_visible(qtbot, env, record_factory):
    from catalog_organizer.gui.panels.search_panel import SearchPanel
    db, _ = env([record_factory(file_id="A"), record_factory(file_id="B")])
    panel = SearchPanel(db)
    qtbot.addWidget(panel)
    panel.check_all_visible()
    assert panel.checked_file_ids() == ["A", "B"]
    db.close()


def test_new_search_clears_ticks(qtbot, env, record_factory):
    """Ticks belong to a result set — carrying them across a filter change
    would let the user add records they can no longer see."""
    from catalog_organizer.gui.panels.search_panel import SearchPanel
    db, _ = env([record_factory(file_id="A", main_category="ring"),
                 record_factory(file_id="B", main_category="earring")])
    panel = SearchPanel(db)
    qtbot.addWidget(panel)
    panel.check_all_visible()
    panel.set_category("earring")
    panel.run_search()
    assert panel.checked_file_ids() == []
    db.close()


def test_add_checked_to_selection(qtbot, env, record_factory):
    from catalog_organizer.gui.panels.search_panel import SearchPanel
    db, _ = env([record_factory(file_id="A"), record_factory(file_id="B")])
    sid = create_selection(db.conn, "S")
    panel = SearchPanel(db)
    qtbot.addWidget(panel)
    panel.set_checked("A", True)
    assert panel.add_checked_to_selection(sid) == 1
    assert selection_file_ids(db.conn, sid) == ["A"]
    db.close()


def test_add_all_matching_ignores_ticks(qtbot, env, record_factory):
    """Nothing ticked, but every filter match must still be added — this is
    the 1000 → 100 workflow's actual tool."""
    from catalog_organizer.gui.panels.search_panel import SearchPanel
    db, _ = env([record_factory(file_id="A", main_category="ring"),
                 record_factory(file_id="B", main_category="ring"),
                 record_factory(file_id="C", main_category="earring")])
    sid = create_selection(db.conn, "S")
    panel = SearchPanel(db)
    qtbot.addWidget(panel)
    panel.set_category("ring")
    panel.run_search()
    assert panel.add_all_matching_to_selection(sid) == 2
    assert selection_file_ids(db.conn, sid) == ["A", "B"]
    db.close()


def test_publish_picks_up_items_added_from_the_search_tab(qtbot, env,
                                                         record_factory):
    """The reported bug: add products to a list in Search, go to Publish, and
    the list is empty — so the PDF button has nothing to render.

    Search does emit `selectionsChanged` on an add, and the main window does
    connect it. But it connects to `refresh_selections`, which rebuilds the
    list of list NAMES and then re-highlights the current row with signals
    blocked — so `_on_selection_changed` never fires and the items behind the
    selection are never re-read."""
    from catalog_organizer.gui.panels.publish_panel import PublishPanel
    from catalog_organizer.gui.panels.search_panel import SearchPanel

    db, _ = env([record_factory(file_id="A"), record_factory(file_id="B")])
    sid = create_selection(db.conn, "S")

    publish = PublishPanel(db)
    qtbot.addWidget(publish)
    publish.select_selection(sid)
    assert publish.item_file_ids() == []

    search = SearchPanel(db)
    qtbot.addWidget(search)
    search.set_checked("A", True)
    search.add_checked_to_selection(sid)

    publish.refresh_selections()          # exactly what the signal invokes
    assert publish.item_file_ids() == ["A"]
    db.close()


# ---------------------------------------------------------------- Publish

def _panel(qtbot, db, records, name="S"):
    from catalog_organizer.gui.panels.publish_panel import PublishPanel
    sid = create_selection(db.conn, name)
    add_items(db.conn, sid, [r.file_id for r in records])
    panel = PublishPanel(db)
    qtbot.addWidget(panel)
    panel.select_selection(sid)
    return panel, sid


def test_publish_lists_selections(qtbot, env, record_factory):
    from catalog_organizer.gui.panels.publish_panel import PublishPanel
    db, _ = env([record_factory(file_id="A")])
    create_selection(db.conn, "Kış 2026")
    create_selection(db.conn, "Etsy ilk parti")
    panel = PublishPanel(db)
    qtbot.addWidget(panel)
    assert panel.selection_names() == ["Etsy ilk parti", "Kış 2026"]
    db.close()


def test_publish_shows_items_of_selected_list(qtbot, env, record_factory):
    db, _ = env([record_factory(file_id="A"), record_factory(file_id="B")])
    panel, _ = _panel(qtbot, db, [record_factory(file_id="A")])
    assert panel.item_file_ids() == ["A"]
    db.close()


def test_publish_create_selection_emits_signal(qtbot, env, record_factory):
    from catalog_organizer.gui.panels.publish_panel import PublishPanel
    db, _ = env([record_factory(file_id="A")])
    panel = PublishPanel(db)
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.selectionsChanged, timeout=1000):
        sid = panel.create_selection("Yeni")
    assert panel.current_selection_id() == sid
    assert [s.name for s in list_selections(db.conn)] == ["Yeni"]
    db.close()


def test_publish_remove_checked(qtbot, env, record_factory):
    recs = [record_factory(file_id="A"), record_factory(file_id="B")]
    db, _ = env(recs)
    panel, sid = _panel(qtbot, db, recs)
    panel.set_item_checked("A", True)
    assert panel.remove_checked_items() == 1
    assert selection_file_ids(db.conn, sid) == ["B"]
    assert panel.item_file_ids() == ["B"]
    db.close()


def test_publish_warns_but_does_not_block(qtbot, env, record_factory):
    recs = [
        record_factory(file_id="A", design_complete=False,
                       sellability="not_sellable"),
        record_factory(file_id="B", design_complete=True, sellability="sellable"),
    ]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs)
    assert panel.item_file_ids() == ["A", "B"]   # not blocked
    assert panel.warning_file_ids() == ["A"]     # but flagged
    db.close()


def test_publish_needs_review_also_warns(qtbot, env, record_factory):
    recs = [record_factory(file_id="A", design_complete=True,
                           sellability="needs_review")]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs)
    assert panel.warning_file_ids() == ["A"]
    db.close()


def test_publish_override_applies_and_survives_resync(qtbot, env, record_factory):
    recs = [record_factory(file_id="A", main_category="ring")]
    db, jsonl = env(recs)
    panel, _ = _panel(qtbot, db, recs)
    panel.apply_override("A", "main_category", "pendant")
    assert get_product(db.conn, "A").main_category == "pendant"
    sync_from_jsonl(db.conn, jsonl)
    assert get_product(db.conn, "A").main_category == "pendant"
    db.close()


def test_publish_reset_override(qtbot, env, record_factory):
    recs = [record_factory(file_id="A", main_category="ring")]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs)
    panel.apply_override("A", "brand", "atasay")
    assert panel.override_fields("A") == {"brand": "atasay"}
    panel.reset_override("A", "brand")
    assert panel.override_fields("A") == {}
    db.close()


def test_publish_override_rejects_bad_field(qtbot, env, record_factory):
    recs = [record_factory(file_id="A")]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs)
    with pytest.raises(ValueError):
        panel.apply_override("A", "sellability", "sellable")
    db.close()


# ---------------------------------------------------------------- uçtan uca

def test_generate_then_export_end_to_end(qtbot, env, record_factory, monkeypatch,
                                         tmp_path):
    """Ürün → seçim → ilan metni → fiyat → WooCommerce CSV."""
    from catalog_organizer.core.schemas import MetalWeights
    import catalog_organizer.gui.panels.publish_panel as pp

    recs = [record_factory(file_id="A", main_category="pendant",
                           metal_weights=MetalWeights(silver_925_g=5.0,
                                                      gold_14k_yellow_g=6.3))]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs, name="Kis 2026")

    monkeypatch.setattr(
        "catalog_organizer.vlm.text_provider.create_text_provider",
        lambda settings: FakeProvider())

    errors = panel.generate_for_selection("woocommerce")
    assert errors == {}

    listing = get_listing(db.conn, "A", "woocommerce")
    assert listing.title == "Kalp Kolye Ucu"
    assert len(listing.variants) == 6
    assert listing.status == "draft"

    panel.set_item_checked("A", True)
    assert panel.approve_checked() == 1
    assert get_listing(db.conn, "A", "woocommerce").status == "approved"

    results = panel.export_selection("woocommerce", out_dir=tmp_path / "out")
    assert len(results) == 1
    result = results[0]
    assert result.row_count == 7            # 1 parent + 6 variations
    assert result.file_path.name.endswith("-kis-2026-woocommerce.csv")

    with result.file_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["Type"] == "variable"
    assert rows[-1]["Type"] == "variation, virtual, downloadable"
    assert rows[-1]["Regular price"] == "50.00"

    assert get_listing(db.conn, "A", "woocommerce").status == "exported"
    assert list_exports(db.conn)[0]["row_count"] == 7
    db.close()


def test_generate_skips_human_edited_listings(qtbot, env, record_factory,
                                              monkeypatch):
    """Kullanıcının elle yazdığı metin yeniden üretimde ezilmemeli."""
    from catalog_organizer.core.schemas import MetalWeights
    from catalog_organizer.db.listings import update_text

    recs = [record_factory(file_id="A",
                           metal_weights=MetalWeights(silver_925_g=5.0,
                                                      gold_14k_yellow_g=6.3))]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs)
    monkeypatch.setattr(
        "catalog_organizer.vlm.text_provider.create_text_provider",
        lambda settings: FakeProvider())

    panel.generate_for_selection("woocommerce")
    listing = get_listing(db.conn, "A", "woocommerce")
    update_text(db.conn, listing.listing_id, title="Elimle Yazdım")

    panel.generate_for_selection("woocommerce")
    assert get_listing(db.conn, "A", "woocommerce").title == "Elimle Yazdım"
    db.close()


def test_export_only_takes_approved_by_default(qtbot, env, record_factory,
                                               monkeypatch, tmp_path):
    from catalog_organizer.core.schemas import MetalWeights
    recs = [record_factory(file_id="A",
                           metal_weights=MetalWeights(silver_925_g=5.0,
                                                      gold_14k_yellow_g=6.3))]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs)
    monkeypatch.setattr(
        "catalog_organizer.vlm.text_provider.create_text_provider",
        lambda settings: FakeProvider())
    panel.generate_for_selection("woocommerce")

    result = panel.export_selection("woocommerce", out_dir=tmp_path / "o")[0]
    assert result.row_count == 0            # taslak, onaylı değil

    panel.set_item_checked("A", True)
    panel.approve_checked()
    result = panel.export_selection("woocommerce", out_dir=tmp_path / "o")[0]
    assert result.row_count == 7
    db.close()


def test_reprice_updates_without_regenerating(qtbot, env, record_factory,
                                              monkeypatch):
    from catalog_organizer.core.schemas import MetalWeights
    recs = [record_factory(file_id="A",
                           metal_weights=MetalWeights(silver_925_g=5.0,
                                                      gold_14k_yellow_g=5.0))]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs)
    monkeypatch.setattr(
        "catalog_organizer.vlm.text_provider.create_text_provider",
        lambda settings: FakeProvider())
    panel.generate_for_selection("woocommerce")
    assert panel.reprice_selection("woocommerce") == 6
    prices = {v.variant_key: v.price
              for v in get_listing(db.conn, "A", "woocommerce").variants}
    assert prices["gold_14k"] == 28665.00
    db.close()


# ---------------------------------------------------------------- Ayarlar

def test_pricing_tab_round_trips_constants(qtbot, tmp_path, monkeypatch):
    """Kur değişince tek yerden güncellensin diye — sabitler koda gömülü
    değil, bu sekmeden düzenleniyor."""
    import shutil

    import catalog_organizer.core.paths as paths_mod
    import catalog_organizer.gui.panels.settings as settings_mod

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    shutil.copy(paths_mod.config_dir() / "pricing.yaml", cfg_dir / "pricing.yaml")
    monkeypatch.setattr(settings_mod, "config_dir", lambda: cfg_dir)

    tab = settings_mod._PricingTab()
    qtbot.addWidget(tab)
    assert tab._woo_gold_rate.value() == 7000.0
    assert tab._karats[14].value() == 0.585
    assert tab._etsy_who.currentText() == "i_did"

    tab._woo_gold_rate.setValue(9000.0)
    tab._etsy_who.setCurrentText("collective")
    tab.save()

    reloaded = settings_mod._PricingTab()
    qtbot.addWidget(reloaded)
    assert reloaded._woo_gold_rate.value() == 9000.0
    assert reloaded._etsy_who.currentText() == "collective"
    assert reloaded._karats[8].value() == 0.333


def test_pricing_tab_rejects_bad_additions(qtbot, tmp_path, monkeypatch):
    import shutil

    import catalog_organizer.core.paths as paths_mod
    import catalog_organizer.gui.panels.settings as settings_mod

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    shutil.copy(paths_mod.config_dir() / "pricing.yaml", cfg_dir / "pricing.yaml")
    monkeypatch.setattr(settings_mod, "config_dir", lambda: cfg_dir)

    tab = settings_mod._PricingTab()
    qtbot.addWidget(tab)
    tab._etsy_add.setText("20, beş, 5")
    with pytest.raises(ValueError, match="numbers"):
        tab.save()


# ------------------------------------------------- "sonsuz döngü" raporu

def test_generation_runs_only_on_the_ticked_products(qtbot, env,
                                                     record_factory,
                                                     monkeypatch):
    """Ticking three products and pressing Generate ran the WHOLE list.

    Every other action on the panel reads the ticks -- "Remove ticked", the
    approval, the export -- but the generation loop read `self._records`. One
    product is one call to a language model, timed on this machine at 43-94 s,
    so on a long list that alone turned a three-minute job into an hour.
    """
    recs = [record_factory(file_id=x) for x in ("A", "B", "C", "D")]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs)

    seen: list[str] = []

    class Counting(FakeProvider):
        def complete_json(self, system_prompt, user_prompt):
            return dict(WOO_PAYLOAD)

    monkeypatch.setattr(
        "catalog_organizer.vlm.text_provider.create_text_provider",
        lambda settings: Counting())

    panel.set_item_checked("B", True)
    panel.set_item_checked("D", True)
    assert [r.file_id for r in panel.records_to_generate()] == ["B", "D"]

    panel.generate_for_selection("woocommerce",
                                 progress=lambda d, t, f: seen.append(f))
    assert seen == ["B", "D"]
    assert get_listing(db.conn, "B", "woocommerce") is not None
    assert get_listing(db.conn, "A", "woocommerce") is None

    # Nothing ticked still means the whole list, which is what a user who has
    # not ticked anything expects.
    panel.set_item_checked("B", False)
    panel.set_item_checked("D", False)
    assert len(panel.records_to_generate()) == 4
    db.close()


def test_generation_reports_progress_and_can_be_cancelled(qtbot, env,
                                                          record_factory,
                                                          monkeypatch):
    """The other half of the freeze: no progress, and no way out.

    The work is minutes long by nature, so the panel has to be able to say
    where it is and to stop between products. Cancelling cannot interrupt a
    request already in flight -- the provider's own timeout bounds that -- so
    what is pinned here is that it stops at the next product rather than
    running the rest of the list.
    """
    recs = [record_factory(file_id=x) for x in ("A", "B", "C")]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs)
    monkeypatch.setattr(
        "catalog_organizer.vlm.text_provider.create_text_provider",
        lambda settings: FakeProvider())

    steps: list[tuple[int, int, str]] = []

    def _progress(done, total, file_id):
        steps.append((done, total, file_id))
        if done == 2:
            raise InterruptedError("iptal")

    with pytest.raises(InterruptedError):
        panel.generate_for_selection("woocommerce", progress=_progress)

    assert steps == [(1, 3, "A"), (2, 3, "B")]
    assert get_listing(db.conn, "C", "woocommerce") is None
    db.close()


def test_generate_for_selection_touches_no_widgets(qtbot, env,
                                                   record_factory,
                                                   monkeypatch):
    """It runs on a worker thread now, so it must not redraw the list.

    Touching a Qt widget from a non-GUI thread is undefined behaviour; the
    redraw belongs to whichever slot receives the worker's result. This is
    pinned by a test because the call was there before the move and removing
    it leaves no visible trace.
    """
    recs = [record_factory(file_id="A")]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs)
    monkeypatch.setattr(
        "catalog_organizer.vlm.text_provider.create_text_provider",
        lambda settings: FakeProvider())

    calls: list[int] = []
    monkeypatch.setattr(panel, "_render_items",
                        lambda *a, **k: calls.append(1))
    panel.generate_for_selection("woocommerce")
    assert calls == []
    db.close()


def test_the_generate_button_works_off_the_ui_thread(qtbot, env,
                                                     record_factory,
                                                     monkeypatch):
    """End to end through the button, which is where the freeze was.

    The guard that matters: the click must RETURN while the work is still
    running. If it ever goes back to doing the model calls inline, this fails
    the moment the worker is gone.
    """
    import catalog_organizer.gui.panels.publish_panel as pp

    recs = [record_factory(file_id=x) for x in ("A", "B")]
    db, _ = env(recs)
    panel, _ = _panel(qtbot, db, recs)
    monkeypatch.setattr(
        "catalog_organizer.vlm.text_provider.create_text_provider",
        lambda settings: FakeProvider())
    monkeypatch.setattr(pp.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))

    panel._on_generate_clicked()
    worker = panel._listing_worker
    assert isinstance(worker, pp.ListingWorker)
    assert isinstance(worker, pp.QThread)

    qtbot.waitUntil(lambda: panel._listing_worker is None, timeout=10000)
    assert get_listing(db.conn, "A", "woocommerce") is not None
    assert get_listing(db.conn, "B", "woocommerce") is not None
    db.close()


# ------------------------------------------------------- webview komut satiri

def test_the_webview_command_exports_a_named_selection(env, record_factory, monkeypatch, tmp_path):
    """`app webview --selection NAME` resolves the list the Publish tab shows and hands each
    product's source path, category and label to the exporter."""
    from catalog_organizer.db import connection
    from catalog_organizer.webview import cli, export

    recs = [record_factory(file_id=x, main_category="ring") for x in ("A", "B")]
    db, _ = env(recs)
    sid = create_selection(db.conn, "Liste")
    add_items(db.conn, sid, ["A", "B"])
    seen: dict = {}

    def fake_export_many(items, out_dir, progress=None, **_kw):
        seen["items"], seen["out"] = list(items), out_dir
        return [], {"Z": "kasten bozuk"}

    monkeypatch.setattr(export, "export_many", fake_export_many)
    monkeypatch.setattr(connection, "Database", lambda *a, **k: db)

    code = cli.run(["--selection", "Liste", "--out", str(tmp_path)])
    assert sorted(i[0] for i in seen["items"]) == ["A", "B"]
    assert all(i[2] == "ring" for i in seen["items"])
    assert seen["out"] == tmp_path
    assert code == 1, "nothing was exported, so the command must say it failed"


def test_the_webview_command_refuses_an_unknown_selection(env, record_factory, monkeypatch, tmp_path):
    from catalog_organizer.db import connection
    from catalog_organizer.webview import cli

    db, _ = env([record_factory(file_id="A")])
    monkeypatch.setattr(connection, "Database", lambda *a, **k: db)
    with pytest.raises(SystemExit) as info:
        cli.run(["--selection", "yok boyle liste", "--out", str(tmp_path)])
    assert "yok boyle liste" in str(info.value)


def test_the_webview_command_needs_something_to_export(capsys):
    from catalog_organizer.webview import cli

    assert cli.run([]) == 2
