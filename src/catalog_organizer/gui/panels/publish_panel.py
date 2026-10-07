"""Publish tab — selection lists, listing generation, marketplace export.

The Search tab is for exploration ("what do I have"); this one is for
commitment ("these are the ones I'm selling"). They are separate screens
because a 100-item selection would be lost inside a 100k-record result list.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from catalog_organizer.core.paths import data_dir
from catalog_organizer.core.schemas import CatalogRecord
from catalog_organizer.db.connection import Database
from catalog_organizer.db.listings import (
    get_listing, mark_exported, record_export, set_status,
)
from catalog_organizer.db.products import (
    OVERRIDABLE_FIELDS, clear_override, get_overrides, get_products, set_override,
)
from catalog_organizer.db.selections import (
    create_selection, delete_selection, list_selections, remove_items,
    selection_file_ids,
)
from catalog_organizer.export import etsy as etsy_export
from catalog_organizer.export import woocommerce as woo_export
from catalog_organizer.gui import theme
from catalog_organizer.gui.widgets.components import (
    Badge, PageHeader, caption, field_label, page_layout,
)
from catalog_organizer.listing.pricing import load_pricing_config
from catalog_organizer.listing.schemas import slugify

CHANNEL_LABELS = {"woocommerce": "WooCommerce (TR)", "etsy": "Etsy (EN)"}
_ROW_H = 40


def has_warning(rec: CatalogRecord) -> bool:
    """Incomplete design or anything but a clean 'sellable'.

    Shown as a badge, never as a block: neither signal has had its real-world
    accuracy measured yet, and refusing to publish on an unmeasured VLM
    opinion would be worse than letting the user decide.
    """
    return (not rec.design_complete) or rec.sellability != "sellable"



class ListingWorker(QThread):
    """Generates listing copy off the UI thread.

    One product is one call to a language model. Measured against this
    machine's own Ollama they take 43-94 s each, so even a short list is
    minutes of work and a long one is an hour -- which is precisely how long
    the window used to sit frozen with no error, no progress and no way out.
    """

    progressed = pyqtSignal(int, int, str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, panel, records: list[CatalogRecord],
                 channel: str) -> None:
        super().__init__(panel)
        self._panel = panel
        self._records = records
        self._channel = channel
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # type: ignore[override]
        def _progress(done: int, total: int, file_id: str) -> None:
            if self._cancelled:
                # Between products only. A request already in flight runs to
                # its own timeout; there is nothing to interrupt it with.
                raise InterruptedError("kullanici iptal etti")
            self.progressed.emit(done, total, file_id)

        try:
            errors = self._panel.generate_for_selection(
                self._channel, records=self._records, progress=_progress)
        except InterruptedError:
            self.failed.emit("Iptal edildi.")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished_ok.emit(errors)

class PublishPanel(QWidget):
    selectionsChanged = pyqtSignal()

    def __init__(self, db: Database, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._current_id: int | None = None
        self._records: list[CatalogRecord] = []
        self._checked: set[str] = set()
        self._listing_worker: ListingWorker | None = None
        self._build_ui()
        self.refresh_selections()

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = page_layout(self)
        root.addWidget(PageHeader(
            "Publish",
            "Selection lists, generated listing copy and marketplace export.",
        ))

        # left: selection lists
        self._selection_list = QListWidget()
        self._selection_list.currentItemChanged.connect(self._on_selection_changed)
        b_new = QPushButton("New list")
        b_new.setProperty("variant", "primary")
        b_new.clicked.connect(self._on_new_clicked)
        b_del = QPushButton("Delete list")
        b_del.setProperty("variant", "ghost")
        b_del.clicked.connect(self._on_delete_clicked)

        left = QVBoxLayout()
        left.setSpacing(theme.SP_2)
        left.addWidget(field_label("Selection lists"))
        left.addWidget(self._selection_list, 1)
        left.addWidget(b_new)
        left.addWidget(b_del)

        # right: actions + items + detail
        self._cb_channel = QComboBox()
        for key, label in CHANNEL_LABELS.items():
            self._cb_channel.addItem(label, userData=key)

        self.remove_button = QPushButton("Remove ticked")
        self.remove_button.setProperty("variant", "ghost")
        self.remove_button.clicked.connect(self._on_remove_clicked)
        self.edit_button = QPushButton("Correct category / brand")
        self.edit_button.setProperty("variant", "ghost")
        self.edit_button.clicked.connect(self._on_edit_clicked)
        self.generate_button = QPushButton("Generate listing copy")
        self.generate_button.setProperty("variant", "primary")
        self.generate_button.clicked.connect(self._on_generate_clicked)
        self.reprice_button = QPushButton("Recalculate prices")
        self.reprice_button.clicked.connect(self._on_reprice_clicked)
        self.export_button = QPushButton("Export")
        self.export_button.setProperty("variant", "primary")
        self.export_button.clicked.connect(self._on_export_clicked)
        self.catalog_button = QPushButton("PDF katalog")
        self.catalog_button.setToolTip(
            "Seçili listeden basılabilir bir ürün kataloğu üretir: sayfa "
            "başına bir ürün, karşıdan ve çapraz görünüş, altında ID ve "
            "ağırlık. Maden ve taş rengi sorulur.")
        self.catalog_button.clicked.connect(self._on_catalog_clicked)

        actions = QHBoxLayout()
        actions.setSpacing(theme.SP_2)
        actions.addWidget(field_label("Channel"))
        actions.addWidget(self._cb_channel)
        for b in (self.remove_button, self.edit_button, self.generate_button,
                  self.reprice_button, self.export_button,
                  self.catalog_button):
            actions.addWidget(b)
        actions.addStretch(1)

        self._item_list = QListWidget()
        self._item_list.setSpacing(1)
        self._item_list.currentItemChanged.connect(self._on_item_changed)

        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setMaximumHeight(220)

        self._status = caption("")

        right = QVBoxLayout()
        right.setSpacing(theme.SP_2)
        right.addLayout(actions)
        right.addWidget(self._status)
        right.addWidget(self._item_list, 1)
        right.addWidget(self._detail)

        body = QHBoxLayout()
        body.setSpacing(theme.SP_4)
        body.addLayout(left, 1)
        body.addLayout(right, 3)
        root.addLayout(body, 1)

    # ── selection lists ──────────────────────────────────────────────────

    def refresh_selections(self) -> None:
        """Rebuild the list of lists AND re-read the open one's contents.

        Re-reading the contents is not incidental. This is the slot the Search
        tab's `selectionsChanged` lands on, and the whole reason Search emits
        it is that it just put records into one of these lists. Rebuilding only
        the names left the open list showing whatever it held when it was last
        clicked — products added from Search never appeared, and the PDF button
        had nothing to render.

        `_highlight` deliberately blocks signals so re-selecting the same row
        does not churn the UI, which is exactly why the reload has to be
        explicit here."""
        keep = self._current_id
        self._selection_list.blockSignals(True)
        self._selection_list.clear()
        for sel in list_selections(self._db.conn):
            item = QListWidgetItem(f"{sel.name}  ({sel.item_count})")
            item.setData(Qt.ItemDataRole.UserRole, sel.selection_id)
            self._selection_list.addItem(item)
        self._selection_list.blockSignals(False)
        if keep is not None:
            self._highlight(keep)
            self._reload_items()

    def _highlight(self, selection_id: int) -> None:
        for i in range(self._selection_list.count()):
            item = self._selection_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == selection_id:
                self._selection_list.blockSignals(True)
                self._selection_list.setCurrentRow(i)
                self._selection_list.blockSignals(False)
                return

    def selection_names(self) -> list[str]:
        return [s.name for s in list_selections(self._db.conn)]

    def current_selection_id(self) -> int | None:
        return self._current_id

    def select_selection(self, selection_id: int) -> None:
        self._current_id = selection_id
        self._highlight(selection_id)
        self._reload_items()

    def create_selection(self, name: str) -> int:
        sid = create_selection(self._db.conn, name)
        self.refresh_selections()
        self.select_selection(sid)
        self.selectionsChanged.emit()
        return sid

    def _on_new_clicked(self) -> None:
        name, ok = QInputDialog.getText(self, "New selection list", "Name:")
        if not (ok and name.strip()):
            return
        try:
            self.create_selection(name.strip())
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot create list", str(exc))

    def _on_delete_clicked(self) -> None:
        if self._current_id is None:
            return
        confirm = QMessageBox.question(
            self, "Delete list",
            "Delete this selection list? The catalogue records themselves "
            "are not touched.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        delete_selection(self._db.conn, self._current_id)
        self._current_id = None
        self._records = []
        self.refresh_selections()
        self._render_items()
        self.selectionsChanged.emit()

    def _on_selection_changed(self, current, _previous) -> None:
        self._current_id = (
            int(current.data(Qt.ItemDataRole.UserRole)) if current else None
        )
        self._reload_items()

    # ── items ────────────────────────────────────────────────────────────

    def _reload_items(self) -> None:
        self._checked.clear()
        if self._current_id is None:
            self._records = []
        else:
            ids = selection_file_ids(self._db.conn, self._current_id)
            self._records = get_products(self._db.conn, ids)
        self._render_items()

    def _render_items(self) -> None:
        self._item_list.clear()
        self._detail.clear()
        channel = self.current_channel()
        for rec in self._records:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, rec.file_id)
            widget = self._make_row(rec, channel)
            # Explicit height, not widget.sizeHint(): asking a widget for its
            # hint before it has been laid out returns a too-small value and
            # clips the badges off the bottom — exactly the bug that hit the
            # search results rows.
            item.setSizeHint(QSize(0, _ROW_H))
            self._item_list.addItem(item)
            self._item_list.setItemWidget(item, widget)
        warned = len(self.warning_file_ids())
        note = f"  ·  {warned} flagged for review" if warned else ""
        self._status.setText(f"{len(self._records)} product(s){note}")

    def _make_row(self, rec: CatalogRecord, channel: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(theme.SP_2, 4, theme.SP_2, 4)
        layout.setSpacing(theme.SP_3)

        box = QCheckBox()
        box.stateChanged.connect(
            lambda state, fid=rec.file_id: self.set_item_checked(fid, state != 0))
        layout.addWidget(box)

        ident = QLabel(rec.file_id)
        ident.setStyleSheet(
            f"color:{theme.TEXT_PRIMARY}; font-weight:600;"
            f" font-size:{theme.FS_SMALL}px;")
        ident.setMinimumWidth(150)
        layout.addWidget(ident)

        meta = QLabel(f"{rec.main_category} / {rec.subcategory}")
        meta.setStyleSheet(
            f"color:{theme.TEXT_SECONDARY}; font-size:{theme.FS_MICRO}px;")
        meta.setMinimumWidth(160)
        layout.addWidget(meta)

        weight = rec.metal_weights.silver_925_g if rec.metal_weights else None
        w_label = QLabel(f"{weight:.2f} g" if weight else "no weight")
        w_label.setStyleSheet(
            f"color:{theme.TEXT_MUTED if weight else theme.WARNING};"
            f" font-size:{theme.FS_MICRO}px;")
        layout.addWidget(w_label)

        listing = get_listing(self._db.conn, rec.file_id, channel,
                              with_variants=False)
        tone = {"draft": "neutral", "approved": "success",
                "exported": "info"}.get(listing.status if listing else "", "neutral")
        layout.addWidget(Badge(listing.status if listing else "no copy", tone))

        if get_overrides(self._db.conn, rec.file_id):
            layout.addWidget(Badge("corrected", "info"))
        if has_warning(rec):
            layout.addWidget(Badge("check", "warning"))
        layout.addStretch(1)
        return row

    def item_file_ids(self) -> list[str]:
        return [r.file_id for r in self._records]

    def warning_file_ids(self) -> list[str]:
        return [r.file_id for r in self._records if has_warning(r)]

    def set_item_checked(self, file_id: str, checked: bool) -> None:
        if checked:
            self._checked.add(file_id)
        else:
            self._checked.discard(file_id)

    def checked_file_ids(self) -> list[str]:
        return [r.file_id for r in self._records if r.file_id in self._checked]

    def remove_checked_items(self) -> int:
        if self._current_id is None or not self._checked:
            return 0
        n = remove_items(self._db.conn, self._current_id, self.checked_file_ids())
        self.refresh_selections()
        self._reload_items()
        self.selectionsChanged.emit()
        return n

    def _on_remove_clicked(self) -> None:
        self.remove_checked_items()

    def current_channel(self) -> str:
        return str(self._cb_channel.currentData() or "woocommerce")

    def _on_item_changed(self, current, _previous) -> None:
        if current is None:
            self._detail.clear()
            return
        file_id = current.data(Qt.ItemDataRole.UserRole)
        listing = get_listing(self._db.conn, file_id, self.current_channel())
        self._detail.setHtml(self._detail_html(file_id, listing))

    def _detail_html(self, file_id: str, listing) -> str:
        if listing is None:
            return (f"<div style='color:{theme.TEXT_MUTED};"
                    f" font-family:sans-serif; font-size:{theme.FS_SMALL}px;'>"
                    f"No listing copy generated yet for {file_id} on "
                    f"{CHANNEL_LABELS[self.current_channel()]}.</div>")
        prices = "<br>".join(
            f"{v.variant_key}: <b>{v.price:,.2f} {v.currency}</b> "
            f"<span style='color:{theme.TEXT_MUTED}'>{v.sku}</span>"
            for v in listing.variants
        ) or "—"
        return (
            f"<div style='font-family:sans-serif; font-size:{theme.FS_SMALL}px;"
            f" color:{theme.TEXT_PRIMARY};'>"
            f"<div style='font-weight:700; margin-bottom:4px;'>"
            f"{listing.title or '(no title)'}</div>"
            f"<div style='color:{theme.TEXT_SECONDARY}; margin-bottom:8px;'>"
            f"{listing.short_description or ''}</div>"
            f"<div style='color:{theme.TEXT_MUTED};'>tags: "
            f"{', '.join(listing.tags) or '—'}</div>"
            f"<div style='color:{theme.TEXT_MUTED}; margin-bottom:8px;'>"
            f"category: {listing.category_path or '—'}</div>"
            f"<div>{prices}</div></div>"
        )

    # ── manual corrections ───────────────────────────────────────────────

    def apply_override(self, file_id: str, field: str, value: str) -> None:
        set_override(self._db.conn, file_id, field, value)
        self._reload_items()

    def reset_override(self, file_id: str, field: str) -> None:
        clear_override(self._db.conn, file_id, field)
        self._reload_items()

    def override_fields(self, file_id: str) -> dict[str, str]:
        return get_overrides(self._db.conn, file_id)

    def _on_edit_clicked(self) -> None:
        item = self._item_list.currentItem()
        if item is None:
            return
        file_id = item.data(Qt.ItemDataRole.UserRole)
        field, ok = QInputDialog.getItem(
            self, "Correct field", "Field:", list(OVERRIDABLE_FIELDS), 0, False)
        if not ok:
            return
        value, ok = QInputDialog.getText(self, "Correct field", f"New {field}:")
        if not ok:
            return
        if value.strip():
            self.apply_override(file_id, field, value.strip())
        else:
            self.reset_override(file_id, field)

    # ── generation / pricing / export ────────────────────────────────────

    def records_to_generate(self) -> list[CatalogRecord]:
        """The products "Generate listing copy" should actually work on.

        Ticking a few products and pressing the button used to run the whole
        list: the loop read `self._records` while every other action on this
        panel ("Remove ticked", the export) reads the ticks. On a 60-item list
        that turned a three-product job into an hour-long one.
        """
        checked = set(self.checked_file_ids())
        if not checked:
            return list(self._records)
        return [r for r in self._records if r.file_id in checked]

    def generate_for_selection(self, channel: str | None = None, *,
                               records: list[CatalogRecord] | None = None,
                               progress=None) -> dict[str, str]:
        """Generate listing copy for each product. Returns {file_id: error}.

        A single bad model response must not abort the other 99, so failures
        are collected rather than raised.

        Synchronous on purpose: this is the callable the worker thread and the
        tests share, the same arrangement `build_catalog_pdf` uses. The GUI
        must NOT call it directly -- see `_on_generate_clicked` for why.

        `progress(done, total, file_id)` runs after each product and may raise
        to cancel, which is how the worker stops between products. It cannot
        interrupt a request already in flight; the provider's own timeout is
        what bounds that.
        """
        from catalog_organizer.core.config import (
            load_listing_prompt_templates, load_pipeline_settings,
        )
        from catalog_organizer.listing.generator import generate_listing, persist
        from catalog_organizer.vlm.text_provider import create_text_provider

        channel = channel or self.current_channel()
        cfg = load_pricing_config()
        templates = load_listing_prompt_templates()
        provider = create_text_provider(load_pipeline_settings().get("vlm", {}))

        records = self.records_to_generate() if records is None else records
        total = len(records)
        errors: dict[str, str] = {}
        for done, rec in enumerate(records, start=1):
            existing = get_listing(self._db.conn, rec.file_id, channel,
                                   with_variants=False)
            if existing is None or not existing.human_edited:
                # Never silently overwrite text the user wrote by hand.
                try:
                    result = generate_listing(rec, channel, provider,
                                              templates, cfg)
                    persist(self._db.conn, result, cfg,
                            selection_id=self._current_id)
                except Exception as exc:
                    errors[rec.file_id] = str(exc)
            if progress is not None:
                progress(done, total, rec.file_id)
        # NO widget work here. This runs on the worker thread, and touching a
        # Qt widget from one is undefined behaviour -- the list is redrawn by
        # whichever slot receives the worker's result, on the UI thread.
        return errors

    def _on_generate_clicked(self) -> None:
        """Run the generation on a worker thread, with progress and cancel.

        This used to run on the UI thread. Each product is one call to a local
        language model, and those were TIMED against this machine's own Ollama
        at 43 s, 69 s and 94 s -- so a list of thirty is half an hour during
        which the window never repaints and Windows paints it "Not Responding".
        It was reported as an infinite loop; nothing was looping. Every call is
        bounded by the provider's 300 s timeout and the work does finish. The
        UI simply had no way to say so, and no way to be stopped.
        """
        records = self.records_to_generate()
        if not records:
            return
        if self._listing_worker is not None:
            QMessageBox.information(self, "Listing copy",
                                    "Uretim zaten suruyor.")
            return

        progress = QProgressDialog(
            "Liste metni uretiliyor...", "Iptal", 0, len(records), self)
        progress.setWindowTitle("Listing copy")
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)

        worker = ListingWorker(self, records, self.current_channel())
        self._listing_worker = worker

        def _on_progress(done: int, total: int, file_id: str) -> None:
            progress.setValue(done)
            progress.setLabelText(f"{done}/{total} - {file_id}")

        def _cleanup() -> None:
            progress.close()
            self._listing_worker = None
            self._render_items()

        def _on_done(errors: dict) -> None:
            _cleanup()
            done = len(records) - len(errors)
            lines = [f"{done} listing(s) generated."]
            if errors:
                lines += ["", f"{len(errors)} failed:"]
                lines += [f"- {k}: {v}" for k, v in list(errors.items())[:10]]
            QMessageBox.information(self, "Listing copy", "\n".join(lines))

        def _on_failed(message: str) -> None:
            _cleanup()
            QMessageBox.warning(self, "Listing copy", message)

        worker.progressed.connect(_on_progress)
        worker.finished_ok.connect(_on_done)
        worker.failed.connect(_on_failed)
        progress.canceled.connect(worker.cancel)
        worker.start()

    def reprice_selection(self, channel: str | None = None) -> int:
        from catalog_organizer.listing.generator import reprice
        channel = channel or self.current_channel()
        cfg = load_pricing_config()
        n = 0
        for rec in self._records:
            n += reprice(self._db.conn, rec, channel, cfg)
        self._render_items()
        return n

    def _on_reprice_clicked(self) -> None:
        n = self.reprice_selection()
        QMessageBox.information(self, "Prices", f"{n} variant price(s) updated.")

    def approve_checked(self) -> int:
        n = 0
        channel = self.current_channel()
        for file_id in self.checked_file_ids():
            listing = get_listing(self._db.conn, file_id, channel,
                                  with_variants=False)
            if listing is not None:
                set_status(self._db.conn, listing.listing_id, "approved")
                n += 1
        self._render_items()
        return n

    def export_selection(
        self,
        channel: str | None = None,
        *,
        out_dir: Path | None = None,
        include_drafts: bool = False,
    ):
        """Write the marketplace file(s) for the current list."""
        from catalog_organizer.db.listings import list_listings

        channel = channel or self.current_channel()
        statuses = ("draft", "approved") if include_drafts else ("approved",)
        listings = list_listings(
            self._db.conn, channel=channel, statuses=statuses,
            file_ids=self.item_file_ids(),
        )
        out_dir = out_dir or (data_dir() / "exports")
        name = self._current_selection_name() or "selection"
        stamp = date.today().isoformat()
        stem = f"{stamp}-{slugify(name)}-{channel}"

        if channel == "woocommerce":
            results = [woo_export.export_woocommerce(
                listings, out_dir / f"{stem}.csv")]
        else:
            results = [
                etsy_export.export_etsy_csv(listings, out_dir / f"{stem}.csv"),
                etsy_export.export_etsy_json(listings, out_dir / f"{stem}.json"),
            ]

        exported_ids: list[int] = []
        for result in results:
            record_export(
                self._db.conn, channel=channel, selection_id=self._current_id,
                file_path=str(result.file_path), row_count=result.row_count,
                skipped_count=len(result.skipped),
            )
            exported_ids.extend(result.exported_listing_ids)
        mark_exported(self._db.conn, sorted(set(exported_ids)))
        self._render_items()
        return results

    def _current_selection_name(self) -> str | None:
        if self._current_id is None:
            return None
        for sel in list_selections(self._db.conn):
            if sel.selection_id == self._current_id:
                return sel.name
        return None

    def _on_export_clicked(self) -> None:
        if self._current_id is None:
            return
        try:
            results = self.export_selection()
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        lines = [f"{r.file_path.name}: {r.row_count} row(s)" for r in results]
        skipped = {k: v for r in results for k, v in r.skipped.items()}
        if skipped:
            lines.append("")
            lines.append(f"{len(skipped)} product(s) skipped:")
            lines.extend(
                f"· {issues[0]}" for issues in list(skipped.values())[:10])
        lines.append("")
        lines.append("Images are intentionally left blank — the current "
                     "snapshots are grey clay renders for classification, "
                     "not product photos.")
        QMessageBox.information(self, "Export complete", "\n".join(lines))

    # ── PDF catalogue ────────────────────────────────────────────────────

    def catalog_records(self, start: int = 0, count: int | None = None
                        ) -> list[CatalogRecord]:
        records = self._records[start:]
        return records[:count] if count is not None else records

    def build_catalog_pdf(
        self,
        *,
        metal_key: str,
        stone_key: str,
        title: str | None = None,
        out_path: Path | None = None,
        resolution: int = 1400,
        cover: bool = True,
        start: int = 0,
        count: int | None = None,
        progress=None,
    ):
        """Render + write the catalogue synchronously. The GUI path uses a
        worker thread; this is the callable the worker and the tests share."""
        from catalog_organizer.export.catalog_pdf import render_and_export
        from catalog_organizer.listing.schemas import slugify

        records = self.catalog_records(start, count)
        name = title or self._current_selection_name() or "Ürün Kataloğu"
        if out_path is None:
            stem = f"{date.today().isoformat()}-{slugify(name)}-katalog"
            out_path = data_dir() / "exports" / f"{stem}.pdf"
        return render_and_export(
            records, Path(out_path), metal_key=metal_key, stone_key=stone_key,
            title=name, resolution=resolution, cover=cover, progress=progress,
            db_conn=self._db.conn,
        )

    def _on_catalog_clicked(self) -> None:
        from catalog_organizer.gui.panels.catalog_pdf_dialog import (
            CatalogPdfDialog, CatalogWorker, make_progress_dialog,
        )
        from catalog_organizer.listing.schemas import slugify

        if not self._records:
            QMessageBox.information(
                self, "PDF Katalog",
                "Önce bir seçim listesi seçin ve içine ürün ekleyin.")
            return

        name = self._current_selection_name() or "Ürün Kataloğu"
        stem = f"{date.today().isoformat()}-{slugify(name)}-katalog"
        dialog = CatalogPdfDialog(
            default_title=name, item_count=len(self._records),
            out_path=data_dir() / "exports" / f"{stem}.pdf", parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        options = dialog.options()
        start, count = dialog.slice_bounds()
        records = self.catalog_records(start, count)
        if not records:
            QMessageBox.information(self, "PDF Katalog",
                                    "Seçilen aralıkta ürün yok.")
            return

        progress = make_progress_dialog(len(records), self)
        worker = CatalogWorker(records, options, db_conn=self._db.conn, parent=self)
        self._catalog_worker = worker

        def _on_progress(done: int, total: int, file_id: str) -> None:
            progress.setValue(done - 1)
            progress.setLabelText(f"{done}/{total} · {file_id}")

        def _cleanup() -> None:
            progress.close()
            self._catalog_worker = None

        def _on_done(path, errors: dict) -> None:
            _cleanup()
            lines = [f"Katalog yazıldı: {path}",
                     f"{len(records)} sayfa · {options.title}"]
            if errors:
                lines += ["", f"{len(errors)} üründe sorun:"]
                lines += [f"· {k}: {v}" for k, v in list(errors.items())[:8]]
            QMessageBox.information(self, "PDF Katalog", "\n".join(lines))

        def _on_failed(message: str) -> None:
            _cleanup()
            QMessageBox.warning(self, "PDF Katalog", message)

        worker.progressed.connect(_on_progress)
        worker.finished_ok.connect(_on_done)
        worker.failed.connect(_on_failed)
        progress.canceled.connect(worker.cancel)
        worker.start()
