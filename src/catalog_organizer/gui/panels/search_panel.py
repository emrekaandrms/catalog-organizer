"""Search panel — multi-filter + free-text search over the catalogue.

the catalogue-search design notes §5: category / subcategory / stone-status / brand /
sprue filters plus a free-text box over controlled_tags + rich_description.
Filtering runs entirely in RAM via `catalog.search.search_records` — no
search engine is needed at 10-50k records.

UI chrome is English throughout, matching the rest of the app. The mixed
Turkish/English labelling this panel shipped with read as an oversight
rather than a choice; the *content* being searched stays Turkish either way.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from catalog_organizer.core.config import load_categories
from catalog_organizer.db.connection import Database
from catalog_organizer.db.products import (
    UNKNOWN_BRAND,
    count_products,
    distinct_brands,
    get_product,
    search_product_ids,
    search_products,
)
from catalog_organizer.db.selections import add_items, list_selections
from catalog_organizer.db.sync import sync_from_jsonl
from catalog_organizer.core.schemas import CatalogRecord
from catalog_organizer.gui import theme
from catalog_organizer.gui.widgets.components import (
    Badge,
    PageHeader,
    caption,
    field_label,
    page_layout,
)

_ANY = "(any)"
_STONE_STATUS_LABELS = {
    "stone": "With stones",
    "polished": "Polished / no stones",
    "unclear": "Unclear",
}
_SPRUE_LABELS = {"yes": "Has sprue", "no": "No sprue"}
_THUMB_PX = 76
# thumbnail + vertical margins, with headroom for the badge strip
_ROW_H = _THUMB_PX + 34


def _rounded_thumbnail(path: Path, size: int = _THUMB_PX) -> QPixmap | None:
    """Scale a snapshot into a rounded square tile.

    Snapshots are rendered on white, which on a dark surface reads as a
    glaring rectangle. Rounding the corners turns it into something that
    looks like a deliberate product tile instead of an unstyled image.
    """
    if not path.exists():
        return None
    src = QPixmap(str(path))
    if src.isNull():
        return None
    scaled = src.scaled(
        size, size,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )
    out = QPixmap(size, size)
    out.fill(Qt.GlobalColor.transparent)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    clip = QPainterPath()
    clip.addRoundedRect(0, 0, size, size, theme.RADIUS_MD, theme.RADIUS_MD)
    painter.setClipPath(clip)
    painter.drawPixmap(
        (size - scaled.width()) // 2, (size - scaled.height()) // 2, scaled)
    painter.end()
    return out


class _ResultRow(QWidget):
    """Thumbnail + title + meta + state badges for one catalogue record.

    A custom widget rather than an icon-and-text list item so state can be
    shown as coloured badges; the previous plain-text rows gave no way to
    see at a glance which hits still need review.
    """

    def __init__(self, rec: CatalogRecord, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(theme.SP_2, theme.SP_2, theme.SP_2, theme.SP_2)
        row.setSpacing(theme.SP_3)

        self.check = QCheckBox()
        self.check.setToolTip("Tick to include this record when adding to a "
                              "selection list.")
        row.addWidget(self.check, alignment=Qt.AlignmentFlag.AlignVCenter)

        thumb = QLabel()
        thumb.setFixedSize(_THUMB_PX, _THUMB_PX)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = None
        if rec.snapshot_paths:
            pix = _rounded_thumbnail(Path(rec.snapshot_paths[-1]))
        if pix is not None:
            thumb.setPixmap(pix)
        else:
            thumb.setText("—")
            thumb.setStyleSheet(
                f"color: {theme.TEXT_MUTED};"
                f" background: {theme.BG_SUNKEN};"
                f" border-radius: {theme.RADIUS_MD}px;")
        row.addWidget(thumb)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(3)

        title = QLabel(Path(rec.source_path).name)
        title.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.FS_BODY}px;"
            " font-weight: 600;")
        col.addWidget(title)

        bits = [f"{rec.main_category} / {rec.subcategory}"]
        if rec.brand:
            bits.append(rec.brand)
        if rec.rich_description:
            bits.append(rec.rich_description[:70])
        meta = QLabel("  ·  ".join(bits))
        meta.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: {theme.FS_MICRO}px;")
        col.addWidget(meta)

        badges = QHBoxLayout()
        badges.setContentsMargins(0, 0, 0, 0)
        badges.setSpacing(theme.SP_2)
        if rec.needs_manual_review:
            badges.addWidget(Badge("needs review", "warning"))
        else:
            badges.addWidget(Badge("final", "success"))
        if rec.stone_summary.status == "stone":
            badges.addWidget(Badge("stones", "info"))
        if rec.sprue.detected:
            badges.addWidget(Badge("sprue", "neutral"))
        badges.addStretch()
        col.addLayout(badges)
        col.addStretch()

        row.addLayout(col, stretch=1)

    def set_checked(self, checked: bool) -> None:
        """Set the box without echoing a stateChanged back to the panel —
        otherwise 'check all' re-enters set_checked once per row."""
        self.check.blockSignals(True)
        self.check.setChecked(checked)
        self.check.blockSignals(False)


class SearchPanel(QWidget):
    """Catalogue search, backed by SQLite.

    Filtering runs in SQL rather than a Python loop over an in-RAM index:
    at 100k records the loop is the difference between an instant screen and
    a visible stall, and the DB is also what the publish tab reads.
    """

    selectionsChanged = pyqtSignal()

    def __init__(self, db: Database, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db = db
        self._categories = load_categories().get("categories", {})
        self._results: list[CatalogRecord] = []
        self._rows: dict[str, _ResultRow] = {}
        self._checked: set[str] = set()
        self._build_ui()
        self._populate_filters()
        self._run_search()

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = page_layout(self)

        header = PageHeader(
            "Search",
            "Filter the catalogue, or type free text to match tags and "
            "descriptions.",
        )
        b_refresh = QPushButton("⟳  Reload")
        b_refresh.setProperty("variant", "ghost")
        b_refresh.setToolTip(
            "Re-read the catalogue file from disk — use this to pick up "
            "records written by a batch running outside this window.")
        b_refresh.clicked.connect(self._refresh_from_disk)
        header.add_action(b_refresh)
        root.addWidget(header)

        # Filters
        filters = QGridLayout()
        filters.setHorizontalSpacing(theme.SP_3)
        filters.setVerticalSpacing(theme.SP_1)
        self._cb_category = QComboBox()
        self._cb_subcategory = QComboBox()
        self._cb_stone = QComboBox()
        self._cb_brand = QComboBox()
        self._cb_sprue = QComboBox()
        for col, (label, widget) in enumerate((
            ("Category", self._cb_category),
            ("Subcategory", self._cb_subcategory),
            ("Stones", self._cb_stone),
            ("Brand", self._cb_brand),
            ("Sprue", self._cb_sprue),
        )):
            filters.addWidget(field_label(label), 0, col)
            filters.addWidget(widget, 1, col)
            filters.setColumnStretch(col, 1)
        root.addLayout(filters)
        self._cb_category.currentTextChanged.connect(self._on_category_changed)
        for cb in (self._cb_category, self._cb_subcategory, self._cb_stone,
                   self._cb_brand, self._cb_sprue):
            cb.currentIndexChanged.connect(self._run_search)

        # Free text
        text_row = QHBoxLayout()
        text_row.setSpacing(theme.SP_2)
        self._txt = QLineEdit()
        self._txt.setPlaceholderText(
            "free text — e.g.  taşlı haç jesus gurmet zincir")
        self._txt.setClearButtonEnabled(True)
        self._txt.returnPressed.connect(self._run_search)
        b_search = QPushButton("Search")
        b_search.setProperty("variant", "primary")
        b_search.clicked.connect(self._run_search)
        b_clear = QPushButton("Clear")
        b_clear.setProperty("variant", "ghost")
        b_clear.clicked.connect(self._clear_filters)
        text_row.addWidget(self._txt, stretch=1)
        text_row.addWidget(b_search)
        text_row.addWidget(b_clear)
        root.addLayout(text_row)

        # Selection toolbar — this is what turns 1000 analysed files into a
        # 100-item publish list: filter, then add every match in one click.
        sel_row = QHBoxLayout()
        sel_row.setSpacing(theme.SP_2)
        self._cb_selection = QComboBox()
        self._cb_selection.setMinimumWidth(200)
        self._b_add = QPushButton("Add checked")
        self._b_add_all = QPushButton("Add all matching")
        self._b_check_all = QPushButton("Tick all")
        self._b_check_none = QPushButton("Untick all")
        for b in (self._b_add, self._b_add_all):
            b.setProperty("variant", "primary")
        for b in (self._b_check_all, self._b_check_none):
            b.setProperty("variant", "ghost")
        self._b_add.clicked.connect(self._on_add_checked)
        self._b_add_all.clicked.connect(self._on_add_all_matching)
        self._b_check_all.clicked.connect(self.check_all_visible)
        self._b_check_none.clicked.connect(self.uncheck_all)
        sel_row.addWidget(field_label("Selection"))
        sel_row.addWidget(self._cb_selection)
        for b in (self._b_add, self._b_add_all, self._b_check_all,
                  self._b_check_none):
            sel_row.addWidget(b)
        sel_row.addStretch(1)
        root.addLayout(sel_row)

        self._status = caption("")
        root.addWidget(self._status)
        self.refresh_selections()

        # Results | detail
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._list = QListWidget()
        self._list.setSpacing(2)
        self._list.currentItemChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._list)

        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([760, 520])
        root.addWidget(splitter, stretch=1)
        self._show_empty_detail()

    # ── filter population ───────────────────────────────────────────────

    def _populate_filters(self) -> None:
        self._cb_category.blockSignals(True)
        self._cb_category.clear()
        self._cb_category.addItem(_ANY)
        self._cb_category.addItems(sorted(self._categories.keys()))
        self._cb_category.blockSignals(False)
        self._on_category_changed(_ANY)

        for combo, labels in ((self._cb_stone, _STONE_STATUS_LABELS),
                              (self._cb_sprue, _SPRUE_LABELS)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(_ANY)
            for key, label in labels.items():
                combo.addItem(label, userData=key)
            combo.blockSignals(False)

        self._rebuild_brand_combo()

    def _rebuild_brand_combo(self) -> None:
        current = self._cb_brand.currentData() if self._cb_brand.count() else None
        self._cb_brand.blockSignals(True)
        self._cb_brand.clear()
        self._cb_brand.addItem(_ANY)
        self._cb_brand.addItem("Unbranded", userData=UNKNOWN_BRAND)
        for b in distinct_brands(self._db.conn):
            self._cb_brand.addItem(b, userData=b)
        if current:
            idx = self._cb_brand.findData(current)
            if idx >= 0:
                self._cb_brand.setCurrentIndex(idx)
        self._cb_brand.blockSignals(False)

    def _on_category_changed(self, category: str) -> None:
        self._cb_subcategory.blockSignals(True)
        self._cb_subcategory.clear()
        self._cb_subcategory.addItem(_ANY)
        if category and category != _ANY:
            subs = (self._categories.get(category) or {}).get("subcategories", [])
            self._cb_subcategory.addItems(subs)
        self._cb_subcategory.blockSignals(False)

    def _clear_filters(self) -> None:
        for w in (self._cb_category, self._cb_subcategory, self._cb_stone,
                  self._cb_brand, self._cb_sprue):
            w.blockSignals(True)
            w.setCurrentIndex(0)
            w.blockSignals(False)
        self._txt.clear()
        self._run_search()

    # ── search ───────────────────────────────────────────────────────────

    def _refresh_from_disk(self) -> None:
        """Re-sync catalog_master.jsonl into the database.

        Records appended by a batch in another process (the `run-pilot` CLI,
        a second window) are invisible until this runs — this panel silently
        showing a stale, empty catalogue was a real reported bug.
        """
        self.reload_from_disk()

    def reload_from_disk(self, jsonl_path: Path | None = None) -> None:
        sync_from_jsonl(self._db.conn, jsonl_path)
        self._rebuild_brand_combo()
        self._run_search()

    def _combo_value(self, combo: QComboBox) -> str | None:
        if combo.currentIndex() <= 0:
            return None
        data = combo.currentData()
        return data if data is not None else combo.currentText()

    def current_filters(self) -> dict[str, object]:
        """The active filters, shaped so `search_product_ids(**filters)` works
        — that is how 'add all matching' reaches beyond the visible rows."""
        sprue_val = self._combo_value(self._cb_sprue)
        return {
            "category": self._combo_value(self._cb_category),
            "subcategory": self._combo_value(self._cb_subcategory),
            "stone_status": self._combo_value(self._cb_stone),
            "brand": self._combo_value(self._cb_brand),
            "sprue_detected": None if sprue_val is None else sprue_val == "yes",
            "free_text": self._txt.text().strip() or None,
        }

    def _run_search(self) -> None:
        self._results = search_products(self._db.conn, **self.current_filters())
        # Ticks belong to a result set. Carrying them across a filter change
        # would let the user add records they can no longer see.
        self._checked.clear()
        self._rows.clear()
        total = count_products(self._db.conn)
        self._status.setText(f"{len(self._results)} of {total} records match")
        self._render_results()

    def run_search(self) -> None:
        self._run_search()

    def _render_results(self) -> None:
        self._list.clear()
        self._show_empty_detail()
        for rec in self._results:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, rec.file_id)
            widget = _ResultRow(rec)
            widget.check.stateChanged.connect(
                lambda state, fid=rec.file_id:
                    self._on_row_toggled(fid, state != 0)
            )
            self._rows[rec.file_id] = widget
            # Explicit height: title + meta + badge strip + margins. Asking
            # the widget for sizeHint() here returns a pre-layout value and
            # clipped the badges off the bottom of every row.
            item.setSizeHint(QSize(0, _ROW_H))
            self._list.addItem(item)
            self._list.setItemWidget(item, widget)

    # ── selection ────────────────────────────────────────────────────────

    def refresh_selections(self) -> None:
        """Repopulate the target-list dropdown. Called when the publish tab
        creates or deletes a list."""
        current = self._cb_selection.currentData()
        self._cb_selection.blockSignals(True)
        self._cb_selection.clear()
        for sel in list_selections(self._db.conn):
            self._cb_selection.addItem(
                f"{sel.name}  ({sel.item_count})", userData=sel.selection_id)
        if current is not None:
            idx = self._cb_selection.findData(current)
            if idx >= 0:
                self._cb_selection.setCurrentIndex(idx)
        self._cb_selection.blockSignals(False)
        has_any = self._cb_selection.count() > 0
        self._b_add.setEnabled(has_any)
        self._b_add_all.setEnabled(has_any)
        if not has_any:
            self._cb_selection.setToolTip(
                "No selection lists yet — create one in the Publish tab.")

    def current_selection_id(self) -> int | None:
        data = self._cb_selection.currentData()
        return int(data) if data is not None else None

    def _on_row_toggled(self, file_id: str, checked: bool) -> None:
        if checked:
            self._checked.add(file_id)
        else:
            self._checked.discard(file_id)

    def checked_file_ids(self) -> list[str]:
        return [r.file_id for r in self._results if r.file_id in self._checked]

    def result_file_ids(self) -> list[str]:
        return [r.file_id for r in self._results]

    def set_checked(self, file_id: str, checked: bool) -> None:
        self._on_row_toggled(file_id, checked)
        row = self._rows.get(file_id)
        if row is not None:
            row.set_checked(checked)

    def check_all_visible(self) -> None:
        for rec in self._results:
            self.set_checked(rec.file_id, True)

    def uncheck_all(self) -> None:
        for file_id in list(self._checked):
            self.set_checked(file_id, False)

    def add_checked_to_selection(self, selection_id: int) -> int:
        n = add_items(self._db.conn, selection_id, self.checked_file_ids())
        self.refresh_selections()
        self.selectionsChanged.emit()
        return n

    def add_all_matching_to_selection(self, selection_id: int) -> int:
        """Adds every record the current filter matches — not just the ones
        on screen. This is the point of the button: filter down to 137 hits,
        add them all, then prune."""
        ids = search_product_ids(self._db.conn, **self.current_filters())
        n = add_items(self._db.conn, selection_id, ids)
        self.refresh_selections()
        self.selectionsChanged.emit()
        return n

    def _on_add_checked(self) -> None:
        sid = self.current_selection_id()
        if sid is None:
            return
        n = self.add_checked_to_selection(sid)
        self._status.setText(f"{n} record(s) added to the selection")

    def _on_add_all_matching(self) -> None:
        sid = self.current_selection_id()
        if sid is None:
            return
        n = self.add_all_matching_to_selection(sid)
        self._status.setText(f"{n} matching record(s) added to the selection")

    # ── test / programmatic filter access ────────────────────────────────

    def set_category(self, value: str | None) -> None:
        self._cb_category.setCurrentText(value or _ANY)

    def set_free_text(self, value: str | None) -> None:
        self._txt.setText(value or "")

    # ── detail pane ──────────────────────────────────────────────────────

    def _show_empty_detail(self) -> None:
        self._detail.setHtml(
            f"<div style='color:{theme.TEXT_MUTED}; font-family:sans-serif;"
            f" font-size:{theme.FS_SMALL}px; padding:{theme.SP_5}px;'>"
            "Select a result to see its measurements, weights and tags."
            "</div>")

    def _on_selection_changed(self, current: QListWidgetItem | None, _prev) -> None:
        if current is None:
            self._show_empty_detail()
            return
        rec = get_product(self._db.conn, current.data(Qt.ItemDataRole.UserRole))
        if rec is None:
            self._show_empty_detail()
            return
        self._detail.setHtml(self._detail_html(rec))

    def _sprue_line(self, rec: CatalogRecord) -> str:
        """Sprue row: presence, volume, and what the runner itself weighs.

        A detected sprue with no volume (VLM-only) is called out explicitly
        — nothing was subtracted in that case, so the net weights below
        still include the runner and are overstated.
        """
        sp = rec.sprue
        if not sp.detected:
            return "none"
        if not sp.estimated_volume_mm3:
            return ("<span style='color:%s'>present — volume not measured; "
                    "the weights below still INCLUDE it</span>" % theme.WARNING)
        line = f"{sp.estimated_volume_mm3:.1f} mm³ <span style='color:%s'>[%s]</span>" % (
            theme.TEXT_MUTED, sp.source)
        try:
            from catalog_organizer.catalog.writer import _sprue_weights  # noqa: PLC0415
            w = _sprue_weights(sp)
        except Exception:
            w = None
        if w:
            line += (f" → {w['silver_925_g']:.2f} g silver / "
                     f"{w['gold_14k_yellow_g']:.2f} g 14K-Y")
        return line

    def _detail_html(self, rec: CatalogRecord) -> str:
        def row(label: str, value: str) -> str:
            return (
                f"<tr>"
                f"<td style='color:{theme.TEXT_MUTED}; padding:3px 12px 3px 0;"
                f" white-space:nowrap; vertical-align:top;'>{label}</td>"
                f"<td style='color:{theme.TEXT_PRIMARY}; padding:3px 0;'>{value}</td>"
                f"</tr>"
            )

        m = rec.measurements
        weights = []
        if rec.metal_weights:
            for field, val in rec.metal_weights.model_dump().items():
                if val:
                    weights.append(f"{field.replace('_g', '')}: <b>{val:.2f} g</b>")

        parts = [
            f"<div style='font-family:sans-serif; font-size:{theme.FS_SMALL}px;"
            f" color:{theme.TEXT_PRIMARY};'>",
            f"<div style='font-size:{theme.FS_HEADING}px; font-weight:700;"
            f" margin-bottom:2px;'>{Path(rec.source_path).name}</div>",
            f"<div style='color:{theme.TEXT_MUTED}; margin-bottom:14px;'>"
            f"{rec.file_id}</div>",
            "<table cellspacing='0' cellpadding='0' width='100%'>",
            row("Category", f"{rec.main_category} / {rec.subcategory}"),
            row("Brand", rec.brand or "unknown"),
            row("Description", rec.rich_description or "—"),
            row("Stones", rec.stone_summary.status),
            row("Sprue", self._sprue_line(rec)),
            row("Net weights", "<br>".join(weights) if weights else "—"),
            row("Dimensions",
                f"{m.bbox_width_mm:.1f} × {m.bbox_height_mm:.1f} × "
                f"{m.bbox_depth_mm:.1f} mm"),
            row("Net volume", f"{m.volume_mm3:.1f} mm³"),
            row("Tags", ", ".join(rec.controlled_tags)),
            row("Review",
                f"<span style='color:{theme.WARNING}'>"
                f"{rec.review_reason or 'flagged'}</span>"
                if rec.needs_manual_review else "not needed"),
            row("Source", f"<span style='color:{theme.TEXT_MUTED}'>"
                          f"{rec.source_path}</span>"),
            "</table></div>",
        ]
        return "".join(parts)
