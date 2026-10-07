"""Main application window — sidebar navigation + QStackedWidget content."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from catalog_organizer.catalog.index import CatalogIndex
from catalog_organizer.db.connection import Database
from catalog_organizer.db.sync import sync_from_jsonl
from catalog_organizer.gui.panels.analyze_panel import AnalyzePanel
from catalog_organizer.gui.panels.dashboard import DashboardPanel
from catalog_organizer.gui.panels.diagnostics import DiagnosticsPanel
from catalog_organizer.gui.panels.logs import LogsPanel
from catalog_organizer.gui.panels.process import ProcessPanel
from catalog_organizer.gui.panels.publish_panel import PublishPanel
from catalog_organizer.gui.panels.render_panel import RenderPanel
from catalog_organizer.gui.panels.search_panel import SearchPanel
from catalog_organizer.gui.panels.settings import SettingsPanel

# (label displayed in sidebar, internal key)
# Scan/Pilot/Catalog/MovePlan were collapsed into the single Process panel
# (2026-05-18). The app's job is folder → CSV; browse/listing UI is deferred.
#
# Glyphs are drawn from one geometric set at a single optical weight. The
# previous mix (⬛ ⚖ ▶ 🔍 ◔ ⚙ ⌕) combined a full-width block, an emoji that
# rendered in colour, and three different stroke weights, so the column never
# lined up. Labels are English throughout for the same reason — "Ara" sitting
# among English items read as a bug rather than a choice.
_NAV: list[tuple[str, str]] = [
    ("◈   Dashboard",   "dashboard"),
    ("◑   Analyze",     "analyze"),
    ("▷   Process",     "process"),
    ("⌕   Search",      "search"),
    ("◇   Publish",     "publish"),
    ("◐   Render",      "render"),
    ("◎   Diagnostics", "diagnostics"),
    ("⚙   Settings",    "settings"),
    ("≡   Logs",        "logs"),
]


class _PlaceholderPanel(QWidget):
    """Stub for panels not yet implemented."""
    def __init__(self, name: str) -> None:
        super().__init__()
        lbl = QLabel(f"{name} — coming in a future session")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #5a5a7a; font-size: 14px;")
        layout = QVBoxLayout(self)
        layout.addWidget(lbl)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Catalog Organizer")
        self.setMinimumSize(1280, 800)

        self._index = CatalogIndex()
        # Search and Publish read SQLite, not the in-RAM index. Sync once
        # at startup so the catalogue is visible immediately; the Search
        # tab's Reload button re-syncs on demand.
        self._db = Database()
        sync_from_jsonl(self._db.conn)
        self._build_ui()
        self._build_shortcuts()
        self._start_status_refresh()
        # Defer the first-run check so the main window paints first;
        # an immediate modal dialog from inside __init__ looks like a crash.
        QTimer.singleShot(0, self._check_first_run)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.setCentralWidget(central)

        # Sidebar: brand block, nav, footer. 160 px was too narrow for the
        # longest label ("Diagnostics"), which forced a horizontal scrollbar
        # to appear under the nav list.
        from catalog_organizer.gui import theme  # noqa: PLC0415

        sidebar_col = QWidget()
        sidebar_col.setFixedWidth(212)
        sidebar_col.setStyleSheet(f"background: {theme.BG_SIDEBAR};")
        sc = QVBoxLayout(sidebar_col)
        sc.setContentsMargins(0, 0, 0, 0)
        sc.setSpacing(0)

        brand = QWidget()
        bl = QVBoxLayout(brand)
        bl.setContentsMargins(theme.SP_4, theme.SP_4, theme.SP_4, theme.SP_3)
        bl.setSpacing(1)
        brand_title = QLabel("Catalog Organizer")
        brand_title.setObjectName("BrandTitle")
        bl.addWidget(brand_title)
        brand_sub = QLabel("Jewelry CAD catalogue")
        brand_sub.setObjectName("BrandSub")
        bl.addWidget(brand_sub)
        sc.addWidget(brand)

        self._sidebar = QListWidget()
        self._sidebar.setObjectName("Sidebar")
        self._sidebar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._sidebar.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for label, _ in _NAV:
            self._sidebar.addItem(QListWidgetItem(label))
        sc.addWidget(self._sidebar, stretch=1)

        self._sidebar_footer = QLabel("")
        self._sidebar_footer.setObjectName("SidebarFooter")
        self._sidebar_footer.setWordWrap(True)
        sc.addWidget(self._sidebar_footer)

        h.addWidget(sidebar_col)

        # Stacked content area (must exist before connecting sidebar signal)
        self._stack = QStackedWidget()
        h.addWidget(self._stack, stretch=1)

        for _, key in _NAV:
            self._stack.addWidget(self._make_panel(key))

        self._sidebar.currentRowChanged.connect(self._on_nav)
        self._sidebar.setCurrentRow(0)

        # Each tab owns selection lists the other one shows; without this
        # a list created in Publish never appears in Search's dropdown.
        if hasattr(self, "_search_panel") and hasattr(self, "_publish_panel"):
            self._publish_panel.selectionsChanged.connect(
                self._search_panel.refresh_selections)
            self._search_panel.selectionsChanged.connect(
                self._publish_panel.refresh_selections)

        # Status bar. The provider indicator gets a coloured dot so a dead
        # backend is visible without reading the text — this is the one piece
        # of state that silently invalidates every run.
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_dot = QLabel("●")
        self._status_dot.setProperty("tone", "neutral")
        sb.addPermanentWidget(self._status_dot)
        self._status_vlm   = QLabel("VLM: –")
        # Cost widget — live session total in USD. Updates via the
        # CostTracker singleton whenever a classify() call records.
        self._status_cost  = QLabel("$0.0000 · 0 calls")
        self._status_cost.setToolTip("VLM cost this session (resets when a new batch starts)")
        self._status_batch = QLabel("Batch: idle")
        self._status_items = QLabel("Items: 0/0")
        for w in (self._status_vlm, self._status_cost,
                  self._status_batch, self._status_items):
            sb.addPermanentWidget(w)

        # Wire the cost tracker to the status bar
        from catalog_organizer.core.cost_tracker import tracker_singleton  # noqa: PLC0415
        tracker_singleton().changed.connect(self._refresh_cost_widget)
        self._refresh_cost_widget()

    def _make_panel(self, key: str) -> QWidget:
        if key == "dashboard":
            return DashboardPanel(self._index)
        if key == "analyze":
            return AnalyzePanel()
        if key == "process":
            return ProcessPanel(self._index)
        if key == "search":
            self._search_panel = SearchPanel(self._db)
            return self._search_panel
        if key == "publish":
            self._publish_panel = PublishPanel(self._db)
            return self._publish_panel
        if key == "render":
            self._render_panel = RenderPanel(self._db)
            return self._render_panel
        if key == "diagnostics":
            return DiagnosticsPanel()
        if key == "settings":
            return SettingsPanel()
        if key == "logs":
            return LogsPanel()
        return _PlaceholderPanel(key.replace("_", " ").title())

    def _on_nav(self, row: int) -> None:
        self._stack.setCurrentIndex(row)

    # ── Keyboard shortcuts (Ctrl+1 … Ctrl+7) ─────────────────────────────────

    def _build_shortcuts(self) -> None:
        for i in range(1, len(_NAV) + 1):
            def _go(idx: int = i - 1) -> None:
                self._sidebar.setCurrentRow(idx)
            sc = QShortcut(QKeySequence(f"Ctrl+{i}"), self)
            sc.activated.connect(_go)

    # ── Status bar refresh ────────────────────────────────────────────────────

    def _start_status_refresh(self) -> None:
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start(3_000)
        self._refresh_status()

    def _refresh_status(self) -> None:
        records   = self._index.all()
        total     = len(records)
        processed = sum(1 for r in records if r.state in ("final", "needs_review"))
        review    = sum(1 for r in records if r.needs_manual_review)
        self._status_items.setText(f"Items: {processed}/{total}")
        # Sidebar footer doubles as the review backlog counter: roughly half
        # the catalogue lands in needs_review, and it was previously only
        # visible by opening the Dashboard.
        if total:
            self._sidebar_footer.setText(
                f"{total} catalogued · {review} need review"
            )
        else:
            self._sidebar_footer.setText("Catalogue empty")
        # Provider/model label is independent — pull from settings so a
        # provider swap reflects without restarting the app.
        self._refresh_provider_label()

    def _refresh_provider_label(self) -> None:
        """Show 'VLM: openai/gpt-4o' or 'VLM: ollama/qwen3.5...' in the
        status bar. Falls back to a generic '–' if the config can't be
        read (don't crash the bar over a typo)."""
        try:
            from catalog_organizer.core.config import load_pipeline_settings  # noqa: PLC0415
            vlm = load_pipeline_settings().get("vlm", {})
            provider = vlm.get("provider", "ollama")
            block = vlm.get(provider, {})
            model = block.get("model", "?")
            self._status_vlm.setText(f"{provider} · {model}")
            self._set_status_tone("success" if model and model != "?" else "warning")
        except Exception:
            self._status_vlm.setText("VLM: –")
            self._set_status_tone("danger")

    def _set_status_tone(self, tone: str) -> None:
        if self._status_dot.property("tone") == tone:
            return
        self._status_dot.setProperty("tone", tone)
        self._status_dot.style().unpolish(self._status_dot)
        self._status_dot.style().polish(self._status_dot)

    def _refresh_cost_widget(self) -> None:
        """Re-render the live cost label from the tracker snapshot."""
        from catalog_organizer.core.cost_tracker import tracker_singleton  # noqa: PLC0415
        s = tracker_singleton().snapshot()
        self._status_cost.setText(
            f"${s.total_cost_usd:.4f} · {s.total_calls} calls"
        )

    # ── First-run / provider-missing check ────────────────────────────────

    def _check_first_run(self) -> None:
        """If a cloud provider is selected but its API key isn't in keyring
        yet, walk the user to Settings instead of letting them hit Start
        on the Process panel and get a config error. Local Ollama needs no
        check — anyone running the app can spin up a local model."""
        try:
            from catalog_organizer.core.config import load_pipeline_settings  # noqa: PLC0415
            from catalog_organizer.core.secrets import get_api_key  # noqa: PLC0415
        except Exception:
            return
        provider = (load_pipeline_settings().get("vlm", {}).get("provider") or "ollama").lower()
        if provider in ("openai", "minimax") and not get_api_key(provider):
            settings_idx = next(
                (i for i, (_, k) in enumerate(_NAV) if k == "settings"), None,
            )
            QMessageBox.warning(
                self,
                "API key required",
                f"The active VLM provider is <b>{provider}</b> but no API key "
                "is stored in your OS credential manager.<br><br>"
                "Open <b>Settings → VLM Provider</b>, paste your key, click "
                "<i>Test Connection</i>, then <i>Save All</i>.",
            )
            if settings_idx is not None:
                self._sidebar.setCurrentRow(settings_idx)
