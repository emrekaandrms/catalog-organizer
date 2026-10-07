"""Dashboard — headline metrics, catalogue composition, recent activity."""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from catalog_organizer.catalog.index import CatalogIndex
from catalog_organizer.core.paths import cache_dir, data_dir
from catalog_organizer.gui import theme
from catalog_organizer.gui.widgets.components import (
    Card,
    PageHeader,
    StatCard,
    caption,
    page_layout,
)

_LEVEL_TONE = {"ERROR": "danger", "WARN": "warning", "INFO": "neutral"}


class _BreakdownRow(QWidget):
    """One "label — count — proportion bar" line.

    A bar rather than a number alone: the useful question about a category
    is what share of the catalogue it is, and 40 vs 92 is far harder to
    compare as digits than as two lengths.
    """

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SP_3)

        self._label = QLabel(label)
        self._label.setFixedWidth(130)
        self._label.setStyleSheet(
            f"color: {theme.TEXT_SECONDARY}; font-size: {theme.FS_SMALL}px;")
        row.addWidget(self._label)

        track = QFrame()
        track.setFixedHeight(6)
        track.setStyleSheet(
            f"background: {theme.BG_SUNKEN}; border-radius: 3px;")
        tl = QHBoxLayout(track)
        tl.setContentsMargins(0, 0, 0, 0)
        self._fill = QFrame()
        self._fill.setStyleSheet(
            f"background: {theme.INFO}; border-radius: 3px;")
        tl.addWidget(self._fill)
        self._track = track
        tl.addStretch()
        row.addWidget(track, stretch=1)

        self._count = QLabel("0")
        self._count.setFixedWidth(48)
        self._count.setAlignment(Qt.AlignmentFlag.AlignRight
                                 | Qt.AlignmentFlag.AlignVCenter)
        self._count.setStyleSheet(
            f"color: {theme.TEXT_PRIMARY}; font-size: {theme.FS_SMALL}px;"
            " font-weight: 600;")
        row.addWidget(self._count)

    def set_value(self, count: int, total: int) -> None:
        self._count.setText(str(count))
        share = (count / total) if total else 0.0
        width = max(2, int(self._track.width() * share)) if share else 0
        self._fill.setFixedWidth(width)


class DashboardPanel(QWidget):
    def __init__(self, index: CatalogIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._rows: dict[str, _BreakdownRow] = {}
        self._build_ui()
        self._refresh()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(5_000)

    # ── UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = page_layout(self)

        self._header = PageHeader(
            "Dashboard",
            "Catalogue health at a glance — refreshes every 5 seconds.",
        )
        root.addWidget(self._header)

        # KPI grid. Two rows of three keeps each card wide enough to read on
        # a laptop; the old single row of six squeezed them to ~180 px.
        grid = QGridLayout()
        grid.setSpacing(theme.SP_3)
        self._t_scanned   = StatCard("Total scanned", "info")
        self._t_processed = StatCard("Processed", "success")
        self._t_review    = StatCard("Needs review", "warning")
        self._t_failed    = StatCard("Failed", "danger")
        self._t_stones    = StatCard("With stones", "neutral")
        self._t_sprue     = StatCard("With sprue", "neutral")
        for i, tile in enumerate((
            self._t_scanned, self._t_processed, self._t_review,
            self._t_failed, self._t_stones, self._t_sprue,
        )):
            grid.addWidget(tile, i // 3, i % 3)
        root.addLayout(grid)

        # Lower half: composition on the left, activity on the right. The
        # old layout stopped after the events box and left the bottom 55 %
        # of the window empty at 1440x900.
        lower = QHBoxLayout()
        lower.setSpacing(theme.SP_4)

        self._composition = Card("Composition by category")
        for key in ("ring", "pendant", "earring", "necklace",
                    "bracelet", "chain", "charm", "brooch", "other"):
            row = _BreakdownRow(key.replace("_", " ").title())
            self._rows[key] = row
            self._composition.add(row)
        self._composition.body.addStretch()
        lower.addWidget(self._composition, stretch=1)

        activity = Card("Recent activity")
        self._event_list = QListWidget()
        self._event_list.setObjectName("LogView")
        self._event_list.setFrameShape(QFrame.Shape.NoFrame)
        activity.add(self._event_list, stretch=1)
        self._disk_label = caption("Cache: –")
        activity.add(self._disk_label)
        lower.addWidget(activity, stretch=1)

        root.addLayout(lower, stretch=1)

    # ── data ────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        records = self._index.all()
        total     = len(records)
        processed = sum(1 for r in records if r.state in ("final", "needs_review"))
        review    = sum(1 for r in records if r.needs_manual_review)
        failed    = sum(1 for r in records if r.state == "failed")
        stones    = sum(1 for r in records if r.stone_summary.status == "stone")
        sprue     = sum(1 for r in records if r.sprue.detected)

        self._t_scanned.set_value(str(total))
        self._t_processed.set_value(str(processed))
        self._t_review.set_value(str(review))
        self._t_failed.set_value(str(failed))
        self._t_stones.set_value(str(stones))
        self._t_sprue.set_value(str(sprue))

        # A zero failure count is good news, so don't paint it alarm-red;
        # likewise an empty review queue.
        self._t_failed.set_tone("danger" if failed else "success")
        self._t_review.set_tone("warning" if review else "success")

        counts: dict[str, int] = {}
        for r in records:
            key = r.main_category if r.main_category in self._rows else "other"
            counts[key] = counts.get(key, 0) + 1
        for key, row in self._rows.items():
            row.set_value(counts.get(key, 0), total)

        cache = cache_dir()
        if cache.exists():
            size = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file())
            self._disk_label.setText(f"Snapshot cache: {size / 1_048_576:.1f} MB")

        self._refresh_events()

    def _refresh_events(self) -> None:
        audit_path = data_dir() / "audit_log.jsonl"
        if not audit_path.exists():
            return
        lines = audit_path.read_bytes().splitlines()[-40:]
        self._event_list.clear()
        for raw in reversed(lines):
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            level = ev.get("level", "info").upper()
            item = QListWidgetItem(
                f"{level:<5} {ev.get('code', '')} — {ev.get('message', '')[:90]}")
            tone = _LEVEL_TONE.get(level, "neutral")
            if tone == "danger":
                item.setForeground(Qt.GlobalColor.red)
            self._event_list.addItem(item)
