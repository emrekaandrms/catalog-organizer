"""Shared UI building blocks.

Panels compose these instead of hand-rolling layouts, so spacing, type
scale and semantic colour stay identical everywhere. All styling lives in
`gui/theme.py`; these classes only set object names / properties that the
stylesheet targets.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from catalog_organizer.gui import theme


class PageHeader(QWidget):
    """Title + optional subtitle on the left, actions on the right.

    Every panel starts with one so the eye lands in the same place on
    every screen and actions are always in the same corner.
    """

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(theme.SP_3)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.SP_3)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
        self._title = QLabel(title)
        self._title.setObjectName("PageTitle")
        text_col.addWidget(self._title)
        self._subtitle = QLabel(subtitle)
        self._subtitle.setObjectName("PageSubtitle")
        self._subtitle.setVisible(bool(subtitle))
        self._subtitle.setWordWrap(True)
        text_col.addWidget(self._subtitle)
        row.addLayout(text_col, stretch=1)

        self._actions = QHBoxLayout()
        self._actions.setContentsMargins(0, 0, 0, 0)
        self._actions.setSpacing(theme.SP_2)
        row.addLayout(self._actions)
        root.addLayout(row)

        rule = QFrame()
        rule.setObjectName("HeaderRule")
        rule.setFixedHeight(1)
        root.addWidget(rule)

    def add_action(self, widget: QWidget) -> None:
        self._actions.addWidget(widget)

    def set_subtitle(self, text: str) -> None:
        self._subtitle.setText(text)
        self._subtitle.setVisible(bool(text))


class Card(QFrame):
    """A bordered surface one elevation step above the page background.

    `title` renders as a section heading inside the card; omit it for a
    plain container. Content goes into `body`, which is a QVBoxLayout.
    """

    def __init__(
        self,
        title: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(theme.SP_4, theme.SP_4, theme.SP_4, theme.SP_4)
        outer.setSpacing(theme.SP_3)

        if title:
            heading = QLabel(title)
            heading.setObjectName("SectionHeading")
            outer.addWidget(heading)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(theme.SP_2)
        outer.addLayout(self.body)

    def add(self, widget: QWidget, stretch: int = 0) -> None:
        self.body.addWidget(widget, stretch)


class StatCard(QFrame):
    """A single headline metric.

    `tone` colours the value and the accent bar, so "3 failed" reads as a
    problem at a glance while "128 processed" stays neutral. Valid tones
    are the keys of `theme._SEMANTIC_RGB`.
    """

    def __init__(
        self,
        label: str,
        tone: str = "neutral",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("StatCard")
        self.setProperty("tone", tone)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.SP_4, theme.SP_3, theme.SP_4, theme.SP_3)
        layout.setSpacing(theme.SP_2)

        accent = QFrame()
        accent.setObjectName("StatAccent")
        accent.setFixedWidth(26)
        layout.addWidget(accent)

        self._value = QLabel("–")
        self._value.setObjectName("StatValue")
        layout.addWidget(self._value)

        self._label = QLabel(label.upper())
        self._label.setObjectName("StatLabel")
        layout.addWidget(self._label)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(112)

    def set_value(self, value: str) -> None:
        self._value.setText(value)

    def set_tone(self, tone: str) -> None:
        """Re-tone at runtime (e.g. a failure count that is only red when
        non-zero). Qt caches property-based styles, so the widget has to be
        re-polished for the change to take."""
        if self.property("tone") == tone:
            return
        self.setProperty("tone", tone)
        style = self.style()
        for w in (self, self._value):
            style.unpolish(w)
            style.polish(w)


class Badge(QLabel):
    """Small coloured pill for a state word ("needs review", "final")."""

    def __init__(
        self,
        text: str,
        tone: str = "neutral",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self.setObjectName("Badge")
        self.setProperty("tone", tone)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)


def caption(text: str) -> QLabel:
    """Muted micro-copy — hints, counts, footnotes."""
    lbl = QLabel(text)
    lbl.setObjectName("Caption")
    lbl.setWordWrap(True)
    return lbl


def field_label(text: str) -> QLabel:
    """Uppercase-ish small label that sits above an input."""
    lbl = QLabel(text)
    lbl.setObjectName("FieldLabel")
    return lbl


def page_layout(widget: QWidget) -> QVBoxLayout:
    """The standard page margin/spacing every panel uses."""
    root = QVBoxLayout(widget)
    root.setContentsMargins(theme.SP_5, theme.SP_5, theme.SP_5, theme.SP_5)
    root.setSpacing(theme.SP_4)
    return root
