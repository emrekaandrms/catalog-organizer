"""Design system: colour tokens, type scale, and the global stylesheet.

Everything visual is driven from the token table below, so a panel never
hard-codes a hex value. Widgets opt into a role with `setObjectName` or
`setProperty("variant", ...)` and the stylesheet does the rest — see
`gui/widgets/components.py` for the shared building blocks.

Palette notes
-------------
The previous theme was a single flat indigo (#1e1e2e) used for every
surface, which left the UI with no depth and no way to say "this number is
a failure and that one is fine". This one is a neutral slate ramp with
four elevation steps plus real semantic colours, so state can be shown in
colour rather than only in words.

Accent is user-configurable (Settings → Theme). Everything else is fixed:
letting users recolour semantic states would defeat the point of having
them.
"""
from __future__ import annotations

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

ACCENT_DEFAULT = "#2f81f7"
_ACCENT_DEFAULT = ACCENT_DEFAULT  # backwards-compatible alias

# ── Colour tokens ───────────────────────────────────────────────────────────
# Elevation ramp: the further "up" a surface is, the lighter it gets.
BG_BASE      = "#0d1117"   # window
BG_SUNKEN    = "#090c10"   # inputs, log views, anything recessed
BG_SURFACE   = "#151b23"   # cards, panels
BG_SURFACE_2 = "#1c232c"   # hover / raised rows
BG_SIDEBAR   = "#0b0f14"

BORDER        = "#262d38"
BORDER_STRONG = "#333c4a"

TEXT_PRIMARY   = "#e6edf3"
TEXT_SECONDARY = "#9aa7b4"
TEXT_MUTED     = "#6b7785"
TEXT_INVERSE   = "#ffffff"

SUCCESS = "#3fb950"
WARNING = "#d29922"
DANGER  = "#f85149"
INFO    = "#58a6ff"

# Semantic colours also need a low-alpha fill for badges/pills. Qt's QSS
# supports rgba(), so these are kept as tuples the builder formats.
_SEMANTIC_RGB = {
    "success": (63, 185, 80),
    "warning": (210, 153, 34),
    "danger":  (248, 81, 73),
    "info":    (88, 166, 255),
    "neutral": (154, 167, 180),
}

# ── Type scale ──────────────────────────────────────────────────────────────
FONT_FAMILY = '"Segoe UI Variable", "Segoe UI", system-ui, sans-serif'
FONT_MONO   = '"Cascadia Mono", "Consolas", "SF Mono", monospace'

FS_DISPLAY = 28   # KPI numbers
FS_TITLE   = 19   # page title
FS_HEADING = 14   # section heading
FS_BODY    = 13
FS_SMALL   = 12
FS_MICRO   = 11   # captions, status bar

# ── Spacing / radius scale ──────────────────────────────────────────────────
SP_1, SP_2, SP_3, SP_4, SP_5, SP_6 = 4, 8, 12, 16, 24, 32
RADIUS_SM, RADIUS_MD, RADIUS_LG = 4, 8, 12


def _shift(hex_colour: str, amount: int) -> str:
    """Lighten (amount > 0) or darken a hex colour by `amount`/255."""
    c = QColor(hex_colour)
    return QColor(
        max(0, min(255, c.red() + amount)),
        max(0, min(255, c.green() + amount)),
        max(0, min(255, c.blue() + amount)),
    ).name()


def apply_dark_theme(app: QApplication, accent: str = ACCENT_DEFAULT) -> None:
    """Apply the palette, base font and global stylesheet to *app*."""
    app.setStyle("Fusion")

    font = QFont("Segoe UI", FS_BODY - 3)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(BG_BASE))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.Base,            QColor(BG_SUNKEN))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(BG_SURFACE_2))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(BG_SURFACE_2))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.Text,            QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.Button,          QColor(BG_SURFACE))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT_PRIMARY))
    pal.setColor(QPalette.ColorRole.BrightText,      QColor(TEXT_INVERSE))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(accent))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(TEXT_INVERSE))
    pal.setColor(QPalette.ColorRole.Mid,             QColor(BORDER))
    pal.setColor(QPalette.ColorRole.Dark,            QColor(BG_BASE))
    pal.setColor(QPalette.ColorRole.Shadow,          QColor("#000000"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_MUTED))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
                 QColor(TEXT_MUTED))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,
                 QColor(TEXT_MUTED))
    app.setPalette(pal)

    app.setStyleSheet(_build_qss(accent))


def _semantic_rules() -> str:
    """Badge/pill and text-colour rules for each semantic state.

    Generated rather than hand-written so a new state means one entry in
    `_SEMANTIC_RGB`, not five near-identical QSS blocks that drift apart.
    """
    out = []
    for name, (r, g, b) in _SEMANTIC_RGB.items():
        out.append(f"""
QLabel#Badge[tone="{name}"] {{
    color: rgb({r}, {g}, {b});
    background: rgba({r}, {g}, {b}, 0.14);
    border: 1px solid rgba({r}, {g}, {b}, 0.32);
    border-radius: {RADIUS_SM}px;
    padding: 2px {SP_2}px;
    font-size: {FS_MICRO}px;
    font-weight: 600;
}}
QLabel[tone="{name}"] {{ color: rgb({r}, {g}, {b}); }}
QFrame#StatCard[tone="{name}"] #StatValue {{ color: rgb({r}, {g}, {b}); }}
QFrame#StatCard[tone="{name}"] #StatAccent {{ background: rgb({r}, {g}, {b}); }}
""")
    return "\n".join(out)


def _build_qss(accent: str) -> str:
    accent_hover = _shift(accent, 24)
    accent_press = _shift(accent, -28)
    accent_c = QColor(accent)
    ar, ag, ab = accent_c.red(), accent_c.green(), accent_c.blue()

    return f"""
/* ── Base ─────────────────────────────────────────────────────────────── */
QWidget {{
    color: {TEXT_PRIMARY};
    font-family: {FONT_FAMILY};
    font-size: {FS_BODY}px;
}}
QMainWindow, QDialog {{ background: {BG_BASE}; }}

/* ── Sidebar navigation ───────────────────────────────────────────────── */
QListWidget#Sidebar {{
    background: {BG_SIDEBAR};
    border: none;
    border-right: 1px solid {BORDER};
    padding: {SP_2}px {SP_2}px;
    outline: 0;
}}
QListWidget#Sidebar::item {{
    padding: 9px {SP_3}px;
    margin: 2px 0;
    border-radius: {RADIUS_MD}px;
    color: {TEXT_SECONDARY};
}}
QListWidget#Sidebar::item:hover {{
    background: {BG_SURFACE};
    color: {TEXT_PRIMARY};
}}
QListWidget#Sidebar::item:selected {{
    background: rgba({ar}, {ag}, {ab}, 0.16);
    color: {TEXT_PRIMARY};
    border-left: 3px solid {accent};
    padding-left: 9px;
    font-weight: 600;
}}

QLabel#BrandTitle {{
    color: {TEXT_PRIMARY};
    font-size: {FS_HEADING}px;
    font-weight: 700;
    padding: 0;
}}
QLabel#BrandSub {{
    color: {TEXT_MUTED};
    font-size: {FS_MICRO}px;
}}
QLabel#SidebarFooter {{
    color: {TEXT_MUTED};
    font-size: {FS_MICRO}px;
    padding: {SP_2}px {SP_3}px;
}}

/* ── Page header ──────────────────────────────────────────────────────── */
QLabel#PageTitle {{
    color: {TEXT_PRIMARY};
    font-size: {FS_TITLE}px;
    font-weight: 700;
}}
QLabel#PageSubtitle {{
    color: {TEXT_SECONDARY};
    font-size: {FS_SMALL}px;
}}
QFrame#HeaderRule {{
    background: {BORDER};
    max-height: 1px;
    border: none;
}}
QLabel#SectionHeading {{
    color: {TEXT_PRIMARY};
    font-size: {FS_HEADING}px;
    font-weight: 600;
}}
QLabel#Caption {{
    color: {TEXT_MUTED};
    font-size: {FS_MICRO}px;
}}
QLabel#FieldLabel {{
    color: {TEXT_SECONDARY};
    font-size: {FS_MICRO}px;
    font-weight: 600;
}}

/* ── Cards ────────────────────────────────────────────────────────────── */
QFrame#Card, QFrame#KpiCard, QFrame#StatCard {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_LG}px;
}}
QFrame#StatCard:hover {{ border-color: {BORDER_STRONG}; }}
QLabel#StatValue {{
    color: {TEXT_PRIMARY};
    font-size: {FS_DISPLAY}px;
    font-weight: 700;
}}
QLabel#StatLabel {{
    color: {TEXT_SECONDARY};
    font-size: {FS_MICRO}px;
    font-weight: 600;
}}
QFrame#StatAccent {{
    background: {TEXT_MUTED};
    border-radius: 2px;
    max-height: 3px;
    min-height: 3px;
}}

/* ── Buttons ──────────────────────────────────────────────────────────── */
QPushButton {{
    background: {BG_SURFACE_2};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_MD}px;
    padding: 7px {SP_4}px;
    font-size: {FS_SMALL}px;
    font-weight: 600;
}}
QPushButton:hover  {{ background: {_shift(BG_SURFACE_2, 12)}; border-color: {accent}; }}
QPushButton:pressed{{ background: {_shift(BG_SURFACE_2, -8)}; }}
QPushButton:disabled {{
    background: {BG_SURFACE};
    color: {TEXT_MUTED};
    border-color: {BORDER};
}}
QPushButton[variant="primary"] {{
    background: {accent};
    color: {TEXT_INVERSE};
    border: 1px solid {accent};
}}
QPushButton[variant="primary"]:hover   {{ background: {accent_hover}; border-color: {accent_hover}; }}
QPushButton[variant="primary"]:pressed {{ background: {accent_press}; }}
QPushButton[variant="primary"]:disabled {{
    background: {BG_SURFACE}; color: {TEXT_MUTED}; border-color: {BORDER};
}}
QPushButton[variant="ghost"] {{
    background: transparent;
    border: 1px solid transparent;
    color: {TEXT_SECONDARY};
}}
QPushButton[variant="ghost"]:hover {{
    background: {BG_SURFACE_2};
    color: {TEXT_PRIMARY};
}}
QPushButton[variant="danger"] {{
    background: transparent; color: {DANGER}; border: 1px solid rgba(248, 81, 73, 0.4);
}}
QPushButton[variant="danger"]:hover {{ background: rgba(248, 81, 73, 0.12); }}

/* ── Inputs ───────────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background: {BG_SUNKEN};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_MD}px;
    padding: 6px {SP_3}px;
    font-size: {FS_SMALL}px;
    selection-background-color: {accent};
    selection-color: {TEXT_INVERSE};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus {{
    border-color: {accent};
    background: {BG_BASE};
}}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
    color: {TEXT_MUTED}; background: {BG_SURFACE};
}}
/* Only the drop-down *area* is restyled. The arrow glyph itself is left to
   Fusion: Qt's ::down-arrow needs a real image, and the CSS
   border-triangle trick that works on the web renders as a white square
   here. */
QComboBox::drop-down {{ border: none; width: 22px; background: transparent; }}
QComboBox QAbstractItemView {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_MD}px;
    selection-background-color: rgba({ar}, {ag}, {ab}, 0.22);
    selection-color: {TEXT_PRIMARY};
    outline: 0;
    padding: {SP_1}px;
}}
QCheckBox {{ color: {TEXT_SECONDARY}; font-size: {FS_SMALL}px; spacing: {SP_2}px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_SM}px;
    background: {BG_SUNKEN};
}}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}
QCheckBox::indicator:hover {{ border-color: {accent}; }}

/* ── Grouping / tabs ──────────────────────────────────────────────────── */
QGroupBox {{
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_LG}px;
    margin-top: {SP_3}px;
    padding: {SP_4}px {SP_3}px {SP_3}px {SP_3}px;
    font-size: {FS_MICRO}px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {SP_3}px;
    padding: 0 {SP_1}px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: {RADIUS_LG}px;
    top: -1px;
    background: {BG_SURFACE};
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_SECONDARY};
    padding: 8px {SP_4}px;
    border: none;
    border-bottom: 2px solid transparent;
    margin-right: {SP_1}px;
    font-size: {FS_SMALL}px;
    font-weight: 600;
}}
QTabBar::tab:hover     {{ color: {TEXT_PRIMARY}; }}
QTabBar::tab:selected  {{ color: {TEXT_PRIMARY}; border-bottom: 2px solid {accent}; }}

/* ── Lists / tables ───────────────────────────────────────────────────── */
QListWidget, QTreeWidget, QTableWidget {{
    background: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_LG}px;
    outline: 0;
    padding: {SP_1}px;
}}
QListWidget::item, QTreeWidget::item {{
    padding: {SP_2}px;
    border-radius: {RADIUS_MD}px;
    color: {TEXT_PRIMARY};
}}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {BG_SURFACE_2}; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: rgba({ar}, {ag}, {ab}, 0.20);
    color: {TEXT_PRIMARY};
}}
QHeaderView::section {{
    background: {BG_SURFACE_2};
    color: {TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: {SP_2}px {SP_3}px;
    font-size: {FS_MICRO}px;
    font-weight: 600;
}}
QTableWidget {{ gridline-color: {BORDER}; }}

/* Recessed monospace surfaces: logs, raw reports */
QTextEdit {{
    background: {BG_SUNKEN};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_LG}px;
    font-family: {FONT_MONO};
    font-size: {FS_SMALL}px;
    padding: {SP_2}px;
    selection-background-color: {accent};
}}
QListWidget#LogView {{
    background: {BG_SUNKEN};
    font-family: {FONT_MONO};
    font-size: {FS_MICRO}px;
}}

/* ── Progress ─────────────────────────────────────────────────────────── */
QProgressBar {{
    background: {BG_SUNKEN};
    border: 1px solid {BORDER};
    border-radius: {RADIUS_SM}px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {accent}; border-radius: {RADIUS_SM}px; }}

/* ── Status bar ───────────────────────────────────────────────────────── */
QStatusBar {{
    background: {BG_SIDEBAR};
    border-top: 1px solid {BORDER};
}}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{
    color: {TEXT_SECONDARY};
    font-size: {FS_MICRO}px;
    padding: 3px {SP_3}px;
}}

/* ── Scrollbars ───────────────────────────────────────────────────────── */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-height: 28px;
    min-width: 28px;
}}
QScrollBar::handle:hover {{ background: {TEXT_MUTED}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ── Splitter / tooltip ───────────────────────────────────────────────── */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: {SP_3}px; }}
QSplitter::handle:vertical {{ height: {SP_3}px; }}
QToolTip {{
    background: {BG_SURFACE_2};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_STRONG};
    border-radius: {RADIUS_MD}px;
    padding: {SP_2}px;
    font-size: {FS_MICRO}px;
}}

{_semantic_rules()}
"""
