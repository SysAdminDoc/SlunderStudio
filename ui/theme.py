"""
Slunder Studio — Theme Engine
Ink-and-signal desktop theme with an accessible focus system, restrained surfaces,
and complete native Qt control styling.
"""
from PySide6.QtGui import QColor

# ── Ink-and-signal palette ─────────────────────────────────────────────────────

class Palette:
    """Low-glare studio colors with high-contrast text and signal accents."""
    CRUST    = "#07111b"
    MANTLE   = "#0a1724"
    BASE     = "#0e1c2a"
    SURFACE0 = "#142638"
    SURFACE1 = "#263a4d"
    SURFACE2 = "#3b5064"
    OVERLAY0 = "#95a3b3"
    OVERLAY1 = "#9ba8b7"
    SUBTEXT0 = "#aeb9c5"
    SUBTEXT1 = "#c5ced8"
    TEXT     = "#f1f4f8"
    BLUE     = "#a293ff"
    TEAL     = "#58d6ca"
    GREEN    = "#72df9d"
    YELLOW   = "#f9e2af"
    PEACH    = "#f0aa78"
    RED      = "#ff8396"
    MAUVE    = "#b69cff"
    PINK     = "#e7a8d8"
    SAPPHIRE = "#aa9dff"
    SKY      = "#75cfee"
    LAVENDER = "#c0b7ff"
    FLAMINGO = "#eebfcb"
    ROSEWATER = "#f2d6d8"


class ThemeEngine:
    """Helper to get theme colors as a dictionary for dynamic styling."""

    @staticmethod
    def get_colors() -> dict:
        return {
            "background": Palette.CRUST,
            "surface": Palette.MANTLE,
            "surface_hover": Palette.BASE,
            "border": Palette.SURFACE0,
            "text": Palette.TEXT,
            "text_secondary": Palette.SUBTEXT0,
            "accent": Palette.BLUE,
            "accent_hover": Palette.SAPPHIRE,
            "success": Palette.GREEN,
            "warning": Palette.YELLOW,
            "error": Palette.RED,
            "muted": Palette.OVERLAY0,
        }


def rgba(hex_color: str, alpha: int) -> str:
    """Return a Qt stylesheet rgba value without relying on ARGB hex parsing."""
    color = QColor(hex_color)
    if not color.isValid():
        color = QColor("#808080")
    bounded_alpha = max(0, min(255, int(alpha)))
    return (
        f"rgba({color.red()}, {color.green()}, {color.blue()}, "
        f"{bounded_alpha})"
    )


def build_stylesheet(accent: str = Palette.BLUE) -> str:
    """
    Build the complete application stylesheet.
    Pass any hex color as accent to re-theme the entire app.
    """
    p = Palette
    # Derive hover/press variants by adjusting the accent
    accent_hover = p.SAPPHIRE
    accent_press = p.SKY

    return f"""
/* ── Global ─────────────────────────────────────────────────────────────── */
* {{
    font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", Arial, sans-serif;
}}
QMainWindow, QWidget {{
    background-color: {p.BASE};
    color: {p.TEXT};
}}
QMainWindow::separator {{
    background: {p.SURFACE1};
    width: 1px;
    height: 1px;
}}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
QWidget#sidebar {{
    background-color: {p.CRUST};
    border-right: 1px solid {p.SURFACE1};
}}
QLabel#brand {{
    color: {p.TEXT};
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 0px;
}}
QLabel#brandMark {{
    background-color: {accent};
    color: {p.CRUST};
    border-radius: 7px;
    font-size: 15px;
    font-weight: 900;
}}
QLabel#navSection {{
    color: {p.OVERLAY0};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 12px 10px 5px 10px;
}}
QPushButton#sidebarBtn {{
    background: transparent;
    color: {p.SUBTEXT0};
    border: none;
    border-radius: 6px;
    padding: 9px 11px;
    text-align: left;
    font-size: 12px;
    font-weight: 600;
}}
QPushButton#sidebarBtn:hover {{
    background-color: {p.SURFACE0};
    color: {p.TEXT};
}}
QPushButton#sidebarBtn:checked {{
    background-color: {p.SURFACE0};
    color: {p.TEXT};
    border-left: 2px solid {accent};
    padding-left: 9px;
}}
QPushButton#sidebarBtn:focus {{
    border: 2px solid {p.YELLOW};
    padding: 7px 9px;
}}

/* ── Studio shell ──────────────────────────────────────────────────────── */
QFrame#commandBar {{
    background-color: {p.CRUST};
    border-bottom: 1px solid {p.SURFACE1};
}}
QFrame#workspaceHeader {{
    background-color: {p.MANTLE};
    border-bottom: 1px solid {p.SURFACE1};
}}
QLabel#pageEyebrow {{
    color: {accent};
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}}
QLabel#pageTitle {{
    color: {p.TEXT};
    font-size: 26px;
    font-weight: 750;
}}
QLabel#pageSubtitle {{
    color: {p.SUBTEXT0};
    font-size: 12px;
}}
QLabel#projectName {{
    color: {p.TEXT};
    font-size: 12px;
    font-weight: 650;
}}
QLabel#commandMeta {{
    color: {p.SUBTEXT0};
    font-size: 11px;
}}
QLabel#localStatus {{
    background-color: {p.SURFACE0};
    color: {p.TEAL};
    border: 1px solid {p.SURFACE1};
    border-radius: 10px;
    padding: 4px 9px;
    font-size: 10px;
    font-weight: 700;
}}
QLabel#computeStatus {{
    color: {p.GREEN};
    font-size: 11px;
    font-weight: 650;
}}
QFrame#studioSurface {{
    background-color: {p.BASE};
    border: none;
}}
QFrame#sessionPanel {{
    background-color: {p.MANTLE};
    border-left: 1px solid {p.SURFACE1};
}}

/* ── Buttons ────────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {accent};
    color: {p.CRUST};
    border: none;
    padding: 8px 16px;
    border-radius: 5px;
    font-weight: 700;
    font-size: 12px;
    min-height: 18px;
}}
QPushButton:hover {{
    background-color: {accent_hover};
}}
QPushButton:pressed {{
    background-color: {accent_press};
}}
QPushButton:disabled {{
    background-color: {p.SURFACE0};
    color: {p.OVERLAY0};
}}
QPushButton:focus {{
    border: 2px solid {p.YELLOW};
    padding: 6px 14px;
}}
QPushButton#secondaryBtn,
QPushButton[class="secondary"] {{
    background-color: {p.SURFACE0};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
}}
QPushButton#secondaryBtn:hover,
QPushButton[class="secondary"]:hover {{
    background-color: {p.SURFACE1};
    border-color: {p.SURFACE2};
}}
QPushButton[class="success"] {{
    background-color: {p.GREEN};
    color: {p.CRUST};
}}
QPushButton[class="success"]:hover {{
    background-color: {p.TEAL};
    color: {p.CRUST};
}}
QPushButton[class="success"]:disabled {{
    background-color: {p.SURFACE0};
    color: {p.OVERLAY0};
}}
QPushButton#dangerBtn,
QPushButton[class="danger"] {{
    background-color: {p.RED};
    color: {p.CRUST};
}}
QPushButton#dangerBtn:hover,
QPushButton[class="danger"]:hover {{
    background-color: {p.FLAMINGO};
}}
QPushButton#ghostBtn,
QPushButton[class="ghost"] {{
    background: transparent;
    color: {p.SUBTEXT0};
    border: none;
    padding: 6px 12px;
}}
QPushButton#ghostBtn:hover,
QPushButton[class="ghost"]:hover {{
    color: {p.TEXT};
    background-color: {p.SURFACE0};
}}
QPushButton#primaryAction {{
    font-size: 14px;
    min-height: 30px;
    padding: 10px 18px;
}}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {p.CRUST};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
    border-radius: 5px;
    padding: 8px 10px;
    font-size: 13px;
    selection-background-color: {accent};
    selection-color: {p.CRUST};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 2px solid {p.YELLOW};
}}
QLineEdit:disabled, QTextEdit:disabled {{
    background-color: {p.MANTLE};
    color: {p.OVERLAY0};
}}
QTextEdit#primaryEditor {{
    background-color: {p.CRUST};
    border: 1px solid {p.SURFACE1};
    border-radius: 5px;
    padding: 10px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {p.CRUST};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
    border-radius: 5px;
    padding: 6px 9px;
    font-size: 12px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 2px solid {p.YELLOW};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: transparent;
    border: none;
    width: 20px;
}}

/* ── ComboBox (Full ControlTemplate for dark mode) ──────────────────────── */
QComboBox {{
    background-color: {p.CRUST};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
    border-radius: 5px;
    padding: 6px 10px;
    font-size: 12px;
    min-height: 20px;
}}
QComboBox:focus {{
    border: 2px solid {p.YELLOW};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border: none;
    border-left: 1px solid {p.SURFACE1};
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
    background: transparent;
}}
QComboBox::down-arrow {{
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {p.SUBTEXT0};
}}
QComboBox::down-arrow:hover {{
    border-top-color: {p.TEXT};
}}
QComboBox QAbstractItemView {{
    background-color: {p.MANTLE};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
    border-radius: 4px;
    padding: 4px;
    selection-background-color: {accent};
    selection-color: {p.CRUST};
    outline: none;
}}
QComboBox QAbstractItemView::item {{
    padding: 6px 12px;
    min-height: 24px;
}}
QComboBox QAbstractItemView::item:hover {{
    background-color: {p.SURFACE1};
}}

/* ── Sliders ────────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: {p.SURFACE0};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {accent};
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {accent_hover};
}}
QSlider:focus {{
    border: 1px solid {p.YELLOW};
    border-radius: 4px;
}}
QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 3px;
}}

/* ── Progress Bar ───────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {p.SURFACE0};
    border: none;
    border-radius: 6px;
    text-align: center;
    color: {p.TEXT};
    font-size: 12px;
    font-weight: 600;
    min-height: 18px;
}}
QProgressBar::chunk {{
    background: {accent};
    border-radius: 6px;
}}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {p.SURFACE1};
    background: transparent;
    padding: 10px 0 0 0;
}}
QTabBar::tab {{
    background: transparent;
    color: {p.SUBTEXT0};
    padding: 9px 14px;
    border-bottom: 2px solid transparent;
    font-size: 12px;
    font-weight: 650;
}}
QTabBar::tab:hover {{
    color: {p.TEXT};
    background-color: transparent;
}}
QTabBar::tab:selected {{
    color: {p.TEXT};
    border-bottom-color: {accent};
}}
QTabBar::tab:focus {{
    color: {p.YELLOW};
    border: 2px solid {p.YELLOW};
}}

/* ── Group Box ──────────────────────────────────────────────────────────── */
QGroupBox {{
    background: transparent;
    border: none;
    border-top: 1px solid {p.SURFACE1};
    border-radius: 0;
    margin-top: 18px;
    padding: 18px 0 0 0;
    font-weight: 700;
    font-size: 12px;
    color: {p.TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 0;
    padding: 0 8px 0 0;
    color: {p.TEXT};
}}

/* ── Scrollbars (branded thin) ──────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    border: none;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {p.SURFACE1};
    border-radius: 4px;
    min-height: 40px;
}}
QScrollBar::handle:vertical:hover {{
    background: {p.SURFACE2};
}}
QScrollBar::handle:vertical:pressed {{
    background: {accent};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    border: none;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {p.SURFACE1};
    border-radius: 4px;
    min-width: 40px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {p.SURFACE2};
}}
QScrollBar::handle:horizontal:pressed {{
    background: {accent};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ── Tables / Lists / Trees ─────────────────────────────────────────────── */
QTableWidget, QTreeWidget, QListWidget {{
    background-color: {p.CRUST};
    alternate-background-color: {p.MANTLE};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
    border-radius: 5px;
    gridline-color: {p.SURFACE0};
    font-size: 13px;
}}
QTableWidget::item:selected, QTreeWidget::item:selected, QListWidget::item:selected {{
    background-color: {accent};
    color: {p.CRUST};
}}
QHeaderView::section {{
    background-color: {p.MANTLE};
    color: {p.SUBTEXT0};
    border: none;
    border-bottom: 1px solid {p.SURFACE1};
    border-right: 1px solid {p.SURFACE0};
    padding: 6px 10px;
    font-weight: 600;
    font-size: 12px;
}}

/* ── Tooltips ───────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {p.SURFACE0};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ── Status Bar ─────────────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {p.MANTLE};
    color: {p.OVERLAY0};
    border-top: 1px solid {p.SURFACE1};
    font-size: 12px;
    padding: 2px 8px;
}}
QStatusBar::item {{
    border: none;
}}

/* ── Labels ─────────────────────────────────────────────────────────────── */
QLabel {{
    background: transparent;
    color: {p.TEXT};
    font-size: 13px;
}}
QLabel#heading {{
    font-size: 24px;
    font-weight: 750;
    color: {p.TEXT};
}}
QLabel#subheading {{
    font-size: 15px;
    font-weight: 600;
    color: {p.SUBTEXT0};
}}
QLabel#caption {{
    font-size: 11px;
    color: {p.SUBTEXT0};
}}
QLabel#accentLabel {{
    color: {accent};
    font-weight: 600;
}}
QFrame#commandSeparator {{
    background: {p.SURFACE1};
    border: none;
    max-width: 1px;
}}

/* ── Splitter ───────────────────────────────────────────────────────────── */
QSplitter::handle {{
    background: {p.SURFACE1};
}}
QSplitter::handle:horizontal {{
    width: 1px;
}}
QSplitter::handle:vertical {{
    height: 1px;
}}

/* ── Transport Bar ──────────────────────────────────────────────────────── */
QWidget#transportBar {{
    background-color: {p.CRUST};
    border-top: 1px solid {p.SURFACE1};
}}
QPushButton#transportBtn {{
    background: transparent;
    color: {p.SUBTEXT0};
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 6px;
    font-size: 14px;
    min-width: 32px;
    min-height: 32px;
}}
QPushButton#transportBtn:hover {{
    background-color: {p.SURFACE0};
    color: {p.TEXT};
}}
QPushButton#transportBtn:checked {{
    color: {accent};
}}
QPushButton#transportPrimary {{
    background: {accent};
    color: {p.CRUST};
    border: none;
    border-radius: 5px;
    font-size: 16px;
    min-width: 38px;
    min-height: 34px;
}}
QLabel#transportTitle {{
    color: {p.TEXT};
    font-size: 11px;
    font-weight: 700;
}}
QLabel#transportMeta {{
    color: {p.OVERLAY0};
    font-size: 10px;
}}
QLabel#transportTime {{
    color: {p.TEXT};
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 13px;
    font-weight: 650;
}}

/* ── Legacy cards: solid, quiet surfaces ────────────────────────────────── */
QFrame#card {{
    background-color: {p.MANTLE};
    border: 1px solid {p.SURFACE0};
    border-radius: 7px;
    padding: 16px;
}}
QFrame#card:hover {{
    border-color: {p.SURFACE2};
}}
QFrame#accentCard {{
    background-color: {p.SURFACE0};
    border: 1px solid {accent};
    border-radius: 7px;
    padding: 16px;
}}

/* ── Toast Notifications ────────────────────────────────────────────────── */
QFrame#toast {{
    background-color: {p.MANTLE};
    border: 1px solid {p.SURFACE1};
    border-radius: 10px;
    padding: 12px 16px;
}}
QFrame#toastSuccess {{
    background-color: {p.MANTLE};
    border: 1px solid {p.GREEN};
    border-radius: 10px;
    padding: 12px 16px;
}}
QFrame#toastError {{
    background-color: {p.MANTLE};
    border: 1px solid {p.RED};
    border-radius: 10px;
    padding: 12px 16px;
}}
QFrame#toastWarning {{
    background-color: {p.MANTLE};
    border: 1px solid {p.YELLOW};
    border-radius: 10px;
    padding: 12px 16px;
}}

/* ── Menu ───────────────────────────────────────────────────────────────── */
QMenu {{
    background-color: {p.SURFACE0};
    border: 1px solid {p.SURFACE1};
    border-radius: 8px;
    padding: 4px;
}}
QMenu::item {{
    padding: 8px 24px 8px 12px;
    border-radius: 4px;
    font-size: 13px;
}}
QMenu::item:selected {{
    background-color: {p.SURFACE1};
    color: {p.TEXT};
}}
QMenu::separator {{
    height: 1px;
    background: {p.SURFACE1};
    margin: 4px 8px;
}}

/* ── Checkbox / Radio ───────────────────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
    font-size: 13px;
    color: {p.TEXT};
}}
QCheckBox:focus {{
    color: {p.YELLOW};
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {p.SURFACE2};
    border-radius: 4px;
    background: {p.SURFACE0};
}}
QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}
QCheckBox::indicator:hover {{
    border-color: {p.OVERLAY0};
}}
QRadioButton {{
    spacing: 8px;
    font-size: 13px;
}}
QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {p.SURFACE2};
    border-radius: 9px;
    background: {p.SURFACE0};
}}
QRadioButton::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}
"""


