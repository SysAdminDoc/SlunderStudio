"""
Slunder Studio v0.0.2 — Theme Engine
Catppuccin Mocha dark theme with glassmorphism, animations, accent color system,
and full QComboBox ControlTemplate for proper dark rendering.
"""
from PySide6.QtWidgets import QWidget, QGraphicsOpacityEffect
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, QRect, Property
from PySide6.QtGui import QPainter, QColor, QLinearGradient

# ── Catppuccin Mocha Palette ───────────────────────────────────────────────────

class Palette:
    """Catppuccin Mocha color tokens."""
    CRUST    = "#11111b"
    MANTLE   = "#181825"
    BASE     = "#1e1e2e"
    SURFACE0 = "#313244"
    SURFACE1 = "#45475a"
    SURFACE2 = "#585b70"
    OVERLAY0 = "#6c7086"
    OVERLAY1 = "#7f849c"
    SUBTEXT0 = "#a6adc8"
    SUBTEXT1 = "#bac2de"
    TEXT     = "#cdd6f4"
    BLUE     = "#89b4fa"
    TEAL     = "#94e2d5"
    GREEN    = "#a6e3a1"
    YELLOW   = "#f9e2af"
    PEACH    = "#fab387"
    RED      = "#f38ba8"
    MAUVE    = "#cba6f7"
    PINK     = "#f5c2e7"
    SAPPHIRE = "#74c7ec"
    SKY      = "#89dceb"
    LAVENDER = "#b4befe"
    FLAMINGO = "#f2cdcd"
    ROSEWATER = "#f5e0dc"


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
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    outline: none;
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
    background-color: {p.MANTLE};
    border-right: 1px solid {p.SURFACE1};
}}
QPushButton#sidebarBtn {{
    background: transparent;
    color: {p.OVERLAY0};
    border: none;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton#sidebarBtn:hover {{
    background-color: {p.SURFACE0};
    color: {p.TEXT};
}}
QPushButton#sidebarBtn:checked {{
    background-color: {p.SURFACE0};
    color: {accent};
    border-left: 3px solid {accent};
    border-radius: 0px 8px 8px 0px;
}}

/* ── Buttons ────────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {accent};
    color: {p.CRUST};
    border: none;
    padding: 8px 20px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {accent_hover};
}}
QPushButton:pressed {{
    background-color: {accent_press};
}}
QPushButton:disabled {{
    background-color: {p.SURFACE1};
    color: {p.OVERLAY0};
}}
QPushButton#secondaryBtn {{
    background-color: {p.SURFACE0};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
}}
QPushButton#secondaryBtn:hover {{
    background-color: {p.SURFACE1};
    border-color: {p.SURFACE2};
}}
QPushButton#dangerBtn {{
    background-color: {p.RED};
    color: {p.CRUST};
}}
QPushButton#dangerBtn:hover {{
    background-color: #e6667a;
}}
QPushButton#ghostBtn {{
    background: transparent;
    color: {p.SUBTEXT0};
    border: none;
    padding: 6px 12px;
}}
QPushButton#ghostBtn:hover {{
    color: {p.TEXT};
    background-color: {p.SURFACE0};
}}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {p.SURFACE0};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: {accent};
    selection-color: {p.CRUST};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {accent};
}}
QLineEdit:disabled, QTextEdit:disabled {{
    background-color: {p.MANTLE};
    color: {p.OVERLAY0};
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {p.SURFACE0};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {accent};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: transparent;
    border: none;
    width: 20px;
}}

/* ── ComboBox (Full ControlTemplate for dark mode) ──────────────────────── */
QComboBox {{
    background-color: {p.SURFACE0};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    min-height: 20px;
}}
QComboBox:focus {{
    border-color: {accent};
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
    background-color: {p.SURFACE0};
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
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {accent}, stop:1 {p.MAUVE});
    border-radius: 6px;
}}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {p.SURFACE1};
    border-radius: 0 0 8px 8px;
    background: {p.BASE};
    padding: 8px;
}}
QTabBar::tab {{
    background: transparent;
    color: {p.OVERLAY0};
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
    font-size: 13px;
    font-weight: 500;
}}
QTabBar::tab:hover {{
    color: {p.TEXT};
    background-color: {p.SURFACE0};
}}
QTabBar::tab:selected {{
    color: {accent};
    border-bottom-color: {accent};
}}

/* ── Group Box ──────────────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {p.SURFACE1};
    border-radius: 10px;
    margin-top: 16px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    font-size: 13px;
    color: {p.TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: {p.SUBTEXT0};
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
    background-color: {p.BASE};
    alternate-background-color: {p.MANTLE};
    color: {p.TEXT};
    border: 1px solid {p.SURFACE1};
    border-radius: 6px;
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
    color: {p.TEXT};
    font-size: 13px;
}}
QLabel#heading {{
    font-size: 20px;
    font-weight: 700;
    color: {p.TEXT};
}}
QLabel#subheading {{
    font-size: 15px;
    font-weight: 600;
    color: {p.SUBTEXT0};
}}
QLabel#caption {{
    font-size: 11px;
    color: {p.OVERLAY0};
}}
QLabel#accentLabel {{
    color: {accent};
    font-weight: 600;
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
    background-color: {p.MANTLE};
    border-top: 1px solid {p.SURFACE1};
}}
QPushButton#transportBtn {{
    background: transparent;
    color: {p.SUBTEXT0};
    border: none;
    border-radius: 20px;
    padding: 8px;
    font-size: 16px;
    min-width: 40px;
    min-height: 40px;
}}
QPushButton#transportBtn:hover {{
    background-color: {p.SURFACE0};
    color: {p.TEXT};
}}
QPushButton#transportBtn:checked {{
    color: {accent};
}}

/* ── Cards (glassmorphism containers) ───────────────────────────────────── */
QFrame#card {{
    background-color: rgba(49, 50, 68, 180);
    border: 1px solid {p.SURFACE1};
    border-radius: 12px;
    padding: 16px;
}}
QFrame#card:hover {{
    border-color: {p.SURFACE2};
}}
QFrame#accentCard {{
    background-color: rgba(137, 180, 250, 20);
    border: 1px solid rgba(137, 180, 250, 60);
    border-radius: 12px;
    padding: 16px;
}}

/* ── Toast Notifications ────────────────────────────────────────────────── */
QFrame#toast {{
    background-color: {p.SURFACE0};
    border: 1px solid {p.SURFACE1};
    border-radius: 10px;
    padding: 12px 16px;
}}
QFrame#toastSuccess {{
    background-color: {p.SURFACE0};
    border: 1px solid {p.GREEN};
    border-radius: 10px;
    padding: 12px 16px;
}}
QFrame#toastError {{
    background-color: {p.SURFACE0};
    border: 1px solid {p.RED};
    border-radius: 10px;
    padding: 12px 16px;
}}
QFrame#toastWarning {{
    background-color: {p.SURFACE0};
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


# ── Animation Helpers ──────────────────────────────────────────────────────────

def fade_in(widget: QWidget, duration: int = 250):
    """Fade a widget in from transparent to opaque."""
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.start()
    return anim


def fade_out(widget: QWidget, duration: int = 250):
    """Fade a widget out from opaque to transparent."""
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(1.0)
    anim.setEndValue(0.0)
    anim.setEasingCurve(QEasingCurve.Type.InCubic)
    anim.start()
    return anim


def slide_in_right(widget: QWidget, parent_width: int, duration: int = 300):
    """Slide a widget in from the right edge."""
    anim = QPropertyAnimation(widget, b"geometry", widget)
    start = QRect(parent_width, widget.y(), widget.width(), widget.height())
    end = widget.geometry()
    anim.setDuration(duration)
    anim.setStartValue(start)
    anim.setEndValue(end)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.start()
    return anim


def slide_out_right(widget: QWidget, parent_width: int, duration: int = 250):
    """Slide a widget out to the right edge."""
    anim = QPropertyAnimation(widget, b"geometry", widget)
    start = widget.geometry()
    end = QRect(parent_width, widget.y(), widget.width(), widget.height())
    anim.setDuration(duration)
    anim.setStartValue(start)
    anim.setEndValue(end)
    anim.setEasingCurve(QEasingCurve.Type.InCubic)
    anim.start()
    return anim
