"""
Slunder Studio - PySide6 accessibility helpers.
Applies screen-reader names, descriptions, focus rings, and tab order.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QProgressBar,
    QGraphicsView,
    QTabBar,
    QTableWidget,
    QPushButton,
    QToolButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QWidget,
)
from core.i18n import tr
from ui.theme import Palette

FOCUS_RING_COLOR = Palette.YELLOW

CONTROL_TYPES = (
    QAbstractButton,
    QLineEdit,
    QTextEdit,
    QPlainTextEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QCheckBox,
    QTabWidget,
    QListWidget,
    QProgressBar,
    QTabBar,
    QTableWidget,
)

FOCUS_STYLES = {
    QPushButton: f"QPushButton:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QToolButton: f"QToolButton:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QCheckBox: f"QCheckBox:focus {{ color: {FOCUS_RING_COLOR}; }}",
    QLineEdit: f"QLineEdit:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QTextEdit: f"QTextEdit:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QPlainTextEdit: f"QPlainTextEdit:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QComboBox: f"QComboBox:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QSpinBox: f"QSpinBox:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QDoubleSpinBox: f"QDoubleSpinBox:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QSlider: f"QSlider:focus {{ border: 1px solid {FOCUS_RING_COLOR}; border-radius: 4px; }}",
    QListWidget: f"QListWidget:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QGraphicsView: f"QGraphicsView:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QTabBar: f"QTabBar::tab:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QTabWidget: f"QTabWidget:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QTableWidget: f"QTableWidget:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
}


def set_accessible(widget: QWidget | None, name: str, description: str = "") -> None:
    if widget is None:
        return
    widget.setAccessibleName(_clean(name))
    if description:
        widget.setAccessibleDescription(_clean(description))


def install_accessibility(
    root: QWidget,
    context: str,
    named_controls: Iterable[tuple[QWidget | None, str, str]] = (),
    tab_order: Sequence[QWidget | None] | None = None,
    include_descendants: bool = True,
) -> list[QWidget]:
    """Apply baseline accessibility to a view and return interactive controls."""
    set_accessible(root, context, f"{context} workspace")

    named_controls = tuple(named_controls)
    for widget, name, description in named_controls:
        set_accessible(widget, name, description)

    if include_descendants:
        controls = _interactive_controls(root)
    else:
        controls = [
            widget
            for widget, _name, _description in named_controls
            if widget is not None and _is_focusable(widget)
        ]
        if _is_accessibility_control(root) and _is_focusable(root):
            controls.insert(0, root)
    for control in controls:
        if not control.accessibleName():
            set_accessible(control, _fallback_name(control, context), _fallback_description(control))
        elif not control.accessibleDescription():
            control.setAccessibleDescription(_fallback_description(control))
        _install_focus_ring(control)

    if tab_order is None:
        set_tab_order(controls)
    else:
        set_tab_order([w for w in tab_order if w is not None])

    return controls


def set_tab_order(widgets: Sequence[QWidget]) -> None:
    focusable = [w for w in widgets if isinstance(w, QWidget) and _is_focusable(w)]
    for first, second in zip(focusable, focusable[1:]):
        QWidget.setTabOrder(first, second)


def _interactive_controls(root: QWidget) -> list[QWidget]:
    controls: list[QWidget] = []
    for widget in (root, *root.findChildren(QWidget)):
        if _inside_unmanaged_graphics_view(widget, root):
            continue
        if _is_accessibility_control(widget) and _is_focusable(widget):
            controls.append(widget)
    return controls


def _is_accessibility_control(widget: QWidget) -> bool:
    return (
        isinstance(widget, CONTROL_TYPES)
        or (
            isinstance(widget, QGraphicsView)
            and widget.property("accessibility_canvas") is True
        )
    )


def _inside_unmanaged_graphics_view(widget: QWidget, root: QWidget) -> bool:
    parent = widget.parentWidget()
    while parent is not None and parent is not root:
        if (
            isinstance(parent, QGraphicsView)
            and parent.property("accessibility_canvas") is not True
        ):
            return True
        parent = parent.parentWidget()
    return False


def _is_focusable(widget: QWidget) -> bool:
    return widget.focusPolicy() != Qt.FocusPolicy.NoFocus and not widget.isHidden()


def _install_focus_ring(widget: QWidget) -> None:
    style = widget.styleSheet() or ""
    if ":focus" in style:
        return
    for widget_type, focus_style in FOCUS_STYLES.items():
        if isinstance(widget, widget_type):
            widget.setStyleSheet((style + "\n" + focus_style).strip())
            return


def _fallback_name(widget: QWidget, context: str) -> str:
    for candidate in (
        _button_text(widget),
        _placeholder(widget),
        _object_name(widget),
        widget.toolTip(),
        widget.windowTitle(),
    ):
        if candidate:
            return f"{context} {candidate}"
    return tr(
        "accessibility.fallback.control_name",
        context=context,
        control=tr(_fallback_control_key(widget)),
    )


def _fallback_description(widget: QWidget) -> str:
    if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
        return tr("accessibility.fallback.text_input")
    if isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
        return tr("accessibility.fallback.numeric_control")
    if isinstance(widget, QComboBox):
        return tr("accessibility.fallback.option_selector")
    if isinstance(widget, QCheckBox):
        return tr("accessibility.fallback.toggle_setting")
    if isinstance(widget, QTabWidget):
        return tr("accessibility.fallback.panel_switcher")
    if isinstance(widget, QListWidget):
        return tr("accessibility.fallback.selectable_list")
    if isinstance(widget, QTableWidget):
        return tr("accessibility.fallback.editable_table")
    if isinstance(widget, QProgressBar):
        return tr("accessibility.fallback.progress")
    if isinstance(widget, QGraphicsView):
        return tr("accessibility.fallback.keyboard_canvas")
    if isinstance(widget, QAbstractButton):
        return tr("accessibility.fallback.action_button")
    return tr("accessibility.fallback.interactive_control")


def _fallback_control_key(widget: QWidget) -> str:
    if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
        return "accessibility.fallback.text_input_name"
    if isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
        return "accessibility.fallback.numeric_control_name"
    if isinstance(widget, QComboBox):
        return "accessibility.fallback.option_selector_name"
    if isinstance(widget, QCheckBox):
        return "accessibility.fallback.toggle_setting_name"
    if isinstance(widget, QTabWidget):
        return "accessibility.fallback.panel_switcher_name"
    if isinstance(widget, QListWidget):
        return "accessibility.fallback.selectable_list_name"
    if isinstance(widget, QTableWidget):
        return "accessibility.fallback.editable_table_name"
    if isinstance(widget, QProgressBar):
        return "accessibility.fallback.progress_name"
    if isinstance(widget, QGraphicsView):
        return "accessibility.fallback.keyboard_canvas_name"
    if isinstance(widget, QAbstractButton):
        return "accessibility.fallback.action_button_name"
    return "accessibility.fallback.interactive_control_name"


def _button_text(widget: QWidget) -> str:
    if isinstance(widget, QAbstractButton):
        return _clean(widget.text())
    return ""


def _placeholder(widget: QWidget) -> str:
    if isinstance(widget, QLineEdit):
        return widget.placeholderText()
    if isinstance(widget, (QTextEdit, QPlainTextEdit)):
        return widget.placeholderText()
    return ""


def _object_name(widget: QWidget) -> str:
    name = widget.objectName()
    if not name:
        return ""
    return re.sub(r"[_\-]+", " ", name).strip()


def _clean(text: str) -> str:
    text = re.sub(r"[\U00010000-\U0010ffff]", "", str(text))
    text = text.replace("&", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -:\t\r\n")
