"""
Slunder Studio v0.1.27 - PySide6 accessibility helpers.
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
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QWidget,
)

FOCUS_RING_COLOR = "#f9e2af"

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
)

FOCUS_STYLES = {
    QPushButton: f"QPushButton:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QCheckBox: f"QCheckBox:focus {{ color: {FOCUS_RING_COLOR}; }}",
    QLineEdit: f"QLineEdit:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QTextEdit: f"QTextEdit:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QPlainTextEdit: f"QPlainTextEdit:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QComboBox: f"QComboBox:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QSpinBox: f"QSpinBox:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QDoubleSpinBox: f"QDoubleSpinBox:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
    QSlider: f"QSlider:focus {{ border: 1px solid {FOCUS_RING_COLOR}; border-radius: 4px; }}",
    QListWidget: f"QListWidget:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}",
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
) -> list[QWidget]:
    """Apply baseline accessibility to a view and return interactive controls."""
    set_accessible(root, context, f"{context} workspace")

    for widget, name, description in named_controls:
        set_accessible(widget, name, description)

    controls = _interactive_controls(root)
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
    for widget in root.findChildren(QWidget):
        if isinstance(widget, CONTROL_TYPES) and _is_focusable(widget):
            controls.append(widget)
    return controls


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
    return f"{context} {widget.__class__.__name__}"


def _fallback_description(widget: QWidget) -> str:
    if isinstance(widget, (QLineEdit, QTextEdit, QPlainTextEdit)):
        return "Text input"
    if isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider)):
        return "Adjustable numeric control"
    if isinstance(widget, QComboBox):
        return "Option selector"
    if isinstance(widget, QCheckBox):
        return "Toggle setting"
    if isinstance(widget, QTabWidget):
        return "Switches between related panels"
    if isinstance(widget, QListWidget):
        return "Selectable list"
    if isinstance(widget, QProgressBar):
        return "Progress indicator"
    if isinstance(widget, QAbstractButton):
        return "Action button"
    return "Interactive control"


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
