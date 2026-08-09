"""Runtime locale QA helpers for already-built Qt widget trees."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QTabWidget,
    QWidget,
)

from core.i18n import PSEUDO_LOCALE, current_locale, pseudolocalize

_SOURCE_PROPERTY = "_i18n_pseudo_source"


def _pseudo_text(widget: QWidget, text: str) -> str:
    """Preserve the original widget text so a QA pass is idempotent."""
    if not text or text.startswith("［"):
        return text
    source = widget.property(_SOURCE_PROPERTY)
    if not source:
        source = text
        widget.setProperty(_SOURCE_PROPERTY, source)
    return pseudolocalize(str(source))


def apply_pseudolocale(root: QWidget) -> int:
    """Expand visible text in ``root`` when pseudo-locale QA is active.

    Catalog-backed strings are already pseudo-localized by :func:`core.i18n.tr`;
    this pass covers legacy/static widget text until that view is migrated to a
    catalog key as well. It never changes production locales.
    """
    if current_locale() != PSEUDO_LOCALE:
        return 0

    widgets = [root, *root.findChildren(QWidget)]
    changed = 0
    for widget in widgets:
        if isinstance(widget, (QLabel, QAbstractButton)):
            text = widget.text()
            expanded = _pseudo_text(widget, text)
            if expanded != text:
                widget.setText(expanded)
                changed += 1
            for getter, setter in (
                (widget.accessibleName, widget.setAccessibleName),
                (widget.accessibleDescription, widget.setAccessibleDescription),
            ):
                value = getter()
                if value and not value.startswith("［"):
                    setter(pseudolocalize(value))
        if isinstance(widget, QGroupBox):
            text = widget.title()
            expanded = _pseudo_text(widget, text)
            if expanded != text:
                widget.setTitle(expanded)
                changed += 1
            for getter, setter in (
                (widget.accessibleName, widget.setAccessibleName),
                (widget.accessibleDescription, widget.setAccessibleDescription),
            ):
                value = getter()
                if value and not value.startswith("［"):
                    setter(pseudolocalize(value))
        if isinstance(widget, QLineEdit):
            placeholder = widget.placeholderText()
            if placeholder and not placeholder.startswith("［"):
                widget.setPlaceholderText(pseudolocalize(placeholder))
        if isinstance(widget, QComboBox):
            for index in range(widget.count()):
                text = widget.itemText(index)
                expanded = text if text.startswith("［") else pseudolocalize(text)
                if expanded != text:
                    widget.setItemText(index, expanded)
                    changed += 1
        if isinstance(widget, QTabWidget):
            for index in range(widget.count()):
                text = widget.tabText(index)
                if text and not text.startswith("［"):
                    widget.setTabText(index, pseudolocalize(text))
                    changed += 1
    return changed
