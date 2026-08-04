"""Small reusable Qt widgets shared by dense studio layouts."""

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout

from ui.theme import Palette


class EmptyStateWidget(QFrame):
    """A compact, actionable explanation for an empty or filtered surface."""

    action_requested = Signal()

    def __init__(
        self,
        title: str,
        message: str,
        action_text: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("emptyState")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"QFrame#emptyState {{ background: {Palette.SURFACE0}; "
            f"border: 1px dashed {Palette.SURFACE2}; border-radius: 8px; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(6)

        self._title = QLabel()
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet(
            f"color: {Palette.TEXT}; font-size: 13px; font-weight: 700;"
        )
        layout.addWidget(self._title)

        self._message = QLabel()
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setWordWrap(True)
        self._message.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 11px;"
        )
        layout.addWidget(self._message)

        self._action_button = QPushButton()
        self._action_button.setObjectName("emptyStateAction")
        self._action_button.setFixedHeight(30)
        self._action_button.clicked.connect(self.action_requested.emit)
        layout.addWidget(self._action_button, 0, Qt.AlignmentFlag.AlignCenter)

        self.set_state(title, message, action_text)

    @property
    def state(self) -> str:
        """Return ``empty`` or ``no_matches`` for callers and tests."""
        return str(self.property("emptyStateKind") or "empty")

    @property
    def title(self) -> str:
        return self._title.text()

    @property
    def message(self) -> str:
        return self._message.text()

    @property
    def action_button(self) -> QPushButton:
        return self._action_button

    def set_state(
        self,
        title: str,
        message: str,
        action_text: str = "",
        *,
        kind: str = "empty",
    ) -> None:
        """Update copy and action while preserving the widget in its layout."""
        self._title.setText(str(title))
        self._message.setText(str(message))
        self._action_button.setText(str(action_text))
        self._action_button.setVisible(bool(action_text))
        self.setProperty("emptyStateKind", kind)
        self.setAccessibleName(str(title))
        self.setAccessibleDescription(str(message))
        self._action_button.setAccessibleName(str(action_text) or "Empty state action")
        self._action_button.setAccessibleDescription(str(message))
        self.style().unpolish(self)
        self.style().polish(self)

    def set_no_matches(
        self,
        message: str,
        action_text: str = "Clear filters",
    ) -> None:
        """Show a distinct no-results state for a filtered collection."""
        self.set_state("No matches", message, action_text, kind="no_matches")


class ElidedLabel(QLabel):
    """A label that keeps its full value in a tooltip when its layout is tight."""

    def __init__(
        self,
        text: str = "",
        minimum_width: int = 0,
        parent=None,
        elide_mode=Qt.TextElideMode.ElideRight,
    ):
        super().__init__(parent)
        self._full_text = ""
        self._elide_mode = elide_mode
        if minimum_width:
            self.setMinimumWidth(minimum_width)
        self.setText(text)

    @property
    def full_text(self) -> str:
        return self._full_text

    def setText(self, text: str):
        self._full_text = str(text)
        super().setToolTip(self._full_text)
        self._update_elided_text()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elided_text()

    def showEvent(self, event):
        super().showEvent(event)
        self._update_elided_text()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.FontChange:
            self._update_elided_text()

    def _update_elided_text(self):
        width = self.width() or self.minimumWidth() or self.sizeHint().width()
        display_text = QFontMetrics(self.font()).elidedText(
            self._full_text,
            self._elide_mode,
            max(1, width),
        )
        super().setText(display_text)
