"""Small reusable Qt widgets shared by dense studio layouts."""

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

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
            f"color: {Palette.TEXT}; font-size: 9.75pt; font-weight: 700;"
        )
        layout.addWidget(self._title)

        self._message = QLabel()
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setWordWrap(True)
        self._message.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8.25pt;"
        )
        layout.addWidget(self._message)

        self._action_button = QPushButton()
        self._action_button.setObjectName("emptyStateAction")
        self._action_button.setMinimumHeight(30)
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


class OperationProgressWidget(QFrame):
    """Compact progress and cancellation controls for a running operation."""

    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("operationProgress")
        self.setAccessibleName("Operation progress")
        self.setAccessibleDescription(
            "Shows progress for the current operation and can request cancellation."
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self._message_label = QLabel("")
        self._message_label.setMinimumWidth(110)
        self._message_label.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8.25pt;"
        )
        self._message_label.setAccessibleName("Operation status")
        layout.addWidget(self._message_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("operationProgressBar")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setMinimumWidth(140)
        self._progress_bar.setAccessibleName("Operation progress percentage")
        self._progress_bar.setAccessibleDescription(
            "Progress percentage, or an animated indicator while the total is unknown."
        )
        layout.addWidget(self._progress_bar, 1)

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setObjectName("operationCancelButton")
        self._cancel_button.setMinimumHeight(28)
        self._cancel_button.setAccessibleName("Cancel operation")
        self._cancel_button.setAccessibleDescription(
            "Requests cancellation of the running operation."
        )
        self._cancel_button.clicked.connect(self._on_cancel_clicked)
        layout.addWidget(self._cancel_button)
        self.hide()

    @property
    def message_label(self) -> QLabel:
        return self._message_label

    @property
    def progress_bar(self) -> QProgressBar:
        return self._progress_bar

    @property
    def cancel_button(self) -> QPushButton:
        return self._cancel_button

    def start(self, message: str, *, determinate: bool = True) -> None:
        """Show the control and choose a known or unknown total state."""
        self._message_label.setText(str(message))
        self._cancel_button.setEnabled(True)
        if determinate:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(0)
        else:
            self._progress_bar.setRange(0, 0)
        self.show()

    def set_progress(self, value: int, message: str | None = None) -> None:
        """Set a determinate percentage without deriving it from display text."""
        if self._progress_bar.minimum() != 0 or self._progress_bar.maximum() != 100:
            self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(max(0, min(100, int(value))))
        if message is not None:
            self._message_label.setText(str(message))

    def set_step(self, message: str) -> None:
        """Display a worker step while retaining the current progress mode."""
        self._message_label.setText(str(message))

    def finish(self) -> None:
        """Hide the control and reset it for the next operation."""
        self._cancel_button.setEnabled(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._message_label.clear()
        self.hide()

    def mark_cancelling(self) -> None:
        """Prevent duplicate requests while the worker unwinds."""
        self._cancel_button.setEnabled(False)
        self._message_label.setText("Cancelling...")
        self._progress_bar.setRange(0, 0)

    def _on_cancel_clicked(self) -> None:
        self.mark_cancelling()
        self.cancel_requested.emit()


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
