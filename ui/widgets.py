"""Small reusable Qt widgets shared by dense studio layouts."""

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel


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
