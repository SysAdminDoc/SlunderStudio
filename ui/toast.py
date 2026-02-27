"""
Slunder Studio v0.0.2 — Toast Notification System
Slide-in from bottom-right, auto-dismiss, no blocking dialogs.
Supports success/error/warning/info types with color-coded borders.
"""
from PySide6.QtWidgets import QFrame, QLabel, QHBoxLayout, QWidget, QApplication
from PySide6.QtCore import QTimer, QPropertyAnimation, QRect, QEasingCurve, Qt, Signal
from PySide6.QtGui import QFont

from ui.theme import Palette


class Toast(QFrame):
    """A single toast notification that slides in and auto-dismisses."""

    closed = Signal()

    TYPES = {
        "info": {"border": Palette.BLUE, "icon": "\U0001f6c8", "name": "toastInfo"},
        "success": {"border": Palette.GREEN, "icon": "\u2713", "name": "toastSuccess"},
        "warning": {"border": Palette.YELLOW, "icon": "\u26a0", "name": "toastWarning"},
        "error": {"border": Palette.RED, "icon": "\u2717", "name": "toastError"},
    }

    def __init__(self, message: str, toast_type: str = "info", duration_ms: int = 3000, parent=None):
        super().__init__(parent)
        self.duration_ms = duration_ms
        self._anim = None

        config = self.TYPES.get(toast_type, self.TYPES["info"])
        self.setObjectName(config["name"])

        # Style
        self.setStyleSheet(f"""
            QFrame#{config["name"]} {{
                background-color: {Palette.SURFACE0};
                border: 1px solid {config["border"]};
                border-left: 4px solid {config["border"]};
                border-radius: 8px;
                padding: 0px;
            }}
        """)

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # Icon
        icon_label = QLabel(config["icon"])
        icon_label.setStyleSheet(f"color: {config['border']}; font-size: 16px; font-weight: bold; border: none;")
        layout.addWidget(icon_label)

        # Message
        msg_label = QLabel(message)
        msg_label.setStyleSheet(f"color: {Palette.TEXT}; font-size: 13px; border: none;")
        msg_label.setWordWrap(True)
        msg_label.setMaximumWidth(320)
        layout.addWidget(msg_label, 1)

        self.setFixedWidth(380)
        self.adjustSize()

        # Dismiss timer
        if duration_ms > 0:
            self._dismiss_timer = QTimer(self)
            self._dismiss_timer.setSingleShot(True)
            self._dismiss_timer.timeout.connect(self.dismiss)
            self._dismiss_timer.start(duration_ms)

    def slide_in(self, target_rect: QRect):
        """Animate sliding in from the right."""
        start = QRect(target_rect.x() + 400, target_rect.y(), target_rect.width(), target_rect.height())
        self.setGeometry(start)
        self.show()

        self._anim = QPropertyAnimation(self, b"geometry")
        self._anim.setDuration(300)
        self._anim.setStartValue(start)
        self._anim.setEndValue(target_rect)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()

    def dismiss(self):
        """Animate sliding out to the right, then destroy."""
        if self._anim and self._anim.state() == QPropertyAnimation.State.Running:
            return

        current = self.geometry()
        end = QRect(current.x() + 400, current.y(), current.width(), current.height())

        self._anim = QPropertyAnimation(self, b"geometry")
        self._anim.setDuration(250)
        self._anim.setStartValue(current)
        self._anim.setEndValue(end)
        self._anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._anim.finished.connect(self._on_dismissed)
        self._anim.start()

    def _on_dismissed(self):
        self.closed.emit()
        self.deleteLater()


class ToastManager:
    """
    Manages toast positioning and stacking.
    Attach to main window: toast_mgr = ToastManager(main_window)
    """

    MARGIN_RIGHT = 16
    MARGIN_BOTTOM = 16
    SPACING = 8

    def __init__(self, parent: QWidget):
        self.parent = parent
        self._toasts: list[Toast] = []

    def show_toast(self, message: str, toast_type: str = "info", duration_ms: int = 3000):
        """Show a new toast notification."""
        toast = Toast(message, toast_type, duration_ms, parent=self.parent)
        toast.closed.connect(lambda t=toast: self._remove_toast(t))

        self._toasts.append(toast)
        self._reposition()

        target = self._get_toast_rect(len(self._toasts) - 1, toast)
        toast.slide_in(target)

    def info(self, message: str, duration_ms: int = 3000):
        self.show_toast(message, "info", duration_ms)

    def success(self, message: str, duration_ms: int = 3000):
        self.show_toast(message, "success", duration_ms)

    def warning(self, message: str, duration_ms: int = 4000):
        self.show_toast(message, "warning", duration_ms)

    def error(self, message: str, duration_ms: int = 5000):
        self.show_toast(message, "error", duration_ms)

    def _get_toast_rect(self, index: int, toast: Toast) -> QRect:
        """Calculate position for toast at given stack index."""
        parent_rect = self.parent.rect()
        toast_h = toast.sizeHint().height()
        toast_w = toast.width()

        # Stack from bottom-right upward
        y_offset = self.MARGIN_BOTTOM
        for i in range(index):
            if i < len(self._toasts):
                y_offset += self._toasts[i].sizeHint().height() + self.SPACING

        x = parent_rect.width() - toast_w - self.MARGIN_RIGHT
        y = parent_rect.height() - y_offset - toast_h

        return QRect(x, y, toast_w, toast_h)

    def _remove_toast(self, toast: Toast):
        """Remove a dismissed toast and reposition remaining."""
        if toast in self._toasts:
            self._toasts.remove(toast)
            self._reposition()

    def _reposition(self):
        """Reposition all visible toasts after one is removed."""
        for i, toast in enumerate(self._toasts):
            if toast.isVisible():
                target = self._get_toast_rect(i, toast)
                anim = QPropertyAnimation(toast, b"geometry", toast)
                anim.setDuration(200)
                anim.setEndValue(target)
                anim.setEasingCurve(QEasingCurve.Type.OutCubic)
                anim.start()
