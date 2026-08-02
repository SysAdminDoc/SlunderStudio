"""
Slunder Studio — Toast Notification System
Slide-in from bottom-right, auto-dismiss, no blocking dialogs.
Supports success/error/warning/info types with color-coded borders.
"""
import re
import time

from PySide6.QtWidgets import (
    QFrame, QLabel, QHBoxLayout, QWidget, QPushButton,
)
from PySide6.QtCore import QTimer, QPropertyAnimation, QRect, QEasingCurve, Signal

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

    _SOFT_BREAK_RE = re.compile(r"\S{24}(?=\S)")

    def __init__(
        self,
        message: str,
        toast_type: str = "info",
        duration_ms: int = 3000,
        parent=None,
        action_label: str = "",
        action_callback=None,
    ):
        super().__init__(parent)
        self.duration_ms = (
            max(duration_ms, len(message) * 60) if duration_ms > 0 else 0
        )
        self._anim = None
        self._action_callback = action_callback
        self._message = message

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
        msg_label = QLabel(self._soft_break_message(message))
        msg_label.setStyleSheet(f"color: {Palette.TEXT}; font-size: 13px; border: none;")
        msg_label.setWordWrap(True)
        msg_label.setMaximumWidth(320)
        msg_label.setToolTip(message)
        self._message_label = msg_label
        layout.addWidget(msg_label, 1)

        if action_label and action_callback:
            action_btn = QPushButton(action_label)
            action_btn.setFixedHeight(28)
            action_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {config["border"]}; color: {Palette.BASE};
                    border: none; border-radius: 4px; padding: 4px 10px;
                    font-size: 11px; font-weight: 700;
                }}
                QPushButton:hover {{ background: {Palette.TEXT}; }}
            """)
            action_btn.clicked.connect(self._on_action)
            layout.addWidget(action_btn)

        self.setFixedWidth(380)
        self.adjustSize()

        # Dismiss timer
        if self.duration_ms > 0:
            self._dismiss_timer = QTimer(self)
            self._dismiss_timer.setSingleShot(True)
            self._dismiss_timer.timeout.connect(self.dismiss)
            self._dismiss_timer.start(self.duration_ms)

    @classmethod
    def _soft_break_message(cls, message: str) -> str:
        """Allow long paths and tokens to wrap without changing the tooltip value."""
        return cls._SOFT_BREAK_RE.sub(lambda match: f"{match.group(0)}\u200b", message)

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

    def _on_action(self):
        if self._action_callback:
            self._action_callback()
        self.dismiss()

    def enterEvent(self, event):
        """Pause timed dismissal while the pointer is over the toast."""
        super().enterEvent(event)
        timer = getattr(self, "_dismiss_timer", None)
        if timer is not None and timer.isActive():
            self._paused_remaining_ms = timer.remainingTime()
            timer.stop()

    def leaveEvent(self, event):
        """Resume timed dismissal after the pointer leaves the toast."""
        super().leaveEvent(event)
        timer = getattr(self, "_dismiss_timer", None)
        remaining = getattr(self, "_paused_remaining_ms", 0)
        if timer is not None and remaining > 0:
            timer.start(remaining)
            self._paused_remaining_ms = 0

    def dismiss(self):
        """Animate sliding out to the right, then destroy."""
        timer = getattr(self, "_dismiss_timer", None)
        if timer is not None:
            timer.stop()
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
    # WCAG 2.2 SC 2.2.1: timed messages need a non-timed equivalent. Every
    # toast is also appended here so it can be reviewed after it disappears.
    HISTORY_LIMIT = 200

    def __init__(self, parent: QWidget):
        self.parent = parent
        self._toasts: list[Toast] = []
        self._history: list[dict] = []
        self._history_listeners: list = []

    # ── Non-timed alternative ──────────────────────────────────────────────────

    @property
    def history(self) -> list[dict]:
        """Every message shown, newest last. Survives the toast timeout."""
        return list(self._history)

    def latest_message(self) -> str:
        if not self._history:
            return ""
        entry = self._history[-1]
        return f"{entry['type'].capitalize()}: {entry['message']}"

    def on_message(self, callback):
        """Register a listener for the non-timed message log."""
        self._history_listeners.append(callback)

    def clear_history(self):
        self._history.clear()

    def _record(self, message: str, toast_type: str):
        entry = {"message": message, "type": toast_type, "timestamp": time.time()}
        self._history.append(entry)
        del self._history[:-self.HISTORY_LIMIT]
        for callback in list(self._history_listeners):
            try:
                callback(entry)
            except Exception:
                pass

    def show_toast(
        self,
        message: str,
        toast_type: str = "info",
        duration_ms: int = 3000,
        action_label: str = "",
        action_callback=None,
    ):
        """Show a new toast notification."""
        self._record(message, toast_type)
        toast = Toast(
            message,
            toast_type,
            duration_ms,
            parent=self.parent,
            action_label=action_label,
            action_callback=action_callback,
        )
        toast.closed.connect(lambda t=toast: self._remove_toast(t))

        self._toasts.append(toast)
        self._reposition()

        target = self._get_toast_rect(len(self._toasts) - 1, toast)
        toast.slide_in(target)

    def info(self, message: str, duration_ms: int = 3000,
             action_label: str = "", action_callback=None):
        self.show_toast(message, "info", duration_ms, action_label, action_callback)

    def success(self, message: str, duration_ms: int = 3000,
                action_label: str = "", action_callback=None):
        self.show_toast(message, "success", duration_ms, action_label, action_callback)

    def warning(self, message: str, duration_ms: int = 4000,
                action_label: str = "", action_callback=None):
        self.show_toast(message, "warning", duration_ms, action_label, action_callback)

    def error(self, message: str, duration_ms: int = 5000,
              action_label: str = "", action_callback=None):
        self.show_toast(message, "error", duration_ms, action_label, action_callback)

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
