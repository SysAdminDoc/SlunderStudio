import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QApplication, QWidget

from ui.toast import Toast, ToastManager


class ToastBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.host = QWidget()
        self.host.resize(800, 600)
        self.addCleanup(self.host.deleteLater)

    def test_long_tokens_soft_wrap_and_extend_read_time(self):
        message = "C:\\Users\\tester\\AppData\\Local\\SlunderStudio\\generations\\" + "x" * 96
        toast = Toast(message, "error", duration_ms=3000, parent=self.host)
        self.addCleanup(toast.deleteLater)

        self.assertGreater(toast.duration_ms, 3000)
        self.assertTrue(toast._message_label.wordWrap())
        self.assertIn("\u200b", toast._message_label.text())
        self.assertEqual(toast._message_label.toolTip(), message)

    def test_hover_pauses_and_resumes_dismissal_timer(self):
        toast = Toast("Short message", duration_ms=5000, parent=self.host)
        self.addCleanup(toast.deleteLater)
        enter = QEnterEvent(QPointF(1, 1), QPointF(0, 0), QPointF(1, 1))
        leave = QEvent(QEvent.Type.Leave)

        toast.enterEvent(enter)
        self.assertFalse(toast._dismiss_timer.isActive())
        self.assertGreater(toast._paused_remaining_ms, 0)

        toast.leaveEvent(leave)

        self.assertTrue(toast._dismiss_timer.isActive())
        toast.dismiss()

    def test_resize_reposition_contract_is_wired_to_main_window(self):
        manager = ToastManager(self.host)
        toast = Toast("Anchored", duration_ms=0, parent=self.host)
        manager._toasts.append(toast)
        self.addCleanup(toast.deleteLater)
        self.addCleanup(manager._toasts.clear)

        old_rect = manager._get_toast_rect(0, toast)
        self.host.resize(1000, 700)
        new_rect = manager._get_toast_rect(0, toast)

        self.assertNotEqual(old_rect.x(), new_rect.x())
        self.assertNotEqual(old_rect.y(), new_rect.y())
        source = Path(__file__).resolve().parents[1] / "ui" / "main_window.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("def resizeEvent(self, event):", text)
        self.assertIn("self.toast_mgr._reposition()", text)


if __name__ == "__main__":
    unittest.main()
