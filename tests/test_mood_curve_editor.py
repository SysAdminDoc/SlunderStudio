import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QGraphicsItem

from ui.mood_curve_editor import MoodCurveEditor


class MoodCurveKeyboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @staticmethod
    def _press(view, key, modifiers=Qt.KeyboardModifier.NoModifier):
        event = QKeyEvent(QKeyEvent.Type.KeyPress, key, modifiers)
        view.keyPressEvent(event)
        return event

    def test_tab_selects_every_point_and_arrows_nudge_selected_point(self):
        editor = MoodCurveEditor()
        view = editor._view
        self.assertEqual(view.focusPolicy(), Qt.FocusPolicy.StrongFocus)
        self.assertEqual(editor._selected_point_index, 0)
        self.assertTrue(
            all(
                point.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                for point in editor._points
            )
        )
        self.assertTrue(all(point.toolTip() for point in editor._points))

        initial_x = editor._points[0].pos().x()
        initial_y = editor._points[0].pos().y()
        changed = []
        editor.curve_changed.connect(changed.append)

        self._press(view, Qt.Key.Key_Right)
        self._press(view, Qt.Key.Key_Up)

        self.assertGreater(editor._points[0].pos().x(), initial_x)
        self.assertLess(editor._points[0].pos().y(), initial_y)
        self.assertTrue(changed)
        self.assertIn("Curve point 1 of 9", view.accessibleDescription())

        for expected in range(1, len(editor._points)):
            self._press(view, Qt.Key.Key_Tab)
            self.assertEqual(editor._selected_point_index, expected)
        self._press(view, Qt.Key.Key_Tab)
        self.assertEqual(editor._selected_point_index, 0)

    def test_shift_arrows_use_a_larger_step_and_home_end_select(self):
        editor = MoodCurveEditor()
        view = editor._view

        initial_x = editor._points[0].pos().x()
        self._press(view, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        self.assertGreater(editor._points[0].pos().x() - initial_x, 50)

        self._press(view, Qt.Key.Key_End)
        self.assertEqual(editor._selected_point_index, len(editor._points) - 1)
        self._press(view, Qt.Key.Key_Home)
        self.assertEqual(editor._selected_point_index, 0)


if __name__ == "__main__":
    unittest.main()
