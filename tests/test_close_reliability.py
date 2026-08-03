import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.main_window import MainWindow
from ui.project_manager import ProjectDetailPanel


class _CloseEvent:
    def __init__(self):
        self.accepted = False
        self.ignored = False

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


class CloseReliabilityTests(unittest.TestCase):
    def _window(self, autosave):
        window = MainWindow.__new__(MainWindow)
        window._autosave = autosave
        window._gpu_timer = mock.Mock()
        window._model_mgr = mock.Mock()
        window.toast_mgr = mock.Mock()
        window._project_mgr_view = mock.Mock()
        return window

    def test_project_save_failure_is_reported_and_keeps_edit_dirty(self):
        manager = mock.Mock()
        manager.current = SimpleNamespace(notes="old notes")
        manager.save.return_value = False
        view = ProjectDetailPanel.__new__(ProjectDetailPanel)
        view._notes = mock.Mock()
        view._notes.toPlainText.return_value = "unsaved notes"
        view.toast_mgr = mock.Mock()

        with mock.patch("ui.project_manager.get_project_manager", return_value=manager):
            result = view._on_save()

        self.assertFalse(result)
        view.toast_mgr.error.assert_called_once()
        manager.save.assert_called_once_with()

    def test_close_flushes_dirty_work_when_interval_autosave_is_off(self):
        manager = mock.Mock()
        manager.current = object()
        manager.is_dirty = True
        autosave = mock.Mock()
        autosave.enabled = False
        window = self._window(autosave)
        order = []

        def flush():
            order.append("flush")
            manager.is_dirty = False
            return object()

        autosave.flush.side_effect = flush
        window._model_mgr.unload.side_effect = lambda: order.append("unload")
        event = _CloseEvent()

        with (
            mock.patch("core.project.get_project_manager", return_value=manager),
            mock.patch("ui.main_window.shutdown_workers", side_effect=lambda: order.append("join") or True),
            mock.patch("ui.main_window.AudioEngine"),
            mock.patch("core.lyrics_db.LyricsDB"),
        ):
            window.closeEvent(event)

        self.assertTrue(event.accepted)
        self.assertFalse(event.ignored)
        autosave.flush.assert_called_once_with()
        self.assertEqual(order, ["flush", "join", "unload"])

    def test_close_stays_open_when_flush_fails(self):
        manager = mock.Mock()
        manager.current = object()
        manager.is_dirty = True
        autosave = mock.Mock()
        autosave.enabled = False
        autosave.flush.return_value = None
        window = self._window(autosave)
        event = _CloseEvent()

        with (
            mock.patch("core.project.get_project_manager", return_value=manager),
            mock.patch("ui.main_window.shutdown_workers") as shutdown,
        ):
            window.closeEvent(event)

        self.assertTrue(event.ignored)
        self.assertFalse(event.accepted)
        shutdown.assert_not_called()
        window.toast_mgr.error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
