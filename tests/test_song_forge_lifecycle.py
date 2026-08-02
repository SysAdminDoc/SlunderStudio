import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui.song_forge_view import SongForgeView


class _RunningWorker:
    def __init__(self):
        self.cancel_calls = 0
        self.result = None

    def isRunning(self):
        return True

    def cancel(self):
        self.cancel_calls += 1


class SongForgeLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.view = SongForgeView()
        self.addCleanup(self.view.deleteLater)

    def test_cancel_waits_for_terminal_signal_before_allowing_generate(self):
        worker = _RunningWorker()
        self.view._worker = worker
        self.view._is_generating = True

        with mock.patch.object(
            self.view, "_get_lyrics", side_effect=AssertionError("started a second job")
        ), mock.patch("ui.song_forge_view.InferenceWorker") as worker_type:
            self.view._on_cancel()
            self.view._on_generate()

        self.assertEqual(worker.cancel_calls, 1)
        self.assertIs(self.view._worker, worker)
        self.assertTrue(self.view._is_generating)
        worker_type.assert_not_called()

    def test_stale_cancel_signal_cannot_clear_new_worker(self):
        stale = _RunningWorker()
        current = _RunningWorker()
        self.view._worker = current
        self.view._is_generating = True

        with mock.patch.object(self.view, "sender", return_value=stale):
            self.view._on_cancelled()

        self.assertIs(self.view._worker, current)
        self.assertTrue(self.view._is_generating)

    def test_current_cancel_signal_releases_worker(self):
        worker = _RunningWorker()
        self.view._worker = worker
        self.view._is_generating = True
        with mock.patch.object(
            self.view._batch_view, "refresh_recoverable_jobs"
        ):
            self.view._on_cancelled()

        self.assertIsNone(self.view._worker)
        self.assertFalse(self.view._is_generating)


if __name__ == "__main__":
    unittest.main()
