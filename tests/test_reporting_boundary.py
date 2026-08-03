import inspect
import logging
import tempfile
import unittest
from pathlib import Path

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QWidget

from core.logging_setup import (
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    configure_logging,
)
from core.workers import InferenceWorker
from ui.ai_producer_view import AIProducerView
from ui.midi_studio_view import MidiStudioView
from ui.mixer_view import MixerView
from ui.toast import ToastHistoryDialog, ToastManager
from ui.vocal_suite_view import VocalSuiteView


class ReportingBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self):
        self._close_application_handlers()

    @staticmethod
    def _close_application_handlers():
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "_slunderstudio_file_handler", False):
                root.removeHandler(handler)
                handler.close()

    def test_logging_is_rotating_and_records_worker_tracebacks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = configure_logging(temp_dir)
            root_handler = next(
                handler
                for handler in logging.getLogger().handlers
                if getattr(handler, "_slunderstudio_file_handler", False)
            )
            self.assertEqual(root_handler.maxBytes, LOG_MAX_BYTES)
            self.assertEqual(root_handler.backupCount, LOG_BACKUP_COUNT)

            def fail(**_kwargs):
                raise RuntimeError("worker boom")

            worker = InferenceWorker(fail)
            worker.run()

            text = Path(log_path).read_text(encoding="utf-8")
            self.assertIn("Inference worker failed", text)
            self.assertIn("RuntimeError: worker boom", text)
            self._close_application_handlers()

    def test_notification_history_dialog_re_reads_dismissed_messages(self):
        host = QWidget()
        self.addCleanup(host.deleteLater)
        manager = ToastManager(host)
        manager._record("Dismissed error", "error")

        dialog = ToastHistoryDialog(manager, host)
        self.addCleanup(dialog.close)
        self.assertIn("Dismissed error", dialog._history.toPlainText())

        manager._record("Later warning", "warning")
        self._app.processEvents()
        self.assertIn("Later warning", dialog._history.toPlainText())

    def test_every_main_view_constructor_has_a_reporting_channel(self):
        for view_type in (
            MidiStudioView,
            VocalSuiteView,
            MixerView,
            AIProducerView,
        ):
            self.assertIn("toast_mgr", inspect.signature(view_type).parameters)

        source = Path(__file__).resolve().parents[1] / "ui" / "main_window.py"
        text = source.read_text(encoding="utf-8")
        for constructor in (
            "MidiStudioView(toast_mgr=self.toast_mgr)",
            "VocalSuiteView(toast_mgr=self.toast_mgr)",
            "MixerView(toast_mgr=self.toast_mgr)",
            "AIProducerView(toast_mgr=self.toast_mgr)",
        ):
            self.assertIn(constructor, text)


if __name__ == "__main__":
    unittest.main()
