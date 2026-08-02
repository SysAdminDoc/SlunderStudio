import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.lyrics_db import LyricsDB
from core.settings import Settings
from ui.lyrics_view import LyricsView


class LyricsStreamingThreadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self):
        if LyricsDB._instance is not None:
            LyricsDB._instance.close()
        LyricsDB._instance = None
        Settings._instance = None

    def test_streaming_tokens_reach_editor_on_gui_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            with mock.patch("core.settings.get_config_dir", return_value=config_dir), \
                 mock.patch("core.lyrics_db.get_config_dir", return_value=config_dir):
                view = LyricsView()
                token_sent = threading.Event()
                token_threads: list[int] = []
                original_append_token = view._editor.append_token

                def record_token(token: str):
                    token_threads.append(threading.get_ident())
                    original_append_token(token)

                view._editor.append_token = record_token

                def fake_generator(prompt: str, token_cb=None, **_kwargs):
                    self.assertEqual(prompt, "thread-safe lyrics")
                    self.assertIsNotNone(token_cb)
                    token_cb("Hello")
                    token_sent.set()
                    time.sleep(0.1)
                    return {"lyrics": "Hello", "genre": "pop", "mood": "bright"}

                try:
                    view._run_generation(fake_generator, "thread-safe lyrics")
                    self.assertTrue(token_sent.wait(timeout=2))

                    deadline = time.monotonic() + 2
                    while not token_threads and time.monotonic() < deadline:
                        self._app.processEvents()
                        time.sleep(0.01)

                    worker = view._worker
                    if worker is not None:
                        worker.wait(5000)
                    self._app.processEvents()

                    self.assertEqual(token_threads, [threading.get_ident()])
                    self.assertEqual(view._editor.text, "Hello")
                finally:
                    worker = view._worker
                    if worker is not None and worker.isRunning():
                        worker.cancel()
                        worker.wait(5000)
                    if LyricsDB._instance is not None:
                        LyricsDB._instance.close()
                    LyricsDB._instance = None
                    Settings._instance = None
                    view.deleteLater()


if __name__ == "__main__":
    unittest.main()
