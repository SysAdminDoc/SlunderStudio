import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import core.audio_engine as audio_engine
from core.audio_engine import AudioEngine
from ui.song_forge_view import SongForgeView


class _FailingSoundFile:
    @staticmethod
    def read(*_args, **_kwargs):
        raise OSError("file is missing")


class _ToastRecorder:
    def __init__(self):
        self.messages = []

    def show_toast(self, message, toast_type):
        self.messages.append((message, toast_type))


class AudioPlaybackFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_failed_file_load_discards_previous_track(self):
        engine = AudioEngine()
        engine.load_array(np.zeros((16, 2), dtype=np.float32), 8000)

        with mock.patch.object(audio_engine, "_ensure_audio_libs"), mock.patch.object(
            audio_engine, "_sf", _FailingSoundFile
        ):
            self.assertFalse(engine.load_file("missing.wav"))

        self.assertIsNone(engine._audio_data)
        self.assertIsNone(engine._source_path)

    def test_song_forge_does_not_play_when_file_load_fails(self):
        toast = _ToastRecorder()
        view = SongForgeView(toast_mgr=toast)
        try:
            path = "missing-batch-result.wav"
            with mock.patch("ui.song_forge_view.AudioEngine") as engine_type:
                engine = engine_type.return_value
                engine.load_file.return_value = False

                view._play_audio(path)

                engine.load_file.assert_called_once_with(path)
                engine.play.assert_not_called()

            self.assertEqual(
                [(f"Could not load {path}", "error")], toast.messages
            )
        finally:
            view.deleteLater()
            self._app.processEvents()

    def test_load_output_reports_missing_file_size(self):
        view = SongForgeView()
        try:
            view._load_output("missing-output.wav", seed=42)
            self.assertIn("size unavailable", view._output_info.text())
        finally:
            view.deleteLater()
            self._app.processEvents()


if __name__ == "__main__":
    unittest.main()
