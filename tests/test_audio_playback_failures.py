import os
import unittest
from unittest import mock
from types import SimpleNamespace

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

    def test_failed_file_load_preserves_previous_track(self):
        engine = AudioEngine()
        data = np.arange(32, dtype=np.float32).reshape(16, 2)
        engine.load_array(data, 8000)
        engine.seek(0.5)
        before_position = engine.position

        with mock.patch.object(audio_engine, "_ensure_audio_libs"), mock.patch.object(
            audio_engine, "_sf", _FailingSoundFile
        ):
            self.assertFalse(engine.load_file("missing.wav"))

        np.testing.assert_array_equal(engine._audio_data, data)
        self.assertEqual(before_position, engine.position)

    def test_callback_end_emits_playback_finished_once(self):
        engine = AudioEngine()
        engine.cleanup()

        class _CallbackStop(Exception):
            pass

        class _Stream:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]

            def start(self):
                return None

            def stop(self):
                return None

            def close(self):
                return None

        fake_sd = SimpleNamespace(OutputStream=_Stream, CallbackStop=_CallbackStop)
        finished = []
        engine.playback_finished.connect(lambda: finished.append(True))
        engine.load_array(np.zeros((4, 1), dtype=np.float32), 4)

        with mock.patch.object(audio_engine, "_sd", fake_sd), mock.patch.object(
            audio_engine, "_sf", object()
        ):
            engine.play()
            stream = engine._stream
            stream.callback(np.zeros((4, 1), dtype=np.float32), 4, None, None)
            with self.assertRaises(_CallbackStop):
                stream.callback(np.zeros((4, 1), dtype=np.float32), 4, None, None)

            engine._emit_position()
            engine._emit_position()

        self.assertEqual([True], finished)
        self.assertFalse(engine.is_playing)
        engine.cleanup()

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
