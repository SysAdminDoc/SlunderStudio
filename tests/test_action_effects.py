import os
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from engines.sfx_engine import SFXResult
from core.voice_bank import VoiceBank, VoiceProfile
from ui.sfx_view import SFXView
from ui.vocal_suite_view import VocalSuiteView


def _write_wav(path: str, frames: int = 128, sample_rate: int = 8000):
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\0\0" * frames)


class ActionEffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_vocal_export_writes_selected_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.wav")
            target = os.path.join(tmp, "vocal-export.wav")
            _write_wav(source)
            view = VocalSuiteView()
            try:
                view._current_audio_path = source
                with mock.patch(
                    "ui.vocal_suite_view.QFileDialog.getSaveFileName",
                    return_value=(target, "WAV (*.wav)"),
                ):
                    view._on_export()

                self.assertTrue(os.path.isfile(target))
                self.assertIn("Exported vocal WAV", view._status.text())
            finally:
                view.close()
                self._app.processEvents()

    def test_sfx_card_play_loads_its_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "sfx.wav")
            _write_wav(source)
            view = SFXView()
            try:
                result = SFXResult(
                    audio=np.zeros((32, 2), dtype=np.float32),
                    sample_rate=8000,
                    file_path=source,
                )
                with mock.patch("ui.sfx_view.AudioEngine") as engine_type:
                    engine = engine_type.return_value
                    engine.load_file.return_value = True
                    view._add_result_card(result)
                    view._cards[-1].play_requested.emit(result)

                    engine.load_file.assert_called_once_with(source)
                    engine.play.assert_called_once_with()
            finally:
                view.close()
                self._app.processEvents()

    def test_stem_play_loads_selected_stem_array(self):
        view = VocalSuiteView()
        try:
            audio = np.zeros((32, 2), dtype=np.float32)
            view._stem_mixer.load_stems([
                SimpleNamespace(name="vocals", audio=audio),
            ], sample_rate=16000)
            with mock.patch("ui.vocal_suite_view.AudioEngine") as engine_type:
                engine = engine_type.return_value
                engine.load_array.return_value = True
                view._stem_mixer.stem_play.emit("vocals")

                engine.load_array.assert_called_once_with(audio, 16000)
                engine.play.assert_called_once_with()
                self.assertEqual("Playing vocals stem", view._status.text())
        finally:
            view.close()
            self._app.processEvents()

    def test_vocal_suite_exposes_and_persists_unsafe_checkpoint_trust(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            VoiceBank._instance = None
            try:
                with mock.patch("core.voice_bank.get_config_dir", return_value=root):
                    bank = VoiceBank()
                    checkpoint = root / "voice.pth"
                    checkpoint.write_bytes(b"checkpoint")
                    profile = VoiceProfile(
                        name="Reviewable RVC",
                        engine="rvc",
                        model_path=str(checkpoint),
                    )
                    bank.add(profile)

                    view = VocalSuiteView()
                    try:
                        index = view._rvc_voice.findData(profile.id)
                        self.assertGreaterEqual(index, 0)
                        view._rvc_voice.setCurrentIndex(index)
                        self.assertTrue(view._rvc_trust_btn.isEnabled())
                        self.assertIn("may execute code", view._rvc_trust_label.text())

                        view._on_trust_rvc_profile()

                        self.assertTrue(bank.get(profile.id).trusted)
                        self.assertFalse(view._rvc_trust_btn.isEnabled())
                        self.assertIn("explicitly trusted", view._rvc_trust_label.text())
                    finally:
                        view.close()
                        self._app.processEvents()
            finally:
                VoiceBank._instance = None


if __name__ == "__main__":
    unittest.main()
