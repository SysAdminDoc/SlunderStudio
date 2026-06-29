import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.provenance import sidecar_path_for
from core.settings import Settings
from core.voice_bank import VoiceBank
from engines.vocal_tuning import AutoTuneParams, autotune_file, compute_frame_corrections
from ui.vocal_suite_view import VocalSuiteView


class VocalTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self):
        Settings._instance = None
        VoiceBank._instance = None

    def test_frame_corrections_pull_voiced_pitch_to_nearest_semitone(self):
        f0 = np.array([450.0, 450.0, 450.0, 440.0, 440.0, np.nan], dtype=np.float32)
        voiced = np.array([True, True, True, True, True, False])

        corrections = compute_frame_corrections(f0, voiced, strength=1.0)

        self.assertLess(corrections[1], -0.25)
        self.assertAlmostEqual(corrections[4], 0.0, delta=0.05)
        self.assertEqual(corrections[5], 0.0)

    def test_autotune_file_writes_wav_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr = 22050
            t = np.arange(int(sr * 0.6)) / sr
            # A4 is 440 Hz; this is intentionally sharp.
            audio = (0.3 * np.sin(2 * np.pi * 450.0 * t)).astype(np.float32)
            source = root / "sharp_vocal.wav"
            output = root / "sharp_vocal_autotune.wav"
            sf.write(source, audio, sr, subtype="PCM_16")

            result = autotune_file(
                AutoTuneParams(
                    input_path=str(source),
                    output_path=str(output),
                    strength=1.0,
                    fmin_note="A3",
                    fmax_note="A5",
                )
            )

            self.assertTrue(output.is_file())
            self.assertTrue(sidecar_path_for(output).is_file())
            self.assertEqual(result.output_path, str(output))
            self.assertEqual(result.sample_rate, sr)
            self.assertGreater(result.frames_analyzed, 0)
            self.assertGreaterEqual(result.voiced_frames, 0)

    def test_vocal_suite_handoff_prepares_autotune_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "vocal.wav"
            sf.write(source, np.zeros(1024, dtype=np.float32), 22050)

            with self._patched_config(root):
                view = VocalSuiteView()
                try:
                    view.set_audio(str(source))
                    self.assertEqual(str(source), view._autotune_input_label.property("path"))
                    self.assertTrue(view._autotune_apply_btn.isEnabled())
                    self.assertEqual(4, view._tabs.currentIndex())

                    view._autotune_strength.setValue(42)
                    self.assertEqual(0.42, Settings().get("vocal_suite.autotune_strength"))
                finally:
                    view.deleteLater()

    def _patched_config(self, root: Path):
        config_dir = root / "config"
        output_dir = root / "renders"
        model_dir = root / "models"
        trash_dir = root / "trash"
        for path in (config_dir, output_dir, model_dir, trash_dir):
            path.mkdir(parents=True, exist_ok=True)
        Settings._instance = None
        VoiceBank._instance = None
        stack = ExitStack()
        stack.enter_context(mock.patch("core.settings.get_config_dir", return_value=config_dir))
        stack.enter_context(mock.patch("core.settings.get_default_output_dir", return_value=output_dir))
        stack.enter_context(mock.patch("core.settings.get_default_cache_dir", return_value=model_dir))
        stack.enter_context(mock.patch("core.settings.get_trash_dir", return_value=trash_dir))
        stack.enter_context(mock.patch("core.voice_bank.get_config_dir", return_value=config_dir))
        return stack


if __name__ == "__main__":
    unittest.main()
