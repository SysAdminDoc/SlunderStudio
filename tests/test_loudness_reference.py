import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from core.mastering import (
    match_loudness_to_reference,
    measure_lufs,
    measure_short_term_lufs,
)
from ui.mixer_view import MixerView


class LoudnessReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_short_term_lufs_tracks_windows(self):
        sr = 12000
        audio = self._tone(sr, 220.0, 0.15, seconds=5.0)

        profile = measure_short_term_lufs(audio, sr, window_sec=2.0, hop_sec=1.0)

        self.assertGreaterEqual(len(profile), 4)
        self.assertEqual(0.0, profile[0].time_sec)
        self.assertTrue(all(point.lufs > -60.0 for point in profile))

    def test_match_loudness_to_reference_uses_reference_lufs(self):
        sr = 12000
        source = self._stereo(self._tone(sr, 220.0, 0.08, seconds=5.0))
        reference = self._stereo(self._tone(sr, 220.0, 0.28, seconds=5.0))

        result = match_loudness_to_reference(source, sr, reference, sr)

        self.assertGreater(result.gain_db, 0.0)
        self.assertAlmostEqual(result.reference_lufs, result.output_lufs, delta=0.4)
        self.assertAlmostEqual(measure_lufs(result.audio, sr), result.output_lufs, delta=0.01)
        self.assertTrue(result.output_short_term)
        self.assertGreaterEqual(result.average_short_term_delta_db, 0.0)
        self.assertLessEqual(float(np.max(np.abs(result.audio))), 1.0)

    def test_mixer_matches_master_to_loaded_reference(self):
        sr = 44100
        track = self._stereo(self._tone(sr, 220.0, 0.08, seconds=5.0))
        reference = self._stereo(self._tone(sr, 220.0, 0.22, seconds=5.0))
        view = MixerView()
        try:
            view.add_track("Mix", track, sr)
            view.set_reference_track("Reference", reference, sr)

            with mock.patch(
                "ui.mixer_view.QFileDialog.getSaveFileName",
                return_value=("", ""),
            ):
                view._on_master_export()

            self.assertIsNotNone(view._last_loudness_match)
            self.assertIn("Ref:", view._lufs_label.text())
            self.assertIn("ST avg delta", view._lufs_label.text())
            self.assertIn("matched to Reference", view._status.text())
        finally:
            view.deleteLater()

    def _tone(self, sr: int, frequency: float, gain: float, seconds: float) -> np.ndarray:
        t = np.arange(int(sr * seconds), dtype=np.float32) / sr
        return (gain * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)

    def _stereo(self, audio: np.ndarray) -> np.ndarray:
        return np.column_stack([audio, audio * 0.92]).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
