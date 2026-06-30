import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from core.mastering import (
    LUFS_TARGETS,
    match_loudness_to_reference,
    measure_lufs,
    measure_short_term_lufs,
)
from core.settings import Settings
from ui.mixer_view import MixerView
from ui.settings_view import SettingsView


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

    def test_expanded_lufs_targets_are_available_in_mixer_and_settings(self):
        self.assertEqual(-16.0, LUFS_TARGETS["podcast"].lufs)
        self.assertEqual(-24.0, LUFS_TARGETS["broadcast"].lufs)
        self.assertEqual(-27.0, LUFS_TARGETS["cinema"].lufs)

        mixer = MixerView()
        try:
            labels = [mixer._target_combo.itemText(i) for i in range(mixer._target_combo.count())]
            self.assertIn("Podcast stereo (-16 LUFS)", labels)
            self.assertIn("Broadcast (-24 LUFS)", labels)
            self.assertIn("Cinema dialog (-27 LUFS)", labels)

            mixer._set_target_combo_key("cinema")
            self.assertAlmostEqual(-27.0, mixer._lufs_spin.value(), places=2)
            self.assertLessEqual(mixer._lufs_spin.minimum(), -27.0)
        finally:
            mixer.deleteLater()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Settings._instance = None
            with mock.patch("core.settings.get_config_dir", return_value=root / "config"), \
                    mock.patch("core.settings.get_default_output_dir", return_value=root / "renders"), \
                    mock.patch("core.settings.get_default_cache_dir", return_value=root / "models"), \
                    mock.patch("core.settings.get_trash_dir", return_value=root / "trash"):
                settings = SettingsView()
                try:
                    keys = [
                        settings._mastering_target.itemData(i)
                        for i in range(settings._mastering_target.count())
                    ]
                    self.assertIn("podcast", keys)
                    self.assertIn("broadcast", keys)
                    self.assertIn("cinema", keys)
                finally:
                    settings.deleteLater()
                    Settings._instance = None

    def _tone(self, sr: int, frequency: float, gain: float, seconds: float) -> np.ndarray:
        t = np.arange(int(sr * seconds), dtype=np.float32) / sr
        return (gain * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)

    def _stereo(self, audio: np.ndarray) -> np.ndarray:
        return np.column_stack([audio, audio * 0.92]).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
