import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from core.mastering import MasteringResult, apply_mid_side_gain
from ui.mixer_view import MixerView


class MidSideMasteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_apply_mid_side_gain_adjusts_side_energy(self):
        left = np.array([0.4, -0.3, 0.2, -0.1], dtype=np.float32)
        right = np.array([0.1, -0.05, -0.2, 0.05], dtype=np.float32)
        audio = np.column_stack([left, right])

        processed = apply_mid_side_gain(audio, mid_gain_db=0.0, side_gain_db=3.0)

        before_side = np.mean(np.square((audio[:, 0] - audio[:, 1]) * 0.5))
        after_side = np.mean(np.square((processed[:, 0] - processed[:, 1]) * 0.5))
        self.assertGreater(after_side, before_side)
        self.assertEqual(audio.shape, processed.shape)
        self.assertLessEqual(float(np.max(np.abs(processed))), 1.0)

    def test_mixer_passes_mid_side_controls_to_mastering_preset(self):
        sr = 44100
        t = np.arange(sr, dtype=np.float32) / sr
        audio = np.column_stack([
            0.1 * np.sin(2.0 * np.pi * 220.0 * t),
            0.08 * np.sin(2.0 * np.pi * 330.0 * t),
        ]).astype(np.float32)
        captured = {}

        def fake_master(mixed, sample_rate, preset):
            captured["mid"] = preset.ms_mid_gain_db
            captured["side"] = preset.ms_side_gain_db
            captured["target_lufs"] = preset.target_lufs
            return MasteringResult(
                audio=mixed,
                sample_rate=sample_rate,
                input_lufs=-20.0,
                output_lufs=preset.target_lufs,
                peak_db=-1.0,
                preset_name=preset.name,
            )

        view = MixerView()
        try:
            view.add_track("Stereo Mix", audio, sr)
            view._mid_gain_spin.setValue(1.5)
            view._side_gain_spin.setValue(-2.0)
            with mock.patch("ui.mixer_view.master_audio", side_effect=fake_master), \
                    mock.patch("ui.mixer_view.QFileDialog.getSaveFileName", return_value=("", "")):
                view._on_master_export()

            self.assertEqual(1.5, captured["mid"])
            self.assertEqual(-2.0, captured["side"])
            self.assertIn("Mastered", view._status.text())
        finally:
            view.deleteLater()


if __name__ == "__main__":
    unittest.main()
