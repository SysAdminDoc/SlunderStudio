import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from core.mastering import DynamicEQBand, apply_dynamic_eq, suggest_dynamic_eq_curve
from ui.mixer_view import MixerView


class DynamicEQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def _wait_for(cls, predicate, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            cls._app.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        cls._app.processEvents()
        if not predicate():
            raise AssertionError("Timed out waiting for Mixer worker")

    def test_suggest_dynamic_eq_curve_is_stem_aware(self):
        sr = 22050
        audio = self._tone(sr, [(90.0, 0.7), (320.0, 0.35), (3500.0, 0.06)])

        suggestion = suggest_dynamic_eq_curve(audio, sr, "Lead Vocal Stem")

        self.assertEqual("vocal", suggestion.stem_role)
        self.assertTrue(suggestion.bands)
        self.assertTrue(any(band.frequency_hz < 400.0 and band.gain_db < 0 for band in suggestion.bands))
        self.assertTrue(any(band.frequency_hz > 2500.0 and band.gain_db > 0 for band in suggestion.bands))
        self.assertGreater(suggestion.low_ratio, 0.0)

    def test_apply_dynamic_eq_preserves_shape_and_changes_signal(self):
        sr = 22050
        mono = self._tone(sr, [(240.0, 0.55), (2400.0, 0.1)])
        stereo = np.column_stack([mono, mono * 0.75]).astype(np.float32)
        bands = (DynamicEQBand(240.0, -3.0, 1.0, "Test cut"),)

        processed = apply_dynamic_eq(stereo, sr, bands, strength=1.0)

        self.assertEqual(stereo.shape, processed.shape)
        self.assertTrue(np.all(np.isfinite(processed)))
        self.assertFalse(np.allclose(stereo, processed))
        self.assertLessEqual(float(np.max(np.abs(processed))), 1.0)

    def test_analysis_suggests_without_touching_the_audio(self):
        sr = 22050
        audio = self._tone(sr, [(90.0, 0.7), (320.0, 0.35), (3500.0, 0.06)])
        stereo = np.column_stack([audio, audio]).astype(np.float32)
        view = MixerView()
        try:
            view.add_track("Lead Vocal", stereo, sr)
            before = view._tracks[0]["audio"].copy()

            view._on_suggest_dynamic_eq()
            self._wait_for(lambda: view._dynamic_eq_worker is None)

            np.testing.assert_allclose(view._tracks[0]["audio"], before)
            self.assertIn(0, view._dynamic_eq_suggestions)
            self.assertEqual("vocal", view._dynamic_eq_suggestions[0].stem_role)
            self.assertIn("not applied", view._status.text())
            self.assertTrue(view._dynamic_eq_preview_btn.isEnabled())
            self.assertFalse(view._dynamic_eq_revert_btn.isEnabled())
        finally:
            view.deleteLater()

    def test_preview_is_gain_matched_and_reversible(self):
        sr = 22050
        audio = self._tone(sr, [(90.0, 0.7), (320.0, 0.35), (3500.0, 0.06)])
        stereo = np.column_stack([audio, audio]).astype(np.float32)
        view = MixerView()
        try:
            view.add_track("Lead Vocal", stereo, sr)
            before = view._tracks[0]["audio"].copy()
            view._on_suggest_dynamic_eq()
            self._wait_for(lambda: view._dynamic_eq_worker is None)

            view._dynamic_eq_preview_btn.setChecked(True)
            self._wait_for(lambda: view._dynamic_eq_operation_worker is None)
            previewed = view._tracks[0]["audio"]
            self.assertFalse(np.allclose(before, previewed))
            # Gain-matched: the preview compares tone, not level.
            self.assertAlmostEqual(
                float(np.sqrt(np.mean(np.square(previewed)))),
                float(np.sqrt(np.mean(np.square(before)))),
                delta=1e-6,
            )

            view._dynamic_eq_preview_btn.setChecked(False)
            self._wait_for(lambda: view._dynamic_eq_operation_worker is None)
            np.testing.assert_allclose(view._tracks[0]["audio"], before)
        finally:
            view.deleteLater()

    def test_apply_then_revert_restores_the_original(self):
        sr = 22050
        audio = self._tone(sr, [(90.0, 0.7), (320.0, 0.35), (3500.0, 0.06)])
        stereo = np.column_stack([audio, audio]).astype(np.float32)
        view = MixerView()
        try:
            view.add_track("Lead Vocal", stereo, sr)
            before = view._tracks[0]["audio"].copy()
            view._on_suggest_dynamic_eq()
            self._wait_for(lambda: view._dynamic_eq_worker is None)

            view._on_apply_dynamic_eq()
            self._wait_for(lambda: view._dynamic_eq_operation_worker is None)
            self.assertFalse(np.allclose(before, view._tracks[0]["audio"]))
            self.assertTrue(view._dynamic_eq_revert_btn.isEnabled())
            self.assertIn("Revert", view._status.text())

            view._on_revert_dynamic_eq()
            np.testing.assert_allclose(view._tracks[0]["audio"], before)
            self.assertFalse(view._dynamic_eq_revert_btn.isEnabled())
        finally:
            view.deleteLater()

    def test_preview_and_apply_require_analysis_first(self):
        sr = 22050
        audio = self._tone(sr, [(90.0, 0.7), (320.0, 0.35)])
        stereo = np.column_stack([audio, audio]).astype(np.float32)
        view = MixerView()
        try:
            view.add_track("Lead Vocal", stereo, sr)
            before = view._tracks[0]["audio"].copy()

            view._on_apply_dynamic_eq()
            np.testing.assert_allclose(view._tracks[0]["audio"], before)
            self.assertIn("Analyze", view._status.text())

            view._dynamic_eq_preview_btn.setChecked(True)
            self.assertFalse(view._dynamic_eq_preview_btn.isChecked())
            np.testing.assert_allclose(view._tracks[0]["audio"], before)
        finally:
            view.deleteLater()

    def test_mixer_reindexes_dynamic_eq_suggestions_after_remove(self):
        sr = 22050
        vocal = self._tone(sr, [(90.0, 0.7), (320.0, 0.35), (3500.0, 0.06)])
        bass = self._tone(sr, [(70.0, 0.6), (260.0, 0.32)])
        view = MixerView()
        try:
            view.add_track("Lead Vocal", np.column_stack([vocal, vocal]).astype(np.float32), sr)
            view.add_track("Bass", np.column_stack([bass, bass]).astype(np.float32), sr)

            view._on_suggest_dynamic_eq()
            self._wait_for(lambda: view._dynamic_eq_worker is None)
            view._on_remove_track(0)

            self.assertIn(0, view._dynamic_eq_suggestions)
            self.assertEqual("bass", view._dynamic_eq_suggestions[0].stem_role)
            self.assertEqual("Bass", view._tracks[0]["name"])
        finally:
            view.deleteLater()

    def _tone(self, sr: int, components: list[tuple[float, float]]) -> np.ndarray:
        t = np.arange(sr, dtype=np.float32) / sr
        audio = np.zeros_like(t)
        for frequency, gain in components:
            audio += gain * np.sin(2.0 * np.pi * frequency * t)
        peak = float(np.max(np.abs(audio)))
        if peak > 0.95:
            audio = audio / peak * 0.95
        return audio.astype(np.float32)


if __name__ == "__main__":
    unittest.main()
