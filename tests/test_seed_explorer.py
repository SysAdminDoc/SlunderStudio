import os
import tempfile
import unittest
import wave

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from engines.audio_analyzer import QualityScore, score_generation_quality
from ui.batch_view import BatchView
from ui.seed_explorer import SeedExplorer


def _write_test_wav(path: str, freq: float = 440.0, duration: float = 2.0,
                     amplitude: float = 0.3, sr: int = 44100):
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    audio = amplitude * np.sin(2 * np.pi * freq * t)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class QualityScoringTests(unittest.TestCase):
    def test_good_audio_scores_above_60(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "good.wav")
            _write_test_wav(path, freq=440, duration=10.0, amplitude=0.3)
            score = score_generation_quality(path, expected_duration=10.0)
            self.assertGreater(score.total, 60.0)
            self.assertGreater(score.silence, 0.0)
            self.assertGreater(score.clipping, 0.0)

    def test_silent_audio_scores_low(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "silent.wav")
            _write_test_wav(path, amplitude=0.0, duration=5.0)
            score = score_generation_quality(path, expected_duration=5.0)
            self.assertLessEqual(score.total, 40.0)

    def test_clipped_audio_loses_clipping_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clipped.wav")
            _write_test_wav(path, amplitude=0.99, duration=5.0)
            score = score_generation_quality(path, expected_duration=5.0)
            self.assertGreater(score.clipping, 0.0)

    def test_score_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "det.wav")
            _write_test_wav(path, freq=330, duration=3.0, amplitude=0.2)
            s1 = score_generation_quality(path)
            s2 = score_generation_quality(path)
            self.assertEqual(s1.total, s2.total)


class BatchUseBestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_use_best_selects_highest_scored_when_no_star(self):
        with tempfile.TemporaryDirectory() as tmp:
            low_path = os.path.join(tmp, "low.wav")
            _write_test_wav(low_path, amplitude=0.001, duration=1.0)
            high_path = os.path.join(tmp, "high.wav")
            _write_test_wav(high_path, amplitude=0.3, duration=5.0)

            view = BatchView()
            view.add_result(low_path, seed=1)
            view.add_result(high_path, seed=2)

            emitted = []
            view.use_result.connect(emitted.append)
            view._use_best()

            self.assertEqual(len(emitted), 1)
            self.assertEqual(emitted[0], high_path)
            view.deleteLater()

    def test_use_best_prefers_starred_over_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            low_path = os.path.join(tmp, "low.wav")
            _write_test_wav(low_path, amplitude=0.001, duration=1.0)
            high_path = os.path.join(tmp, "high.wav")
            _write_test_wav(high_path, amplitude=0.3, duration=5.0)

            view = BatchView()
            view.add_result(low_path, seed=1)
            view.add_result(high_path, seed=2)
            view._cards[0]._toggle_star()

            emitted = []
            view.use_result.connect(emitted.append)
            view._use_best()

            self.assertEqual(len(emitted), 1)
            self.assertEqual(emitted[0], low_path)
            view.deleteLater()


class SeedExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_distance_slider_syncs_seed_range(self):
        explorer = SeedExplorer()

        explorer._distance_slider.setValue(750)
        self.assertEqual(explorer._range_spin.value(), 750)

        explorer._range_spin.setValue(250)
        self.assertEqual(explorer._distance_slider.value(), 250)

    def test_explore_emits_seed_and_cfg_grid(self):
        explorer = SeedExplorer()
        emitted = []
        explorer.generate_requested.connect(emitted.append)
        explorer._grid_combo.setCurrentIndex(0)
        explorer._seed_spin.setValue(1000)
        explorer._range_spin.setValue(100)
        explorer._cfg_min_spin.setValue(3.0)
        explorer._cfg_max_spin.setValue(5.0)

        explorer._start_exploration()

        self.assertEqual(len(emitted), 1)
        params = emitted[0]
        self.assertEqual(len(params), 4)
        self.assertEqual(params[0]["seed"], 950)
        self.assertEqual(params[-1]["seed"], 1050)
        self.assertEqual(params[0]["cfg_scale"], 3.0)
        self.assertEqual(params[-1]["cfg_scale"], 5.0)


if __name__ == "__main__":
    unittest.main()
