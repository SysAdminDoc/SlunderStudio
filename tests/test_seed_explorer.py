import os
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from engines.audio_analyzer import QualityScore, score_generation_quality
from core.provenance import sidecar_path_for
from ui.batch_view import BatchView, _prepare_batch_card_task
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

    def test_batch_preview_decodes_once_and_scores_that_same_buffer(self):
        audio = np.zeros((128, 2), dtype=np.float32)
        score = QualityScore(total=73.0)
        with mock.patch(
            "ui.batch_view.decode_audio_file",
            return_value=(audio, 44100),
        ) as decode, mock.patch(
            "engines.audio_analyzer.score_audio_buffer",
            return_value=score,
        ) as score_buffer:
            result = _prepare_batch_card_task("variation.wav")

        decode.assert_called_once()
        score_buffer.assert_called_once()
        self.assertIs(score_buffer.call_args.args[0], audio)
        self.assertIs(result["audio"], audio)
        self.assertEqual(73.0, result["quality"].total)


class BatchUseBestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def _wait_for_scores(self, view):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self._app.processEvents()
            if all(card._quality_worker is None for card in view._cards):
                return True
            time.sleep(0.01)
        self._app.processEvents()
        return all(card._quality_worker is None for card in view._cards)

    def test_use_best_selects_highest_scored_when_no_star(self):
        with tempfile.TemporaryDirectory() as tmp:
            low_path = os.path.join(tmp, "low.wav")
            _write_test_wav(low_path, amplitude=0.001, duration=1.0)
            high_path = os.path.join(tmp, "high.wav")
            _write_test_wav(high_path, amplitude=0.3, duration=5.0)

            view = BatchView()
            view.add_result(low_path, seed=1)
            view.add_result(high_path, seed=2)
            self.assertTrue(self._wait_for_scores(view))

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
            self.assertTrue(self._wait_for_scores(view))
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

    def _wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        self._app.processEvents()
        return bool(predicate())

    def test_distance_slider_syncs_seed_range(self):
        explorer = SeedExplorer()

        explorer._distance_slider.setValue(750)
        self.assertEqual(explorer._range_spin.value(), 750)

        explorer._range_spin.setValue(250)
        self.assertEqual(explorer._distance_slider.value(), 250)

    def test_explore_emits_seed_and_timestep_shift_grid(self):
        explorer = SeedExplorer()
        emitted = []
        explorer.generate_requested.connect(emitted.append)
        explorer._grid_combo.setCurrentIndex(0)
        explorer._seed_spin.setValue(1000)
        explorer._range_spin.setValue(100)
        explorer._shift_min_spin.setValue(1.0)
        explorer._shift_max_spin.setValue(3.0)

        explorer._start_exploration()

        self.assertEqual(len(emitted), 1)
        params = emitted[0]
        self.assertEqual(len(params), 4)
        self.assertEqual(params[0]["seed"], 950)
        self.assertEqual(params[-1]["seed"], 1050)
        self.assertEqual(params[0]["shift"], 1.0)
        self.assertEqual(params[-1]["shift"], 3.0)

    def test_export_starred_copies_audio_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "generated.wav"
            destination = root / "exported"
            destination.mkdir()
            _write_test_wav(str(source), duration=0.2)
            sidecar_path_for(source).write_text('{"test": true}', encoding="utf-8")

            explorer = SeedExplorer()
            explorer._cells[0][0].set_result(str(source), seed=123)
            explorer._cells[0][0]._toggle_star()

            with mock.patch(
                "ui.seed_explorer.QFileDialog.getExistingDirectory",
                return_value=str(destination),
                ):
                    explorer._export_starred()

            exported = destination / "seed_0_0_123.wav"
            self.assertTrue(
                self._wait_for(
                    lambda: "Exported 1 starred" in explorer._info.text()
                )
            )
            self.assertTrue(exported.is_file())
            self.assertTrue(sidecar_path_for(exported).is_file())
            self.assertIn("Exported 1 starred", explorer._info.text())

    def test_seed_cell_keyboard_playing_state_is_textual_and_starred(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "generated.wav")
            _write_test_wav(source, duration=0.2)

            explorer = SeedExplorer()
            cell = explorer._cells[0][0]
            cell.set_result(source, seed=123)
            self.assertTrue(
                self._wait_for(lambda: cell._waveform._waveform.has_audio)
            )
            played = []
            explorer.play_requested.connect(played.append)

            cell.keyPressEvent(
                QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
            )

            self.assertEqual(played, [source])
            self.assertIn("Playing", cell._status_label.text())
            self.assertIn("▶", cell._status_label.text())
            self.assertIn("Playing", cell.accessibleDescription())
            self.assertEqual(cell.focusPolicy(), Qt.FocusPolicy.StrongFocus)
            self.assertIn(":focus", cell.styleSheet())

            cell.keyPressEvent(
                QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_S, Qt.KeyboardModifier.NoModifier)
            )
            self.assertTrue(cell.is_starred)
            self.assertIn("Unstar", cell._star_btn.accessibleName())

    def test_seed_cells_are_all_focusable_keyboard_targets(self):
        explorer = SeedExplorer()
        cells = [cell for row in explorer._cells for cell in row]

        self.assertEqual(len(cells), 9)
        self.assertTrue(all(cell.focusPolicy() == Qt.FocusPolicy.StrongFocus for cell in cells))
        self.assertEqual(
            {cell.accessibleName() for cell in cells},
            {
                f"Seed variation row {row + 1}, column {col + 1}"
                for row in range(3)
                for col in range(3)
            },
        )


if __name__ == "__main__":
    unittest.main()
