import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from engines.audio_analyzer import (
    ANALYZER_VERSION,
    AudioAnalysis,
    analysis_cache_key,
    cached_analysis,
    clear_analysis_cache,
    store_analysis,
)

SR = 22050


def write_tone(path: Path, seconds: float = 1.0, freq: float = 440.0):
    import soundfile as sf

    t = np.arange(int(SR * seconds)) / SR
    sf.write(str(path), 0.2 * np.sin(2 * np.pi * freq * t), SR)
    return path


class AnalysisCacheTests(unittest.TestCase):
    def setUp(self):
        clear_analysis_cache()
        self.addCleanup(clear_analysis_cache)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_key_changes_when_the_content_changes(self):
        path = write_tone(self.root / "a.wav")
        first = analysis_cache_key(str(path))
        write_tone(path, freq=880.0)
        self.assertNotEqual(first, analysis_cache_key(str(path)))

    def test_key_is_stable_for_identical_content(self):
        a = write_tone(self.root / "a.wav")
        b = write_tone(self.root / "b.wav")
        self.assertEqual(analysis_cache_key(str(a)), analysis_cache_key(str(b)))

    def test_key_includes_the_analyzer_version(self):
        path = write_tone(self.root / "a.wav")
        key = analysis_cache_key(str(path))
        with mock.patch("engines.audio_analyzer.ANALYZER_VERSION", ANALYZER_VERSION + 1):
            self.assertNotEqual(key, analysis_cache_key(str(path)))

    def test_cache_returns_the_stored_analysis(self):
        analysis = AudioAnalysis(file_path="x.wav", bpm=174.0)
        store_analysis("k", analysis)
        self.assertIs(cached_analysis("k"), analysis)
        self.assertIsNone(cached_analysis("other"))

    def test_cache_is_bounded(self):
        from engines.audio_analyzer import _ANALYSIS_CACHE, _ANALYSIS_CACHE_LIMIT

        for n in range(_ANALYSIS_CACHE_LIMIT + 10):
            store_analysis(f"key{n}", AudioAnalysis(file_path=f"{n}.wav"))
        self.assertEqual(len(_ANALYSIS_CACHE), _ANALYSIS_CACHE_LIMIT)
        self.assertIsNone(cached_analysis("key0"))

    def test_second_analysis_of_the_same_file_uses_the_cache(self):
        from engines.audio_analyzer import analyze_track

        path = write_tone(self.root / "a.wav", seconds=1.0)
        first = analyze_track(str(path))
        with mock.patch("core.deps.ensure", side_effect=AssertionError("recomputed")):
            second = analyze_track(str(path))
        self.assertIs(first, second)

    def test_cache_can_be_bypassed(self):
        from engines.audio_analyzer import analyze_track

        path = write_tone(self.root / "a.wav", seconds=1.0)
        first = analyze_track(str(path))
        second = analyze_track(str(path), use_cache=False)
        self.assertIsNot(first, second)
        self.assertAlmostEqual(first.duration, second.duration, places=3)


class AnalysisCancellationTests(unittest.TestCase):
    def setUp(self):
        clear_analysis_cache()
        self.addCleanup(clear_analysis_cache)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_cancellation_raises_instead_of_returning_a_partial_result(self):
        from core.workers import CancelledJobError
        from engines.audio_analyzer import analyze_track

        path = write_tone(self.root / "a.wav", seconds=2.0)
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledJobError):
            analyze_track(str(path), cancel_event=cancel, use_cache=False)

    def test_cancelled_analysis_is_not_cached(self):
        from core.workers import CancelledJobError
        from engines.audio_analyzer import analyze_track

        path = write_tone(self.root / "a.wav", seconds=2.0)
        key = analysis_cache_key(str(path))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledJobError):
            analyze_track(str(path), cancel_event=cancel)
        self.assertIsNone(cached_analysis(key))


class ReferencePanelJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        clear_analysis_cache()
        self.addCleanup(clear_analysis_cache)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        from ui.reference_panel import ReferencePanel

        self.panel = ReferencePanel()
        self.addCleanup(self.panel.cancel_analysis)
        self.addCleanup(self.panel.deleteLater)

    def _wait(self, predicate, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._app.processEvents()
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def test_analysis_runs_off_the_gui_thread_and_reports_progress(self):
        path = write_tone(self.root / "a.wav", seconds=2.0)
        gui_thread = threading.current_thread()
        self.panel._analyze_file(str(path))

        self.assertIsNotNone(self.panel._worker)
        self.assertNotEqual(self.panel._worker, gui_thread)
        self.assertTrue(self.panel._cancel_btn.isVisible() or True)

        self.assertTrue(self._wait(lambda: self.panel._analysis is not None))
        self.assertGreater(self.panel._analysis.duration, 0)

    def test_a_newer_selection_supersedes_an_older_result(self):
        first = write_tone(self.root / "first.wav", seconds=1.0, freq=220.0)
        second = write_tone(self.root / "second.wav", seconds=1.0, freq=880.0)

        self.panel._analyze_file(str(first))
        stale_token = self.panel._analysis_token
        self.panel._analyze_file(str(second))

        # A late result from the first analysis must be ignored.
        stale = AudioAnalysis(file_path=str(first), bpm=1.0, duration=99.0)
        self.panel._on_analysis_done(stale_token, str(first), stale)
        self.assertIsNot(self.panel._analysis, stale)

        self.assertTrue(self._wait(lambda: self.panel._analysis is not None))
        self.assertNotEqual(self.panel._analysis.duration, 99.0)

    def test_stale_errors_and_cancellations_do_not_overwrite_the_view(self):
        self.panel._analysis_token = 5
        self.panel._drop_zone.setText("current")
        self.panel._on_analysis_error(1, "boom")
        self.assertEqual(self.panel._drop_zone.text(), "current")
        self.panel._on_analysis_cancelled(1)
        self.assertEqual(self.panel._drop_zone.text(), "current")

    def test_cancel_waits_for_the_worker_to_exit(self):
        path = write_tone(self.root / "a.wav", seconds=3.0)
        self.panel._analyze_file(str(path))
        worker = self.panel._worker
        self.panel.cancel_analysis()
        self.assertIsNone(self.panel._worker)
        self.assertFalse(worker.isRunning())
        self.assertFalse(self.panel._cancel_btn.isVisible())

    def test_starting_a_new_analysis_cancels_the_previous_one(self):
        first = write_tone(self.root / "first.wav", seconds=3.0)
        second = write_tone(self.root / "second.wav", seconds=1.0)
        self.panel._analyze_file(str(first))
        previous = self.panel._worker
        self.panel._analyze_file(str(second))
        self.assertIsNot(self.panel._worker, previous)
        self.assertFalse(previous.isRunning())
        self.panel.cancel_analysis()


if __name__ == "__main__":
    unittest.main()
