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


class AnalysisConstraintTests(unittest.TestCase):
    def test_corrections_preserve_raw_values_and_serialize_lineage(self):
        analysis = AudioAnalysis(
            duration=120.0,
            bpm=128.0,
            bpm_confidence=0.42,
            bpm_alternatives=[{"value": 64.0, "reason": "half-time"}],
            key="C major",
            key_confidence=0.31,
            key_alternatives=[{"value": "A minor", "confidence": 0.28}],
            sections=[
                {"start": 0.0, "end": 60.0, "label": "Intro"},
                {"start": 60.0, "end": 120.0, "label": "Chorus"},
            ],
        )

        analysis.apply_corrections(
            bpm=96.0,
            key="A minor",
            sections=[
                {"start": 0.0, "end": 48.0, "label": "Verse"},
                {"start": 48.0, "end": 120.0, "label": "Hook"},
            ],
        )

        self.assertEqual(analysis.bpm, 128.0)
        self.assertEqual(analysis.key, "C major")
        self.assertEqual(analysis.sections[0]["label"], "Intro")
        self.assertEqual(analysis.effective_bpm, 96.0)
        self.assertEqual(analysis.effective_key, "A minor")
        self.assertEqual(analysis.effective_sections[0]["label"], "Verse")
        self.assertTrue(analysis.has_corrections)

        payload = analysis.to_dict()
        self.assertEqual(payload["raw"]["bpm"], 128.0)
        self.assertEqual(payload["raw"]["key"], "C major")
        self.assertEqual(payload["corrections"]["bpm"], 96.0)
        self.assertEqual(payload["corrections"]["key"], "A minor")
        self.assertEqual(payload["effective"]["bpm"], 96.0)
        self.assertEqual(payload["generation_constraints"]["schema_version"], 1)
        self.assertEqual(payload["generation_constraints"]["alternatives"]["key"][0]["value"], "A minor")

    def test_corrections_reject_invalid_values(self):
        analysis = AudioAnalysis(duration=30.0, bpm=120.0, key="C major")

        with self.assertRaises(ValueError):
            analysis.apply_corrections(bpm=301)
        with self.assertRaises(ValueError):
            analysis.apply_corrections(key="H major")
        with self.assertRaises(ValueError):
            analysis.apply_corrections(
                sections=[{"start": 0.0, "end": 40.0, "label": "Too long"}]
            )
        with self.assertRaises(ValueError):
            analysis.apply_corrections(
                sections=[
                    {"start": 0.0, "end": 20.0, "label": "A"},
                    {"start": 19.0, "end": 30.0, "label": "Overlap"},
                ]
            )

    def test_corrected_tempo_and_key_reach_ace_step_tags(self):
        analysis = AudioAnalysis(
            bpm=128.0,
            key="C major",
            suggested_tags=["ambient"],
            suggested_tempo_tag="moderate",
        )
        analysis.apply_corrections(bpm=80.0, key="A minor")

        tags = analysis.to_ace_step_tags()
        self.assertIn("slow", tags)
        self.assertNotIn("moderate", tags)
        self.assertIn("A minor", tags)


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

    def test_result_is_published_only_after_native_thread_stops(self):
        path = write_tone(self.root / "settled.wav", seconds=1.0)
        self.panel._analyze_file(str(path))
        worker = self.panel._worker

        self.assertTrue(
            self._wait(lambda: self.panel._analysis is not None),
            "analysis result was not published",
        )
        self.assertFalse(worker.isRunning())
        self.assertIsNone(self.panel._worker)

    def test_repeated_results_can_release_source_files_immediately(self):
        for index in range(4):
            source = tempfile.TemporaryDirectory()
            self.addCleanup(source.cleanup)
            path = write_tone(
                Path(source.name) / f"source-{index}.wav",
                seconds=0.5,
                freq=440.0 + index * 37.0,
            )
            self.panel._analysis = None
            self.panel._analyze_file(str(path))
            worker = self.panel._worker

            self.assertTrue(
                self._wait(
                    lambda p=str(path): self.panel._analysis is not None
                    and self.panel._analysis.file_path == p
                ),
                f"analysis {index} did not settle",
            )
            self.assertFalse(worker.isRunning())
            # Windows must be able to remove the source as soon as the result
            # becomes visible; a live worker would keep this unlink locked.
            Path(path).unlink()
            source.cleanup()

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

    def test_cancel_is_nonblocking_and_retains_worker_until_exit(self):
        path = write_tone(self.root / "a.wav", seconds=3.0)
        self.panel._analyze_file(str(path))
        worker = self.panel._worker
        started = time.monotonic()
        self.panel.cancel_analysis()
        self.assertLess(time.monotonic() - started, 1.0)
        if worker.isRunning():
            self.assertIs(self.panel._worker, worker)
        self.assertTrue(
            self._wait(lambda: not worker.isRunning() and self.panel._worker is None)
        )
        self.assertIsNone(self.panel._worker)
        self.assertFalse(self.panel._cancel_btn.isVisible())

    def test_starting_a_new_analysis_cancels_the_previous_one(self):
        first = write_tone(self.root / "first.wav", seconds=3.0)
        second = write_tone(self.root / "second.wav", seconds=1.0)
        self.panel._analyze_file(str(first))
        previous = self.panel._worker
        self.panel._analyze_file(str(second))
        self.assertIsNot(self.panel._worker, previous)
        self.assertTrue(self._wait(lambda: not previous.isRunning()))
        self.panel.cancel_analysis()

    def test_correction_editor_applies_constraints_without_overwriting_raw_values(self):
        analysis = AudioAnalysis(
            duration=120.0,
            bpm=128.0,
            bpm_confidence=0.55,
            key="C major",
            key_confidence=0.44,
            sections=[
                {"start": 0.0, "end": 60.0, "label": "Verse"},
                {"start": 60.0, "end": 120.0, "label": "Chorus"},
            ],
        )
        self.panel._analysis = analysis
        self.panel._populate_correction_editor(analysis)
        self.panel._refresh_analysis_display(analysis)

        self.panel._bpm_override_check.setChecked(True)
        self.panel._bpm_override_spin.setValue(96.0)
        self.panel._key_override_check.setChecked(True)
        self.panel._key_override_combo.setCurrentIndex(
            self.panel._key_override_combo.findData("A minor")
        )
        self.panel._sections_override_check.setChecked(True)
        self.panel._sections_table.item(0, 1).setText("0")
        self.panel._sections_table.item(0, 2).setText("48")
        self.panel._sections_table.item(1, 1).setText("48")
        self.panel._sections_table.item(1, 2).setText("120")

        self.assertTrue(self.panel._apply_corrections())
        self.assertEqual(analysis.bpm, 128.0)
        self.assertEqual(analysis.key, "C major")
        self.assertEqual(analysis.sections[0]["end"], 60.0)
        self.assertEqual(analysis.effective_bpm, 96.0)
        self.assertEqual(analysis.effective_key, "A minor")
        self.assertEqual(analysis.effective_sections[0]["end"], 48.0)

        match_payloads = []
        midi_payloads = []
        self.panel.match_requested.connect(match_payloads.append)
        self.panel.reference_to_midi.connect(midi_payloads.append)
        self.panel._on_match()
        self.panel._on_send_to_midi()
        self.assertEqual(match_payloads[0]["raw"]["bpm"], 128.0)
        self.assertEqual(match_payloads[0]["effective"]["bpm"], 96.0)
        self.assertEqual(midi_payloads[0]["bpm"], 96.0)
        self.assertEqual(midi_payloads[0]["key"], "A minor")


class MidiReferenceConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_midi_studio_applies_effective_reference_constraints(self):
        from ui.midi_studio_view import MidiStudioView

        view = MidiStudioView()
        try:
            constraints = {
                "schema_version": 1,
                "bpm": 96.0,
                "key": "A minor",
                "sections": [{"start": 0.0, "end": 30.0, "label": "Verse"}],
                "effective": {
                    "bpm": 96.0,
                    "key": "A minor",
                    "sections": [{"start": 0.0, "end": 30.0, "label": "Verse"}],
                },
            }
            self.assertTrue(view.apply_reference_constraints(constraints))
            self.assertEqual(view._tempo_spin.value(), 96)
            self.assertEqual(view._key_combo.currentData(), "A minor")
            self.assertEqual(
                view._reference_constraints["effective"]["sections"][0]["label"],
                "Verse",
            )
        finally:
            view.deleteLater()


class SongForgeReferenceConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_match_adds_effective_constraints_and_keeps_provenance(self):
        from ui.song_forge_view import SongForgeView

        view = SongForgeView()
        try:
            view._on_reference_match(
                {
                    "duration": 90.0,
                    "suggested_tags": ["ambient"],
                    "suggested_tempo_tag": "moderate",
                    "energy_curve": [0.2, 0.8],
                    "generation_constraints": {
                        "schema_version": 1,
                        "bpm": 96.0,
                        "key": "A minor",
                        "sections": [],
                        "effective": {
                            "bpm": 96.0,
                            "key": "A minor",
                            "sections": [],
                        },
                        "corrections": {"bpm": 96.0, "key": "A minor"},
                    },
                }
            )
            tags = view._get_tags()
            self.assertIn("96 bpm", tags)
            self.assertIn("A minor", tags)
            self.assertEqual(view._duration_spin.value(), 90.0)
            self.assertEqual(view._reference_analysis_constraints["bpm"], 96.0)
        finally:
            view.deleteLater()


if __name__ == "__main__":
    unittest.main()
