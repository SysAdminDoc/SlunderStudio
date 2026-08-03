import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from core.midi_utils import MidiData, NoteData, TrackData
from core.workers import CancelledJobError
from engines.fluidsynth_engine import MidiRenderCancelled, MidiRenderResult
from engines.midi_llm_engine import MidiGenParams, MidiGenResult, generate_midi
from ui.midi_studio_view import MidiStudioView


def _midi_fixture() -> MidiData:
    return MidiData(
        tracks=[
            TrackData(
                name="Piano",
                channel=0,
                notes=[NoteData(pitch=60, start=0.0, end=0.25)],
            ),
            TrackData(
                name="Bass",
                channel=1,
                notes=[NoteData(pitch=36, start=0.0, end=0.25)],
            ),
        ],
        duration=0.25,
    )


class MidiRenderJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        self.app.processEvents()
        return bool(predicate())

    def test_render_worker_passes_complete_mixer_snapshot(self):
        view = MidiStudioView()
        try:
            view.set_midi_data(_midi_fixture())
            view._mixer._on_mute(1, True)
            view._mixer._on_solo(0, True)
            view._mixer._on_solo(1, True)
            view._mixer._strips[0]._volume_slider.setValue(64)
            view._mixer._strips[0]._pan_slider.setValue(-32)

            captured = {}
            audio = np.zeros((64, 2), dtype=np.float32)

            def render(_midi, **kwargs):
                captured.update(kwargs)
                return MidiRenderResult(audio=audio)

            with tempfile.TemporaryDirectory() as tmp, patch(
                "core.settings.get_config_dir",
                return_value=tmp,
            ), patch(
                "engines.fluidsynth_engine.render_midi_to_audio",
                side_effect=render,
            ):
                view._on_render()
                self.assertTrue(self._wait_for(lambda: view._render_worker is None))

            self.assertEqual({1}, captured["mute_tracks"])
            self.assertEqual({0, 1}, captured["solo_tracks"])
            self.assertAlmostEqual(64 / 127.0, captured["track_mix"][0]["volume"])
            self.assertAlmostEqual(-32 / 64.0, captured["track_mix"][0]["pan"])
            self.assertTrue(view._waveform.has_audio)
            self.assertIn("Rendered:", view._status.text())
            self.assertTrue(view._contract_result.is_success)
        finally:
            view.close()
            self.app.processEvents()

    def test_render_worker_cancellation_is_reported_as_cancelled(self):
        view = MidiStudioView()
        started = threading.Event()
        try:
            view.set_midi_data(_midi_fixture())

            def slow_render(_midi, *, cancel_event=None, **_kwargs):
                started.set()
                while cancel_event is not None and not cancel_event.is_set():
                    time.sleep(0.005)
                raise MidiRenderCancelled("stop")

            with tempfile.TemporaryDirectory() as tmp, patch(
                "core.settings.get_config_dir",
                return_value=tmp,
            ), patch(
                "engines.fluidsynth_engine.render_midi_to_audio",
                side_effect=slow_render,
            ):
                view._on_render()
                self.assertTrue(started.wait(2.0))
                view._on_render()
                self.assertTrue(self._wait_for(lambda: view._render_worker is None))

            self.assertTrue(view._contract_result.is_cancelled)
            self.assertIn("cancelled", view._status.text().lower())
            self.assertTrue(view._render_btn.isEnabled())
        finally:
            view.close()
            self.app.processEvents()

    def test_render_task_rejects_pre_cancelled_job(self):
        view = MidiStudioView()
        try:
            event = threading.Event()
            event.set()
            with self.assertRaises(CancelledJobError):
                view._run_render(
                    _midi_fixture(),
                    str(tempfile.gettempdir()) + "\\slunder-midi-test.wav",
                    set(),
                    set(),
                    {},
                    cancel_event=event,
                )
        finally:
            view.close()
            self.app.processEvents()

    def test_selected_loaded_midi_model_receives_worker_cancellation_contract(self):
        calls = {}
        midi = _midi_fixture()

        class LoadedEngine:
            is_loaded = True
            model_id = "midi-llm-1b"

            def generate(self, params, progress_callback=None, cancel_event=None):
                calls["params"] = params
                calls["progress_callback"] = progress_callback
                calls["cancel_event"] = cancel_event
                return MidiGenResult(midi_data=midi)

        cancel_event = threading.Event()
        with patch("engines.midi_llm_engine.get_engine", return_value=LoadedEngine()):
            result = generate_midi(
                MidiGenParams(prompt="a bright piano motif"),
                progress_callback=lambda *_args: None,
                model_id="midi-llm-1b",
                cancel_event=cancel_event,
            )

        self.assertTrue(result.is_success)
        self.assertIs(calls["cancel_event"], cancel_event)
        self.assertIsNotNone(calls["progress_callback"])

    def test_demo_generation_reports_pre_cancelled_request(self):
        cancel_event = threading.Event()
        cancel_event.set()
        class UnloadedEngine:
            is_loaded = False
            model_id = ""

        with patch("engines.midi_llm_engine.get_engine", return_value=UnloadedEngine()):
            result = generate_midi(
                MidiGenParams(prompt="cancel me", allow_demo_output=True),
                cancel_event=cancel_event,
            )

        self.assertTrue(result.is_cancelled)
        self.assertFalse(result.can_route)


if __name__ == "__main__":
    unittest.main()
