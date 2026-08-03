import unittest
import threading

import numpy as np

from core.midi_utils import MidiData, NoteData, TrackData
from engines.fluidsynth_engine import (
    FluidSynthEngine,
    MidiRenderCancelled,
    RenderSettings,
    render_midi_simple,
)


class _StickyTailSynth:
    """Small fake that retains signal until the engine explicitly resets it."""

    def __init__(self):
        self.active = False
        self.system_reset_calls = 0

    def system_reset(self):
        self.system_reset_calls += 1
        self.active = False

    def program_select(self, *_args):
        pass

    def noteon(self, *_args):
        self.active = True

    def noteoff(self, *_args):
        pass

    def cc(self, *_args):
        pass

    def get_samples(self, frame_count):
        value = 1200 if self.active else 0
        return np.full(frame_count * 2, value, dtype=np.int16).tobytes()


class _RecordingSynth:
    def __init__(self):
        self.calls = []

    def system_reset(self):
        self.calls.append(("reset",))

    def program_select(self, channel, *_args):
        self.calls.append(("program", channel))

    def noteon(self, channel, pitch, velocity):
        self.calls.append(("on", channel, pitch, velocity))

    def noteoff(self, channel, pitch):
        self.calls.append(("off", channel, pitch))

    def cc(self, channel, controller, value):
        self.calls.append(("cc", channel, controller, value))

    def get_samples(self, frame_count):
        return np.zeros(frame_count * 2, dtype=np.int16).tobytes()


def _midi_with_note():
    return MidiData(
        tracks=[TrackData(notes=[NoteData(start=0.0, end=0.25)])],
        duration=0.25,
    )


def _midi_with_two_tracks():
    return MidiData(
        tracks=[
            TrackData(
                name="Left",
                channel=0,
                notes=[NoteData(pitch=60, start=0.0, end=0.25)],
            ),
            TrackData(
                name="Right",
                channel=1,
                notes=[NoteData(pitch=64, start=0.0, end=0.25)],
            ),
        ],
        duration=0.25,
    )


class FluidSynthEngineTests(unittest.TestCase):
    def _engine(self, channels=2):
        engine = FluidSynthEngine()
        engine._synth = _StickyTailSynth()
        engine._soundfont_id = 1
        engine._settings = RenderSettings(sample_rate=8, channels=channels)
        return engine

    def test_render_resets_shared_synth_before_next_render(self):
        engine = self._engine()
        first = engine.render_to_numpy(_midi_with_note())
        second = engine.render_to_numpy(MidiData(duration=0.25))

        self.assertGreater(np.max(np.abs(first)), 0.0)
        self.assertTrue(np.array_equal(second, np.zeros_like(second)))
        self.assertGreaterEqual(engine._synth.system_reset_calls, 4)

    def test_mono_render_downmixes_fluidsynth_stereo_output(self):
        engine = self._engine(channels=1)

        audio = engine.render_to_numpy(_midi_with_note())

        self.assertEqual(audio.shape[1], 1)
        self.assertTrue(np.allclose(audio[:, 0], 1200 / 32768.0))

    def test_render_honors_multiple_solos_mutes_and_track_mix(self):
        engine = FluidSynthEngine()
        synth = _RecordingSynth()
        engine._synth = synth
        engine._soundfont_id = 1
        engine._settings = RenderSettings(sample_rate=8)

        engine.render_to_numpy(
            _midi_with_two_tracks(),
            mute_tracks={1},
            solo_tracks={0, 1},
            track_mix={
                0: {"volume": 0.5, "pan": -1.0},
                1: {"volume": 0.25, "pan": 1.0},
            },
        )

        note_on_channels = {
            call[1] for call in synth.calls if call[0] == "on"
        }
        self.assertEqual({0}, note_on_channels)
        controls = {
            (call[1], call[2]): call[3]
            for call in synth.calls
            if call[0] == "cc" and call[2] in {7, 10}
        }
        self.assertEqual(64, controls[(0, 7)])
        self.assertEqual(0, controls[(0, 10)])

        synth.calls.clear()
        engine.render_to_numpy(
            _midi_with_two_tracks(),
            solo_tracks={0, 1},
        )
        self.assertEqual({0, 1}, {
            call[1] for call in synth.calls if call[0] == "on"
        })

    def test_simple_renderer_honors_solo_and_pan(self):
        audio = render_midi_simple(
            _midi_with_two_tracks(),
            sample_rate=1000,
            solo_tracks={0},
            track_mix={0: {"pan": -1.0}},
        )

        self.assertGreater(np.max(np.abs(audio[:, 0])), 0.0)
        self.assertTrue(np.allclose(audio[:, 1], 0.0))

    def test_simple_renderer_fails_fast_when_cancelled(self):
        cancel_event = threading.Event()
        cancel_event.set()
        with self.assertRaises(MidiRenderCancelled):
            render_midi_simple(_midi_with_note(), cancel_event=cancel_event)


if __name__ == "__main__":
    unittest.main()
