import unittest

import numpy as np

from core.midi_utils import MidiData, NoteData, TrackData
from engines.fluidsynth_engine import FluidSynthEngine, RenderSettings


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


def _midi_with_note():
    return MidiData(
        tracks=[TrackData(notes=[NoteData(start=0.0, end=0.25)])],
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


if __name__ == "__main__":
    unittest.main()
