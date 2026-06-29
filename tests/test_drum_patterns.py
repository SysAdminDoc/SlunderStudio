import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from engines.midi_llm_engine import (
    DRUM_CLOSED_HAT,
    DRUM_SNARE,
    MidiGenParams,
    build_generation_prompt,
    generate_demo_midi,
    generate_drum_track,
    select_drum_groove,
)
from ui.midi_studio_view import MidiStudioView


class DrumPatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_auto_groove_selects_from_style_and_instruments(self):
        jazz = MidiGenParams(style="jazz swing trio", instruments=["Piano", "Drums"])
        dance = MidiGenParams(style="electronic house", instruments=["Synth", "Drums"])

        self.assertEqual("Swing Shuffle", select_drum_groove(jazz, rng=None).name)
        self.assertEqual("Four-on-the-Floor", select_drum_groove(dance, rng=None).name)

    def test_prompt_includes_requested_drum_groove(self):
        prompt = build_generation_prompt(
            MidiGenParams(
                prompt="tight halftime beat",
                instruments=["Piano"],
                drum_groove="Hip-Hop Half-Time",
            )
        )

        self.assertIn("Drum Groove: Hip-Hop Half-Time", prompt)
        self.assertIn("ghost notes", prompt)

    def test_drum_track_applies_swing_ghosts_and_velocity_humanize(self):
        params = MidiGenParams(
            tempo=120,
            duration_bars=4,
            instruments=["Drums"],
            drum_groove="Swing Shuffle",
            seed=9,
        )
        track = generate_drum_track(params)

        self.assertIsNotNone(track)
        self.assertTrue(track.is_drum)
        self.assertEqual(9, track.channel)

        closed_hats = [note for note in track.notes if note.pitch == DRUM_CLOSED_HAT]
        ghost_snares = [
            note for note in track.notes
            if note.pitch == DRUM_SNARE and note.velocity <= 46
        ]
        offbeat_hat = min(
            note for note in closed_hats
            if 0.27 < note.start < 0.35
        )

        self.assertGreater(offbeat_hat.start, 0.28)
        self.assertGreaterEqual(len(ghost_snares), 2)
        self.assertGreater(len({note.velocity for note in closed_hats}), 1)

    def test_demo_generator_adds_drums_only_when_requested(self):
        band = generate_demo_midi(
            MidiGenParams(
                instruments=["Piano", "Bass", "Drums"],
                drum_groove="Trap Hats",
                duration_bars=2,
                seed=4,
            )
        )
        piano = generate_demo_midi(
            MidiGenParams(instruments=["Piano"], drum_groove="Auto", duration_bars=2, seed=4)
        )

        drum_tracks = [track for track in band.tracks if track.is_drum]
        self.assertEqual(1, len(drum_tracks))
        self.assertTrue(drum_tracks[0].name.startswith("Drums - Trap Hats"))
        self.assertFalse(any(track.is_drum for track in piano.tracks))

    def test_midi_studio_params_include_selected_groove(self):
        view = MidiStudioView()
        try:
            view._groove_combo.setCurrentText("Straight Rock")
            params = view._build_params()
            self.assertEqual("Straight Rock", params.drum_groove)
        finally:
            view.deleteLater()


if __name__ == "__main__":
    unittest.main()
