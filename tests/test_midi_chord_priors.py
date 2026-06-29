import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from engines.midi_llm_engine import (
    MidiGenParams,
    build_generation_prompt,
    generate_demo_midi,
    parse_chord_progression,
)
from ui.midi_studio_view import MidiStudioView


class MidiChordPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_roman_progression_parser_handles_common_priors(self):
        self.assertEqual([0, 4, 5, 3], parse_chord_progression("I-V-vi-IV"))
        self.assertEqual([1, 4, 0], parse_chord_progression("ii-V-I"))
        self.assertEqual([0, 5, 2, 6], parse_chord_progression("i-VI-III-VII", is_minor=True))

    def test_prompt_includes_requested_chord_progression(self):
        prompt = build_generation_prompt(
            MidiGenParams(
                prompt="upbeat pop cue",
                key="C major",
                duration_bars=4,
                chord_progression="I-V-vi-IV",
            )
        )

        self.assertIn("Chord Progression: I-V-vi-IV", prompt)
        self.assertIn("one chord per bar", prompt)

    def test_demo_generator_uses_progression_roots_and_diatonic_vi(self):
        midi = generate_demo_midi(
            MidiGenParams(
                key="C major",
                tempo=120,
                duration_bars=4,
                chord_progression="I-V-vi-IV",
                seed=7,
            )
        )

        piano = midi.tracks[0]
        bar_starts = sorted({round(note.start, 3) for note in piano.notes})
        roots = [
            min(note.pitch for note in piano.notes if round(note.start, 3) == start)
            for start in bar_starts
        ]
        vi_bar_notes = sorted(
            note.pitch for note in piano.notes if round(note.start, 3) == bar_starts[2]
        )

        self.assertEqual([48, 55, 57, 53], roots)
        self.assertEqual([57, 60, 64], vi_bar_notes)

    def test_midi_studio_params_include_selected_progression(self):
        view = MidiStudioView()
        try:
            view._progression_combo.setCurrentText("ii-V-I")
            params = view._build_params()
            self.assertEqual("ii-V-I", params.chord_progression)
        finally:
            view.deleteLater()


if __name__ == "__main__":
    unittest.main()
