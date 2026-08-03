import unittest

from core.midi_utils import (
    CCEvent,
    MidiData,
    NoteData,
    TrackData,
    apply_swing_to_notes,
    get_pitch_range,
    get_program_name,
    get_time_range,
    humanize_note_velocities,
    quantize_notes,
    scale_velocity,
    transpose_notes,
)


class MidiUtilsTests(unittest.TestCase):
    def test_data_models_report_names_counts_and_duration(self):
        note = NoteData(pitch=60, start=1.0, end=1.5, velocity=100)
        track = TrackData(
            name="Piano",
            notes=[note, NoteData(pitch=64, start=0.0, end=2.0)],
            cc_events=[CCEvent(controller=64, value=127)],
        )
        midi = MidiData(tracks=[track], tempo=96)

        self.assertEqual(note.name, "C4")
        self.assertAlmostEqual(note.duration, 0.5)
        self.assertEqual(track.note_count, 2)
        self.assertEqual(track.duration, 2.0)
        self.assertEqual(midi.track_count, 1)
        self.assertEqual(midi.total_notes, 2)

    def test_program_names_cover_drums_and_unknown_programs(self):
        self.assertEqual(get_program_name(0), "Acoustic Grand Piano")
        self.assertEqual(get_program_name(0, is_drum=True), "Drums")
        self.assertEqual(get_program_name(127), "Program 127")

    def test_quantize_rounds_to_grid_and_keeps_minimum_duration(self):
        notes = [NoteData(pitch=60, start=0.12, end=0.13)]

        quantized = quantize_notes(notes, grid=0.5, tempo=120)

        self.assertEqual(quantized[0].start, 0.0)
        self.assertEqual(quantized[0].end, 0.25)
        self.assertEqual(notes[0].start, 0.12)

    def test_swing_is_deterministic_and_noop_for_zero_amount(self):
        notes = [
            NoteData(pitch=60, start=0.125, end=0.2),
            NoteData(pitch=62, start=0.25, end=0.35),
        ]
        swung = apply_swing_to_notes(notes, grid=0.25, tempo=120, amount=0.5)
        unchanged = apply_swing_to_notes(notes, grid=0.25, tempo=120, amount=0.0)

        self.assertGreater(swung[0].start, notes[0].start)
        self.assertEqual(swung[1].start, notes[1].start)
        self.assertEqual(unchanged, notes)

    def test_humanize_is_seeded_and_clamped(self):
        notes = [NoteData(pitch=60, velocity=2), NoteData(pitch=62, velocity=126)]

        first = humanize_note_velocities(notes, amount=12, seed=7)
        second = humanize_note_velocities(notes, amount=12, seed=7)

        self.assertEqual(first, second)
        self.assertTrue(all(1 <= note.velocity <= 127 for note in first))
        self.assertEqual(humanize_note_velocities(notes, amount=0), notes)

    def test_transpose_scale_and_ranges_clamp_without_mutating_source(self):
        notes = [NoteData(pitch=1, start=0.5, end=1.0, velocity=2), NoteData(pitch=126, start=0.0, end=2.0, velocity=126)]

        transposed = transpose_notes(notes, -5)
        scaled = scale_velocity(notes, 2.0)

        self.assertEqual([note.pitch for note in transposed], [0, 121])
        self.assertEqual([note.velocity for note in scaled], [4, 127])
        self.assertEqual(get_pitch_range(notes), (1, 126))
        self.assertEqual(get_pitch_range([]), (60, 72))
        self.assertEqual(get_time_range(notes), (0.0, 2.0))
        self.assertEqual(get_time_range([]), (0.0, 4.0))
        self.assertEqual(notes[0].pitch, 1)


if __name__ == "__main__":
    unittest.main()
