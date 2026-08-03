import importlib.util
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent

from core.midi_utils import (
    CCEvent,
    MidiData,
    NoteData,
    TrackData,
    apply_swing_to_notes,
    humanize_note_velocities,
    load_midi,
    save_midi,
)
from ui.piano_roll import (
    KEY_WIDTH,
    NOTE_HEIGHT,
    PIXELS_PER_BEAT,
    TOTAL_KEYS,
    PianoRollScene,
    PianoRollWidget,
)


class PianoRollEditingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_swing_delays_offbeat_grid_notes(self):
        notes = [
            NoteData(pitch=60, start=0.0, end=0.2, velocity=90),
            NoteData(pitch=62, start=0.25, end=0.45, velocity=90),
        ]

        swung = apply_swing_to_notes(notes, grid=0.5, tempo=120.0, amount=0.5)

        self.assertEqual(0.0, swung[0].start)
        self.assertAlmostEqual(0.375, swung[1].start, places=6)
        self.assertAlmostEqual(0.575, swung[1].end, places=6)

    def test_velocity_humanize_is_seeded_and_clamped(self):
        notes = [
            NoteData(pitch=60, start=0.0, end=0.5, velocity=2),
            NoteData(pitch=62, start=0.5, end=1.0, velocity=126),
        ]

        first = humanize_note_velocities(notes, amount=12, seed=3)
        second = humanize_note_velocities(notes, amount=12, seed=3)

        self.assertEqual([note.velocity for note in first], [note.velocity for note in second])
        self.assertTrue(all(1 <= note.velocity <= 127 for note in first))

    @unittest.skipUnless(
        importlib.util.find_spec("pretty_midi") is not None,
        "optional pretty-midi dependency is not installed",
    )
    def test_save_and_load_preserves_control_changes(self):
        track = TrackData(
            name="Synth",
            program=80,
            channel=1,
            notes=[NoteData(pitch=64, start=0.0, end=1.0, velocity=90, channel=1)],
            cc_events=[
                CCEvent(controller=1, value=32, time=0.0, channel=1),
                CCEvent(controller=11, value=96, time=0.5, channel=1),
            ],
        )
        midi = MidiData(tracks=[track], tempo=120.0, duration=1.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cc.mid"
            save_midi(midi, str(path))
            loaded = load_midi(str(path))

        events = loaded.tracks[0].cc_events
        self.assertEqual([(1, 32), (11, 96)], [(event.controller, event.value) for event in events])

    def test_piano_roll_actions_update_notes_and_cc_lane(self):
        widget = PianoRollWidget()
        track = TrackData(
            name="Piano",
            program=0,
            channel=2,
            notes=[
                NoteData(pitch=60, start=0.03, end=0.47, velocity=80, channel=2),
                NoteData(pitch=62, start=0.25, end=0.7, velocity=80, channel=2),
            ],
        )
        try:
            widget.load_track(track, tempo=120.0, bars=1)
            widget._snap_combo.setCurrentText("1/8")
            widget._on_quantize()
            self.assertEqual([0.0, 0.25], [note.start for note in track.notes])

            widget._swing_spin.setValue(50)
            widget._on_apply_swing()
            self.assertAlmostEqual(0.375, track.notes[1].start, places=6)

            widget._humanize_spin.setValue(10)
            widget._on_humanize_velocity()
            self.assertEqual(2, len(track.notes))
            self.assertTrue(all(1 <= note.velocity <= 127 for note in track.notes))

            widget._automation_lane._controller_combo.setCurrentText("Expression (CC11)")
            widget._automation_lane._beat_spin.setValue(2.0)
            widget._automation_lane._value_spin.setValue(99)
            widget._automation_lane._on_add_event()

            self.assertEqual(1, len(track.cc_events))
            self.assertEqual(11, track.cc_events[0].controller)
            self.assertEqual(99, track.cc_events[0].value)
            self.assertEqual(2, track.cc_events[0].channel)
            self.assertAlmostEqual(1.0, track.cc_events[0].time, places=6)
        finally:
            widget.deleteLater()

    def test_scene_rect_includes_pitch_and_bar_label_gutters(self):
        scene = PianoRollScene()
        self.addCleanup(scene.deleteLater)

        expected_width = 16 * 4 * PIXELS_PER_BEAT + KEY_WIDTH
        expected_height = TOTAL_KEYS * NOTE_HEIGHT + 20
        rect = scene.sceneRect()

        self.assertEqual(-KEY_WIDTH, rect.left())
        self.assertEqual(-20, rect.top())
        self.assertEqual(expected_width, rect.width())
        self.assertEqual(expected_height, rect.height())

    def test_destructive_edits_restore_exact_note_and_cc_state(self):
        widget = PianoRollWidget()
        track = TrackData(
            name="Undoable Piano",
            channel=2,
            notes=[
                NoteData(pitch=60, start=0.03, end=0.47, velocity=80, channel=2),
                NoteData(pitch=62, start=0.25, end=0.7, velocity=110, channel=2),
            ],
            cc_events=[
                CCEvent(controller=1, value=32, time=0.0, channel=2),
                CCEvent(controller=11, value=96, time=0.5, channel=2),
            ],
        )

        def assert_undo_restores(operation):
            notes_before = deepcopy(track.notes)
            cc_before = deepcopy(track.cc_events)
            operation()
            self.assertTrue(widget.undo())
            self.assertEqual(notes_before, track.notes)
            self.assertEqual(cc_before, track.cc_events)
            self.assertFalse(widget._undo_stack)

        try:
            widget.load_track(track, tempo=120.0, bars=1)
            widget._snap_combo.setCurrentText("1/8")
            assert_undo_restores(widget._on_quantize)

            widget._swing_spin.setValue(50)
            assert_undo_restores(widget._on_apply_swing)

            widget._humanize_spin.setValue(32)
            assert_undo_restores(widget._on_humanize_velocity)

            widget._scene._note_items[0].setSelected(True)
            assert_undo_restores(widget._scene.delete_selected)

            widget._automation_lane._controller_combo.setCurrentText("Expression (CC11)")
            assert_undo_restores(widget._automation_lane._on_clear_lane)

            widget._on_quantize()
            event = QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Z,
                Qt.KeyboardModifier.ControlModifier,
            )
            widget._view.keyPressEvent(event)
            self.assertEqual(
                [NoteData(pitch=60, start=0.03, end=0.47, velocity=80, channel=2),
                 NoteData(pitch=62, start=0.25, end=0.7, velocity=110, channel=2)],
                track.notes,
            )
            self.assertEqual(
                [CCEvent(controller=1, value=32, time=0.0, channel=2),
                 CCEvent(controller=11, value=96, time=0.5, channel=2)],
                track.cc_events,
            )
            self.assertTrue(event.isAccepted())
        finally:
            widget.deleteLater()


if __name__ == "__main__":
    unittest.main()
