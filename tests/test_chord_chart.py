import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.chord_chart import (
    detect_chord_segments,
    format_chordpro,
    format_crd,
    save_chord_chart,
)
from core.midi_utils import MidiData, NoteData, TrackData
from ui.midi_studio_view import MidiStudioView


class ChordChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_detects_bar_level_chords_from_midi_notes(self):
        midi = self._four_chord_midi()
        names = [segment.name for segment in detect_chord_segments(midi)]

        self.assertEqual(["C", "G", "Am", "F"], names)

    def test_chordpro_export_merges_sectioned_lyrics(self):
        text = format_chordpro(
            self._four_chord_midi(),
            title="Demo Song",
            lyrics="[Verse]\nLine one\nLine two",
        )

        self.assertIn("{title: Demo Song}", text)
        self.assertIn("{tempo: 120}", text)
        self.assertIn("{comment: Verse}", text)
        self.assertIn("[C]Line one", text)
        self.assertIn("[G]Line two", text)
        self.assertIn("[Am] [F]", text)

    def test_crd_export_places_chords_above_lyrics(self):
        text = format_crd(
            self._four_chord_midi(),
            title="Demo Song",
            lyrics="[Chorus]\nHook line",
        )

        self.assertIn("Title: Demo Song", text)
        self.assertIn("[Chorus]", text)
        self.assertIn("C\nHook line", text)
        self.assertIn("G | Am | F", text)

    def test_save_chord_chart_writes_selected_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sheet.crd"
            output = save_chord_chart(
                self._four_chord_midi(),
                str(target),
                title="Sheet",
                lyrics="First line",
            )

            self.assertEqual(str(target), output)
            self.assertIn("Title: Sheet", target.read_text(encoding="utf-8"))

    def test_midi_studio_exports_chart_with_pasted_lyrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "midi_chart"
            view = MidiStudioView()
            try:
                view.set_midi_data(self._four_chord_midi())
                view._chart_lyrics.setPlainText("[Verse]\nLine one")
                with mock.patch(
                    "ui.midi_studio_view.QFileDialog.getSaveFileName",
                    return_value=(str(target), "Chord sheet (*.crd)"),
                ):
                    view._on_export_chart()

                saved = target.with_suffix(".crd")
                self.assertTrue(saved.is_file())
                self.assertIn("Line one", saved.read_text(encoding="utf-8"))
                self.assertIn("Exported chart:", view._status.text())
            finally:
                view.deleteLater()

    def _four_chord_midi(self) -> MidiData:
        bar_dur = 2.0
        chords = [
            [60, 64, 67],
            [55, 59, 62],
            [57, 60, 64],
            [53, 57, 60],
        ]
        track = TrackData(name="Piano", program=0, channel=0)
        for bar, pitches in enumerate(chords):
            start = bar * bar_dur
            for pitch in pitches:
                track.notes.append(
                    NoteData(pitch=pitch, start=start, end=start + bar_dur, velocity=96)
                )
        return MidiData(
            tracks=[track],
            tempo=120.0,
            time_signature=(4, 4),
            duration=bar_dur * len(chords),
        )


if __name__ == "__main__":
    unittest.main()
