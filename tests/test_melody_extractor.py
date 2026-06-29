import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.midi_utils import NoteData, load_midi
from core.provenance import sidecar_path_for
from core.settings import Settings
from core.voice_bank import VoiceBank
from engines.melody_extractor import (
    LyricMelodyParams,
    align_lyrics_to_notes,
    generate_lyric_melody,
    lyric_units_from_text,
    notes_from_pitch_frames,
)
from ui.vocal_suite_view import VocalSuiteView


class MelodyExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self):
        Settings._instance = None
        VoiceBank._instance = None

    def test_pitch_frames_merge_into_quantized_midi_notes(self):
        f0 = np.array([440.0, 440.0, np.nan, 493.88, 493.88, 493.88], dtype=np.float32)
        voiced = np.array([True, True, False, True, True, True])

        notes = notes_from_pitch_frames(
            f0,
            voiced,
            sr=100,
            hop_length=10,
            min_duration=0.15,
            max_merge_gap=0.02,
        )

        self.assertEqual([69, 71], [note.pitch for note in notes])
        self.assertGreaterEqual(notes[0].duration, 0.15)

    def test_lyrics_are_tokenized_and_aligned_to_notes(self):
        units = lyric_units_from_text("Hey, you / broken-hearted")
        notes = [
            NoteData(pitch=60, start=0.0, end=0.5),
            NoteData(pitch=62, start=0.5, end=1.0),
            NoteData(pitch=64, start=1.0, end=1.5),
        ]

        aligned = align_lyrics_to_notes(notes, units)

        self.assertEqual(["Hey", "you", "broken-hearted"], [entry["text"] for entry in aligned])
        self.assertEqual(62, aligned[1]["pitch"])

    def test_generate_lyric_melody_writes_midi_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sr = 22050
            t = np.arange(int(sr * 1.2)) / sr
            audio = (0.35 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            source = root / "hum.wav"
            output = root / "hum_melody.mid"
            sf.write(source, audio, sr, subtype="PCM_16")

            with self._patched_config(root):
                result = generate_lyric_melody(
                    LyricMelodyParams(
                        input_path=str(source),
                        lyrics="hello local singer",
                        tempo=96,
                        output_midi_path=str(output),
                        render_diffsinger=False,
                        fmin_note="A3",
                        fmax_note="A5",
                    )
                )

            midi = load_midi(str(output))
            self.assertEqual(str(output), result.midi_path)
            self.assertTrue(output.is_file())
            self.assertTrue(sidecar_path_for(output).is_file())
            self.assertGreater(result.notes_count, 0)
            self.assertEqual(96, result.tempo)
            self.assertGreater(midi.total_notes, 0)
            self.assertAlmostEqual(96, midi.tempo, delta=0.5)

    def test_vocal_suite_handoff_prepares_melody_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "hummed.wav"
            sf.write(source, np.zeros(2048, dtype=np.float32), 22050)

            with self._patched_config(root):
                view = VocalSuiteView()
                try:
                    view.set_audio(str(source))
                    self.assertEqual(str(source), view._melody_input_label.property("path"))
                    self.assertTrue(view._melody_generate_btn.isEnabled())
                    self.assertEqual(5, view._tabs.currentIndex())
                finally:
                    view.deleteLater()

    def _patched_config(self, root: Path):
        config_dir = root / "config"
        output_dir = root / "renders"
        model_dir = root / "models"
        trash_dir = root / "trash"
        for path in (config_dir, output_dir, model_dir, trash_dir):
            path.mkdir(parents=True, exist_ok=True)
        Settings._instance = None
        VoiceBank._instance = None
        stack = ExitStack()
        stack.enter_context(mock.patch("core.settings.get_config_dir", return_value=config_dir))
        stack.enter_context(mock.patch("core.settings.get_default_output_dir", return_value=output_dir))
        stack.enter_context(mock.patch("core.settings.get_default_cache_dir", return_value=model_dir))
        stack.enter_context(mock.patch("core.settings.get_trash_dir", return_value=trash_dir))
        stack.enter_context(mock.patch("core.voice_bank.get_config_dir", return_value=config_dir))
        return stack


if __name__ == "__main__":
    unittest.main()
