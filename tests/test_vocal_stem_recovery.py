import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.job_state import extract_output_paths
from core.provenance import read_provenance_sidecar
from engines.ace_step_engine import recover_song_vocal_stem
from engines.demucs_engine import (
    SeparationResult,
    StemResult,
    VocalStemRecoveryResult,
    recover_vocal_stem,
)
from ui.song_forge_view import SongForgeView


class VocalStemRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_recover_vocal_stem_writes_song_forge_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "song.wav"
            vocal = root / "vocals.wav"
            sr = 8000
            audio = np.zeros((sr // 2, 2), dtype=np.float32)
            sf.write(source, audio, sr, subtype="PCM_16")
            sf.write(vocal, audio, sr, subtype="PCM_16")

            def fake_separator(input_path, model_name="htdemucs", progress_callback=None):
                return SeparationResult(
                    stems=[
                        StemResult(name="vocals", file_path=str(vocal), sample_rate=sr),
                    ],
                    sample_rate=sr,
                    duration=0.5,
                    separation_time=0.25,
                    model_name=model_name,
                )

            result = recover_vocal_stem(
                str(source),
                model_name="fake-demucs",
                separation_fn=fake_separator,
            )

            provenance = read_provenance_sidecar(result.path)
            self.assertEqual(str(vocal), result.path)
            self.assertTrue(result.provenance_path)
            self.assertEqual("song_forge", provenance["module"])
            self.assertEqual("recover_vocal_stem", provenance["operation"])
            self.assertEqual([str(source)], provenance["source_paths"])
            self.assertEqual("vocals", provenance["parameters"]["stem_name"])

    def test_song_generation_recovery_metadata_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = root / "song.wav"
            vocal = root / "vocals.wav"
            sf.write(song, np.zeros((1024, 2), dtype=np.float32), 22050)
            sf.write(vocal, np.zeros((1024, 2), dtype=np.float32), 22050)

            with mock.patch(
                "engines.demucs_engine.recover_vocal_stem",
                return_value=VocalStemRecoveryResult(
                    path=str(vocal),
                    provenance_path=str(vocal) + ".provenance.json",
                    model_name="fake-demucs",
                ),
            ):
                payload = recover_song_vocal_stem(str(song))

            self.assertEqual(str(vocal), payload["vocal_stem_path"])
            self.assertEqual("", payload["vocal_stem_error"])

    def test_job_output_extraction_includes_recovered_vocal_stem(self):
        payload = {
            "audio_path": "song.wav",
            "vocal_stem_path": "vocals.wav",
            "vocal_stem_provenance_path": "vocals.wav.provenance.json",
        }

        paths = extract_output_paths(payload)

        self.assertIn("song.wav", paths)
        self.assertIn("vocals.wav", paths)
        self.assertIn("vocals.wav.provenance.json", paths)

    def test_song_forge_routes_recovered_vocal_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = root / "song.wav"
            vocal = root / "vocals.wav"
            sf.write(song, np.zeros((1024, 2), dtype=np.float32), 22050)
            sf.write(vocal, np.zeros((1024, 2), dtype=np.float32), 22050)

            view = SongForgeView()
            emitted = []
            try:
                view.send_to_vocals.connect(emitted.append)
                view._load_output(str(song), seed=123, vocal_stem_path=str(vocal))
                self.assertTrue(view._to_vocal_stem_btn.isEnabled())
                self.assertIn("vocals.wav", view._output_info.text())

                view._on_send_vocal_stem()

                self.assertEqual([str(vocal)], emitted)
            finally:
                view.deleteLater()


if __name__ == "__main__":
    unittest.main()
