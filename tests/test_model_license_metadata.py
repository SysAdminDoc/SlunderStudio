import os
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.audio_export import ExportSettings, export_audio, get_export_license_warnings
from core.model_manager import (
    BUILTIN_MODELS,
    COMMERCIAL_USE_LIMITED,
    COMMERCIAL_USE_NON_COMMERCIAL,
)
from core.provenance import read_provenance_sidecar, write_provenance_sidecar
from ui.model_hub import ModelCard


def _write_wav(path: Path, duration: float = 0.1, sample_rate: int = 24000):
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 220.0 * t)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class ModelLicenseMetadataTests(unittest.TestCase):
    def test_builtin_registry_marks_restricted_models(self):
        stable = BUILTIN_MODELS["stable-audio-open"]
        musicgen = BUILTIN_MODELS["musicgen-medium"]

        self.assertEqual(stable.commercial_use, COMMERCIAL_USE_LIMITED)
        self.assertTrue(stable.gated)
        self.assertTrue(stable.requires_export_warning)
        self.assertIn("Commercial use is limited", stable.license_warning)
        self.assertEqual(musicgen.commercial_use, COMMERCIAL_USE_NON_COMMERCIAL)
        self.assertIn("not be cleared for commercial use", musicgen.license_warning)

    def test_model_card_displays_license_access_and_commercial_status(self):
        app = QApplication.instance() or QApplication([])
        card = ModelCard(BUILTIN_MODELS["stable-audio-open"])
        try:
            self.assertIn("License: Stability Community", card._rights_label.text())
            self.assertIn("Commercial: Limited", card._rights_label.text())
            self.assertIn("Gated / token required", card._rights_label.text())
            self.assertIsNotNone(card._license_warning)
            self.assertIn("Commercial use is limited", card._license_warning.text())
        finally:
            card.deleteLater()

    def test_provenance_sidecar_contains_model_license_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "musicgen.wav"
            artifact.write_bytes(b"audio")

            write_provenance_sidecar(
                artifact,
                module="song_forge",
                operation="generate",
                model_id="musicgen-medium",
                export_format="wav",
            )
            data = read_provenance_sidecar(artifact)

            self.assertEqual(data["model"]["license"], "CC-BY-NC")
            self.assertEqual(data["model"]["commercial_use"], COMMERCIAL_USE_NON_COMMERCIAL)
            self.assertTrue(data["model"]["requires_export_warning"])
            self.assertIn("commercial use", data["model"]["license_warning"])

    def test_export_carries_source_model_license_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            target = Path(tmp) / "export.wav"
            _write_wav(source)
            write_provenance_sidecar(
                source,
                module="song_forge",
                operation="generate",
                model_id="musicgen-medium",
                export_format="wav",
            )

            warnings = get_export_license_warnings(str(source))
            output = export_audio(str(source), str(target), ExportSettings(format="wav"))
            data = read_provenance_sidecar(output)

            self.assertTrue(warnings)
            self.assertEqual(data["extra"]["license_warnings"], warnings)
            self.assertEqual(
                data["extra"]["source_model_license"]["commercial_use"],
                COMMERCIAL_USE_NON_COMMERCIAL,
            )


if __name__ == "__main__":
    unittest.main()
