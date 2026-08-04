import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.audio_export import ExportSettings, export_audio, get_export_license_warnings
from core.model_manager import (
    BUILTIN_MODELS,
    COMMERCIAL_USE_LIMITED,
    COMMERCIAL_USE_NON_COMMERCIAL,
    ModelStatus,
)
from core.provenance import read_provenance_sidecar, write_provenance_sidecar
from ui.model_hub import ExecutableModelConsentDialog, ModelCard


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
            self.assertIn("Pinned", card._trust_label.text())
            self.assertIn("OMS signature: unsigned", card._trust_label.text())
        finally:
            card.deleteLater()

    def test_model_card_does_not_claim_pinning_for_package_managed_models(self):
        app = QApplication.instance() or QApplication([])
        card = ModelCard(BUILTIN_MODELS["diffsinger"])
        try:
            self.assertNotIn("Pinned", card._trust_label.text())
            self.assertIn("Package-managed", card._trust_label.text())
            self.assertIn("no pinned revision", card._trust_label.text())
        finally:
            card.deleteLater()

    def test_model_card_description_is_not_clipped_and_has_full_tooltip(self):
        app = QApplication.instance() or QApplication([])
        info = BUILTIN_MODELS["stable-audio-open"]
        card = ModelCard(info)
        try:
            description = card._description_label
            card.resize(320, 500)
            card.layout().activate()
            self.assertTrue(description.wordWrap())
            self.assertGreater(description.height(), 40)
            self.assertGreater(description.maximumHeight(), 40)
            self.assertEqual(description.toolTip(), info.description)
            self.assertEqual(description.accessibleDescription(), info.description)
        finally:
            card.deleteLater()

    def test_executable_model_consent_dialog_requires_acknowledgement(self):
        app = QApplication.instance() or QApplication([])
        info = BUILTIN_MODELS["musicgen-medium"]
        dialog = ExecutableModelConsentDialog(info)
        card = ModelCard(info)
        try:
            self.assertFalse(dialog._approve.isEnabled())
            dialog._ack.setChecked(True)
            self.assertTrue(dialog._approve.isEnabled())

            card.update_status(ModelStatus.DOWNLOADED)
            self.assertFalse(card._consent_btn.isHidden())
            self.assertIn(info.revision, dialog._details.text())
        finally:
            dialog.deleteLater()
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
            self.assertEqual("unsigned", data["model"]["signature_status"])

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

    def test_metadata_failure_is_indeterminate_and_export_still_warns(self):
        class FailingModelManager:
            def __init__(self):
                raise RuntimeError("registry unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            target = root / "export.wav"
            _write_wav(source)

            with mock.patch(
                "core.model_manager.ModelManager",
                FailingModelManager,
            ):
                write_provenance_sidecar(
                    source,
                    module="song_forge",
                    operation="generate",
                    model_id="unavailable-model",
                    export_format="wav",
                )

            source_data = read_provenance_sidecar(source)
            model = source_data["model"]
            self.assertEqual(model["metadata_status"], "indeterminate")
            self.assertEqual(model["license"], "unknown")
            self.assertEqual(model["commercial_use"], "unknown")
            self.assertTrue(model["requires_export_warning"])
            self.assertTrue(get_export_license_warnings(str(source)))

            export_audio(str(source), str(target), ExportSettings(format="wav"))
            exported = read_provenance_sidecar(target)
            self.assertTrue(exported["extra"]["license_warnings"])
            self.assertEqual(
                exported["extra"]["source_model_license"]["metadata_status"],
                "indeterminate",
            )


if __name__ == "__main__":
    unittest.main()
