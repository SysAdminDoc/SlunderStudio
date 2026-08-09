import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.separator_registry import (
    COMMERCIAL_USE_UNKNOWN,
    LONG_FILE_THRESHOLD_SECONDS,
    SEPARATOR_CHECKPOINTS,
    checkpoint_id_for_demucs_model,
    get_separator_checkpoint,
    separator_artifact_policy,
)
from core.provenance import read_provenance_sidecar
from engines.audio_separator_engine import AudioSeparatorEngine
from engines.demucs_engine import restore_native_audio


class SeparatorRegistryTests(unittest.TestCase):
    def test_every_checkpoint_declares_run_capabilities_and_license(self):
        self.assertGreaterEqual(len(SEPARATOR_CHECKPOINTS), 3)
        for checkpoint in SEPARATOR_CHECKPOINTS.values():
            with self.subTest(checkpoint=checkpoint.id):
                self.assertTrue(checkpoint.stems)
                self.assertTrue(checkpoint.model_filename)
                self.assertTrue(checkpoint.checkpoint_license)
                self.assertTrue(checkpoint.chunking)
                self.assertTrue(checkpoint.quality)
                self.assertTrue(checkpoint.speed)
                self.assertGreater(checkpoint.vram_gb, 0)
                self.assertGreater(checkpoint.ram_gb, 0)

    def test_checkpoint_terms_stay_unknown_when_not_published(self):
        checkpoint = get_separator_checkpoint("audio-separator-bs-roformer")
        self.assertEqual(COMMERCIAL_USE_UNKNOWN, checkpoint.commercial_use)
        self.assertIn("verify", checkpoint.checkpoint_license.lower())

    def test_long_file_policy_has_a_stable_boundary_and_warning(self):
        checkpoint = get_separator_checkpoint("audio-separator-bs-roformer")

        boundary = separator_artifact_policy(
            checkpoint,
            LONG_FILE_THRESHOLD_SECONDS,
        )
        self.assertFalse(boundary["long_file"])
        self.assertEqual("native_rate_duration", boundary["policy"])
        self.assertEqual("", boundary["warning"])

        long_file = separator_artifact_policy(
            checkpoint,
            LONG_FILE_THRESHOLD_SECONDS + 0.01,
        )
        self.assertTrue(long_file["long_file"])
        self.assertFalse(long_file["crossfade"])
        self.assertEqual("warn_no_crossfade", long_file["policy"])
        self.assertEqual("long_file_no_crossfade", long_file["warning"])

    def test_legacy_demucs_names_resolve_to_declared_checkpoints(self):
        self.assertEqual("demucs-htdemucs", checkpoint_id_for_demucs_model("htdemucs"))
        self.assertEqual("demucs-htdemucs-6s", checkpoint_id_for_demucs_model("htdemucs_6s"))


class AudioSeparatorAdapterTests(unittest.TestCase):
    def test_adapter_preserves_outputs_and_checkpoint_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            audio = np.zeros((800, 2), dtype=np.float32)
            sf.write(input_path, audio, 8000, subtype="PCM_16")

            class FakeSeparator:
                def __init__(self):
                    self.output_dir = str(root / "outputs")

                def separate(self, _input):
                    output_dir = Path(self.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    vocals = output_dir / "input_(Vocals).wav"
                    instrumental = output_dir / "input_(Instrumental).wav"
                    rendered = np.ones((400, 2), dtype=np.float32) * 0.125
                    sf.write(vocals, rendered, 16000, subtype="PCM_16")
                    sf.write(instrumental, rendered, 16000, subtype="PCM_16")
                    return [str(vocals), str(instrumental)]

            engine = AudioSeparatorEngine()
            engine._output_dir = root / "outputs"
            engine._separator = FakeSeparator()
            engine._checkpoint = get_separator_checkpoint("audio-separator-bs-roformer")
            original_output_dir = engine._separator.output_dir

            result = engine.separate(str(input_path))

            self.assertTrue(result.is_success)
            self.assertEqual("audio-separator", result.backend_id)
            self.assertEqual("audio-separator-bs-roformer", result.checkpoint_id)
            self.assertEqual(original_output_dir, engine._separator.output_dir)
            self.assertEqual({"vocals", "instrumental"}, {stem.name for stem in result.stems})
            self.assertEqual(8000, result.sample_rate)
            self.assertEqual(8000, result.source_sample_rate)
            self.assertEqual(0.1, result.source_duration)
            self.assertEqual("preserve_input_rate_duration", result.artifact_policy["mode"])
            self.assertFalse(result.long_file_warning)
            self.assertTrue(all(stem.file_path for stem in result.stems))
            self.assertTrue(all(stem.provenance_path for stem in result.stems))
            for stem in result.stems:
                self.assertEqual((800, 2), stem.audio.shape)
                self.assertEqual(8000, stem.sample_rate)
                info = sf.info(stem.file_path)
                self.assertEqual(8000, info.samplerate)
                self.assertEqual(800, info.frames)
            provenance = read_provenance_sidecar(result.vocals.file_path)
            self.assertEqual("audio-separator-bs-roformer", provenance["model"]["id"])
            self.assertEqual(
                "unknown",
                provenance["parameters"]["checkpoint"]["commercial_use"],
            )
            self.assertEqual(
                "preserve_input_rate_duration",
                provenance["parameters"]["artifact_policy"]["mode"],
            )
            self.assertEqual(8000, provenance["parameters"]["source_sample_rate"])

    def test_restore_native_audio_resamples_and_pads_exactly(self):
        rendered = np.ones((3, 1), dtype=np.float32)

        restored = restore_native_audio(
            rendered,
            source_sample_rate=8000,
            source_frames=8,
            rendered_sample_rate=16000,
        )

        self.assertEqual((8, 1), restored.shape)
        self.assertEqual(np.float32, restored.dtype)
        self.assertTrue(np.all(np.isfinite(restored)))

    def test_load_model_uses_optional_dependency_without_installing_it(self):
        fake_module = types.ModuleType("audio_separator")
        fake_separator_module = types.ModuleType("audio_separator.separator")

        class FakeSeparator:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def load_model(self, model_filename):
                self.model_filename = model_filename

        fake_separator_module.Separator = FakeSeparator
        fake_module.separator = fake_separator_module
        with mock.patch.dict(
            sys.modules,
            {
                "audio_separator": fake_module,
                "audio_separator.separator": fake_separator_module,
            },
        ), mock.patch("engines.audio_separator_engine.ensure") as ensure, mock.patch(
            "core.model_manager.ModelManager"
        ) as manager:
            manager.return_value.is_offline = False
            engine = AudioSeparatorEngine()
            engine.load_model("audio-separator-bs-roformer")

        ensure.assert_called_once_with("audio_separator", pip_name="audio-separator")
        self.assertEqual("audio-separator-bs-roformer", engine.checkpoint_id)


if __name__ == "__main__":
    unittest.main()
