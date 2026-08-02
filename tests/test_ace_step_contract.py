import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
try:
    import torch
except ImportError:  # Optional ACE-Step engine profile is not in the core lock.
    torch = None

from core.ace_step_contract import (
    ACE_STEP_ADAPTER,
    ACE_STEP_CAPABILITIES,
    ACE_STEP_IGNORE_PATTERNS,
    ACE_STEP_MODEL_ID,
    ACE_STEP_REVISION,
    ACE_STEP_SAMPLE_RATE,
    ACE_STEP_SOURCE,
)
from core.model_manager import BUILTIN_MODELS, ModelManager
from core.trash import TrashManager
from engines.ace_step_engine import (
    ACEStepEngine,
    GenerationParams,
    validate_ace_step_runtime,
)


class _PipelineOutput:
    def __init__(self):
        self.audios = torch.zeros((1, 2, 480), dtype=torch.float32)


class _RecordingPipeline:
    sample_rate = ACE_STEP_SAMPLE_RATE

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _PipelineOutput()


@unittest.skipUnless(torch is not None, "optional torch dependency is not installed")
class AceStepContractTests(unittest.TestCase):
    def _source_audio(self, root: Path, seconds: float = 10.0) -> Path:
        path = root / "source.wav"
        samples = int(seconds * ACE_STEP_SAMPLE_RATE)
        audio = np.full((samples, 2), 0.125, dtype=np.float32)
        sf.write(path, audio, ACE_STEP_SAMPLE_RATE, subtype="FLOAT")
        return path

    def _engine(self, root: Path) -> tuple[ACEStepEngine, _RecordingPipeline]:
        pipeline = _RecordingPipeline()
        engine = ACEStepEngine()
        engine._pipeline = pipeline
        engine._model_loaded = True
        engine._output_dir = root
        return engine, pipeline

    def test_registry_matches_the_immutable_diffusers_contract(self):
        info = BUILTIN_MODELS[ACE_STEP_MODEL_ID]
        self.assertEqual(info.source, ACE_STEP_SOURCE)
        self.assertEqual(info.revision, ACE_STEP_REVISION)
        self.assertEqual(info.license, "MIT")
        self.assertEqual(info.tags, list(ACE_STEP_CAPABILITIES))
        self.assertEqual(info.ignore_patterns, list(ACE_STEP_IGNORE_PATTERNS))
        self.assertFalse(info.requires_remote_code)
        self.assertFalse(info.allows_unsafe_weights)
        self.assertIn("silence_latent.pt", info.ignore_patterns)
        self.assertEqual(ACE_STEP_ADAPTER, "diffusers.AceStepPipeline")

    def test_runtime_rejects_transformers_outside_ace_step_range(self):
        compatible = {
            "torch": "2.11.0",
            "transformers": "4.57.6",
            "diffusers": "0.39.0",
            "accelerate": "1.14.0",
        }
        with mock.patch(
            "engines.ace_step_engine.importlib.metadata.version",
            side_effect=lambda package: compatible[package],
        ):
            self.assertEqual(validate_ace_step_runtime(), compatible)

        incompatible = dict(compatible, transformers="5.0.0")
        with mock.patch(
            "engines.ace_step_engine.importlib.metadata.version",
            side_effect=lambda package: incompatible[package],
        ):
            with self.assertRaisesRegex(RuntimeError, r"transformers<4\.58\.0"):
                validate_ace_step_runtime()

    def test_loader_is_local_safetensors_only_and_selects_cpu(self):
        loaded = {}

        class FakeLoadedPipeline:
            def to(self, device):
                loaded["device"] = device
                return self

        class FakeAceStepPipeline:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                loaded["path"] = path
                loaded["kwargs"] = kwargs
                return FakeLoadedPipeline()

        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.AceStepPipeline = FakeAceStepPipeline

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.dict(sys.modules, {"diffusers": fake_diffusers}),
                mock.patch(
                    "engines.ace_step_engine.validate_ace_step_runtime",
                    return_value={},
                ),
                mock.patch("torch.cuda.is_available", return_value=False),
                mock.patch("torch.backends.mps.is_available", return_value=False),
                mock.patch(
                    "core.model_manager.ModelManager.is_offline",
                    new_callable=lambda: property(lambda _self: False),
                ),
            ):
                engine = ACEStepEngine()
                engine.load(tmp)

        self.assertEqual(loaded["path"], tmp)
        self.assertTrue(loaded["kwargs"]["local_files_only"])
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertEqual(loaded["kwargs"]["torch_dtype"], torch.float32)
        self.assertEqual(loaded["device"], "cpu")

    def test_cover_passes_decoded_source_as_reference_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_audio(root)
            engine, pipeline = self._engine(root)

            result = engine.generate(
                GenerationParams(
                    task_type="cover",
                    source_audio_path=str(source),
                    duration=10.0,
                    style_tags="dream pop",
                    seed=7,
                    audio_cover_strength=0.8,
                )
            )

            call = pipeline.calls[-1]
            self.assertEqual(call["task_type"], "cover")
            self.assertEqual(call["audio_cover_strength"], 0.8)
            self.assertEqual(tuple(call["reference_audio"].shape), (2, 480_000))
            self.assertNotIn("src_audio", call)
            sidecar = json.loads(Path(result.provenance_path).read_text())
            self.assertEqual(sidecar["source_paths"], [str(source)])
            self.assertEqual(sidecar["extra"]["requested_task"], "cover")

    def test_repaint_passes_source_tensor_and_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_audio(root)
            engine, pipeline = self._engine(root)

            engine.generate(
                GenerationParams(
                    task_type="repaint",
                    source_audio_path=str(source),
                    duration=10.0,
                    repaint_start=2.0,
                    repaint_end=5.0,
                    seed=11,
                )
            )

            call = pipeline.calls[-1]
            self.assertEqual(call["task_type"], "repaint")
            self.assertEqual(call["repainting_start"], 2.0)
            self.assertEqual(call["repainting_end"], 5.0)
            self.assertEqual(tuple(call["src_audio"].shape), (2, 480_000))

    def test_extend_pads_source_and_repaints_only_the_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._source_audio(root)
            engine, pipeline = self._engine(root)

            result = engine.extend(
                str(source),
                GenerationParams(style_tags="ambient"),
                extend_duration=5.0,
            )

            call = pipeline.calls[-1]
            self.assertEqual(call["task_type"], "repaint")
            self.assertEqual(call["audio_duration"], 15.0)
            self.assertEqual(call["repainting_start"], 10.0)
            self.assertEqual(call["repainting_end"], 15.0)
            self.assertEqual(tuple(call["src_audio"].shape), (2, 720_000))
            self.assertTrue(torch.all(call["src_audio"][:, 480_000:] == 0))
            self.assertEqual(result.params.task_type, "extend")

    def test_source_tasks_reject_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine, _ = self._engine(Path(tmp))
            for task in ("cover", "repaint", "extend"):
                with self.subTest(task=task):
                    with self.assertRaisesRegex(ValueError, "requires a source"):
                        engine.generate(
                            GenerationParams(task_type=task, duration=10.0)
                        )
            with self.assertRaises(FileNotFoundError):
                engine.extend("missing.wav", GenerationParams())
            with self.assertRaises(FileNotFoundError):
                engine.retake("missing.wav", 1.0, 2.0, GenerationParams())

    def test_legacy_cache_is_migrated_to_recoverable_trash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mgr = ModelManager()
            previous_settings = mgr._settings
            previous_trash = mgr._trash
            previous_current_id = mgr._current_model_id
            previous_current = mgr._current_model
            try:
                mgr._settings = type(
                    "SettingsStub",
                    (),
                    {
                        "get": lambda _self, key, default=None: (
                            str(root / "models")
                            if key == "model_hub.cache_dir"
                            else default
                        )
                    },
                )()
                mgr._trash = TrashManager(
                    trash_dir=root / "trash",
                    retention_days=30,
                )
                mgr._current_model_id = None
                mgr._current_model = None

                cache = mgr.get_cache_dir(ACE_STEP_MODEL_ID)
                cache.mkdir(parents=True)
                (cache / "legacy.bin").write_bytes(b"legacy")
                (cache / mgr.COMPLETE_MARKER).write_text(
                    json.dumps({
                        "source": "ACE-Step/ACE-Step-v1-3.5B",
                        "revision": "82cd0d7b6322bd28cd4e830fe675ddb6180ce36c",
                        "resolved_revision": (
                            "82cd0d7b6322bd28cd4e830fe675ddb6180ce36c"
                        ),
                    }),
                    encoding="utf-8",
                )

                entry = mgr._quarantine_incompatible_cache(
                    ACE_STEP_MODEL_ID,
                    cache,
                )
                self.assertIsNotNone(entry)
                self.assertFalse(cache.exists())
                self.assertTrue(Path(entry.trash_path).is_dir())
                self.assertEqual(entry.metadata["new_source"], ACE_STEP_SOURCE)
                self.assertEqual(entry.metadata["new_revision"], ACE_STEP_REVISION)
            finally:
                mgr._settings = previous_settings
                mgr._trash = previous_trash
                mgr._current_model_id = previous_current_id
                mgr._current_model = previous_current


if __name__ == "__main__":
    unittest.main()
