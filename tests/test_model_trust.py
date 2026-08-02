import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.model_manager import (
    BUILTIN_MODELS,
    COMMERCIAL_USE_ALLOWED,
    EXECUTABLE_MODEL_WARNING,
    ModelInfo,
    ModelManager,
    ModelCategory,
    ModelSecurityError,
    OfflineModeError,
    is_commit_sha,
)
from core.voice_bank import VoiceProfile
from engines.demucs_engine import DemucsEngine, separate_stems
from engines.midi_llm_engine import MidiLLMEngine
from engines.rvc_engine import RVCEngine


class ModelTrustTests(unittest.TestCase):
    def test_builtin_huggingface_sources_use_immutable_revisions(self):
        for model_id, info in BUILTIN_MODELS.items():
            if not info.source or info.pip_managed:
                continue
            with self.subTest(model_id=model_id):
                self.assertTrue(is_commit_sha(info.revision), info.revision)

    def test_download_manifest_records_hashes_and_detects_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager()
            old_settings = mgr._settings
            old_registry = mgr._registry
            try:
                mgr._settings = type(
                    "SettingsStub",
                    (),
                    {"get": lambda _self, key, default=None: tmp if key == "model_hub.cache_dir" else default},
                )()
                mgr._registry = {
                    "test-model": ModelInfo(
                        model_id="test-model",
                        name="Test Model",
                        description="Test",
                        category=ModelCategory.EXTRAS,
                        vram_gb=1.0,
                        disk_gb=0.001,
                        license="MIT",
                        source="example/model",
                        revision="a" * 40,
                        loader_module="engines.sfx_engine",
                        loader_fn="load_model",
                        commercial_use=COMMERCIAL_USE_ALLOWED,
                        commercial_use_note="Test license note",
                        license_url="https://example.com/license",
                    )
                }
                cache_dir = mgr.get_cache_dir("test-model")
                cache_dir.mkdir(parents=True)
                model_file = cache_dir / "weights.safetensors"
                model_file.write_bytes(b"safe weights")

                mgr._write_complete_marker(
                    "test-model",
                    cache_dir,
                    resolved_path=str(cache_dir),
                    resolved_revision="a" * 40,
                )

                ok, reason = mgr.verify_download("test-model")
                self.assertTrue(ok, reason)
                manifest = mgr.get_download_manifest("test-model")
                self.assertEqual(manifest["license"], "MIT")
                self.assertEqual(manifest["license_url"], "https://example.com/license")
                self.assertEqual(manifest["commercial_use"], "allowed")
                self.assertEqual(manifest["commercial_use_label"], "Allowed")
                self.assertFalse(manifest["requires_export_warning"])
                self.assertEqual(manifest["revision"], "a" * 40)
                self.assertEqual(manifest["resolved_revision"], "a" * 40)
                self.assertIn("weights.safetensors", manifest["file_hashes"])
                self.assertEqual(manifest["serialization"], "safetensors")

                model_file.write_bytes(b"changed")
                ok, reason = mgr.verify_download("test-model")
                self.assertFalse(ok)
                self.assertIn("Hash mismatch", reason)
            finally:
                mgr._settings = old_settings
                mgr._registry = old_registry

    def test_verification_rejects_unhashed_and_undeclared_executable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager()
            old_settings = mgr._settings
            old_registry = mgr._registry
            try:
                mgr._settings = type(
                    "SettingsStub",
                    (),
                    {"get": lambda _self, key, default=None: tmp if key == "model_hub.cache_dir" else default},
                )()
                info = ModelInfo(
                    model_id="safe-model",
                    name="Safe Model",
                    description="Test",
                    category=ModelCategory.EXTRAS,
                    vram_gb=1.0,
                    disk_gb=0.001,
                    license="MIT",
                    source="example/safe",
                    revision="b" * 40,
                    loader_module="engines.sfx_engine",
                    loader_fn="load_model",
                )
                mgr._registry = {"safe-model": info}
                cache_dir = mgr.get_cache_dir("safe-model")
                cache_dir.mkdir(parents=True)
                (cache_dir / "weights.safetensors").write_bytes(b"safe")
                mgr._write_complete_marker(
                    "safe-model",
                    cache_dir,
                    resolved_revision=info.revision,
                )

                (cache_dir / "injected.py").write_text("raise SystemExit", encoding="utf-8")
                ok, reason = mgr.verify_download("safe-model")
                self.assertFalse(ok)
                self.assertIn("File count mismatch", reason)

                mgr._write_complete_marker(
                    "safe-model",
                    cache_dir,
                    resolved_revision=info.revision,
                )
                ok, reason = mgr.verify_download("safe-model")
                self.assertFalse(ok)
                self.assertIn("Undeclared executable model code", reason)
            finally:
                mgr._settings = old_settings
                mgr._registry = old_registry

    def test_executable_model_requires_explicit_per_revision_consent(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "model_hub.cache_dir": tmp,
                "model_hub.execution_consents": {},
            }

            class SettingsStub:
                def get(self, key, default=None):
                    return state.get(key, default)

                def set(self, key, value):
                    state[key] = value

            mgr = ModelManager()
            old_settings = mgr._settings
            old_registry = mgr._registry
            try:
                info = ModelInfo(
                    model_id="custom-code",
                    name="Custom Code",
                    description="Test",
                    category=ModelCategory.EXTRAS,
                    vram_gb=1.0,
                    disk_gb=0.001,
                    license="MIT",
                    source="example/custom-code",
                    revision="c" * 40,
                    loader_module="engines.sfx_engine",
                    loader_fn="load_model",
                    requires_remote_code=True,
                )
                mgr._settings = SettingsStub()
                mgr._registry = {"custom-code": info}
                cache_dir = mgr.get_cache_dir("custom-code")
                cache_dir.mkdir(parents=True)
                (cache_dir / "weights.safetensors").write_bytes(b"safe")
                (cache_dir / "modeling_custom.py").write_text("# reviewed", encoding="utf-8")
                mgr._write_complete_marker(
                    "custom-code",
                    cache_dir,
                    resolved_revision=info.revision,
                )

                with self.assertRaises(ModelSecurityError) as ctx:
                    mgr.require_verified_model("custom-code")
                self.assertIn(EXECUTABLE_MODEL_WARNING, str(ctx.exception))

                with self.assertRaises(ModelSecurityError):
                    mgr.approve_executable_model(
                        "custom-code",
                        info.revision,
                        acknowledged=False,
                    )

                mgr.approve_executable_model(
                    "custom-code",
                    info.revision,
                    acknowledged=True,
                )
                manifest = mgr.require_verified_model("custom-code")
                self.assertEqual(manifest["revision"], info.revision)

                info.revision = "d" * 40
                with self.assertRaises(ModelSecurityError):
                    mgr.require_verified_model("custom-code")
            finally:
                mgr._settings = old_settings
                mgr._registry = old_registry

    def test_unverified_cache_never_reaches_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = ModelManager()
            old_settings = mgr._settings
            old_registry = mgr._registry
            loader = MagicMock()
            try:
                mgr._settings = type(
                    "SettingsStub",
                    (),
                    {
                        "get": lambda _self, key, default=None: (
                            tmp if key == "model_hub.cache_dir"
                            else True if key == "model_hub.offline_mode"
                            else default
                        )
                    },
                )()
                mgr._registry = {
                    "offline-model": ModelInfo(
                        model_id="offline-model",
                        name="Offline Model",
                        description="Test",
                        category=ModelCategory.EXTRAS,
                        vram_gb=1.0,
                        disk_gb=0.001,
                        license="MIT",
                        source="example/offline",
                        revision="e" * 40,
                        loader_module="engines.sfx_engine",
                        loader_fn="load_model",
                    )
                }

                with self.assertRaises(ModelSecurityError):
                    mgr.load_model("offline-model", loader_fn=loader)
                loader.assert_not_called()
            finally:
                mgr._settings = old_settings
                mgr._registry = old_registry

    def test_midi_transformers_loader_is_local_and_disables_remote_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = MagicMock()
            model.to.return_value = model
            tokenizer = MagicMock()
            engine = MidiLLMEngine()
            tokenizer_loader = MagicMock(return_value=tokenizer)
            model_loader = MagicMock(return_value=model)
            transformers = types.ModuleType("transformers")
            transformers.AutoTokenizer = types.SimpleNamespace(
                from_pretrained=tokenizer_loader
            )
            transformers.AutoModelForCausalLM = types.SimpleNamespace(
                from_pretrained=model_loader
            )
            with patch.dict(sys.modules, {"transformers": transformers}):
                engine.load_model(tmp, device="cpu")

            tokenizer_kwargs = tokenizer_loader.call_args.kwargs
            model_kwargs = model_loader.call_args.kwargs
            self.assertTrue(tokenizer_kwargs["local_files_only"])
            self.assertFalse(tokenizer_kwargs["trust_remote_code"])
            self.assertTrue(model_kwargs["local_files_only"])
            self.assertFalse(model_kwargs["trust_remote_code"])
            self.assertTrue(model_kwargs["use_safetensors"])

    def test_rvc_rejects_untrusted_pickle_checkpoint_before_torch_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "voice.pth"
            checkpoint.write_bytes(b"not a real checkpoint")
            engine = RVCEngine()
            profile = VoiceProfile(
                name="Untrusted",
                engine="rvc",
                model_path=str(checkpoint),
                trusted=False,
                owner_name="Singer",
                consent_status="confirmed",
                consent_source="Self-recorded / my voice",
                consent_scope="Clone + conversion",
                language="en",
                permitted_uses=["voice-conversion"],
            )

            with self.assertRaises(RuntimeError) as ctx:
                engine.load_model(profile, device="cpu")

            self.assertIn("unsafe local checkpoint", str(ctx.exception))


    def test_offline_mode_blocks_download_model(self):
        mgr = ModelManager()
        old_settings = mgr._settings

        try:
            mgr._settings = type(
                "SettingsStub",
                (),
                {
                    "get": lambda _self, key, default=None: (
                        True if key == "model_hub.offline_mode" else default
                    )
                },
            )()

            with self.assertRaises(OfflineModeError) as ctx:
                mgr.download_model("ace-step-v1.5")

            self.assertIn("Offline Mode", str(ctx.exception))
        finally:
            mgr._settings = old_settings

    @patch("core.model_manager.ModelManager.is_offline", new_callable=lambda: property(lambda self: True))
    def test_offline_mode_skips_hf_revision_resolution(self, _):
        mgr = ModelManager()
        info = mgr.get_model_info("ace-step-v1.5")
        self.assertIsNotNone(info)

        with patch("huggingface_hub.HfApi") as mock_api:
            result = mgr._resolve_hf_revision(info)
            mock_api.assert_not_called()
            self.assertEqual(result, info.revision)

    @patch("core.model_manager.ModelManager.is_offline", new_callable=lambda: property(lambda self: True))
    def test_offline_mode_snapshot_download_not_called(self, _):
        mgr = ModelManager()
        with patch("huggingface_hub.snapshot_download") as mock_sd:
            with self.assertRaises(OfflineModeError):
                mgr.download_model("ace-step-v1.5")
            mock_sd.assert_not_called()

    @patch("core.model_manager.ModelManager.is_offline", new_callable=lambda: property(lambda self: True))
    def test_demucs_offline_mode_rejects_uncached_checkpoint_before_get_model(self, _):
        # Keep the fake torch module below from affecting ModelManager's one-time
        # runtime-package probe.
        ModelManager()
        with tempfile.TemporaryDirectory() as tmp:
            torch_module = types.ModuleType("torch")
            torch_module.hub = types.SimpleNamespace(
                get_dir=lambda: str(Path(tmp) / "torch")
            )
            pretrained_module = types.ModuleType("demucs.pretrained")
            pretrained_module.REMOTE_ROOT = Path(tmp) / "demucs-remote"
            pretrained_module.get_model = MagicMock()
            demucs_module = types.ModuleType("demucs")
            demucs_module.pretrained = pretrained_module

            with patch.dict(
                sys.modules,
                {
                    "torch": torch_module,
                    "demucs": demucs_module,
                    "demucs.pretrained": pretrained_module,
                },
            ), patch("engines.demucs_engine.get_demucs", return_value=DemucsEngine()):
                with self.assertRaises(OfflineModeError):
                    separate_stems(str(Path(tmp) / "input.wav"))

            pretrained_module.get_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
