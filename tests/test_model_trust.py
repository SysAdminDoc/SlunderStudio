import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.model_manager import (
    COMMERCIAL_USE_ALLOWED,
    ModelInfo,
    ModelManager,
    ModelCategory,
    OfflineModeError,
)
from core.voice_bank import VoiceProfile
from engines.rvc_engine import RVCEngine


class ModelTrustTests(unittest.TestCase):
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
                        revision="abc123",
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
                    resolved_revision="abc123resolved",
                )

                ok, reason = mgr.verify_download("test-model")
                self.assertTrue(ok, reason)
                manifest = mgr.get_download_manifest("test-model")
                self.assertEqual(manifest["license"], "MIT")
                self.assertEqual(manifest["license_url"], "https://example.com/license")
                self.assertEqual(manifest["commercial_use"], "allowed")
                self.assertEqual(manifest["commercial_use_label"], "Allowed")
                self.assertFalse(manifest["requires_export_warning"])
                self.assertEqual(manifest["revision"], "abc123")
                self.assertEqual(manifest["resolved_revision"], "abc123resolved")
                self.assertIn("weights.safetensors", manifest["file_hashes"])

                model_file.write_bytes(b"changed")
                ok, reason = mgr.verify_download("test-model")
                self.assertFalse(ok)
                self.assertIn("Hash mismatch", reason)
            finally:
                mgr._settings = old_settings
                mgr._registry = old_registry

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


if __name__ == "__main__":
    unittest.main()
