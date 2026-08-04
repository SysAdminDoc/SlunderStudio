import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.model_manager import (
    COMMERCIAL_USE_ALLOWED,
    ModelCategory,
    ModelInfo,
    ModelManager,
    ModelStatus,
    ModelUpdateError,
)
from core.trash import TrashManager


class _SettingsStub:
    def __init__(self, cache_dir: Path):
        self.values = {
            "model_hub.cache_dir": str(cache_dir),
            "model_hub.offline_mode": False,
            "model_hub.update_checks": {},
            "model_hub.update_backups": {},
            "model_hub.installed_revisions": {},
        }

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class _FakeHuggingFace:
    def __init__(self, target: str):
        self.target = target

    def model_info(self, _source):
        return SimpleNamespace(
            sha=self.target,
            cardData={"release_notes": ["Improved vocal texture and reduced memory use."]},
            lastModified="2026-08-03T12:00:00Z",
        )

    def list_repo_commits(self, _source, **_kwargs):
        return [SimpleNamespace(title="Improve vocal texture", message="Improve vocal texture")]


class ModelUpdateTests(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager()
        self.old_settings = self.manager._settings
        self.old_registry = self.manager._registry
        self.old_status = self.manager._status
        self.old_trash = self.manager._trash
        self.old_errors = self.manager._model_errors
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.settings = _SettingsStub(root / "models")
        self.manager._settings = self.settings
        self.manager._trash = TrashManager(root / "trash", retention_days=30)
        self.info = ModelInfo(
            model_id="update-model",
            name="Update Model",
            description="test model",
            category=ModelCategory.EXTRAS,
            vram_gb=1.0,
            disk_gb=0.001,
            license="MIT",
            source="example/update-model",
            revision="a" * 40,
            loader_module="engines.sfx_engine",
            loader_fn="load_model",
            commercial_use=COMMERCIAL_USE_ALLOWED,
        )
        self.manager._registry = {self.info.model_id: self.info}
        self.manager._status = {self.info.model_id: ModelStatus.NOT_DOWNLOADED}
        self.manager._readiness_cache.clear()

    def tearDown(self):
        self.manager._settings = self.old_settings
        self.manager._registry = self.old_registry
        self.manager._status = self.old_status
        self.manager._trash = self.old_trash
        self.manager._model_errors = self.old_errors
        self.tmp.cleanup()

    def _write_cache(self, revision: str, contents: bytes):
        cache = self.manager.get_cache_dir(self.info.model_id)
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "weights.safetensors").write_bytes(contents)
        self.manager._write_complete_marker(
            self.info.model_id,
            cache,
            resolved_revision=revision,
            revision=revision,
        )
        return cache

    def test_check_requires_immutable_target_and_persists_release_notes(self):
        target = "b" * 40
        result = self.manager.check_for_updates(
            [self.info.model_id],
            api=_FakeHuggingFace(target),
        )[self.info.model_id]

        self.assertTrue(result.available)
        self.assertEqual(target, result.target_revision)
        self.assertIn("Improved vocal texture", result.release_notes[0])
        self.assertEqual(target, self.settings.get("model_hub.update_checks")[self.info.model_id]["target_revision"])

        invalid_api = mock.Mock()
        invalid_api.model_info.return_value = SimpleNamespace(sha="main")
        invalid = self.manager.check_for_updates([self.info.model_id], api=invalid_api)[self.info.model_id]
        self.assertEqual("error", invalid.status)
        self.assertIn("immutable", invalid.error)

    def test_update_installs_checked_revision_and_rolls_back_last_good_cache(self):
        self._write_cache("a" * 40, b"last good")
        target = "b" * 40
        self.manager.check_for_updates([self.info.model_id], api=_FakeHuggingFace(target))

        def fake_download(model_id, revision, **_kwargs):
            cache = self.manager.get_cache_dir(model_id)
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "weights.safetensors").write_bytes(b"new candidate")
            self.manager._write_complete_marker(
                model_id,
                cache,
                resolved_revision=revision,
                revision=revision,
            )
            return True

        with mock.patch.object(self.manager, "_download_at_revision", side_effect=fake_download):
            health = self.manager.install_model_update(self.info.model_id)

        self.assertTrue(health.healthy)
        self.assertEqual(target, self.manager.get_model_info(self.info.model_id).revision)
        self.assertEqual(ModelStatus.DOWNLOADED, self.manager.get_status(self.info.model_id))
        self.assertEqual(target, self.manager.get_model_rollback(self.info.model_id)["target_revision"])
        # The old cache is recoverable in trash while the target is installed.
        self.assertEqual(
            b"new candidate",
            (self.manager.get_cache_dir(self.info.model_id) / "weights.safetensors").read_bytes(),
        )

        rolled_back = self.manager.rollback_model_update(self.info.model_id)

        self.assertTrue(rolled_back.healthy)
        self.assertEqual("a" * 40, self.manager.get_model_info(self.info.model_id).revision)
        self.assertEqual(b"last good", (self.manager.get_cache_dir(self.info.model_id) / "weights.safetensors").read_bytes())
        self.assertIsNone(self.manager.get_model_rollback(self.info.model_id))

    def test_failed_health_validation_restores_last_good_cache(self):
        self._write_cache("a" * 40, b"last good")
        target = "b" * 40
        self.manager.check_for_updates([self.info.model_id], api=_FakeHuggingFace(target))

        def fake_download(model_id, revision, **_kwargs):
            cache = self.manager.get_cache_dir(model_id)
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "weights.safetensors").write_bytes(b"bad runtime")
            self.manager._write_complete_marker(
                model_id,
                cache,
                resolved_revision=revision,
                revision=revision,
            )

        with mock.patch.object(self.manager, "_download_at_revision", side_effect=fake_download):
            with self.assertRaisesRegex(ModelUpdateError, "Health validation failed"):
                self.manager.install_model_update(
                    self.info.model_id,
                    runtime_check=lambda *_args: False,
                )

        self.assertEqual("a" * 40, self.manager.get_model_info(self.info.model_id).revision)
        self.assertEqual(ModelStatus.DOWNLOADED, self.manager.get_status(self.info.model_id))
        self.assertEqual(b"last good", (self.manager.get_cache_dir(self.info.model_id) / "weights.safetensors").read_bytes())


if __name__ == "__main__":
    unittest.main()
