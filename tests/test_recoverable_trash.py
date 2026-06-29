import tempfile
import time
import unittest
import contextlib
import io
from pathlib import Path
from unittest import mock

from core.model_manager import ModelCategory, ModelInfo, ModelManager, ModelStatus
from core.project import ProjectManager
from core.trash import TrashError, TrashManager


class RecoverableTrashTests(unittest.TestCase):
    def test_trash_manifest_restore_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "generated.wav"
            source.write_bytes(b"audio")
            trash = TrashManager(root / "trash", retention_days=0)

            entry = trash.trash_path(
                source,
                category="generated_asset",
                label="generated.wav",
                metadata={"module": "sfx"},
            )

            self.assertFalse(source.exists())
            self.assertTrue(Path(entry.trash_path).exists())
            self.assertTrue(Path(entry.manifest_path).exists())
            self.assertEqual(entry.metadata["module"], "sfx")

            restored = trash.restore(entry.id)
            self.assertEqual(Path(restored.original_path), source)
            self.assertTrue(source.exists())
            self.assertFalse(Path(restored.manifest_path).exists())

            source.write_bytes(b"audio2")
            expired = trash.trash_path(
                source,
                category="generated_asset",
                label="generated.wav",
            )
            removed = trash.cleanup_expired(now=time.time() + 2)
            self.assertIn(expired.id, removed)
            self.assertFalse(Path(expired.manifest_path).parent.exists())

    def test_missing_path_delete_reports_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            trash = TrashManager(Path(tmp) / "trash")
            with self.assertRaises(TrashError):
                trash.trash_path(
                    Path(tmp) / "missing",
                    category="project",
                    label="missing",
                )

    def test_project_delete_moves_directory_and_restores_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            ProjectManager._instance = None
            try:
                with mock.patch("core.project.get_config_dir", return_value=config_dir):
                    mgr = ProjectManager()
                    mgr._trash = TrashManager(Path(tmp) / "trash")
                    project = mgr.create("Recoverable Project")
                    project_dir = Path(mgr._index[project.id]["path"])

                    entry = mgr.delete(project.id)

                    self.assertIsNotNone(entry)
                    self.assertFalse(project_dir.exists())
                    self.assertNotIn(project.id, mgr._index)

                    self.assertTrue(mgr.restore_deleted_project(entry.id))
                    self.assertTrue(project_dir.exists())
                    self.assertIn(project.id, mgr._index)
                    self.assertEqual(mgr._index[project.id]["name"], "Recoverable Project")
            finally:
                ProjectManager._instance = None

    def test_project_delete_failure_keeps_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            ProjectManager._instance = None
            try:
                with mock.patch("core.project.get_config_dir", return_value=config_dir):
                    mgr = ProjectManager()
                    mgr._trash = TrashManager(Path(tmp) / "trash")
                    mgr._index["missing"] = {
                        "name": "Missing",
                        "path": str(config_dir / "projects" / "missing"),
                        "updated_at": time.time(),
                    }

                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertIsNone(mgr.delete("missing"))
                    self.assertIn("missing", mgr._index)
            finally:
                ProjectManager._instance = None

    def test_model_cache_delete_and_restore_refreshes_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mgr = ModelManager()
            old_settings = mgr._settings
            old_registry = mgr._registry
            old_status = mgr._status
            old_trash = mgr._trash
            try:
                mgr._settings = type(
                    "SettingsStub",
                    (),
                    {
                        "get": lambda _self, key, default=None: (
                            str(root / "models") if key == "model_hub.cache_dir" else default
                        )
                    },
                )()
                mgr._registry = {
                    "trash-model": ModelInfo(
                        model_id="trash-model",
                        name="Trash Model",
                        description="Test",
                        category=ModelCategory.EXTRAS,
                        vram_gb=1.0,
                        disk_gb=0.001,
                        license="MIT",
                        source="example/model",
                        loader_module="engines.sfx_engine",
                        loader_fn="load_model",
                    )
                }
                mgr._status = {"trash-model": ModelStatus.DOWNLOADED}
                mgr._trash = TrashManager(root / "trash")

                cache_dir = mgr.get_cache_dir("trash-model")
                cache_dir.mkdir(parents=True)
                (cache_dir / "weights.safetensors").write_bytes(b"weights")
                mgr._write_complete_marker("trash-model", cache_dir)

                entry = mgr.delete_model_cache("trash-model")

                self.assertIsNotNone(entry)
                self.assertFalse(cache_dir.exists())
                self.assertEqual(mgr.get_status("trash-model"), ModelStatus.NOT_DOWNLOADED)

                self.assertTrue(mgr.restore_model_cache(entry.id))
                self.assertTrue(cache_dir.exists())
                self.assertEqual(mgr.get_status("trash-model"), ModelStatus.DOWNLOADED)
            finally:
                mgr._settings = old_settings
                mgr._registry = old_registry
                mgr._status = old_status
                mgr._trash = old_trash


if __name__ == "__main__":
    unittest.main()
