import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from core.project import PROJECT_SCHEMA_VERSION, ProjectManager
from core.settings import APP_VERSION, SETTINGS_SCHEMA_VERSION, Settings
from core.trash import TrashManager


class SchemaMigrationTests(unittest.TestCase):
    def tearDown(self):
        Settings._instance = None
        ProjectManager._instance = None

    def test_settings_old_schema_migrates_with_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            config_path = config_dir / "config.json"
            config_path.write_text(json.dumps({
                "version": "0.1.0",
                "general": {
                    "output_dir": str(root / "renders"),
                    "audio_format": "flac",
                },
                "model_hub": {
                    "cache_dir": str(root / "models"),
                },
            }), encoding="utf-8")

            Settings._instance = None
            with mock.patch("core.settings.get_config_dir", return_value=config_dir):
                settings = Settings()

            status = settings.repair_status
            saved = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(status["status"], "migrated")
            self.assertEqual(saved["schema_version"], SETTINGS_SCHEMA_VERSION)
            self.assertEqual(saved["version"], APP_VERSION)
            self.assertEqual(saved["general"]["audio_format"], "flac")
            self.assertEqual(saved["general"]["trash_retention_days"], 30)
            self.assertTrue(status["backup_paths"])
            self.assertTrue(Path(status["backup_paths"][0]).is_file())

    def test_settings_corrupt_json_restores_defaults_with_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp) / "config"
            config_dir.mkdir()
            config_path = config_dir / "config.json"
            config_path.write_text("{not-json", encoding="utf-8")

            Settings._instance = None
            with mock.patch("core.settings.get_config_dir", return_value=config_dir), \
                    mock.patch("core.settings.get_default_output_dir", return_value=Path(tmp) / "renders"), \
                    mock.patch("core.settings.get_default_cache_dir", return_value=Path(tmp) / "models"):
                settings = Settings()

            status = settings.repair_status
            saved = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(status["status"], "repaired")
            self.assertEqual(saved["schema_version"], SETTINGS_SCHEMA_VERSION)
            self.assertEqual(saved["version"], APP_VERSION)
            self.assertEqual(saved["general"]["output_dir"], str(Path(tmp) / "renders"))
            self.assertTrue(status["backup_paths"])
            self.assertTrue(Path(status["backup_paths"][0]).is_file())

    def test_project_old_schema_migrates_with_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            project_dir = config_dir / "projects" / "proj_old"
            project_dir.mkdir(parents=True)
            now = time.time()
            (config_dir / "projects" / "index.json").write_text(json.dumps({
                "proj_old": {
                    "name": "Old Project",
                    "path": str(project_dir),
                    "updated_at": now,
                }
            }), encoding="utf-8")
            (project_dir / "project.json").write_text(json.dumps({
                "id": "proj_old",
                "name": "Old Project",
                "created_at": now,
                "updated_at": now,
                "tempo": 96,
                "time_signature": [3, 4],
                "assets": [],
            }), encoding="utf-8")

            ProjectManager._instance = None
            try:
                with mock.patch("core.project.get_config_dir", return_value=config_dir):
                    mgr = ProjectManager()
                    mgr._trash = TrashManager(root / "trash")
                    project = mgr.open("proj_old")
            finally:
                ProjectManager._instance = None

            status = mgr.last_repair_status
            saved = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))

            self.assertIsNotNone(project)
            self.assertEqual(project.schema_version, PROJECT_SCHEMA_VERSION)
            self.assertEqual(project.app_version, APP_VERSION)
            self.assertEqual(project.time_signature, (3, 4))
            self.assertEqual(status["status"], "migrated")
            self.assertEqual(saved["schema_version"], PROJECT_SCHEMA_VERSION)
            self.assertEqual(saved["app_version"], APP_VERSION)
            self.assertTrue(status["backup_paths"])
            self.assertTrue(Path(status["backup_paths"][0]).is_file())

    def test_project_corrupt_json_reports_repair_with_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            project_dir = config_dir / "projects" / "proj_bad"
            project_dir.mkdir(parents=True)
            (config_dir / "projects" / "index.json").write_text(json.dumps({
                "proj_bad": {
                    "name": "Bad Project",
                    "path": str(project_dir),
                    "updated_at": time.time(),
                }
            }), encoding="utf-8")
            (project_dir / "project.json").write_text("{bad-json", encoding="utf-8")

            ProjectManager._instance = None
            try:
                with mock.patch("core.project.get_config_dir", return_value=config_dir):
                    mgr = ProjectManager()
                    mgr._trash = TrashManager(root / "trash")
                    project = mgr.open("proj_bad")
            finally:
                ProjectManager._instance = None

            status = mgr.last_repair_status
            self.assertIsNone(project)
            self.assertEqual(status["status"], "repaired")
            self.assertIn("unreadable", " ".join(status["messages"]))
            self.assertTrue(status["backup_paths"])
            self.assertTrue(Path(status["backup_paths"][0]).is_file())


if __name__ == "__main__":
    unittest.main()
