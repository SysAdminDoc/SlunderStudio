import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from core.project import PROJECT_SCHEMA_VERSION, ProjectManager
from core.provenance import read_provenance_sidecar, write_provenance_sidecar
from core.trash import TrashManager


class ProjectAssetRecoveryTests(unittest.TestCase):
    def tearDown(self):
        ProjectManager._instance = None

    def test_same_named_assets_and_sidecars_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first" / "take.wav"
            second = root / "second" / "take.wav"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"first audio")
            second.write_bytes(b"second audio")
            write_provenance_sidecar(
                first,
                module="song_forge",
                operation="generate",
                prompt="first",
            )
            write_provenance_sidecar(
                second,
                module="song_forge",
                operation="generate",
                prompt="second",
            )

            mgr = self._manager(root)
            project = mgr.create("Collision Test")
            first_id = mgr.import_asset(str(first), "audio", "song_forge")
            second_id = mgr.import_asset(str(second), "audio", "song_forge")

            self.assertNotEqual(first_id, second_id)
            self.assertEqual(2, len(project.assets))
            first_asset, second_asset = project.assets
            first_dest = Path(first_asset.file_path)
            second_dest = Path(second_asset.file_path)
            self.assertNotEqual(first_dest, second_dest)
            self.assertEqual(b"first audio", first_dest.read_bytes())
            self.assertEqual(b"second audio", second_dest.read_bytes())
            self.assertEqual(b"first audio", first.read_bytes())
            self.assertEqual(b"second audio", second.read_bytes())
            self.assertTrue(first_dest.name.startswith(f"{first_id}__"))
            self.assertTrue(second_dest.name.startswith(f"{second_id}__"))

            first_provenance = read_provenance_sidecar(first_dest)
            second_provenance = read_provenance_sidecar(second_dest)
            self.assertEqual("first", first_provenance["prompt"])
            self.assertEqual("second", second_provenance["prompt"])
            self.assertNotEqual(
                first_asset.provenance_path,
                second_asset.provenance_path,
            )

            saved = json.loads(
                (
                    root
                    / "config"
                    / "projects"
                    / project.id
                    / "project.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                {first_id, second_id},
                {asset["id"] for asset in saved["assets"]},
            )

    def test_asset_display_name_cannot_escape_storage_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            source.write_bytes(b"safe")
            mgr = self._manager(root)
            project = mgr.create("Traversal Test")

            asset_id = mgr.import_asset(
                str(source),
                "audio",
                "mixer",
                name=r"..\..\escape.wav",
            )

            asset = next(item for item in project.assets if item.id == asset_id)
            assets_dir = (
                root / "config" / "projects" / project.id / "assets"
            ).resolve()
            destination = Path(asset.file_path).resolve()
            self.assertEqual(assets_dir, destination.parent)
            self.assertFalse((root / "escape.wav").exists())
            self.assertEqual(b"safe", destination.read_bytes())

    def test_failed_metadata_save_rolls_back_new_asset_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            source.write_bytes(b"transaction")
            mgr = self._manager(root)
            project = mgr.create("Rollback Test")

            with mock.patch.object(mgr, "save", return_value=False):
                with self.assertRaises(OSError):
                    mgr.import_asset(str(source), "audio", "mixer")

            self.assertEqual([], project.assets)
            assets_dir = (
                root / "config" / "projects" / project.id / "assets"
            )
            self.assertEqual([], list(assets_dir.iterdir()))
            self.assertEqual(b"transaction", source.read_bytes())

    def test_failed_delete_metadata_save_restores_asset_file_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            source.write_bytes(b"delete rollback")
            mgr = self._manager(root)
            project = mgr.create("Delete Rollback")
            asset_id = mgr.import_asset(str(source), "audio", "mixer")
            asset = project.assets[0]
            stored_path = Path(asset.file_path)

            with mock.patch.object(mgr, "save", return_value=False):
                self.assertIsNone(mgr.delete_asset(asset_id))

            self.assertTrue(stored_path.exists())
            self.assertEqual([asset_id], [item.id for item in project.assets])
            self.assertEqual([], mgr._trash.list_entries())

    def test_exclusive_copy_never_removes_or_replaces_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"new")
            destination.write_bytes(b"existing")

            with self.assertRaises(FileExistsError):
                ProjectManager._copy_file_exclusive(source, destination)

            self.assertEqual(b"existing", destination.read_bytes())
            self.assertEqual(b"new", source.read_bytes())

    def test_corrupt_index_is_rebuilt_from_valid_project_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            projects_dir = root / "config" / "projects"
            first_dir = projects_dir / "proj_first"
            second_dir = projects_dir / "proj_second"
            first_dir.mkdir(parents=True)
            second_dir.mkdir(parents=True)
            self._write_project(first_dir, "proj_first", "First")
            self._write_project(second_dir, "proj_second", "Second")
            (projects_dir / "index.json").write_text(
                "{not-json",
                encoding="utf-8",
            )

            mgr = self._manager(root)

            self.assertEqual(
                {"proj_first", "proj_second"},
                {item["id"] for item in mgr.list_projects()},
            )
            repaired_index = json.loads(
                (projects_dir / "index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {"proj_first", "proj_second"},
                set(repaired_index),
            )
            self.assertTrue(list(
                (projects_dir / "backups").glob("index.json.*.corrupt.bak")
            ))
            status = mgr.last_repair_status
            self.assertEqual("repaired", status["status"])
            self.assertIn("Recovered project First", " ".join(status["messages"]))

    def test_incomplete_index_is_merged_and_paths_are_canonicalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            projects_dir = root / "config" / "projects"
            first_dir = projects_dir / "proj_first"
            second_dir = projects_dir / "proj_second"
            first_dir.mkdir(parents=True)
            second_dir.mkdir(parents=True)
            self._write_project(first_dir, "proj_first", "First")
            self._write_project(second_dir, "proj_second", "Second")
            (projects_dir / "index.json").write_text(json.dumps({
                "proj_first": {
                    "name": "Wrong",
                    "path": str(root / "outside"),
                    "updated_at": 0,
                }
            }), encoding="utf-8")

            mgr = self._manager(root)
            indexed = {item["id"]: item for item in mgr.list_projects()}

            self.assertEqual({"proj_first", "proj_second"}, set(indexed))
            self.assertEqual(str(first_dir.resolve()), indexed["proj_first"]["path"])
            self.assertEqual("First", indexed["proj_first"]["name"])
            self.assertIsNotNone(mgr.open("proj_second"))

    def test_project_metadata_is_restored_from_latest_valid_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project_dir = root / "config" / "projects" / "proj_backup"
            backup_dir = project_dir / "backups"
            backup_dir.mkdir(parents=True)
            (project_dir / "project.json").write_text(
                "{corrupt",
                encoding="utf-8",
            )
            valid = self._project_payload("proj_backup", "Recovered")
            backup = backup_dir / "project.json.20260728_120000.pre-save.bak"
            backup.write_text(json.dumps(valid), encoding="utf-8")

            mgr = self._manager(root)
            initial_status = mgr.last_repair_status
            restored = json.loads(
                (project_dir / "project.json").read_text(encoding="utf-8")
            )
            project = mgr.open("proj_backup")

            self.assertEqual("proj_backup", restored["id"])
            self.assertIsNotNone(project)
            self.assertEqual("Recovered", project.name)
            self.assertIn(
                "Restored project proj_backup",
                " ".join(initial_status["messages"]),
            )
            self.assertTrue(list(
                backup_dir.glob("project.json.*.corrupt.bak")
            ))

    def _manager(self, root: Path) -> ProjectManager:
        ProjectManager._instance = None
        patcher = mock.patch(
            "core.project.get_config_dir",
            return_value=root / "config",
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        manager = ProjectManager()
        manager._trash = TrashManager(root / "trash")
        return manager

    @classmethod
    def _write_project(
        cls,
        project_dir: Path,
        project_id: str,
        name: str,
    ) -> None:
        (project_dir / "project.json").write_text(
            json.dumps(cls._project_payload(project_id, name)),
            encoding="utf-8",
        )

    @staticmethod
    def _project_payload(project_id: str, name: str) -> dict:
        now = time.time()
        return {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "app_version": "0.1.30",
            "id": project_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "tempo": 120,
            "key": "C major",
            "time_signature": [4, 4],
            "assets": [],
            "versions": [],
            "mixer_state": {},
            "lyrics_text": "",
            "notes": "",
        }


if __name__ == "__main__":
    unittest.main()
