import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.autosave import MAX_INTERVAL_SECONDS, MIN_INTERVAL_SECONDS, AutosaveCoordinator, resolve_interval
from core.credentials import MemoryCredentialStore
from core.project import (
    VERSION_KIND_AUTO,
    VERSION_KIND_MANUAL,
    VERSION_KIND_PRE_RESTORE,
    ProjectManager,
)
from core.settings import Settings


class ProjectVersionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        config_dir = self.root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch("core.settings.get_config_dir", return_value=config_dir))
        stack.enter_context(mock.patch("core.settings.get_default_output_dir", return_value=self.root / "out"))
        stack.enter_context(mock.patch("core.settings.get_default_cache_dir", return_value=self.root / "models"))
        stack.enter_context(mock.patch("core.project.get_config_dir", return_value=str(config_dir)))
        stack.enter_context(mock.patch("core.credentials.get_credential_store",
                                       return_value=MemoryCredentialStore()))
        Settings._instance = None
        ProjectManager._instance = None
        self.addCleanup(setattr, Settings, "_instance", None)
        self.addCleanup(setattr, ProjectManager, "_instance", None)

        self.settings = Settings()
        self.mgr = ProjectManager()
        self.project = self.mgr.create("Trap Metal Demo")

    # ── Dirty tracking ─────────────────────────────────────────────────────────

    def test_new_project_is_clean_until_edited(self):
        self.assertFalse(self.mgr.is_dirty)
        self.project.notes = "verse idea"
        self.assertTrue(self.mgr.is_dirty)
        self.mgr.save()
        self.assertFalse(self.mgr.is_dirty)

    def test_reopening_a_project_is_clean(self):
        self.project.notes = "chorus"
        self.mgr.save()
        reopened = self.mgr.open(self.project.id)
        self.assertIsNotNone(reopened)
        self.assertFalse(self.mgr.is_dirty)

    # ── Versions ───────────────────────────────────────────────────────────────

    def test_version_snapshot_captures_the_current_state(self):
        self.project.notes = "first take"
        version = self.mgr.create_version("First take")
        self.assertIsNotNone(version)

        payload = self.mgr.read_version_payload(version.version)
        self.assertEqual(payload["notes"], "first take")

        # A later edit must not change the stored snapshot.
        self.project.notes = "second take"
        self.mgr.save()
        payload = self.mgr.read_version_payload(version.version)
        self.assertEqual(payload["notes"], "first take")

    def test_preview_does_not_change_the_open_project(self):
        self.project.notes = "original"
        self.project.tempo = 96.0
        v1 = self.mgr.create_version("Original")
        self.project.notes = "changed"
        self.project.tempo = 174.0
        self.mgr.save()

        preview = self.mgr.version_preview(v1.version)
        self.assertEqual(preview["notes"], "original")
        self.assertEqual(preview["tempo"], 96.0)
        self.assertEqual(self.mgr.current.notes, "changed")
        self.assertEqual(self.mgr.current.tempo, 174.0)

    def test_restore_snapshots_current_state_first(self):
        self.project.notes = "take one"
        v1 = self.mgr.create_version("Take one")
        self.project.notes = "take two"
        self.mgr.save()

        restored = self.mgr.restore_version(v1.version)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.notes, "take one")
        self.assertEqual(restored.id, self.project.id)

        pre = [v for v in restored.versions if v.kind == VERSION_KIND_PRE_RESTORE]
        self.assertEqual(len(pre), 1)
        undo = self.mgr.read_version_payload(pre[0].version)
        self.assertEqual(undo["notes"], "take two")

    def test_restore_survives_reopening_the_project(self):
        self.project.notes = "keeper"
        v1 = self.mgr.create_version("Keeper")
        self.project.notes = "scratch"
        self.mgr.save()
        self.mgr.restore_version(v1.version)

        self.mgr.close()
        reopened = self.mgr.open(self.project.id)
        self.assertEqual(reopened.notes, "keeper")

    def test_missing_snapshot_cannot_be_restored(self):
        v1 = self.mgr.create_version("Gone")
        path = self.mgr.version_dir(self.project.id, v1.version) / "project.json"
        path.unlink()
        self.assertIsNone(self.mgr.version_preview(v1.version))
        self.assertIsNone(self.mgr.restore_version(v1.version))

    # ── Retention ──────────────────────────────────────────────────────────────

    def test_retention_prunes_oldest_autosaves_first(self):
        self.settings.set("general.max_project_versions", 4)
        for i in range(3):
            self.project.notes = f"manual {i}"
            self.mgr.create_version(f"Manual {i}", kind=VERSION_KIND_MANUAL)
        for i in range(4):
            self.project.notes = f"auto {i}"
            self.mgr.create_version(f"Auto {i}", kind=VERSION_KIND_AUTO)

        kinds = [v.kind for v in self.mgr.current.versions]
        self.assertEqual(len(kinds), 4)
        self.assertEqual(kinds.count(VERSION_KIND_AUTO), 1)
        self.assertEqual(kinds.count(VERSION_KIND_MANUAL), 3)

        # Pruned snapshot directories are gone, kept ones remain.
        for ver in self.mgr.current.versions:
            self.assertTrue(
                (self.mgr.version_dir(self.project.id, ver.version) / "project.json").is_file()
            )

    def test_pre_restore_versions_are_never_pruned(self):
        self.settings.set("general.max_project_versions", 2)
        v1 = self.mgr.create_version("Base", kind=VERSION_KIND_MANUAL)
        self.project.notes = "changed"
        self.mgr.restore_version(v1.version)
        for i in range(5):
            self.project.notes = f"auto {i}"
            self.mgr.create_version(f"Auto {i}", kind=VERSION_KIND_AUTO)

        kinds = [v.kind for v in self.mgr.current.versions]
        self.assertIn(VERSION_KIND_PRE_RESTORE, kinds)

    # ── Autosave coordinator ───────────────────────────────────────────────────

    def test_interval_is_clamped(self):
        self.settings.set("general.auto_save_interval", 1)
        self.assertEqual(resolve_interval(self.settings), MIN_INTERVAL_SECONDS)
        self.settings.set("general.auto_save_interval", 99999)
        self.assertEqual(resolve_interval(self.settings), MAX_INTERVAL_SECONDS)
        self.settings.set("general.auto_save_interval", "nonsense")
        self.assertEqual(resolve_interval(self.settings), 60)

    def test_tick_saves_only_when_dirty(self):
        coordinator = AutosaveCoordinator(self.mgr, self.settings)
        skipped: list[str] = []
        saved: list[int] = []
        coordinator.skipped.connect(skipped.append)
        coordinator.autosaved.connect(lambda v, d: saved.append(v))

        self.assertIsNone(coordinator.tick())
        self.assertTrue(skipped)
        self.assertEqual(saved, [])

        self.project.notes = "riff idea"
        version = coordinator.tick()
        self.assertIsNotNone(version)
        self.assertEqual(version.kind, VERSION_KIND_AUTO)
        self.assertEqual(saved, [version.version])
        self.assertFalse(self.mgr.is_dirty)

        # Clean again: nothing further is written.
        self.assertIsNone(coordinator.tick())

    def test_autosaved_state_survives_a_simulated_crash(self):
        coordinator = AutosaveCoordinator(self.mgr, self.settings)
        self.project.notes = "unsaved chorus"
        self.project.tempo = 155.0
        self.assertIsNotNone(coordinator.tick())

        # Simulate a crash: drop all in-memory state and reload from disk.
        ProjectManager._instance = None
        fresh = ProjectManager()
        recovered = fresh.open(self.project.id)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.notes, "unsaved chorus")
        self.assertEqual(recovered.tempo, 155.0)
        self.assertTrue(
            any(v.kind == VERSION_KIND_AUTO for v in recovered.versions)
        )

    def test_disabled_autosave_does_nothing(self):
        self.settings.set("general.auto_save_enabled", False)
        coordinator = AutosaveCoordinator(self.mgr, self.settings)
        self.project.notes = "will not be saved by the timer"
        self.assertIsNone(coordinator.tick())
        coordinator.start()
        self.assertFalse(coordinator.is_active())

    def test_interval_change_restarts_the_timer(self):
        coordinator = AutosaveCoordinator(self.mgr, self.settings)
        coordinator.start()
        self.assertTrue(coordinator.is_active())
        self.settings.set("general.auto_save_interval", 30)
        self.assertEqual(coordinator.interval_seconds, 30)
        self.assertTrue(coordinator.is_active())
        coordinator.stop()

    def test_legacy_versions_without_kind_still_load(self):
        v1 = self.mgr.create_version("Legacy")
        meta = Path(self.mgr._projects_dir) / self.project.id / "project.json"
        data = json.loads(meta.read_text(encoding="utf-8"))
        for entry in data["versions"]:
            entry.pop("kind", None)
            entry["auto_save"] = True
        meta.write_text(json.dumps(data, indent=2), encoding="utf-8")

        self.mgr.close()
        reopened = self.mgr.open(self.project.id)
        self.assertEqual(reopened.versions[0].kind, VERSION_KIND_AUTO)
        self.assertEqual(reopened.versions[0].version, v1.version)


if __name__ == "__main__":
    unittest.main()
