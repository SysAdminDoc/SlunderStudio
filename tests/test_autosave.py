import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.autosave import (
    MAX_INTERVAL_SECONDS,
    MIN_INTERVAL_SECONDS,
    AutosaveCoordinator,
    resolve_interval,
)
from core.project import ProjectVersion


class _Settings:
    def __init__(self, **values):
        self.values = {
            "general.auto_save_interval": 60,
            "general.auto_save_enabled": True,
            **values,
        }
        self.callbacks = []

    def get(self, key, default=None):
        return self.values.get(key, default)

    def on_change(self, callback):
        self.callbacks.append(callback)

    def set(self, key, value):
        old = self.values.get(key)
        self.values[key] = value
        for callback in self.callbacks:
            callback(key, value, old)


class _Projects:
    def __init__(self):
        self.current = None
        self.is_dirty = False
        self.next_version = ProjectVersion(version=4, description="Autosaved")
        self.failure = None

    def autosave(self):
        if self.failure:
            raise self.failure
        self.is_dirty = False
        return self.next_version


class AutosaveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_resolve_interval_clamps_invalid_and_extreme_values(self):
        settings = _Settings(**{"general.auto_save_interval": 1})
        self.assertEqual(resolve_interval(settings), MIN_INTERVAL_SECONDS)
        settings.values["general.auto_save_interval"] = 999999
        self.assertEqual(resolve_interval(settings), MAX_INTERVAL_SECONDS)
        settings.values["general.auto_save_interval"] = "invalid"
        self.assertEqual(resolve_interval(settings), 60)

    def test_tick_reports_skip_reasons_and_saves_dirty_project(self):
        settings = _Settings()
        projects = _Projects()
        coordinator = AutosaveCoordinator(projects, settings)
        skipped = []
        saved = []
        coordinator.skipped.connect(skipped.append)
        coordinator.autosaved.connect(lambda version, description: saved.append((version, description)))

        self.assertIsNone(coordinator.tick())
        self.assertIn("No project is open", skipped[-1])
        projects.current = SimpleNamespace(id="project-1")
        self.assertIsNone(coordinator.tick())
        self.assertIn("No unsaved changes", skipped[-1])

        projects.is_dirty = True
        version = coordinator.tick()
        self.assertEqual(version, projects.next_version)
        self.assertEqual(saved, [(4, "Autosaved")])
        self.assertGreater(coordinator.last_autosave, 0)
        coordinator.deleteLater()

    def test_flush_runs_when_interval_autosave_is_disabled(self):
        settings = _Settings(**{"general.auto_save_enabled": False})
        projects = _Projects()
        projects.current = SimpleNamespace(id="project-1")
        projects.is_dirty = True
        coordinator = AutosaveCoordinator(projects, settings)

        self.assertIsNone(coordinator.tick())
        projects.is_dirty = True
        self.assertEqual(coordinator.flush(), projects.next_version)
        self.assertFalse(projects.is_dirty)
        coordinator.start()
        self.assertFalse(coordinator.is_active())
        coordinator.deleteLater()

    def test_save_exception_is_reported_and_reentrancy_is_skipped(self):
        settings = _Settings()
        projects = _Projects()
        projects.current = SimpleNamespace(id="project-1")
        projects.is_dirty = True
        projects.failure = RuntimeError("disk full")
        coordinator = AutosaveCoordinator(projects, settings)
        failures = []
        skipped = []
        coordinator.autosave_failed.connect(failures.append)
        coordinator.skipped.connect(skipped.append)

        self.assertIsNone(coordinator.tick())
        self.assertEqual(failures, ["RuntimeError: disk full"])
        coordinator._running = True
        self.assertIsNone(coordinator.flush())
        self.assertIn("already running", skipped[-1])
        coordinator.deleteLater()

    def test_setting_changes_restart_or_stop_active_timer(self):
        settings = _Settings()
        coordinator = AutosaveCoordinator(_Projects(), settings)
        coordinator.start()
        self.assertTrue(coordinator.is_active())
        settings.set("general.auto_save_interval", 30)
        self.assertEqual(coordinator.interval_seconds, 30)
        self.assertTrue(coordinator.is_active())
        settings.set("general.auto_save_enabled", False)
        self.assertFalse(coordinator.is_active())
        coordinator.deleteLater()


if __name__ == "__main__":
    unittest.main()
