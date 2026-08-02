import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.credentials import MemoryCredentialStore
from core.job_state import JobStatus, JobStore
from core.project import ProjectManager
from core.retention import (
    CATEGORIES,
    CATEGORY_CRASH_LOGS,
    CATEGORY_JOBS,
    CATEGORY_JOB_LOGS,
    CATEGORY_PROJECT_VERSIONS,
    CATEGORY_SETTINGS_BACKUPS,
    DEFAULT_POLICIES,
    RecoveryCenter,
    RetentionItem,
    RetentionPolicy,
    load_policy,
    plan_cleanup,
)
from core.settings import Settings

DAY = 86400.0


def item(identifier: str, age_days: float, size: int = 0,
         protected: bool = False, now: float = 0.0) -> RetentionItem:
    return RetentionItem(
        category="test",
        identifier=identifier,
        label=identifier,
        timestamp=(now or time.time()) - age_days * DAY,
        size_bytes=size,
        protected=protected,
    )


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.now = 1_800_000_000.0

    def test_age_limit_keeps_the_recent_and_drops_the_old(self):
        items = [item("new", 1, now=self.now), item("old", 45, now=self.now)]
        plan = plan_cleanup(items, RetentionPolicy("test", max_age_days=30), now=self.now)
        self.assertEqual([i.identifier for i in plan.remove], ["old"])
        self.assertEqual([i.identifier for i in plan.keep], ["new"])

    def test_count_limit_keeps_the_newest(self):
        items = [item(f"i{n}", n, now=self.now) for n in range(5)]
        plan = plan_cleanup(items, RetentionPolicy("test", max_count=2), now=self.now)
        self.assertEqual([i.identifier for i in plan.keep], ["i0", "i1"])
        self.assertEqual(len(plan.remove), 3)

    def test_size_limit_trims_the_oldest_over_budget(self):
        items = [item(f"i{n}", n, size=6_000_000, now=self.now) for n in range(4)]
        plan = plan_cleanup(items, RetentionPolicy("test", max_total_mb=12), now=self.now)
        self.assertEqual([i.identifier for i in plan.keep], ["i0", "i1"])
        self.assertEqual(len(plan.remove), 2)

    def test_size_limit_keeps_at_least_one_item(self):
        items = [item("huge", 1, size=999_000_000, now=self.now)]
        plan = plan_cleanup(items, RetentionPolicy("test", max_total_mb=1), now=self.now)
        self.assertEqual(len(plan.keep), 1)
        self.assertEqual(plan.remove, [])

    def test_protected_items_are_never_selected(self):
        items = [
            item("locked", 900, protected=True, now=self.now),
            item("old", 900, now=self.now),
        ]
        plan = plan_cleanup(items, RetentionPolicy("test", max_age_days=1, max_count=1),
                            now=self.now)
        self.assertEqual([i.identifier for i in plan.protected], ["locked"])
        self.assertEqual([i.identifier for i in plan.remove], ["old"])

    def test_zero_means_no_limit(self):
        items = [item(f"i{n}", n * 100, size=10 ** 9, now=self.now) for n in range(5)]
        plan = plan_cleanup(items, RetentionPolicy("test"), now=self.now)
        self.assertEqual(plan.remove, [])
        self.assertEqual(len(plan.keep), 5)

    def test_every_category_has_a_safe_default(self):
        for category in CATEGORIES:
            with self.subTest(category=category):
                policy = DEFAULT_POLICIES[category]
                self.assertTrue(
                    policy.max_age_days or policy.max_count or policy.max_total_mb,
                    "category has no bound at all",
                )
                self.assertIn(policy.category, CATEGORIES)


class RecoveryCenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = self.root / "config"
        self.config.mkdir(parents=True, exist_ok=True)

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch("core.settings.get_config_dir", return_value=self.config))
        stack.enter_context(mock.patch("core.settings.get_default_output_dir", return_value=self.root / "out"))
        stack.enter_context(mock.patch("core.settings.get_default_cache_dir", return_value=self.root / "models"))
        stack.enter_context(mock.patch("core.project.get_config_dir", return_value=str(self.config)))
        stack.enter_context(mock.patch("core.retention.get_config_dir", return_value=self.config))
        stack.enter_context(mock.patch("core.credentials.get_credential_store",
                                       return_value=MemoryCredentialStore()))
        Settings._instance = None
        ProjectManager._instance = None
        self.addCleanup(setattr, Settings, "_instance", None)
        self.addCleanup(setattr, ProjectManager, "_instance", None)

        self.settings = Settings()
        self.jobs = JobStore(self.config / "jobs")
        self.projects = ProjectManager()
        self.center = RecoveryCenter(self.settings, job_store=self.jobs,
                                     project_manager=self.projects)

    def _age(self, path: Path, days: float):
        stamp = time.time() - days * DAY
        os.utime(path, (stamp, stamp))

    def test_active_and_recoverable_jobs_are_protected(self):
        done = self.jobs.create("render", "finished")
        self.jobs.mark_completed(done.id)
        running = self.jobs.create("render", "running")
        self.jobs.mark_running(running.id)
        stale = self.jobs.create("render", "stale")
        self.jobs.mark_running(stale.id)
        self.jobs.recover_stale_jobs()

        items = self.center.collect(CATEGORY_JOBS)
        protected = {i.identifier for i in items if i.protected}
        self.assertIn(running.id, protected)
        self.assertIn(stale.id, protected)
        self.assertNotIn(done.id, protected)

        plan = plan_cleanup(
            items,
            RetentionPolicy(CATEGORY_JOBS, max_age_days=1),
            now=time.time() + 10 * DAY,
        )
        self.assertEqual([i.identifier for i in plan.remove], [done.id])

    def test_preview_removes_nothing(self):
        logs = self.config / "jobs" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        old = logs / "old.log"
        old.write_text("x" * 100, encoding="utf-8")
        self._age(old, 400)

        plan = self.center.preview(CATEGORY_JOB_LOGS)
        self.assertEqual([i.identifier for i in plan.remove], ["old.log"])
        self.assertTrue(old.exists(), "preview must not delete anything")

        self.center.clean(CATEGORY_JOB_LOGS)
        self.assertFalse(old.exists())

    def test_current_crash_log_is_never_removed(self):
        current = self.config / "crash.log"
        current.write_text("boom", encoding="utf-8")
        self._age(current, 9999)
        rotated = self.config / "crash.20200101.log"
        rotated.write_text("older boom", encoding="utf-8")
        self._age(rotated, 9999)

        removed = self.center.clean(CATEGORY_CRASH_LOGS)
        self.assertTrue(current.exists())
        self.assertFalse(rotated.exists())
        self.assertEqual([i.identifier for i in removed], ["crash.20200101.log"])

    def test_settings_backups_are_bounded_by_count(self):
        backups = self.config / "backups"
        backups.mkdir(parents=True, exist_ok=True)
        for n in range(60):
            path = backups / f"config.json.{n:03d}.pre-save.bak"
            path.write_text("{}", encoding="utf-8")
            self._age(path, n)

        self.settings.set("retention.settings_backups.max_count", 10)
        self.center.clean(CATEGORY_SETTINGS_BACKUPS)
        self.assertEqual(len(list(backups.glob("*.bak"))), 10)

    def test_project_versions_protect_newest_and_pre_restore(self):
        project = self.projects.create("Retention Project")
        base = self.projects.create_version("Base")
        project.notes = "changed"
        self.projects.restore_version(base.version)
        for n in range(6):
            project.notes = f"auto {n}"
            self.projects.create_version(f"Auto {n}", auto_save=True)

        items = self.center.collect(CATEGORY_PROJECT_VERSIONS)
        reasons = {i.identifier: i.protected_reason for i in items if i.protected}
        self.assertIn("pre-restore snapshot", reasons.values())
        self.assertIn("newest version", reasons.values())

        self.settings.set("retention.project_versions.max_count", 2)
        self.center.clean(CATEGORY_PROJECT_VERSIONS)
        remaining = self.projects.current.versions
        self.assertTrue(any(v.kind == "pre-restore" for v in remaining))

    def test_policy_reads_settings_and_falls_back_on_junk(self):
        self.settings.set("retention.crash_logs.max_count", 7)
        self.assertEqual(load_policy(CATEGORY_CRASH_LOGS, self.settings).max_count, 7)
        self.settings.set("retention.crash_logs.max_age_days", "not a number")
        self.assertEqual(
            load_policy(CATEGORY_CRASH_LOGS, self.settings).max_age_days,
            DEFAULT_POLICIES[CATEGORY_CRASH_LOGS].max_age_days,
        )

    def test_preview_all_covers_every_category(self):
        plans = self.center.preview_all()
        self.assertEqual(set(plans), set(CATEGORIES))
        for category, plan in plans.items():
            with self.subTest(category=category):
                self.assertEqual(plan.category, category)
                self.assertIsInstance(plan.summary(), str)

    def test_job_store_refuses_to_delete_an_active_record(self):
        running = self.jobs.create("render", "running")
        self.jobs.mark_running(running.id)
        self.assertFalse(self.jobs.delete(running.id))
        self.assertIsNotNone(self.jobs.get(running.id))

        done = self.jobs.create("render", "done")
        self.jobs.mark_completed(done.id)
        self.assertTrue(self.jobs.delete(done.id))
        self.assertIsNone(self.jobs.get(done.id))


if __name__ == "__main__":
    unittest.main()
