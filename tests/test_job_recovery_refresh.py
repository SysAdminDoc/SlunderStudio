import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.job_state import JobStatus, JobStore
from ui.batch_view import BatchView


class JobRecoveryRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_batch_banner_refresh_does_not_recover_live_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = JobStore(root / "jobs", cleanup_roots=[root])
            with mock.patch("ui.batch_view.JobStore", return_value=store):
                view = BatchView()
            self.addCleanup(view.deleteLater)

            record = store.create("song_generation", "Live render")
            store.mark_running(record.id, "Rendering")

            view.refresh_recoverable_jobs()

            current = store.get(record.id)
            self.assertEqual(current.status, JobStatus.RUNNING)


if __name__ == "__main__":
    unittest.main()
