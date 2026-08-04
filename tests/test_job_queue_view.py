import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.job_state import JobStatus, JobStore
from ui.job_queue_view import JobQueueView
from ui.song_forge_view import SongForgeView


class JobQueueViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_startup_recovers_stale_jobs_and_exports_completed_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = JobStore(root / "jobs")
            running = store.create("song_generation", "Interrupted")
            store.mark_running(running.id, "Rendering")
            output = root / "render.wav"
            output.write_bytes(b"audio")
            completed = store.create("song_generation", "Completed")
            store.mark_completed(completed.id, outputs={"paths": [str(output)]})

            view = JobQueueView(job_store=store)
            try:
                self.assertEqual(store.get(running.id).status, JobStatus.RECOVERABLE)
                self.assertGreaterEqual(view._jobs.count(), 2)
                self.assertEqual(view._outputs.count(), 1)
                written = view.export_selected_to(root / "exports")
                self.assertEqual(Path(written[0]).read_bytes(), b"audio")
            finally:
                view.deleteLater()

    def test_retry_emits_new_queued_record_for_registered_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            failed = store.create(
                "song_generation",
                "Failed Song",
                metadata={"replay": {"lyrics": "hello"}},
            )
            store.mark_failed(failed.id, "failed", metadata=failed.metadata)
            view = JobQueueView(job_store=store)
            requeued = []
            view.job_requeued.connect(requeued.append)
            try:
                item = next(
                    view._jobs.item(row)
                    for row in range(view._jobs.count())
                    if view._jobs.item(row).data(Qt.ItemDataRole.UserRole) == failed.id
                )
                view._jobs.setCurrentItem(item)
                view._requeue_selected(resume=False)

                self.assertEqual(len(requeued), 1)
                self.assertEqual(requeued[0].status, JobStatus.QUEUED)
                self.assertEqual(requeued[0].metadata["replay"]["lyrics"], "hello")
                self.assertEqual(store.get(failed.id).status, JobStatus.FAILED)
            finally:
                view.deleteLater()

    def test_song_forge_replay_restores_zero_seed_and_advanced_tags(self):
        view = SongForgeView()
        try:
            record = SimpleNamespace(
                id="queued-song",
                kind="song_generation",
                label="Resume: Song",
                metadata={
                    "replay": {
                        "advanced": True,
                        "lyrics": "[Verse] hello",
                        "style_tags": "dark synth-pop, female vocals",
                        "duration": 180,
                        "shift": 3.5,
                        "infer_steps": 10,
                        "seed": 0,
                        "batch_count": 2,
                        "long_form": True,
                        "mode": "batch",
                    }
                },
            )
            with mock.patch.object(view, "_on_generate") as generate:
                view._on_queue_job_requeued(record)

            self.assertEqual(view._queue_resume_job_id, record.id)
            self.assertEqual(view._seed_spin.value(), 0)
            self.assertEqual(view._batch_spin.value(), 2)
            self.assertEqual(view._tag_browser.get_tags(), "dark synth-pop, female vocals")
            generate.assert_called_once_with()
        finally:
            view.deleteLater()


if __name__ == "__main__":
    unittest.main()
