import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import core.workers as workers
from core.job_state import JobStatus, JobStore
from core.workers import CancelledJobError, InferenceWorker, active_workers


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def _wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def test_cancel_during_run_marks_job_cancelled_and_emits_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs", cleanup_roots=[Path(tmp)])
            started = threading.Event()
            stopped = threading.Event()
            cancelled = []

            def task(cancel_event=None, **_kwargs):
                started.set()
                while not cancel_event.is_set():
                    time.sleep(0.005)
                return {"partial": True}

            with mock.patch.object(workers, "JobLog", return_value=mock.Mock()):
                worker = InferenceWorker(
                    task,
                    job_kind="worker-test",
                    job_label="Cancellation",
                    job_store=store,
                )
                worker.cancelled.connect(lambda: cancelled.append(True))
                worker.thread_stopped.connect(stopped.set)
                worker.start()

                self.assertTrue(started.wait(2), "worker did not start")
                worker.cancel()
                self.assertTrue(
                    self._wait_for(stopped.is_set, timeout=3),
                    "worker did not report thread stop",
                )
                worker.wait(3000)

            record = store.get(worker.job_id)
            self.assertFalse(worker.isRunning())
            self.assertTrue(cancelled)
            self.assertEqual(record.status, JobStatus.CANCELLED)
            self.assertNotIn(worker, active_workers())

    def test_cancelled_job_error_preserves_verified_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kept = root / "kept.wav"
            removed = root / "removed.wav"
            kept.write_bytes(b"finished")
            removed.write_bytes(b"partial")
            store = JobStore(root / "jobs", cleanup_roots=[root])

            def task(**_kwargs):
                raise CancelledJobError(
                    "stopped after first variation",
                    outputs=[str(kept), str(removed)],
                    preserved=[str(kept)],
                )

            with mock.patch.object(workers, "JobLog", return_value=mock.Mock()):
                worker = InferenceWorker(
                    task,
                    job_kind="worker-test",
                    job_label="Preserved output",
                    job_store=store,
                )
                worker.run()

            record = store.get(worker.job_id)
            self.assertTrue(kept.exists())
            self.assertFalse(removed.exists())
            self.assertEqual(record.status, JobStatus.CANCELLED)
            self.assertEqual(record.outputs["preserved_paths"], [str(kept)])

    def test_semantic_failure_is_recorded_without_being_reported_as_exception(self):
        result = SimpleNamespace(
            is_success=False,
            error="model returned no audio",
            job_metadata=lambda: {"stage": "render"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            finished = []
            errors = []
            with mock.patch.object(workers, "JobLog", return_value=mock.Mock()):
                worker = InferenceWorker(
                    lambda **_kwargs: result,
                    job_kind="worker-test",
                    job_label="Semantic failure",
                    job_store=store,
                )
                worker.finished.connect(finished.append)
                worker.error.connect(errors.append)
                worker.run()

            record = store.get(worker.job_id)
            self.assertEqual(finished, [result])
            self.assertEqual(errors, [])
            self.assertEqual(record.status, JobStatus.FAILED)
            self.assertEqual(record.error, "model returned no audio")
            self.assertEqual(record.metadata, {"stage": "render"})

    def test_exception_failure_emits_error_and_marks_job_failed(self):
        errors = []
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            with mock.patch.object(workers, "JobLog", return_value=mock.Mock()):
                worker = InferenceWorker(
                    lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad input")),
                    job_kind="worker-test",
                    job_label="Exception",
                    job_store=store,
                )
                worker.error.connect(errors.append)
                worker.run()

            record = store.get(worker.job_id)
            self.assertEqual(errors, ["ValueError: bad input"])
            self.assertEqual(record.status, JobStatus.FAILED)
            self.assertIn("ValueError: bad input", record.error)

    def test_progress_signal_is_live_but_job_persistence_is_throttled(self):
        store = mock.Mock()
        store.create.return_value = SimpleNamespace(id="job-1")
        emitted = []
        with mock.patch.object(workers, "JobLog", return_value=mock.Mock()), mock.patch.object(
            workers.time,
            "monotonic",
            side_effect=[10.0, 10.05, 10.11],
        ):
            worker = InferenceWorker(
                lambda **_kwargs: None,
                job_kind="worker-test",
                job_label="Progress",
                job_store=store,
            )
            worker.progress.connect(emitted.append)
            worker._emit_progress(10)
            worker._emit_progress(20)
            worker._emit_progress(30)

        self.assertEqual(emitted, [10, 20, 30])
        self.assertEqual(
            store.update_progress.call_args_list,
            [mock.call("job-1", 10), mock.call("job-1", 30)],
        )


if __name__ == "__main__":
    unittest.main()
