import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.admission import (
    AdmissionBusyError,
    AdmissionCancelledError,
    AdmissionController,
    admission_kind_for_job,
)
from core.job_state import JobStatus, JobStore
from core.workers import InferenceWorker


class AdmissionControllerTests(unittest.TestCase):
    def test_downloads_are_bounded_and_duplicate_model_keys_are_rejected(self):
        controller = AdmissionController(max_downloads=1, max_inference=1)
        first = controller.acquire(AdmissionController.DOWNLOAD, key="alpha")
        waiting = threading.Event()
        acquired = threading.Event()
        result = []

        def acquire_second():
            waiting.set()
            lease = controller.acquire(AdmissionController.DOWNLOAD, key="beta")
            result.append(lease)
            acquired.set()

        thread = threading.Thread(target=acquire_second)
        thread.start()
        self.assertTrue(waiting.wait(timeout=2))
        time.sleep(0.05)
        self.assertFalse(acquired.is_set())
        with self.assertRaises(AdmissionBusyError):
            controller.acquire(AdmissionController.DOWNLOAD, key="alpha")
        self.assertEqual(1, controller.snapshot().active[AdmissionController.DOWNLOAD])

        first.release()
        self.assertTrue(acquired.wait(timeout=2))
        result[0].release()
        thread.join(timeout=2)
        self.assertEqual(
            0,
            controller.snapshot().active[AdmissionController.DOWNLOAD],
        )

    def test_waiting_inference_honors_cancellation_and_releases_state(self):
        controller = AdmissionController(max_downloads=1, max_inference=1)
        first = controller.acquire(AdmissionController.INFERENCE)
        cancel = threading.Event()
        result = []

        def wait_for_slot():
            try:
                controller.acquire(
                    AdmissionController.INFERENCE,
                    cancel_event=cancel,
                )
            except AdmissionCancelledError as exc:
                result.append(exc)

        thread = threading.Thread(target=wait_for_slot)
        thread.start()
        time.sleep(0.05)
        cancel.set()
        thread.join(timeout=2)
        first.release()

        self.assertEqual(1, len(result))
        self.assertEqual(0, controller.snapshot().queued[AdmissionController.INFERENCE])

    def test_job_kind_mapping_only_admits_model_work(self):
        self.assertEqual("inference", admission_kind_for_job("song_generation"))
        self.assertEqual("inference", admission_kind_for_job("model_update"))
        self.assertIsNone(admission_kind_for_job("audio_export"))
        self.assertIsNone(admission_kind_for_job("mixer_import"))

    def test_inference_worker_waits_for_central_capacity(self):
        controller = AdmissionController(max_downloads=1, max_inference=1)
        first = controller.acquire(AdmissionController.INFERENCE)
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            started = threading.Event()
            worker = InferenceWorker(
                lambda **_kwargs: started.set(),
                job_kind="song_generation",
                job_label="Queued generation",
                job_store=store,
            )
            with mock.patch("core.workers.global_admission_controller", return_value=controller):
                thread = threading.Thread(target=worker.run)
                thread.start()
                time.sleep(0.05)
                self.assertFalse(started.is_set())
                worker.cancel()
                thread.join(timeout=2)

            first.release()
            self.assertFalse(started.is_set())
            self.assertEqual(JobStatus.CANCELLED, store.get(worker.job_id).status)


if __name__ == "__main__":
    unittest.main()
