import json
import os
import subprocess
import sys
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
    @staticmethod
    def _process_environment():
        root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(root), environment.get("PYTHONPATH", "")) if item
        )
        environment["QT_QPA_PLATFORM"] = "offscreen"
        return root, environment

    def _spawn_slot_holder(self, state_path, marker_path, kind, key, seconds):
        root, environment = self._process_environment()
        script = """
import os
import sys
import time
from pathlib import Path

from core.admission import AdmissionController

controller = AdmissionController(
    max_downloads=1,
    max_inference=1,
    shared_state_path=Path(sys.argv[1]),
    heartbeat_interval=0.05,
    stale_after_seconds=0.3,
)
with controller.acquire(sys.argv[3], key=sys.argv[4]):
    Path(sys.argv[2]).write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(float(sys.argv[5]))
"""
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(state_path),
                str(marker_path),
                kind,
                key,
                str(seconds),
            ],
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    @staticmethod
    def _wait_for_path(path: Path, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return True
            time.sleep(0.02)
        return path.exists()

    def _assert_processes_share_slot(self, kind, key_one, key_two):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / f"{kind}.json"
            first_marker = root / "first.acquired"
            second_marker = root / "second.acquired"
            first = self._spawn_slot_holder(
                state_path,
                first_marker,
                kind,
                key_one,
                0.6,
            )
            second = None
            try:
                self.assertTrue(self._wait_for_path(first_marker))
                second = self._spawn_slot_holder(
                    state_path,
                    second_marker,
                    kind,
                    key_two,
                    0.1,
                )
                time.sleep(0.15)
                self.assertFalse(second_marker.exists())
                first_stdout, first_stderr = first.communicate(timeout=5)
                second_stdout, second_stderr = second.communicate(timeout=5)
                self.assertEqual(
                    0,
                    first.returncode,
                    f"first process failed: {first_stdout}\n{first_stderr}",
                )
                self.assertEqual(
                    0,
                    second.returncode,
                    f"second process failed: {second_stdout}\n{second_stderr}",
                )
                self.assertTrue(second_marker.exists())
            finally:
                if first.poll() is None:
                    first.kill()
                    first.wait(timeout=5)
                if second is not None and second.poll() is None:
                    second.kill()
                    second.wait(timeout=5)

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

    def test_independent_processes_share_inference_and_download_capacity(self):
        self._assert_processes_share_slot(
            AdmissionController.INFERENCE,
            "",
            "",
        )
        self._assert_processes_share_slot(
            AdmissionController.DOWNLOAD,
            "model-one",
            "model-two",
        )

    def test_dead_process_lease_is_reclaimed_before_new_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "stale.json"
            marker = root / "acquired"
            process = self._spawn_slot_holder(
                state_path,
                marker,
                AdmissionController.INFERENCE,
                "",
                60,
            )
            try:
                self.assertTrue(self._wait_for_path(marker))
                process.terminate()
                process.wait(timeout=5)

                controller = AdmissionController(
                    max_downloads=1,
                    max_inference=1,
                    shared_state_path=state_path,
                    heartbeat_interval=0.05,
                    stale_after_seconds=60,
                )
                lease = controller.acquire(AdmissionController.INFERENCE, timeout=2)
                lease.release()
                self.assertEqual(
                    0,
                    controller.snapshot().active[AdmissionController.INFERENCE],
                )
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

    def test_job_ledger_updates_are_serialized_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_root = Path(tmp) / "jobs"
            root, environment = self._process_environment()
            script = """
import sys
from pathlib import Path

from core.job_state import JobStore

store = JobStore(Path(sys.argv[1]))
record = store.create("song_generation", sys.argv[2], inputs={"worker": sys.argv[2]})
store.mark_completed(record.id, metadata={"worker": sys.argv[2]})
"""
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", script, str(jobs_root), f"worker-{index}"],
                    cwd=root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for index in range(8)
            ]
            for process in processes:
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(
                    0,
                    process.returncode,
                    f"ledger process failed: {stdout}\n{stderr}",
                )

            payload = json.loads((jobs_root / "jobs.json").read_text(encoding="utf-8"))
            self.assertEqual(1, payload["schema_version"])
            self.assertEqual(8, len(payload["jobs"]))
            self.assertEqual(
                {f"worker-{index}" for index in range(8)},
                {record["label"] for record in payload["jobs"]},
            )
            self.assertTrue(all(record["status"] == JobStatus.COMPLETED for record in payload["jobs"]))


if __name__ == "__main__":
    unittest.main()
