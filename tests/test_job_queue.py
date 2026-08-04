import tempfile
import unittest
from pathlib import Path

from core.job_queue import estimate_job_resources, export_selected_outputs
from core.job_state import JobStatus, JobStore
from core.workers import InferenceWorker


class JobQueueTests(unittest.TestCase):
    def test_requeue_retains_audit_record_and_marks_retry_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            failed = store.create(
                "song_generation",
                "Song render",
                inputs={"duration": 180, "batch_count": 2},
                metadata={"replay": {"lyrics": "[Verse] hello"}},
            )
            store.mark_failed(failed.id, "render failed", metadata=failed.metadata)

            retry = store.requeue(failed.id)

            self.assertIsNotNone(retry)
            self.assertNotEqual(retry.id, failed.id)
            self.assertEqual(store.get(failed.id).status, JobStatus.FAILED)
            self.assertEqual(retry.status, JobStatus.QUEUED)
            self.assertEqual(retry.metadata["retry_of"], failed.id)
            self.assertFalse(retry.metadata["resume"])
            self.assertEqual(retry.metadata["replay"]["lyrics"], "[Verse] hello")

    def test_retry_adopts_queued_record_and_preserves_replay_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            created = store.create(
                "song_generation",
                "Interrupted song",
                metadata={"replay": {"seed": 0}},
            )

            def fail(**_kwargs):
                raise RuntimeError("engine unavailable")

            failed_worker = InferenceWorker(
                fail,
                job_kind=created.kind,
                job_label=created.label,
                job_inputs=created.inputs,
                job_metadata=created.metadata,
                job_store=store,
            )
            failed_worker.run()
            failed = store.get(failed_worker.job_id)
            resumed = store.requeue(failed.id, resume=False)
            ran = []

            worker = InferenceWorker(
                lambda **_kwargs: ran.append(True),
                job_store=store,
                resume_job_id=resumed.id,
            )
            worker.run()

            self.assertEqual(ran, [True])
            self.assertEqual(worker.job_id, resumed.id)
            self.assertEqual(store.get(resumed.id).status, JobStatus.COMPLETED)
            self.assertEqual(store.get(failed.id).metadata["replay"]["seed"], 0)
            self.assertEqual(store.list_records(kind="song_generation")[0].id, resumed.id)

            interrupted = store.create("song_generation", "Interrupted")
            store.mark_running(interrupted.id, "Rendering")
            store.recover_stale_jobs()
            resume = store.requeue(interrupted.id, resume=True)
            self.assertTrue(resume.metadata["resume"])

    def test_resource_estimate_uses_declared_values_then_bounded_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            derived = store.create(
                "song_generation",
                "Derived",
                inputs={"duration": 180, "batch_count": 2},
            )
            declared = store.create(
                "song_generation",
                "Declared",
                inputs={"duration": 1},
                metadata={
                    "resource_estimate": {
                        "duration_minutes": 4.5,
                        "output_gb": 0.75,
                        "ram_gb": 8,
                        "vram_gb": 6,
                        "basis": "Measured on RTX 3080",
                    }
                },
            )

            derived_estimate = estimate_job_resources(derived)
            declared_estimate = estimate_job_resources(declared)

            self.assertEqual(derived_estimate.duration_minutes, 6.0)
            self.assertGreater(derived_estimate.output_gb, 0)
            self.assertEqual(declared_estimate.duration_minutes, 4.5)
            self.assertEqual(declared_estimate.output_gb, 0.75)
            self.assertEqual(declared_estimate.basis, "Measured on RTX 3080")

    def test_export_selected_outputs_is_declared_and_collision_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "render.wav"
            source.write_bytes(b"render")
            destination = root / "exports"
            destination.mkdir()
            (destination / source.name).write_bytes(b"existing")
            store = JobStore(root / "jobs")
            record = store.create("song_generation", "Completed render")
            store.mark_completed(record.id, outputs={"paths": [str(source)]})
            record = store.get(record.id)

            written = export_selected_outputs([record], [source], destination)

            self.assertEqual(written, [str(destination / "render (2).wav")])
            self.assertEqual(Path(written[0]).read_bytes(), b"render")
            with self.assertRaises(ValueError):
                export_selected_outputs([record], [root / "not-declared.wav"], destination)


if __name__ == "__main__":
    unittest.main()
