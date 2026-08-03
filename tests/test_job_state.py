import tempfile
import threading
import time
import unittest
from pathlib import Path

from core.job_state import JobLog, JobStatus, JobStore
from core.workers import (
    DownloadWorker,
    InferenceWorker,
    CancelledJobError,
    active_workers,
    shutdown_workers,
)
from engines.ace_step_engine import ACEStepEngine, GenerationParams, GenerationResult
from engines.demucs_engine import SeparationResult, StemResult


class JobStateTests(unittest.TestCase):
    def test_shutdown_workers_cancels_and_joins_running_inference(self):
        started = threading.Event()

        def task(cancel_event=None, **_kwargs):
            started.set()
            while not cancel_event.is_set():
                time.sleep(0.005)
            return None

        worker = InferenceWorker(task)
        worker.start()
        self.assertTrue(started.wait(timeout=2))
        self.assertIn(worker, active_workers())

        self.assertTrue(shutdown_workers(timeout_ms=2_000))
        self.assertFalse(worker.isRunning())
        self.assertNotIn(worker, active_workers())

    def test_failed_separation_result_marks_job_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            failed = SeparationResult(error="corrupt audio")

            worker = InferenceWorker(
                lambda **_kwargs: failed,
                job_kind="stem_separation",
                job_label="Failed separation",
                job_store=store,
            )
            worker.run()
            record = store.get(worker.job_id)

        self.assertFalse(failed.is_success)
        self.assertIsNotNone(record)
        self.assertEqual(record.status, JobStatus.FAILED)
        self.assertEqual(record.error, "corrupt audio")

    def test_separation_with_stems_is_successful(self):
        result = SeparationResult(stems=[StemResult(name="vocals")])

        self.assertTrue(result.is_success)

    def test_stale_active_jobs_become_recoverable_on_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp))
            record = store.create(
                "song_generation",
                "Interrupted render",
                inputs={"duration": 180},
            )
            store.mark_running(record.id, "Rendering")

            recovered = store.recover_stale_jobs()
            current = store.get(record.id)

        self.assertEqual(len(recovered), 1)
        self.assertEqual(current.status, JobStatus.RECOVERABLE)
        self.assertTrue(current.recoverable)

    def test_cancelled_inference_worker_records_state_and_removes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            audio = tmp_path / "render.wav"
            sidecar = tmp_path / "render.wav.provenance.json"
            store = JobStore(tmp_path / "jobs", cleanup_roots=[tmp_path])

            def task(progress_cb=None, step_cb=None, log_cb=None, cancel_event=None):
                audio.write_bytes(b"partial audio")
                sidecar.write_text("{}", encoding="utf-8")
                if progress_cb:
                    progress_cb(50)
                return {"audio_path": str(audio), "provenance_path": str(sidecar)}

            worker = InferenceWorker(
                task,
                job_kind="song_generation",
                job_label="Cancellation test",
                job_store=store,
            )
            worker.cancel()
            worker.run()
            record = store.get(worker.job_id)
            self.assertIsNotNone(record)
            self.assertEqual(record.status, JobStatus.CANCELLED)
            self.assertFalse(audio.exists())
            self.assertFalse(sidecar.exists())

    def test_cleanup_rejects_outside_directories_relative_and_malformed_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            owned_root = root / "owned"
            owned_root.mkdir()
            owned_file = owned_root / "partial.wav"
            owned_file.write_bytes(b"partial")
            owned_directory = owned_root / "folder"
            owned_directory.mkdir()
            outside_file = root / "outside.wav"
            outside_file.write_bytes(b"keep")
            store = JobStore(root / "jobs", cleanup_roots=[owned_root])

            with self.assertLogs("core.job_state", level="WARNING") as captured:
                removed = store.cleanup_outputs({
                    "paths": [
                        str(owned_file),
                        str(owned_directory),
                        str(outside_file),
                        "relative.wav",
                        "\0invalid",
                        42,
                    ]
                })

            self.assertEqual(removed, [str(owned_file.resolve())])
            self.assertFalse(owned_file.exists())
            self.assertTrue(owned_directory.is_dir())
            self.assertTrue(outside_file.exists())
            messages = "\n".join(captured.output)
            self.assertIn("path is not a regular file", messages)
            self.assertIn("path is outside approved roots", messages)
            self.assertIn("path is not absolute", messages)
            self.assertIn("malformed output record", messages)
            self.assertIn("malformed path", messages)

    def test_cleanup_rejects_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            owned_root = root / "owned"
            outside_root = root / "outside"
            owned_root.mkdir()
            outside_root.mkdir()
            outside_file = outside_root / "keep.wav"
            outside_file.write_bytes(b"keep")
            store = JobStore(root / "jobs", cleanup_roots=[owned_root])

            traversal = owned_root / ".." / "outside" / "keep.wav"
            with self.assertLogs("core.job_state", level="WARNING") as captured:
                removed = store.cleanup_outputs([str(traversal)])
            self.assertEqual(removed, [])
            self.assertTrue(outside_file.exists())
            self.assertIn("outside approved roots", "\n".join(captured.output))

            symlink = owned_root / "escape.wav"
            try:
                symlink.symlink_to(outside_file)
            except (NotImplementedError, OSError):
                return
            with self.assertLogs("core.job_state", level="WARNING") as captured:
                removed = store.cleanup_outputs([str(symlink)])
            self.assertEqual(removed, [])
            self.assertTrue(outside_file.exists())
            self.assertTrue(symlink.is_symlink())
            self.assertIn("symbolic links", "\n".join(captured.output))

    def test_download_worker_cancel_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store = JobStore(tmp_path / "jobs")

            def download_fn(model_id, progress_cb=None, speed_cb=None, downloaded_cb=None, cancel_event=None):
                (tmp_path / "partial.bin").write_bytes(b"partial")
                if progress_cb:
                    progress_cb(25)

            worker = DownloadWorker(
                download_fn,
                "test-model",
                model_name="Test Model",
                job_store=store,
            )
            worker.cancel()
            worker.run()
            record = store.get(worker.job_id)
            self.assertIsNotNone(record)
            self.assertEqual(record.status, JobStatus.CANCELLED)
            self.assertTrue(record.recoverable)
            self.assertEqual(record.outputs["model_id"], "test-model")

    def test_long_form_cancel_cleans_rendered_section_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            engine = ACEStepEngine()
            engine._pipeline = object()
            engine._model_loaded = True
            engine._output_dir = tmp_path
            cancel_event = threading.Event()
            rendered: list[Path] = []

            def fake_generate(params, progress_cb=None, cancel_event=None):
                index = len(rendered)
                audio = tmp_path / f"section_{index}.wav"
                sidecar = tmp_path / f"section_{index}.wav.provenance.json"
                audio.write_bytes(b"partial")
                sidecar.write_text("{}", encoding="utf-8")
                rendered.append(audio)
                cancel_event.set()
                return GenerationResult(
                    audio_path=str(audio),
                    provenance_path=str(sidecar),
                    seed=params.seed,
                    params=params,
                )

            engine.generate = fake_generate
            params = GenerationParams(
                lyrics="[Verse]\nLine one\n\n[Chorus]\nLine two",
                style_tags="dark trap",
                duration=180,
                section_crossfade=0,
            )

            with self.assertRaises(CancelledJobError):
                engine.generate_long_form(params, cancel_event=cancel_event)

            for path in rendered:
                self.assertFalse(path.exists())
                self.assertFalse(Path(str(path) + ".provenance.json").exists())


    def test_job_log_persists_bounded_redacted_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = JobLog("test-job-123", root=Path(tmp))
            log.set_device_info({"gpu": "RTX 3080", "vram_gb": 10.0})
            log.set_model_info({"model_id": "ace-step-v1.5"})
            log.add_redact_pattern("secret-lyrics")

            log.info("Starting generation")
            log.info("Using prompt: secret-lyrics about love")
            log.warn("GPU memory low")
            log.error("Generation failed: OOM")

            saved = log.save()
            self.assertTrue(saved.exists())

            import json
            data = json.loads(saved.read_text())
            self.assertEqual(data["job_id"], "test-job-123")
            self.assertEqual(data["device"]["gpu"], "RTX 3080")
            self.assertEqual(data["model"]["model_id"], "ace-step-v1.5")
            self.assertEqual(data["entry_count"], 4)
            self.assertEqual(len(data["entries"]), 4)

            prompt_entry = data["entries"][1]
            self.assertNotIn("secret-lyrics", prompt_entry["m"])
            self.assertIn("[REDACTED]", prompt_entry["m"])

    def test_job_log_redacts_hf_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = JobLog("token-test", root=Path(tmp))
            log.info("Token is hf_abcdefgh12345678")
            summary = log.summary()
            self.assertNotIn("hf_abcdefgh12345678", summary[0]["m"])
            self.assertIn("[REDACTED_HF_TOKEN]", summary[0]["m"])

    def test_job_log_respects_max_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = JobLog("big-job", root=Path(tmp))
            for i in range(300):
                log.info(f"Entry {i}")
            self.assertLessEqual(log.entry_count, 200)


if __name__ == "__main__":
    unittest.main()
