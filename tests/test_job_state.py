import tempfile
import threading
import unittest
from pathlib import Path

from core.job_state import JobLog, JobStatus, JobStore
from core.workers import DownloadWorker, InferenceWorker, CancelledJobError
from engines.ace_step_engine import ACEStepEngine, GenerationParams, GenerationResult


class JobStateTests(unittest.TestCase):
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
            store = JobStore(tmp_path / "jobs")

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
