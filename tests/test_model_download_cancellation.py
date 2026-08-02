import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.job_state import JobStatus, JobStore
from core.model_manager import ModelManager, ModelStatus
from core.workers import CancelledJobError, DownloadWorker
from ui.model_hub import ModelHubView


class _ToastRecorder:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)


class ModelDownloadCancellationTests(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager()
        self.info = self.manager.get_model_info("llama-3.1-8b-q4")
        self.assertIsNotNone(self.info)

    def test_cancelled_transfer_stops_before_next_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "model"
            cache.mkdir()
            cancel_event = threading.Event()
            fetched = []

            def fake_snapshot_download(**kwargs):
                progress = kwargs["tqdm_class"](total=3)
                try:
                    progress.update(1)
                    (cache / "partial.bin").write_bytes(b"partial")
                    fetched.append("first")
                    cancel_event.set()
                    progress.update(1)
                    fetched.append("second")
                finally:
                    progress.close()
                return str(cache)

            with (
                mock.patch.object(self.manager, "get_cache_dir", return_value=cache),
                mock.patch.object(
                    self.manager, "_resolve_hf_revision", return_value="resolved"
                ),
                mock.patch("huggingface_hub.snapshot_download", fake_snapshot_download),
            ):
                with self.assertRaises(CancelledJobError):
                    self.manager.download_model(
                        self.info.model_id, cancel_event=cancel_event
                    )

            self.assertEqual(["first"], fetched)
            self.assertFalse((cache / self.manager.COMPLETE_MARKER).exists())
            self.assertEqual(
                ModelStatus.PARTIAL, self.manager.get_status(self.info.model_id)
            )

    def test_post_completion_cancel_keeps_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "model"
            cache.mkdir()
            cancel_event = threading.Event()

            def fake_snapshot_download(**kwargs):
                progress = kwargs["tqdm_class"](total=1)
                try:
                    progress.update(1)
                finally:
                    progress.close()
                (cache / "weights.gguf").write_bytes(b"complete")
                cancel_event.set()
                return str(cache)

            with (
                mock.patch.object(self.manager, "get_cache_dir", return_value=cache),
                mock.patch.object(
                    self.manager, "_resolve_hf_revision", return_value="resolved"
                ),
                mock.patch("huggingface_hub.snapshot_download", fake_snapshot_download),
            ):
                result = self.manager.download_model(
                    self.info.model_id, cancel_event=cancel_event
                )

            self.assertTrue(result)
            self.assertTrue((cache / self.manager.COMPLETE_MARKER).is_file())
            self.assertEqual(
                ModelStatus.DOWNLOADED, self.manager.get_status(self.info.model_id)
            )

    def test_download_worker_keeps_explicitly_completed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")

            def complete(_model_id, cancel_event=None, **_kwargs):
                cancel_event.set()
                return True

            worker = DownloadWorker(
                complete,
                "completed-model",
                job_store=store,
            )
            worker.run()

            self.assertEqual(JobStatus.COMPLETED, store.get(worker.job_id).status)

    def test_resume_while_stopping_reports_visible_feedback(self):
        model_id = "llama-3.1-8b-q4"
        worker = mock.Mock()
        card = mock.Mock()
        toast = _ToastRecorder()
        view = ModelHubView.__new__(ModelHubView)
        view._workers = {model_id: worker}
        view._stopping_downloads = set()
        view._cards = {model_id: card}
        view.toast_mgr = toast
        view._mgr = mock.Mock()
        view._mgr.get_model_info.return_value = SimpleNamespace(name="Lyrics model")

        view._cancel_download(model_id)
        view._start_download(model_id)

        worker.cancel.assert_called_once_with()
        card.set_download_stopping.assert_called_once_with()
        self.assertIn(model_id, view._stopping_downloads)
        self.assertIn("still stopping", toast.messages[-1])


if __name__ == "__main__":
    unittest.main()
