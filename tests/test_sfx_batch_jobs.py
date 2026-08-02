import os
import tempfile

import numpy as np
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.engine_contract import (
    CAP_SFX_GENERATE,
    ArtifactKind,
    EngineArtifact,
    EngineBatchResult,
    EngineRunResult,
    RunOutcome,
)
from core.job_state import JobStore
from core.workers import CancelledJobError, InferenceWorker


def _run(path: str, outcome=RunOutcome.MODEL) -> EngineRunResult:
    return EngineRunResult(
        capability_id=CAP_SFX_GENERATE,
        outcome=outcome,
        artifacts=[EngineArtifact(kind=ArtifactKind.AUDIO, path=path)],
    )


class CancellationPreservesCompletedWorkTests(unittest.TestCase):
    """Cancelling a batch must not destroy variations that already finished."""

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.jobs = JobStore(self.root / "jobs", cleanup_roots=[self.root])

    def _artifact(self, name: str) -> str:
        path = self.root / name
        path.write_bytes(b"audio")
        return str(path)

    def _wait(self, predicate, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._app.processEvents()
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def test_preserved_outputs_survive_and_partials_are_removed(self):
        done = self._artifact("done.wav")
        partial = self._artifact("partial.wav")

        def task(**kwargs):
            raise CancelledJobError(
                "cancelled",
                outputs=[done, partial],
                preserved=[done],
            )

        worker = InferenceWorker(task, job_kind="sfx", job_label="batch",
                                 job_store=self.jobs)
        seen = []
        worker.cancelled.connect(lambda: seen.append(True))
        worker.start()
        self.assertTrue(self._wait(lambda: seen))
        worker.wait(5000)

        self.assertTrue(os.path.isfile(done), "completed variation was deleted")
        self.assertFalse(os.path.isfile(partial), "partial was not cleaned")

        record = self.jobs.get(worker.job_id)
        self.assertEqual(record.status, "cancelled")
        self.assertIn(done, record.outputs.get("preserved_paths", []))

    def test_without_preserved_everything_is_still_cleaned(self):
        first = self._artifact("a.wav")
        second = self._artifact("b.wav")

        def task(**kwargs):
            raise CancelledJobError("cancelled", outputs=[first, second])

        worker = InferenceWorker(task, job_kind="sfx", job_label="batch",
                                 job_store=self.jobs)
        seen = []
        worker.cancelled.connect(lambda: seen.append(True))
        worker.start()
        self.assertTrue(self._wait(lambda: seen))
        worker.wait(5000)

        self.assertFalse(os.path.isfile(first))
        self.assertFalse(os.path.isfile(second))

    def test_partial_result_is_available_after_cancellation(self):
        done = self._artifact("done.wav")
        batch = EngineBatchResult(CAP_SFX_GENERATE, [_run(done)])

        def task(**kwargs):
            raise CancelledJobError(
                "cancelled", outputs=[done], preserved=[done], result=batch,
            )

        worker = InferenceWorker(task, job_kind="sfx", job_label="batch",
                                 job_store=self.jobs)
        seen = []
        worker.cancelled.connect(lambda: seen.append(True))
        worker.start()
        self.assertTrue(self._wait(lambda: seen))
        worker.wait(5000)
        self.assertIs(worker.result, batch)


class SFXViewCancellationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        from ui.sfx_view import SFXView

        self.view = SFXView()
        self.addCleanup(self.view.deleteLater)

    def test_cancelled_batch_keeps_and_shows_completed_variations(self):
        from engines.sfx_engine import SFXResult

        path = self.root / "kept.wav"
        path.write_bytes(b"audio")
        run = _run(str(path))
        run.source_result = SFXResult(
            file_path=str(path), duration=2.0,
            audio=np.zeros(16, dtype=np.float32),
        )
        batch = EngineBatchResult(CAP_SFX_GENERATE, [run])

        worker = mock.Mock()
        worker.result = batch
        self.view._generation_worker = worker

        self.view._on_generation_cancelled()

        self.assertEqual(len(self.view._results), 1)
        self.assertIn("kept 1", self.view._status.text())
        self.assertIn("retry", self.view._status.text().lower())

    def test_a_completed_run_whose_file_vanished_is_not_advertised(self):
        from engines.sfx_engine import SFXResult

        missing = str(self.root / "gone.wav")
        run = _run(missing)
        run.source_result = SFXResult(
            file_path=missing, duration=2.0,
            audio=np.zeros(16, dtype=np.float32),
        )
        batch = EngineBatchResult(CAP_SFX_GENERATE, [run])

        worker = mock.Mock()
        worker.result = batch
        self.view._generation_worker = worker

        self.view._on_generation_cancelled()

        self.assertEqual(len(self.view._results), 0)
        self.assertEqual(self.view._status.text(), "SFX generation cancelled")

    def test_cancellation_without_any_result_reports_plainly(self):
        self.view._generation_worker = None
        self.view._on_generation_cancelled()
        self.assertEqual(self.view._status.text(), "SFX generation cancelled")

    def test_batch_task_reports_progress_per_variation(self):
        from engines.sfx_engine import SFXParams, SFXResult

        steps = []
        calls = {"n": 0}

        def fake_generate(item, progress_callback=None):
            calls["n"] += 1
            if progress_callback:
                progress_callback(1.0, "rendering")
            out = self.root / f"v{calls['n']}.wav"
            out.write_bytes(b"audio")
            return SFXResult(
                file_path=str(out), duration=1.0,
                audio=np.zeros(16, dtype=np.float32),
            )

        params = SFXParams(prompt="rain", batch_size=3, allow_demo_output=True)
        with mock.patch("engines.sfx_engine.generate_sfx", fake_generate):
            batch = self.view._run_generation_batch(
                params, "stable-audio-open",
                progress_cb=lambda _p: None,
                step_cb=steps.append,
            )

        self.assertEqual(len(batch.runs), 3)
        self.assertTrue(any(s.startswith("1/3") for s in steps))
        self.assertTrue(any(s.startswith("3/3") for s in steps))

    def test_batch_task_cancels_between_variations_and_keeps_the_rest(self):
        from engines.sfx_engine import SFXParams, SFXResult

        cancel = threading.Event()
        made = []

        def fake_generate(item, progress_callback=None):
            out = self.root / f"c{len(made)}.wav"
            out.write_bytes(b"audio")
            made.append(str(out))
            cancel.set()  # cancel after the first variation completes
            return SFXResult(
                file_path=str(out), duration=1.0,
                audio=np.zeros(16, dtype=np.float32),
            )

        params = SFXParams(prompt="rain", batch_size=4, allow_demo_output=True)
        with mock.patch("engines.sfx_engine.generate_sfx", fake_generate):
            with self.assertRaises(CancelledJobError) as ctx:
                self.view._run_generation_batch(
                    params, "stable-audio-open", cancel_event=cancel)

        error = ctx.exception
        self.assertEqual(len(made), 1)
        self.assertEqual(list(error.preserved), made)
        self.assertIn("1 of 4", str(error))
        self.assertIsInstance(error.result, EngineBatchResult)


if __name__ == "__main__":
    unittest.main()
