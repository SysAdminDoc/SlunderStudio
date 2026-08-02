import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.job_state import JobStore
from core.workers import InferenceWorker
from engines.ace_step_engine import (
    ACEStepEngine,
    GenerationParams,
    GenerationResult,
    generate_seed_grid,
)


class AceStepCancellationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.jobs = JobStore(self.root / "jobs", cleanup_roots=[self.root])

    def _render_stub(self, root: Path, calls: list[str]):
        def render(_params, cancel_event=None, **_kwargs):
            index = len(calls)
            audio = root / f"render_{index}.wav"
            sidecar = root / f"render_{index}.wav.provenance.json"
            audio.write_bytes(b"completed audio")
            sidecar.write_text("{}", encoding="utf-8")
            calls.extend([str(audio), str(sidecar)])
            cancel_event.set()
            return GenerationResult(
                audio_path=str(audio),
                provenance_path=str(sidecar),
                seed=index,
                duration=5.0,
            )

        return render

    def test_cancelled_batch_preserves_verified_variation(self):
        engine = ACEStepEngine()
        engine._model_loaded = True
        calls = []

        def task(**kwargs):
            return engine.generate_batch(
                GenerationParams(duration=5.0),
                count=3,
                cancel_event=kwargs["cancel_event"],
            )

        with mock.patch.object(engine, "generate", side_effect=self._render_stub(self.root, calls)):
            worker = InferenceWorker(
                task,
                job_kind="song_generation",
                job_label="ACE-Step batch",
                job_store=self.jobs,
            )
            worker.start()
            self.assertTrue(worker.wait(10000))

        record = self.jobs.get(worker.job_id)
        self.assertEqual(record.status, "cancelled")
        self.assertEqual(len(calls), 2)
        self.assertTrue(Path(calls[0]).is_file())
        self.assertTrue(Path(calls[1]).is_file())
        self.assertEqual(record.outputs["preserved_paths"], calls)
        self.assertEqual(len(worker.result), 1)

    def test_cancelled_seed_grid_preserves_completed_cell(self):
        engine = ACEStepEngine()
        engine._model_loaded = True
        calls = []
        manager = mock.Mock()
        manager.load_model.return_value = engine

        params = [
            {"row": 0, "col": 0, "seed": 10, "shift": 1.0},
            {"row": 0, "col": 1, "seed": 11, "shift": 1.0},
        ]

        with (
            mock.patch("core.model_manager.ModelManager", return_value=manager),
            mock.patch.object(
                engine,
                "generate",
                side_effect=self._render_stub(self.root, calls),
            ),
        ):
            worker = InferenceWorker(
                generate_seed_grid,
                lyrics="lyrics",
                style_tags="style",
                params_list=params,
                duration=5.0,
                job_kind="song_generation",
                job_label="ACE-Step seed grid",
                job_store=self.jobs,
            )
            worker.start()
            self.assertTrue(worker.wait(10000))

        record = self.jobs.get(worker.job_id)
        self.assertEqual(record.status, "cancelled")
        self.assertEqual(len(calls), 2)
        self.assertTrue(Path(calls[0]).is_file())
        self.assertTrue(Path(calls[1]).is_file())
        self.assertEqual(record.outputs["preserved_paths"], calls)
        self.assertEqual(len(worker.result["results"]), 1)


if __name__ == "__main__":
    unittest.main()
