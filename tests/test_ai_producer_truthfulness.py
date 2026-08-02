import os
import tempfile
import threading
import unittest
import wave
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import numpy as np

from core.job_state import JobStatus, JobStore
from core.model_manager import ModelManager
from core.workers import InferenceWorker
from engines.ai_producer import (
    AIProducer,
    PipelineStage,
    PipelineStep,
    ProducerBrief,
    ProducerResult,
    produce_song,
)
from engines.lyrics_engine import LyricsLLM


def _write_wav(path: str | Path, *, frames: int = 128, sample_rate: int = 8000) -> str:
    target = str(path)
    pcm = np.zeros((frames, 2), dtype=np.int16)
    with wave.open(target, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return target


class AIProducerTruthfulnessTests(unittest.TestCase):
    def test_lyrics_fallback_is_reported_as_demo_and_degraded(self):
        producer = AIProducer()
        brief = self._brief()
        result = ProducerResult(brief=brief)
        producer._current_result = result
        plan = {"genre": "pop", "mood": "bright"}

        with patch.object(
            ModelManager,
            "current_model",
            new_callable=PropertyMock,
            return_value=None,
        ):
            step = producer._run_stage(
                PipelineStage.LYRICS,
                result,
                lambda: producer._generate_lyrics(plan, brief),
                None,
                None,
                None,
                None,
            )

        self.assertEqual("complete", step.status)
        self.assertEqual("demo", result.output_kind)
        self.assertTrue(result.is_degraded)
        self.assertIn("Lyrics: explicit demo fallback was used", result.degraded_reasons)

    def test_lyrics_stage_uses_loaded_model_with_built_prompts(self):
        producer = AIProducer()
        brief = self._brief()
        result = ProducerResult(brief=brief)
        producer._current_result = result
        plan = {"genre": "pop", "mood": "bright"}
        engine = LyricsLLM()
        engine._model = object()
        engine._backend = "stub"
        generated = []

        def generate(system_prompt, user_prompt):
            generated.append((system_prompt, user_prompt))
            return "[Verse 1]\nModel lyrics"

        engine.generate = generate

        with patch.object(
            ModelManager,
            "current_model",
            new_callable=PropertyMock,
            return_value=engine,
        ), patch.object(
            ModelManager,
            "current_model_id",
            new_callable=PropertyMock,
            return_value="lyrics-test",
        ):
            step = producer._run_stage(
                PipelineStage.LYRICS,
                result,
                lambda: producer._generate_lyrics(plan, brief),
                None,
                None,
                None,
                None,
            )

        self.assertEqual("complete", step.status)
        self.assertEqual("model", result.output_kind)
        self.assertFalse(result.is_degraded)
        self.assertEqual("[Verse 1]\nModel lyrics", result.lyrics_text)
        self.assertEqual(1, len(generated))
        self.assertIn("GENRE: Pop", generated[0][0])
        self.assertIn("Write song lyrics about: truthful test song", generated[0][1])

    def test_required_error_dictionary_stops_dependent_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            producer = AIProducer()
            producer._output_dir = tmp
            master = Mock(return_value={"master_path": "must-not-run.wav"})

            with self._stub_early_stages(producer, Path(tmp)):
                with patch.object(
                    producer,
                    "_mix",
                    return_value={"error": "decoder rejected the song"},
                ):
                    with patch.object(producer, "_master", master):
                        result = producer.produce(self._brief())

            self.assertEqual(PipelineStage.FAILED, result.stage)
            self.assertFalse(result.is_success)
            self.assertFalse(result.can_export)
            self.assertEqual(
                "failed",
                result.get_step(PipelineStage.MIXING).status,
            )
            self.assertIsNone(result.get_step(PipelineStage.MASTERING))
            master.assert_not_called()
            self.assertEqual(
                "failed",
                result.job_metadata()["pipeline"]["outcome"],
            )

    def test_success_requires_an_owned_new_readable_final_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            producer = AIProducer()
            producer._output_dir = tmp
            stale = _write_wav(Path(tmp) / "prior-run.wav")

            def stale_master(result, _brief):
                result.final_audio_path = stale
                return {"master_path": stale}

            with self._stub_early_stages(producer, Path(tmp)):
                with patch.object(producer, "_mix", side_effect=self._mix_stub(Path(tmp))):
                    with patch.object(producer, "_master", side_effect=stale_master):
                        result = producer.produce(self._brief())

            self.assertEqual(PipelineStage.FAILED, result.stage)
            self.assertFalse(result.can_export)
            master_step = result.get_step(PipelineStage.MASTERING)
            self.assertEqual("failed", master_step.status)
            self.assertIn("new, readable", master_step.error)

    def test_optional_failure_is_visible_degradation_not_false_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            producer = AIProducer()
            producer._output_dir = tmp
            brief = self._brief(include_sfx=True)

            with self._stub_early_stages(
                producer,
                Path(tmp),
                plan_extra={"sfx_prompt": "gentle rain"},
            ):
                with patch.object(
                    producer,
                    "_add_sfx",
                    return_value={"error": "SFX model is unavailable"},
                ):
                    with patch.object(producer, "_mix", side_effect=self._mix_stub(Path(tmp))):
                        with patch.object(
                            producer,
                            "_master",
                            side_effect=self._master_stub(Path(tmp)),
                        ):
                            result = producer.produce(brief)

            self.assertTrue(result.is_success)
            self.assertTrue(result.is_degraded)
            self.assertFalse(result.is_demo)
            self.assertIsNone(result.error)
            self.assertEqual("failed", result.get_step(PipelineStage.SFX).status)
            self.assertIn("Sfx failed", result.degraded_reasons[0])
            self.assertEqual(
                "degraded",
                result.job_metadata()["pipeline"]["outcome"],
            )

    def test_cancellation_stops_at_the_active_stage_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            producer = AIProducer()
            producer._output_dir = tmp
            cancel_event = threading.Event()

            def cancel_during_plan(_brief):
                cancel_event.set()
                return {"tempo": 120, "key": "C", "style_tags": []}

            with patch.object(producer, "_plan", side_effect=cancel_during_plan):
                result = producer.produce(
                    self._brief(),
                    cancel_event=cancel_event,
                )

            self.assertTrue(result.cancelled)
            self.assertEqual(PipelineStage.CANCELLED, result.stage)
            self.assertEqual(
                "cancelled",
                result.get_step(PipelineStage.PLANNING).status,
            )
            self.assertIsNone(result.get_step(PipelineStage.LYRICS))
            self.assertFalse(result.can_export)

    def test_progress_and_detailed_stage_messages_reach_callers(self):
        with tempfile.TemporaryDirectory() as tmp:
            producer = AIProducer()
            producer._output_dir = tmp
            progress = []
            messages = []

            with self._stub_early_stages(producer, Path(tmp)):
                with patch.object(producer, "_mix", side_effect=self._mix_stub(Path(tmp))):
                    with patch.object(
                        producer,
                        "_master",
                        side_effect=self._master_stub(Path(tmp)),
                    ):
                        with patch(
                            "engines.ai_producer.get_producer",
                            return_value=producer,
                        ):
                            result = produce_song(
                                self._brief(),
                                progress_cb=progress.append,
                                step_cb=messages.append,
                            )

            self.assertTrue(result.is_success)
            self.assertEqual(100, progress[-1])
            self.assertTrue(any("Planning: running" == item for item in messages))
            self.assertTrue(any("Mastering: complete" == item for item in messages))
            self.assertEqual(sorted(progress), progress)

    def test_worker_persists_semantic_failure_and_stage_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp), cleanup_roots=[Path(tmp)])
            result = ProducerResult(
                stage=PipelineStage.FAILED,
                error="Failed at mixing: no layers",
                steps=[
                    PipelineStep(
                        stage=PipelineStage.MIXING,
                        status="failed",
                        error="no layers",
                    )
                ],
            )

            def task(**_kwargs):
                return result

            worker = InferenceWorker(
                task,
                job_kind="ai_producer",
                job_label="truthfulness test",
                job_store=store,
            )
            worker.run()

            record = store.get(worker.job_id)
            self.assertEqual(JobStatus.FAILED, record.status)
            self.assertEqual("failed", record.metadata["pipeline"]["outcome"])
            self.assertEqual(
                "no layers",
                record.metadata["pipeline"]["stages"][0]["error"],
            )

    def test_rerun_reset_revokes_stale_export_state(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        from ui.ai_producer_view import AIProducerView

        app = QApplication.instance() or QApplication([])
        view = AIProducerView()
        view._result = ProducerResult(final_audio_path="stale.wav")
        view._export_btn.setEnabled(True)
        view._retry_btn.setEnabled(True)

        view._reset_for_run()

        self.assertIsNone(view._result)
        self.assertFalse(view._export_btn.isEnabled())
        self.assertEqual(0, view._progress.value())
        self.assertTrue(all(
            indicator._status == "pending"
            for indicator in view._stage_indicators.values()
        ))
        view.close()
        app.processEvents()

    @staticmethod
    def _brief(*, include_sfx: bool = False) -> ProducerBrief:
        return ProducerBrief(
            prompt="truthful test song",
            genre="pop",
            duration_seconds=1.0,
            vocal_style="none",
            include_sfx=include_sfx,
        )

    @staticmethod
    def _mix_stub(root: Path):
        def mix(result, _brief):
            path = _write_wav(root / f"mix-{result.run_id}.wav")
            result.add_artifact(path)
            return {"mix_path": path}

        return mix

    @staticmethod
    def _master_stub(root: Path):
        def master(result, _brief):
            path = _write_wav(root / f"master-{result.run_id}.wav")
            result.mastered_audio_path = path
            result.final_audio_path = path
            result.add_artifact(path)
            return {"master_path": path, "output_lufs": -14.0}

        return master

    @staticmethod
    def _stub_early_stages(
        producer: AIProducer,
        root: Path,
        *,
        plan_extra: dict | None = None,
    ):
        stack = ExitStack()
        plan = {
            "tempo": 120,
            "key": "C major",
            "style_tags": ["pop"],
            "sfx_prompt": "",
            **(plan_extra or {}),
        }

        def song(_plan, result, _brief, **_kwargs):
            path = _write_wav(root / f"song-{result.run_id}.wav")
            result.song_audio_path = path
            result.add_artifact(path)
            return {"audio_path": path}

        stack.enter_context(patch.object(producer, "_plan", return_value=plan))
        stack.enter_context(patch.object(
            producer,
            "_generate_lyrics",
            return_value={"lyrics": "[Verse]\nTest"},
        ))
        stack.enter_context(patch.object(
            producer,
            "_select_style",
            return_value={"tags": ["pop"], "tempo": 120, "key": "C major"},
        ))
        stack.enter_context(patch.object(producer, "_generate_song", side_effect=song))
        return stack


if __name__ == "__main__":
    unittest.main()
