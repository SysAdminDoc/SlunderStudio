import os
import os
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.midi_utils import MidiData
from engines.rvc_engine import (
    GPTSoVITSEngine,
    RVCEngine,
    VoiceCloneParams,
    VoiceConvertParams,
)
from engines.sfx_engine import SFXEngine, SFXParams
from engines.ai_producer import AIProducer, ProducerBrief, PipelineStage
from engines.fluidsynth_engine import MidiRenderResult, render_midi_to_audio
from engines.diffsinger_engine import DiffSingerEngine, SingParams
from ui.midi_studio_view import MidiStudioView


def _write_wav(path: str, duration: float = 12.0, sample_rate: int = 24000):
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    audio = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class DemoOutputContractTests(unittest.TestCase):
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
        self._app.processEvents()
        return bool(predicate())

    def test_vocal_suite_live_operation_labels_do_not_claim_demo_output(self):
        source = (Path(__file__).resolve().parents[1] / "ui" / "vocal_suite_view.py").read_text(
            encoding="utf-8"
        )
        for phrase in (
            "Cancelling RVC demo",
            "GPT-SoVITS demo...",
            "Preparing consent-ready GPT-SoVITS demo",
            "Produces declared outputs:",
        ):
            self.assertNotIn(phrase, source)

    def test_fluidsynth_failure_marks_sine_preview_as_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "preview.wav")
            with patch(
                "engines.fluidsynth_engine.get_fluidsynth",
                side_effect=RuntimeError("FluidSynth DLL missing"),
            ):
                result = render_midi_to_audio(
                    MidiData(),
                    output_path=output_path,
                    return_metadata=True,
                )

            self.assertIsInstance(result, MidiRenderResult)
            self.assertTrue(result.is_demo)
            self.assertIn("FluidSynth DLL missing", result.fallback_reason)
            self.assertTrue(os.path.isfile(output_path))

    def test_midi_view_disables_routing_for_sine_preview(self):
        view = MidiStudioView()
        try:
            view._midi_data = MidiData()
            preview = MidiRenderResult(
                audio=np.zeros((128, 2), dtype=np.float32),
                output_kind="demo",
                fallback_reason="RuntimeError: FluidSynth DLL missing",
            )
            with tempfile.TemporaryDirectory() as tmp:
                with patch(
                    "core.settings.get_config_dir",
                    return_value=tmp,
                ), patch(
                    "engines.fluidsynth_engine.render_midi_to_audio",
                    return_value=preview,
                ):
                    view._on_render()
                    self.assertTrue(self._wait_for(lambda: view._render_worker is None))

            self.assertIn("Preview render (sine)", view._status.text())
            self.assertIn("FluidSynth DLL missing", view._status.text())
            self.assertFalse(view._to_forge_btn.isEnabled())
            self.assertFalse(view._to_vocals_btn.isEnabled())
        finally:
            view.close()
            self._app.processEvents()

    def test_sfx_requires_explicit_demo_when_model_is_unloaded(self):
        engine = SFXEngine()

        result = engine.generate(SFXParams(prompt="rain on glass"))

        self.assertIsNotNone(result.error)
        self.assertFalse(result.is_success)
        self.assertFalse(result.can_route)
        self.assertEqual(result.output_kind, "error")

    def test_sfx_demo_output_is_marked_and_opted_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SFXEngine()
            engine._output_dir = tmp

            result = engine.generate(SFXParams(
                prompt="soft chime",
                duration=0.5,
                allow_demo_output=True,
            ))

            self.assertIsNone(result.error)
            self.assertTrue(result.is_success)
            self.assertTrue(result.is_demo)
            self.assertEqual(result.output_kind, "demo")
            self.assertTrue(result.can_route)
            self.assertTrue(os.path.isfile(result.file_path))

    def test_rvc_placeholder_conversion_is_not_reported_as_model_success(self):
        engine = RVCEngine()
        engine._model = object()
        audio = np.zeros(1600, dtype=np.float32)

        result = engine.convert(VoiceConvertParams(
            input_audio=audio,
            allow_demo_output=True,
        ))

        self.assertIsNotNone(result.error)
        self.assertFalse(result.is_success)
        self.assertFalse(result.can_route)
        self.assertEqual(result.output_kind, "error")
        self.assertIsNone(result.audio)
        self.assertIn("no placeholder audio", result.error.lower())

    def test_gpt_sovits_placeholder_synthesis_is_not_reported_as_model_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = os.path.join(tmp, "voice.wav")
            _write_wav(ref_path)
            engine = GPTSoVITSEngine()
            engine._sovits_model = object()

            result = engine.clone(VoiceCloneParams(
                text="Hello from the test",
                ref_audio_path=ref_path,
                ref_text="Hello from the test",
            ))

            self.assertIsNotNone(result.error)
            self.assertFalse(result.is_success)
            self.assertFalse(result.can_route)
            self.assertEqual(result.output_kind, "error")
            self.assertIsNone(result.audio)
            self.assertIn("no placeholder audio", result.error.lower())

    def test_diffsinger_without_dictionary_fails_closed(self):
        engine = DiffSingerEngine()
        engine._session = object()
        engine._sample_rate = 24000
        engine._hop_size = 256

        with self.assertRaises(RuntimeError) as ctx:
            engine.synthesize(SingParams(lyrics="la"))

        self.assertIn("phoneme dictionary", str(ctx.exception))

    def test_ai_producer_stops_on_song_failure_without_demo_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            producer = AIProducer()
            producer._output_dir = tmp

            brief = ProducerBrief(
                prompt="test song",
                genre="pop",
                duration_seconds=5.0,
                demo_fallback=False,
            )

            with patch("engines.ai_producer.AIProducer._generate_lyrics",
                        return_value={"lyrics": "[Verse 1]\ntest"}):
                with patch("engines.ai_producer.AIProducer._select_style",
                            return_value={"tags": ["pop"], "tempo": 120, "key": "C major"}):
                    result = producer.produce(brief)

            song_step = result.get_step(PipelineStage.SONG_GEN)
            self.assertIsNotNone(song_step)
            self.assertEqual(song_step.status, "failed")
            self.assertIsNotNone(result.error)
            self.assertIn("Song generation failed", result.error)
            self.assertIsNone(result.final_audio_path)

    def test_ai_producer_continues_with_demo_fallback_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            producer = AIProducer()
            producer._output_dir = tmp

            brief = ProducerBrief(
                prompt="test song",
                genre="pop",
                duration_seconds=1.0,
                demo_fallback=True,
                include_sfx=False,
                vocal_style="none",
            )

            with patch("engines.ai_producer.AIProducer._generate_lyrics",
                        return_value={"lyrics": "[Verse 1]\ntest"}):
                with patch("engines.ai_producer.AIProducer._select_style",
                            return_value={"tags": ["pop"], "tempo": 120, "key": "C major"}):
                    result = producer.produce(brief)

            song_step = result.get_step(PipelineStage.SONG_GEN)
            self.assertIsNotNone(song_step)
            self.assertEqual(song_step.status, "complete")
            self.assertTrue(song_step.output_data.get("fallback"))
            self.assertTrue(song_step.output_data.get("demo"))

            self.assertIsNotNone(result.song_audio_path)
            self.assertIn("demo", os.path.basename(result.song_audio_path))


if __name__ == "__main__":
    unittest.main()
