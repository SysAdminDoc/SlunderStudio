import os
import tempfile
import unittest
import wave

import numpy as np

from engines.rvc_engine import (
    GPTSoVITSEngine,
    RVCEngine,
    VoiceCloneParams,
    VoiceConvertParams,
)
from engines.sfx_engine import SFXEngine, SFXParams


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

        result = engine.convert(VoiceConvertParams(input_audio=audio))

        self.assertIsNotNone(result.error)
        self.assertFalse(result.is_success)
        self.assertFalse(result.can_route)
        self.assertEqual(result.output_kind, "error")

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


if __name__ == "__main__":
    unittest.main()
