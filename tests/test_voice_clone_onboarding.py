import os
import tempfile
import unittest
import wave

import numpy as np

from engines.rvc_engine import GPTSoVITSEngine, VoiceCloneParams, assess_clone_reference


def _write_wav(path: str, duration: float, amplitude: float = 0.2, sample_rate: int = 24000):
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    audio = amplitude * np.sin(2 * np.pi * 220.0 * t)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


class VoiceCloneOnboardingTests(unittest.TestCase):
    def test_reference_quality_passes_clean_10_to_30_second_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "voice.wav")
            _write_wav(path, duration=12.0)

            report = assess_clone_reference(path)

            self.assertEqual(report.status, "pass")
            self.assertTrue(report.can_onboard)
            self.assertGreaterEqual(report.score, 90)
            self.assertAlmostEqual(report.duration, 12.0, places=1)

    def test_reference_quality_rejects_short_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "short.wav")
            _write_wav(path, duration=5.0)

            report = assess_clone_reference(path)

            self.assertEqual(report.status, "fail")
            self.assertFalse(report.can_onboard)
            self.assertIn("too short", " ".join(report.issues))

    def test_clone_rejects_failed_guardrails_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "short.wav")
            _write_wav(path, duration=4.0)
            engine = GPTSoVITSEngine()
            engine._sovits_model = object()

            result = engine.clone(VoiceCloneParams(text="Hello", ref_audio_path=path, ref_text="Hello"))

            self.assertIsNotNone(result.error)
            self.assertIn("guardrails", result.error)


if __name__ == "__main__":
    unittest.main()
