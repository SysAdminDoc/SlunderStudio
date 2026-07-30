import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core.audio_buffers import (
    AudioBufferError,
    normalize_channel_layout,
    resample_audio,
)
from core.audio_export import ExportSettings, export_audio


class MixerResamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_polyphase_resampling_preserves_duration_and_tone_pitch(self):
        source_rate = 44100
        target_rate = 48000
        t = np.arange(source_rate, dtype=np.float32) / source_rate
        source = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        output = resample_audio(source, source_rate, target_rate)

        self.assertEqual(target_rate, len(output))
        spectrum = np.abs(np.fft.rfft(output * np.hanning(len(output))))
        frequencies = np.fft.rfftfreq(len(output), 1.0 / target_rate)
        peak_hz = frequencies[int(np.argmax(spectrum))]
        self.assertAlmostEqual(440.0, peak_hz, delta=1.0)

    def test_mixer_resamples_before_length_and_sum(self):
        from ui.mixer_view import MixerView

        view = MixerView(project_sample_rate=48000)
        first = self._tone(44100, 440.0, 1.0, gain=0.1)
        second = self._tone(48000, 660.0, 1.0, gain=0.1)
        view.add_track("44k tone", first, 44100)
        view.add_track("48k tone", second, 48000)

        self.assertEqual([48000, 48000], [
            len(track["audio"]) for track in view._tracks
        ])
        self.assertEqual({48000}, {track["sr"] for track in view._tracks})

        mixed = view._get_mixed_audio()
        self.assertEqual((48000, 2), mixed.shape)
        spectrum = np.abs(np.fft.rfft(mixed[:, 0] * np.hanning(len(mixed))))
        frequencies = np.fft.rfftfreq(len(mixed), 1.0 / 48000)
        strongest = np.argpartition(spectrum, -8)[-8:]
        strongest_hz = frequencies[strongest]
        self.assertLess(float(np.min(np.abs(strongest_hz - 440.0))), 1.1)
        self.assertLess(float(np.min(np.abs(strongest_hz - 660.0))), 1.1)
        view.close()

    def test_mixed_rate_impulses_remain_aligned_and_output_is_deterministic(self):
        from ui.mixer_view import MixerView

        view = MixerView(project_sample_rate=48000)
        first = np.zeros(44100, dtype=np.float32)
        second = np.zeros(48000, dtype=np.float32)
        first[11025] = 0.25
        second[12000] = 0.25
        view.add_track("44k impulse", first, 44100)
        view.add_track("48k impulse", second, 48000)

        mixed_once = view._get_mixed_audio()
        mixed_twice = view._get_mixed_audio()

        self.assertLessEqual(abs(int(np.argmax(mixed_once[:, 0])) - 12000), 1)
        self.assertTrue(np.array_equal(mixed_once, mixed_twice))
        view.close()

    def test_channel_layouts_are_normalized_with_explicit_downmix(self):
        mono = np.array([0.25, -0.25], dtype=np.float32)
        stereo = normalize_channel_layout(mono)
        self.assertEqual((2, 2), stereo.shape)
        self.assertTrue(np.array_equal(stereo[:, 0], stereo[:, 1]))

        surround = np.tile(
            np.arange(1, 7, dtype=np.float32),
            (3, 1),
        )
        downmix = normalize_channel_layout(surround)
        expected_left = 1 + (3 * np.sqrt(0.5)) + (4 * 0.5) + (5 * np.sqrt(0.5))
        expected_right = 2 + (3 * np.sqrt(0.5)) + (4 * 0.5) + (6 * np.sqrt(0.5))
        self.assertTrue(np.allclose(downmix[:, 0], expected_left, atol=1e-6))
        self.assertTrue(np.allclose(downmix[:, 1], expected_right, atol=1e-6))

    def test_invalid_rates_and_samples_are_rejected_before_mixing(self):
        from ui.mixer_view import MixerView

        view = MixerView(project_sample_rate=48000)
        with self.assertRaises(AudioBufferError):
            view.add_track("bad rate", np.zeros(32, dtype=np.float32), 0)
        with self.assertRaises(AudioBufferError):
            view.add_track(
                "bad samples",
                np.array([0.0, np.nan], dtype=np.float32),
                48000,
            )
        self.assertEqual([], view._tracks)
        view.close()

    def test_mastering_receives_the_explicit_project_rate(self):
        from ui.mixer_view import MixerView

        view = MixerView(project_sample_rate=48000)
        view.add_track("source", self._tone(44100, 220.0, 0.05), 44100)
        captured = {}

        def fake_master(audio, sample_rate, _preset):
            captured["shape"] = audio.shape
            captured["sample_rate"] = sample_rate
            return SimpleNamespace(error="test stop")

        with patch("ui.mixer_view.master_audio", side_effect=fake_master):
            view._on_master_export()

        self.assertEqual(48000, captured["sample_rate"])
        self.assertEqual(2, captured["shape"][1])
        view.close()

    def test_export_uses_the_same_duration_preserving_resampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.wav"
            output = Path(tmp) / "output.wav"
            sf.write(source, self._tone(44100, 330.0, 1.0), 44100)

            exported = export_audio(
                str(source),
                str(output),
                ExportSettings(
                    format="wav",
                    sample_rate=48000,
                    bit_depth=24,
                ),
            )
            decoded, sample_rate = sf.read(exported, dtype="float32")

            self.assertEqual(48000, sample_rate)
            self.assertEqual(48000, len(decoded))
            self.assertTrue(Path(f"{exported}.provenance.json").is_file())

    @staticmethod
    def _tone(
        sample_rate: int,
        frequency: float,
        seconds: float,
        *,
        gain: float = 0.2,
    ) -> np.ndarray:
        timeline = np.arange(
            int(round(sample_rate * seconds)),
            dtype=np.float32,
        ) / sample_rate
        return (gain * np.sin(2 * np.pi * frequency * timeline)).astype(np.float32)


if __name__ == "__main__":
    unittest.main()
