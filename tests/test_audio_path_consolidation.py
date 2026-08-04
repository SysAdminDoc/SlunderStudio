import ast
import tempfile
import unittest
from pathlib import Path
from threading import Event

import numpy as np
import soundfile as sf

from core.audio_buffers import mixdown_audio
from core.audio_export import write_audio_file
from core.workers import CancelledJobError


class SharedAudioPathTests(unittest.TestCase):
    def test_writer_honors_rate_channels_and_bit_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stereo = np.zeros((257, 2), dtype=np.float32)
            mono = np.zeros(257, dtype=np.float32)

            write_audio_file(
                root / "stereo24.wav",
                stereo,
                32000,
                file_format="wav",
                bit_depth=24,
                channels=2,
            )
            write_audio_file(
                root / "mono16.wav",
                mono,
                22050,
                file_format="wav",
                bit_depth=16,
                channels=1,
            )
            write_audio_file(
                root / "float32.wav",
                stereo,
                48000,
                file_format="wav",
                bit_depth=32,
                channels=2,
            )

            stereo_info = sf.info(root / "stereo24.wav")
            mono_info = sf.info(root / "mono16.wav")
            float_info = sf.info(root / "float32.wav")

        self.assertEqual((32000, 2, 257, "PCM_24"), (
            stereo_info.samplerate,
            stereo_info.channels,
            stereo_info.frames,
            stereo_info.subtype,
        ))
        self.assertEqual((22050, 1, 257, "PCM_16"), (
            mono_info.samplerate,
            mono_info.channels,
            mono_info.frames,
            mono_info.subtype,
        ))
        self.assertEqual("FLOAT", float_info.subtype)

    def test_mixdown_applies_solo_mute_volume_and_pan(self):
        muted = np.ones(8, dtype=np.float32)
        selected = np.full((4, 2), 0.5, dtype=np.float32)

        mixed = mixdown_audio([
            (muted, 1.0, -1.0, True, False),
            (selected, 0.5, 1.0, False, True),
        ])

        self.assertEqual((8, 2), mixed.shape)
        np.testing.assert_allclose(mixed[:4, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(mixed[:4, 1], 0.25, atol=1e-6)
        np.testing.assert_allclose(mixed[4:], 0.0, atol=1e-6)

    def test_mixdown_peak_normalization_and_cancellation_are_shared(self):
        layer = np.ones((4, 2), dtype=np.float32)
        mixed = mixdown_audio([
            (layer, 1.0, 0.0, False, False),
            (layer, 1.0, 0.0, False, False),
        ])
        self.assertLessEqual(float(np.max(np.abs(mixed))), 1.0)
        self.assertTrue(np.allclose(mixed, 1.0))

        cancel = Event()
        cancel.set()
        with self.assertRaises(CancelledJobError):
            mixdown_audio([(layer, 1.0, 0.0, False, False)], cancel_event=cancel)


class AudioWriterStructureTests(unittest.TestCase):
    def test_runtime_sources_have_no_duplicate_audio_writers(self):
        root = Path(__file__).resolve().parents[1]
        duplicate_writers = []
        direct_soundfile_writers = []
        for directory in ("core", "engines", "ui"):
            for path in (root / directory).rglob("*.py"):
                if path == root / "core" / "audio_export.py":
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    function = node.func
                    if (
                        isinstance(function, ast.Attribute)
                        and isinstance(function.value, ast.Name)
                        and function.value.id == "wave"
                        and function.attr == "open"
                    ):
                        mode = "rb"
                        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                            mode = str(node.args[1].value)
                        for keyword in node.keywords:
                            if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
                                mode = str(keyword.value.value)
                        if "w" in mode:
                            duplicate_writers.append(str(path.relative_to(root)))
                    if (
                        isinstance(function, ast.Attribute)
                        and isinstance(function.value, ast.Name)
                        and function.value.id == "sf"
                        and function.attr == "write"
                    ):
                        direct_soundfile_writers.append(str(path.relative_to(root)))

        self.assertEqual([], duplicate_writers)
        self.assertEqual([], direct_soundfile_writers)

    def test_signal_chain_uses_the_canonical_resampler(self):
        root = Path(__file__).resolve().parents[1]
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for directory in ("core", "engines", "ui")
            for path in (root / directory).rglob("*.py")
        )
        self.assertNotIn("librosa.resample", sources)
        self.assertNotIn("torchaudio.functional.resample", sources)


if __name__ == "__main__":
    unittest.main()
