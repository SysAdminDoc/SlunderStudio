import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

from engines.diffsinger_engine import DiffSingerEngine, SingParams


def notes(*spans):
    return [
        {"pitch": pitch, "start": start, "end": end, "text": "a"}
        for pitch, start, end in spans
    ]


class FrameTimingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.model = self.root / "model.onnx"
        self.model.write_bytes(b"not a real model")

    def _write_config(self, payload: dict, name: str = "dsconfig.yaml"):
        path = self.root / name
        if name.endswith(".json"):
            path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            path.write_text(
                "\n".join(f"{k}: {v}" for k, v in payload.items()), encoding="utf-8"
            )
        return path

    def _engine(self, payload: dict, name: str = "dsconfig.yaml") -> DiffSingerEngine:
        self._write_config(payload, name)
        engine = DiffSingerEngine()
        config = engine._read_model_config(str(self.model))
        engine._sample_rate, engine._hop_size = engine._resolve_frame_timing(
            config, str(self.model)
        )
        return engine

    def test_frame_period_comes_from_the_model_config(self):
        engine = self._engine({"audio_sample_rate": 44100, "hop_size": 512})
        self.assertAlmostEqual(engine.frame_period_sec, 512 / 44100)

        engine = self._engine({"audio_sample_rate": 48000, "hop_size": 256})
        self.assertAlmostEqual(engine.frame_period_sec, 256 / 48000)

    def test_alternate_config_key_names_are_accepted(self):
        engine = self._engine({"sampling_rate": 44100, "hop_length": 512})
        self.assertAlmostEqual(engine.frame_period_sec, 512 / 44100)

    def test_json_config_is_read(self):
        engine = self._engine(
            {"audio_sample_rate": 44100, "hop_size": 512}, name="config.json")
        self.assertAlmostEqual(engine.frame_period_sec, 512 / 44100)

    def test_missing_metadata_fails_explicitly(self):
        engine = DiffSingerEngine()
        with self.assertRaises(RuntimeError) as ctx:
            engine._resolve_frame_timing({}, str(self.model))
        self.assertIn("frame timing", str(ctx.exception))

    def test_invalid_metadata_fails_explicitly(self):
        engine = DiffSingerEngine()
        for bad in (
            {"audio_sample_rate": 0, "hop_size": 512},
            {"audio_sample_rate": 44100, "hop_size": -1},
            {"audio_sample_rate": "forty four one", "hop_size": 512},
        ):
            with self.subTest(config=bad):
                with self.assertRaises(RuntimeError):
                    engine._resolve_frame_timing(bad, str(self.model))

    def test_frame_timing_is_unavailable_before_a_model_loads(self):
        engine = DiffSingerEngine()
        with self.assertRaises(RuntimeError):
            _ = engine.frame_period_sec

    def test_known_pitch_events_align_within_one_frame(self):
        for sample_rate, hop_size in ((44100, 512), (48000, 256), (44100, 128)):
            with self.subTest(sample_rate=sample_rate, hop_size=hop_size):
                engine = self._engine(
                    {"audio_sample_rate": sample_rate, "hop_size": hop_size})
                period = engine.frame_period_sec
                events = notes((60, 0.0, 0.5), (67, 1.0, 1.5), (72, 2.0, 2.5))
                total_frames = engine.time_to_frame(2.5)
                curve = engine.build_f0_curve(events, total_frames)[0]

                for note in events:
                    expected = int(round(note["start"] / period))
                    voiced = np.flatnonzero(
                        np.isclose(
                            curve,
                            440.0 * 2 ** ((note["pitch"] - 69) / 12.0),
                            rtol=1e-5,
                        )
                    )
                    self.assertTrue(voiced.size, f"pitch {note['pitch']} missing")
                    self.assertLessEqual(
                        abs(int(voiced[0]) - expected), 1,
                        f"pitch {note['pitch']} off by more than one frame",
                    )

    def test_curve_length_tracks_the_note_span(self):
        engine = self._engine({"audio_sample_rate": 44100, "hop_size": 512})
        frames = engine.time_to_frame(3.0)
        self.assertAlmostEqual(frames * engine.frame_period_sec, 3.0, delta=engine.frame_period_sec)

    def test_notes_are_clamped_inside_the_curve(self):
        engine = self._engine({"audio_sample_rate": 44100, "hop_size": 512})
        curve = engine.build_f0_curve(notes((60, 0.0, 99.0)), 10)
        self.assertEqual(curve.shape, (1, 10))
        self.assertTrue(np.all(curve[0] > 0))

    def test_unload_clears_frame_timing(self):
        engine = self._engine({"audio_sample_rate": 44100, "hop_size": 512})
        engine.unload_model()
        with self.assertRaises(RuntimeError):
            _ = engine.frame_period_sec

    def test_synthesis_and_wav_export_use_model_sample_rate(self):
        engine = self._engine({"audio_sample_rate": 24000, "hop_size": 256})

        class _Input:
            name = "f0"

        class _Session:
            def get_inputs(self):
                return [_Input()]

            def run(self, _outputs, _inputs):
                return [np.zeros(240, dtype=np.float32)]

        engine._session = _Session()
        engine._model_path = str(self.model)
        engine._phonemizer = lambda _lyrics: ["la"]
        engine._phoneme_dictionary = {"la": 7}
        params = SingParams(
            lyrics="la",
            notes=notes((60, 0.0, 0.25)),
            pitch_shift=1,
            gender=0.5,
        )
        shifted = np.zeros(240, dtype=np.float32)
        with (
            mock.patch.object(
                engine, "_pitch_shift", return_value=shifted
            ) as pitch_shift,
            mock.patch.object(
                engine, "_apply_gender", return_value=shifted
            ) as apply_gender,
        ):
            result = engine.synthesize(params)

        self.assertIsNone(result.error)
        self.assertEqual(24000, result.sample_rate)
        self.assertAlmostEqual(240 / 24000, result.duration)
        self.assertEqual(1, pitch_shift.call_count)
        np.testing.assert_array_equal(pitch_shift.call_args.args[0], shifted)
        self.assertEqual((1, 24000), pitch_shift.call_args.args[1:])
        self.assertEqual(1, apply_gender.call_count)
        np.testing.assert_array_equal(apply_gender.call_args.args[0], shifted)
        self.assertEqual((0.5, 24000), apply_gender.call_args.args[1:])

        engine._output_dir = str(self.root)
        output = engine.save_output(result, name="model-rate")
        with wave.open(output, "rb") as handle:
            self.assertEqual(24000, handle.getframerate())

    def test_dictionary_ids_replace_hash_tokens_and_durations_are_clamped(self):
        engine = self._engine({"audio_sample_rate": 24000, "hop_size": 256})

        class _Input:
            def __init__(self, name):
                self.name = name

        class _Session:
            def get_inputs(self):
                return [_Input("tokens"), _Input("durations")]

        engine._session = _Session()
        engine._phoneme_dictionary = {"a": 11, "b": 23, "SP": 2}
        inputs = engine._prepare_inputs(
            ["a", "b", "a"],
            notes((60, 0.0, 0.01)),
            SingParams(),
        )

        np.testing.assert_array_equal(inputs["tokens"], [[11, 23, 11]])
        self.assertTrue(np.all(inputs["durations"] >= 1))

    def test_text_dictionary_supports_comments_and_both_column_orders(self):
        dictionary = self.root / "dsdict.txt"
        dictionary.write_text(
            "# symbol id\nSP 2\n11 a\n\nb\n",
            encoding="utf-8",
        )

        self.assertEqual(
            {"SP": 2, "a": 11, "b": 12},
            DiffSingerEngine._parse_phoneme_dictionary(str(dictionary)),
        )


if __name__ == "__main__":
    unittest.main()
