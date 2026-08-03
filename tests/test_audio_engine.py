import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import core.audio_engine as audio_engine
from core.audio_engine import (
    AudioEngine,
    AudioOutputDevice,
    enumerate_output_devices,
    format_time,
)


class _CallbackStop(Exception):
    pass


class _Stream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class AudioEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.engine = AudioEngine()
        self.engine.cleanup()
        self.engine._volume = 1.0
        self.engine._loop_enabled = False
        self.engine._loop_start = 0
        self.engine._loop_end = 0
        self.engine.set_output_device("")

    def tearDown(self):
        self.engine.cleanup()

    def _fake_sd(self):
        return SimpleNamespace(OutputStream=_Stream, CallbackStop=_CallbackStop)

    def test_callback_scales_volume_and_advances_position(self):
        self.engine.load_array(np.arange(6, dtype=np.float32).reshape(-1, 1), 10)
        self.engine._is_playing = True
        self.engine.volume = 0.5
        outdata = np.zeros((3, 1), dtype=np.float32)

        with mock.patch.object(audio_engine, "_sd", self._fake_sd()):
            self.engine._callback(outdata, 3)

        np.testing.assert_allclose(outdata[:, 0], [0.0, 0.5, 1.0])
        self.assertEqual(self.engine._position, 3)

    def test_callback_zero_fills_tail_and_stops_on_next_block(self):
        self.engine.load_array(np.arange(5, dtype=np.float32).reshape(-1, 1), 10)
        self.engine._is_playing = True
        self.engine.seek(0.3)
        outdata = np.full((4, 1), -1.0, dtype=np.float32)

        with mock.patch.object(audio_engine, "_sd", self._fake_sd()):
            self.engine._callback(outdata, 4)
            np.testing.assert_allclose(outdata[:, 0], [3.0, 4.0, 0.0, 0.0])
            self.assertEqual(self.engine._position, 5)

            outdata.fill(-1.0)
            with self.assertRaises(_CallbackStop):
                self.engine._callback(outdata, 4)

        np.testing.assert_allclose(outdata, 0.0)
        self.assertFalse(self.engine._is_playing)

    def test_callback_wraps_across_the_loop_end_without_silence(self):
        self.engine.load_array(np.arange(8, dtype=np.float32).reshape(-1, 1), 10)
        self.engine.set_loop(True, start_sec=0.2, end_sec=0.5)
        self.engine._is_playing = True
        self.engine.seek(0.4)
        outdata = np.zeros((4, 1), dtype=np.float32)

        with mock.patch.object(audio_engine, "_sd", self._fake_sd()):
            self.engine._callback(outdata, 4)

        np.testing.assert_allclose(outdata[:, 0], [4.0, 2.0, 3.0, 4.0])
        self.assertEqual(self.engine._position, 2)
        self.assertTrue(self.engine.loop_enabled)

    def test_callback_outputs_silence_while_paused(self):
        self.engine.load_array(np.ones((4, 1), dtype=np.float32), 10)
        self.engine._is_playing = True
        self.engine._is_paused = True
        self.engine._position = 2
        outdata = np.full((2, 1), -1.0, dtype=np.float32)

        with mock.patch.object(audio_engine, "_sd", self._fake_sd()):
            self.engine._callback(outdata, 2)

        np.testing.assert_allclose(outdata, 0.0)
        self.assertEqual(self.engine._position, 2)

    def test_play_pause_seek_and_loop_controls(self):
        self.engine.load_array(np.zeros((20, 2), dtype=np.float32), 10)
        fake_sd = self._fake_sd()

        with mock.patch.object(audio_engine, "_sd", fake_sd), mock.patch.object(
            audio_engine, "_sf", object()
        ):
            self.engine.play()
            self.assertTrue(self.engine.is_playing)
            self.assertIsInstance(self.engine._stream, _Stream)
            self.assertIs(self.engine._stream.callback.__self__, self.engine)
            self.assertIs(
                self.engine._stream.callback.__func__,
                AudioEngine._callback,
            )

            self.engine.pause()
            self.assertTrue(self.engine.is_paused)
            self.assertFalse(self.engine.is_playing)
            self.engine.play()
            self.assertTrue(self.engine.is_playing)

            self.engine.seek(-3.0)
            self.assertEqual(self.engine._position, 0)
            self.engine.seek(99.0)
            self.assertEqual(self.engine._position, 20)
            self.engine.seek_relative(-0.5)
            self.assertEqual(self.engine._position, 15)

            self.engine.set_loop(True, start_sec=0.5, end_sec=1.5)
            self.assertEqual(self.engine._loop_start, 5)
            self.assertEqual(self.engine._loop_end, 15)
            self.engine.set_loop(False)
            self.assertFalse(self.engine.loop_enabled)

    def test_enumerate_output_devices_filters_inputs_and_names_host_api(self):
        fake_sd = SimpleNamespace(
            query_hostapis=lambda: [
                {"name": "MME"},
                {"name": "Windows WASAPI"},
            ],
            query_devices=lambda: [
                {
                    "name": "Microphone",
                    "hostapi": 0,
                    "max_output_channels": 0,
                    "default_samplerate": 44100,
                },
                {
                    "name": "Speakers",
                    "hostapi": 1,
                    "max_output_channels": 2,
                    "default_samplerate": 48000,
                },
            ],
        )

        devices, error = enumerate_output_devices(fake_sd)

        self.assertIsNone(error)
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].index, 1)
        self.assertEqual(devices[0].host_api, "Windows WASAPI")
        self.assertEqual(devices[0].label, "Speakers (Windows WASAPI)")
        self.assertEqual(devices[0].identity, "Windows WASAPI::Speakers")

    def test_play_passes_persisted_output_device_index_to_stream(self):
        device = AudioOutputDevice(
            index=7,
            name="Headphones",
            host_api="Windows WASAPI",
            max_output_channels=2,
            default_sample_rate=48000,
        )
        self.engine.set_output_device(device.identity)
        fake_sd = self._fake_sd()
        self.engine.load_array(np.zeros((20, 2), dtype=np.float32), 10)

        with mock.patch.object(audio_engine, "_sd", fake_sd), mock.patch.object(
            audio_engine,
            "enumerate_output_devices",
            return_value=([device], None),
        ), mock.patch.object(audio_engine, "_sf", object()):
            self.engine.play()

        self.assertEqual(self.engine._stream.kwargs["device"], 7)

    def test_unavailable_saved_device_uses_default_and_reports_fallback(self):
        self.engine.set_output_device("Windows WASAPI::Dock speakers")
        fake_sd = self._fake_sd()
        messages = []
        self.engine.output_device_status.connect(messages.append)
        self.engine.load_array(np.zeros((20, 2), dtype=np.float32), 10)

        with mock.patch.object(audio_engine, "_sd", fake_sd), mock.patch.object(
            audio_engine,
            "enumerate_output_devices",
            return_value=([], None),
        ), mock.patch.object(audio_engine, "_sf", object()):
            self.engine.play()

        self.assertTrue(messages)
        self.assertIn("Dock speakers (Windows WASAPI)", messages[-1])
        self.assertIn("system default", messages[-1])
        self.assertIsNone(self.engine._stream.kwargs["device"])

    def test_load_file_failure_leaves_existing_audio_untouched(self):
        data = np.arange(12, dtype=np.float32).reshape(6, 2)
        self.engine.load_array(data, 8000)
        self.engine.seek(0.25)
        before_source = self.engine._source_path
        before_position = self.engine._position

        class _FailingSoundFile:
            @staticmethod
            def read(*_args, **_kwargs):
                raise OSError("file is missing")

        with mock.patch.object(audio_engine, "_ensure_audio_libs"), mock.patch.object(
            audio_engine, "_sf", _FailingSoundFile
        ):
            self.assertFalse(self.engine.load_file("missing.wav"))

        np.testing.assert_array_equal(self.engine._audio_data, data)
        self.assertEqual(self.engine._sample_rate, 8000)
        self.assertEqual(self.engine._position, before_position)
        self.assertEqual(self.engine._source_path, before_source)

    def test_waveform_generation_reports_mono_envelope(self):
        data = np.array([[0.1, -0.3], [0.8, 0.2], [-0.4, 0.5]], dtype=np.float32)
        waveform = []
        self.engine.waveform_ready.connect(waveform.append)

        self.engine.load_array(data, 48000)

        self.assertEqual(len(waveform), 1)
        np.testing.assert_allclose(waveform[0], [0.1, 0.5, 0.05], atol=1e-6)
        self.assertAlmostEqual(self.engine.duration, 3 / 48000)

    def test_format_time_clamps_negative_and_supports_hours(self):
        self.assertEqual(format_time(-1), "0:00")
        self.assertEqual(format_time(0), "0:00")
        self.assertEqual(format_time(65.9), "1:05")
        self.assertEqual(format_time(3661), "1:01:01")


if __name__ == "__main__":
    unittest.main()
