import os
import tempfile
import time
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import QApplication

from core.midi_utils import MidiData
from core.settings import Settings
from core.voice_bank import VoiceBank
from engines.ai_producer import PipelineStage, ProducerResult
from engines.rvc_engine import VoiceResult
from ui.ai_producer_view import AIProducerView
from ui.midi_studio_view import MidiStudioView
from ui.vocal_suite_view import VocalSuiteView
from ui.waveform_widget import MiniWaveform, WaveformWidget


class WaveformContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        self.app.processEvents()
        return bool(predicate())

    def test_canonical_loader_renders_file_mono_and_both_stereo_layouts(self):
        widget = WaveformWidget()
        try:
            mono = np.linspace(-0.5, 0.5, 4096, dtype=np.float32)
            self.assertTrue(widget.load_audio(mono, 8000))
            self.assertEqual((4096,), widget._audio_data.shape)
            self.assertAlmostEqual(4096 / 8000, widget.duration)

            stereo = np.column_stack((mono, -mono))
            self.assertTrue(widget.load_audio(stereo, 8000))
            self.assertEqual((4096, 2), widget._audio_data.shape)

            channels_first = stereo.T
            self.assertTrue(widget.load_audio(channels_first, 8000))
            self.assertEqual((4096, 2), widget._audio_data.shape)

            short_channels_first = channels_first[:, :8]
            self.assertTrue(widget.load_audio(short_channels_first, 8000))
            self.assertEqual((8, 2), widget._audio_data.shape)

            with tempfile.TemporaryDirectory() as tmp:
                audio_path = Path(tmp) / "preview.wav"
                sf.write(audio_path, stereo, 8000, subtype="PCM_16")
                self.assertTrue(widget.load_audio(audio_path))
                self.assertTrue(self._wait_for(lambda: widget.has_audio))
                self.assertEqual((4096, 2), widget._audio_data.shape)
                self.assertEqual(8000, widget._sample_rate)
        finally:
            widget.close()

    def test_failed_load_clears_stale_preview_and_reports_the_error(self):
        widget = WaveformWidget()
        try:
            self.assertTrue(widget.load_audio(np.zeros(4096, dtype=np.float32), 8000))
            self.assertTrue(widget.has_audio)

            self.assertFalse(widget.load_audio("missing-waveform-contract.wav"))

            self.assertFalse(widget.has_audio)
            self.assertEqual(0.0, widget.duration)
            self.assertIn("not found", widget.last_error.lower())
            self.assertIn("Error:", widget._info_label.text())
        finally:
            widget.close()

    def test_mini_waveform_skips_spectrogram_and_releases_source_audio(self):
        mini = MiniWaveform()
        try:
            audio = np.sin(np.linspace(0, 20 * np.pi, 4096)).astype(np.float32)
            with patch("librosa.feature.melspectrogram") as melspectrogram:
                self.assertTrue(mini.load_audio(audio, 8000))

            melspectrogram.assert_not_called()
            self.assertTrue(mini._waveform.has_audio)
            self.assertIsNone(mini._waveform._audio_data)
        finally:
            mini.close()

    def test_spectrogram_is_lazy_and_cached_for_each_load(self):
        widget = WaveformWidget()
        try:
            import librosa

            audio = np.sin(np.linspace(0, 20 * np.pi, 4096)).astype(np.float32)
            original = librosa.feature.melspectrogram
            with patch(
                "librosa.feature.melspectrogram",
                side_effect=original,
            ) as melspectrogram:
                self.assertTrue(widget.load_audio(audio, 8000))
                melspectrogram.assert_not_called()

                widget._set_mode("spectrogram")
                self.assertTrue(self._wait_for(lambda: widget._spectrogram_ready))
                self.assertEqual(1, melspectrogram.call_count)
                widget._set_mode("waveform")
                widget._set_mode("spectrogram")
                self.assertEqual(1, melspectrogram.call_count)
        finally:
            widget.close()

    def test_midi_and_ai_producer_success_paths_reach_waveform_display(self):
        stereo = np.zeros((4096, 2), dtype=np.float32)
        midi_view = MidiStudioView()
        ai_view = AIProducerView()
        try:
            midi_view._midi_data = MidiData()
            with tempfile.TemporaryDirectory() as tmp:
                with patch(
                    "core.settings.get_config_dir",
                    return_value=Path(tmp),
                ), patch(
                    "engines.fluidsynth_engine.render_midi_to_audio",
                    return_value=stereo,
                ):
                    midi_view._on_render()

                self.assertTrue(midi_view._waveform.has_audio)
                self.assertEqual((4096, 2), midi_view._waveform._audio_data.shape)
                self.assertEqual(1, midi_view._tabs.currentIndex())

                master_path = Path(tmp) / "producer-master.wav"
                sf.write(master_path, stereo, 8000, subtype="PCM_16")
                result = ProducerResult(
                    final_audio_path=str(master_path),
                    artifact_paths=[str(master_path)],
                    stage=PipelineStage.COMPLETE,
                )
                ai_view._display_result(result)

                self.assertTrue(self._wait_for(lambda: ai_view._waveform.has_audio))
                self.assertTrue(ai_view._waveform.has_audio)
                self.assertTrue(ai_view._export_btn.isEnabled())
        finally:
            midi_view.close()
            ai_view.close()
            self.app.processEvents()

    def test_vocal_clone_and_routed_file_paths_reach_waveform_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stereo = np.zeros((4096, 2), dtype=np.float32)
            routed_path = root / "routed-vocal.wav"
            sf.write(routed_path, stereo, 16000, subtype="PCM_16")

            with self._patched_config(root):
                view = VocalSuiteView()
                try:
                    view._on_clone_generated({
                        "result": VoiceResult(
                            audio=stereo,
                            sample_rate=16000,
                            duration=4096 / 16000,
                        ),
                        "path": str(routed_path),
                    })
                    self.assertTrue(view._clone_waveform.has_audio)
                    self.assertEqual(
                        (4096, 2),
                        view._clone_waveform._audio_data.shape,
                    )

                    view.set_audio(str(routed_path))
                    self.assertTrue(
                        self._wait_for(
                            lambda: (
                                view._melody_waveform.has_audio
                                and view._autotune_waveform.has_audio
                            )
                        )
                    )
                    self.assertTrue(view._melody_waveform.has_audio)
                    self.assertTrue(view._autotune_waveform.has_audio)
                    self.assertEqual(
                        str(routed_path),
                        view._autotune_input_label.property("path"),
                    )
                finally:
                    view.close()
                    self.app.processEvents()

    @contextmanager
    def _patched_config(self, root: Path):
        config_dir = root / "config"
        output_dir = root / "renders"
        model_dir = root / "models"
        trash_dir = root / "trash"
        for path in (config_dir, output_dir, model_dir, trash_dir):
            path.mkdir(parents=True, exist_ok=True)

        Settings._instance = None
        VoiceBank._instance = None
        with ExitStack() as stack:
            stack.enter_context(
                patch("core.settings.get_config_dir", return_value=config_dir)
            )
            stack.enter_context(
                patch("core.settings.get_default_output_dir", return_value=output_dir)
            )
            stack.enter_context(
                patch("core.settings.get_default_cache_dir", return_value=model_dir)
            )
            stack.enter_context(
                patch("core.settings.get_trash_dir", return_value=trash_dir)
            )
            stack.enter_context(
                patch("core.voice_bank.get_config_dir", return_value=config_dir)
            )
            yield
        Settings._instance = None
        VoiceBank._instance = None


if __name__ == "__main__":
    unittest.main()
