import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.audio_engine import AudioEngine, AudioOutputDevice
from core.settings import Settings
from ui.settings_view import SettingsView


class AudioOutputDeviceSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        Settings._instance = None
        AudioEngine._instance = None

    def tearDown(self):
        if AudioEngine._instance is not None:
            AudioEngine._instance.cleanup()
        AudioEngine._instance = None
        Settings._instance = None

    def test_device_choice_persists_and_refreshes_without_restart(self):
        first = AudioOutputDevice(
            index=2,
            name="Studio monitors",
            host_api="Windows WASAPI",
            max_output_channels=2,
            default_sample_rate=48000,
        )
        second = AudioOutputDevice(
            index=4,
            name="USB headphones",
            host_api="MME",
            max_output_channels=2,
            default_sample_rate=44100,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            output_dir = root / "renders"
            model_dir = root / "models"
            trash_dir = root / "trash"
            for path in (config_dir, output_dir, model_dir, trash_dir):
                path.mkdir(parents=True)

            with ExitStack() as stack:
                stack.enter_context(mock.patch("core.settings.get_config_dir", return_value=config_dir))
                stack.enter_context(mock.patch("core.settings.get_default_output_dir", return_value=output_dir))
                stack.enter_context(mock.patch("core.settings.get_default_cache_dir", return_value=model_dir))
                stack.enter_context(mock.patch("core.settings.get_trash_dir", return_value=trash_dir))
                stack.enter_context(
                    mock.patch(
                        "ui.settings_view.enumerate_output_devices",
                        side_effect=[([first], None), ([second], None)],
                    )
                )

                view = SettingsView()
                try:
                    first_index = view._audio_device_combo.findData(first.identity)
                    self.assertGreaterEqual(first_index, 0)
                    self.assertIn("Windows WASAPI", view._audio_device_combo.itemText(first_index))

                    view._audio_device_combo.setCurrentIndex(first_index)
                    self.assertEqual(
                        Settings().get("general.audio_output_device"),
                        first.identity,
                    )
                    self.assertEqual(AudioEngine().output_device_identity, first.identity)

                    # The second query is the refresh action; it takes effect in
                    # this live view and retains the saved choice as unavailable.
                    view._refresh_audio_devices()
                    second_index = view._audio_device_combo.findData(second.identity)
                    self.assertGreaterEqual(second_index, 0)
                    self.assertIn("MME", view._audio_device_combo.itemText(second_index))
                    self.assertIn("Unavailable", view._audio_device_combo.currentText())
                    self.assertIn("unavailable", view._audio_device_status.text())

                    # Selecting the explicit default remains an immediate,
                    # persisted choice and updates the shared transport.
                    view._audio_device_combo.setCurrentIndex(0)
                    self.assertEqual(Settings().get("general.audio_output_device"), "")
                    self.assertEqual(AudioEngine().output_device_identity, "")
                finally:
                    view.deleteLater()


if __name__ == "__main__":
    unittest.main()
