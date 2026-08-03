import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from core.settings import APP_VERSION, SETTINGS_SCHEMA_VERSION, Settings


class SettingsTests(unittest.TestCase):
    def tearDown(self):
        Settings._instance = None

    def _isolated_settings(self, root: Path):
        config_dir = root / "config"
        output_dir = root / "renders"
        model_dir = root / "models"
        trash_dir = root / "trash"
        for path in (config_dir, output_dir, model_dir, trash_dir):
            path.mkdir(parents=True, exist_ok=True)
        stack = ExitStack()
        stack.enter_context(mock.patch("core.settings.get_config_dir", return_value=config_dir))
        stack.enter_context(mock.patch("core.settings.get_default_output_dir", return_value=output_dir))
        stack.enter_context(mock.patch("core.settings.get_default_cache_dir", return_value=model_dir))
        stack.enter_context(mock.patch("core.settings.get_trash_dir", return_value=trash_dir))
        return stack, config_dir

    def test_defaults_include_audio_output_and_nested_round_trip_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack, config_dir = self._isolated_settings(Path(tmp))
            with stack:
                settings = Settings()
                self.assertEqual(settings.get("schema_version"), SETTINGS_SCHEMA_VERSION)
                self.assertEqual(settings.get("version"), APP_VERSION)
                self.assertEqual(settings.get("general.audio_output_device"), "")

                identity = "Windows WASAPI::Studio monitors"
                settings.set("general.audio_output_device", identity)
                saved = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["general"]["audio_output_device"], identity)

                Settings._instance = None
                reloaded = Settings()
                self.assertEqual(reloaded.get("general.audio_output_device"), identity)

    def test_callbacks_and_section_reads_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack, _config_dir = self._isolated_settings(Path(tmp))
            with stack:
                settings = Settings()
                events = []
                settings.on_change(lambda key, new, old: events.append((key, new, old)))
                settings.set("general.sample_rate", 44100)
                self.assertEqual(events[-1], ("general.sample_rate", 44100, 48000))

                section = settings.get_section("general")
                section["sample_rate"] = 1
                self.assertEqual(settings.get("general.sample_rate"), 44100)
                settings.remove_callback(settings._callbacks[0])
                settings.set("general.sample_rate", 48000)
                self.assertEqual(len(events), 1)

    def test_legacy_config_migrates_and_preserves_unknown_compatible_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stack, config_dir = self._isolated_settings(root)
            config_path = config_dir / "config.json"
            config_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "version": "0.1.0",
                    "general": {"audio_format": "flac", "custom_value": "keep"},
                    "song_forge": {"cfg_scale": 4.0},
                }),
                encoding="utf-8",
            )
            with stack:
                settings = Settings()
                saved = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(settings.repair_status["status"], "migrated")
            self.assertEqual(saved["schema_version"], SETTINGS_SCHEMA_VERSION)
            self.assertEqual(saved["version"], APP_VERSION)
            self.assertEqual(saved["general"]["audio_format"], "flac")
            self.assertEqual(saved["general"]["custom_value"], "keep")
            self.assertNotIn("cfg_scale", saved["song_forge"])
            self.assertIn("audio_output_device", saved["general"])

    def test_corrupt_config_restores_defaults_and_records_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stack, config_dir = self._isolated_settings(root)
            config_path = config_dir / "config.json"
            config_path.write_text("{bad", encoding="utf-8")
            with stack:
                settings = Settings()

            self.assertEqual(settings.repair_status["status"], "repaired")
            self.assertTrue(settings.repair_status["backup_paths"])
            self.assertTrue(Path(settings.repair_status["backup_paths"][0]).is_file())
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["general"]["audio_output_device"], "")


if __name__ == "__main__":
    unittest.main()
