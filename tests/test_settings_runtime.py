import os
import inspect
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QHBoxLayout

from core.audio_export import configured_export_settings
from core.device import configured_cuda_index, configured_torch_device
from core.engine_contract import ModelReadiness
from core.model_manager import ModelInfo, ModelManager, ModelCategory
from core.settings import Settings, get_configured_output_dir
from ui.onboarding import OnboardingWizard, SystemCheckPage, model_readiness_label


ROOT = Path(__file__).resolve().parents[1]


class SettingsRuntimeTests(unittest.TestCase):
    def tearDown(self):
        Settings._instance = None
        ModelManager._instance = None

    def _isolated_settings(self, root: Path):
        config = root / "config"
        output = root / "renders"
        models = root / "models"
        trash = root / "trash"
        for path in (config, output, models, trash):
            path.mkdir(parents=True, exist_ok=True)
        stack = ExitStack()
        stack.enter_context(mock.patch("core.settings.get_config_dir", return_value=config))
        stack.enter_context(mock.patch("core.settings.get_default_output_dir", return_value=output))
        stack.enter_context(mock.patch("core.settings.get_default_cache_dir", return_value=models))
        stack.enter_context(mock.patch("core.settings.get_trash_dir", return_value=trash))
        return stack

    def test_configured_output_and_export_defaults_follow_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._isolated_settings(root):
                settings = Settings()
                configured = root / "chosen-renders"
                settings.set("general.output_dir", str(configured))
                settings.set("general.audio_format", "flac")
                settings.set("general.sample_rate", 44100)
                settings.set("general.bit_depth", 32)

                self.assertEqual(configured, get_configured_output_dir())
                export = configured_export_settings()
                self.assertEqual("flac", export.format)
                self.assertEqual(44100, export.sample_rate)
                self.assertEqual(32, export.bit_depth)

    def test_gpu_selection_is_clamped_and_used_by_torch_engines(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 3),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self._isolated_settings(Path(tmp)):
                settings = Settings()
                settings.set("general.gpu_device", 99)
                self.assertEqual(2, configured_cuda_index(fake_torch))
                self.assertEqual("cuda:2", configured_torch_device(fake_torch))

    def test_model_download_respects_cache_admission_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._isolated_settings(Path(tmp)):
                manager = ModelManager()
                manager._settings.set("general.max_cache_gb", 1.0)
                manager._registry = {
                    "fixture": ModelInfo(
                        model_id="fixture",
                        name="Fixture",
                        description="fixture",
                        category=ModelCategory.EXTRAS,
                        vram_gb=1.0,
                        disk_gb=2.0,
                        license="MIT",
                        source="fixture/source",
                        loader_module="fixture.engine",
                        loader_fn="load_model",
                    )
                }
                with (
                    mock.patch.object(manager, "_validate_registry_revision"),
                    mock.patch.object(manager, "_is_model_cached", return_value=False),
                    mock.patch.object(manager, "get_total_disk_usage", return_value=0.5),
                ):
                    with self.assertRaisesRegex(RuntimeError, "cache limit is 1.0 GB"):
                        manager.download_model("fixture")

    def test_readiness_labels_keep_lifecycle_states_distinct(self):
        cases = [
            (ModelReadiness("id", False, False, False, False, "not_downloaded"), "not downloaded"),
            (ModelReadiness("id", False, False, False, False, "not_downloaded"), "offline"),
            (ModelReadiness("id", True, True, True, False, "downloaded"), "downloaded / loadable"),
            (ModelReadiness("id", True, True, False, False, "downloaded"), "installed / not loadable"),
            (ModelReadiness("id", True, True, True, True, "loaded"), "loaded"),
            (ModelReadiness("id", True, False, False, False, "error"), "error"),
        ]
        for readiness, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(expected, model_readiness_label(readiness, expected == "offline"))

    def test_audited_visible_settings_have_non_ui_runtime_consumers(self):
        consumers = {
            "general.output_dir": ["core/settings.py", "engines/ace_step_engine.py"],
            "general.audio_format": ["core/audio_export.py"],
            "general.gpu_device": ["core/device.py", "core/model_manager.py"],
            "general.experience_level": ["ui/song_forge_view.py"],
            "general.max_cache_gb": ["core/model_manager.py"],
            "song_forge.timestep_shift": ["ui/song_forge_view.py"],
            "song_forge.inference_steps": ["ui/song_forge_view.py"],
            "song_forge.batch_count": ["ui/song_forge_view.py"],
            "song_forge.default_duration": ["ui/song_forge_view.py"],
            "midi_studio.default_bpm": ["ui/midi_studio_view.py"],
            "production.mastering_target": ["ui/mixer_view.py"],
            "production.mastering_auto_eq": ["ui/mixer_view.py", "core/mastering.py"],
            "production.mastering_auto_compress": ["ui/mixer_view.py", "core/mastering.py"],
        }
        for key, paths in consumers.items():
            with self.subTest(key=key):
                self.assertTrue(
                    any(key in (ROOT / path).read_text(encoding="utf-8") for path in paths),
                    f"No runtime consumer found for {key}",
                )

    def test_onboarding_dismissal_stays_incomplete_and_explicit_skip_is_distinct(self):
        app = QApplication.instance() or QApplication([])
        del app
        with tempfile.TemporaryDirectory() as tmp:
            with self._isolated_settings(Path(tmp)), mock.patch(
                "ui.onboarding.ModelManager",
                return_value=SimpleNamespace(get_core_models=lambda: [], is_offline=False),
            ):
                settings = Settings()
                wizard = OnboardingWizard()
                wizard.reject()
                self.assertFalse(settings.get("general.onboarding_complete"))
                self.assertFalse(settings.get("general.onboarding_skipped"))
                wizard._skip()
                self.assertFalse(settings.get("general.onboarding_complete"))
                self.assertTrue(settings.get("general.onboarding_skipped"))
                wizard._finish()
                self.assertTrue(settings.get("general.onboarding_complete"))
                self.assertFalse(settings.get("general.onboarding_skipped"))
                wizard.deleteLater()

        import main

        self.assertNotIn(
            'settings.set("general.onboarding_complete", True)',
            inspect.getsource(main._launch_app),
        )

    def test_system_check_rows_are_replaced_before_redraw(self):
        app = QApplication.instance() or QApplication([])
        del app
        page = SystemCheckPage()
        page._checks_layout.addLayout(QHBoxLayout())
        page._checks_layout.addLayout(QHBoxLayout())
        page._clear_check_rows()
        self.assertEqual(0, page._checks_layout.count())
        page.deleteLater()


if __name__ == "__main__":
    unittest.main()
