import os
import inspect
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QHBoxLayout, QPushButton

from core.audio_export import configured_export_settings
from core.device import configured_cuda_index, configured_torch_device
from core.engine_contract import ModelReadiness
from core.model_manager import ModelInfo, ModelManager, ModelCategory
from core.settings import Settings, get_configured_output_dir
from ui.model_hub import ModelHubView
from ui.onboarding import (
    OnboardingWizard,
    SystemCheckPage,
    check_system,
    model_readiness_label,
    run_dependency_setup,
)


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

    def test_onboarding_saves_preferences_and_carries_model_handoff(self):
        app = QApplication.instance() or QApplication([])
        del app
        core_info = ModelInfo(
            model_id="core-fixture",
            name="Core Fixture",
            description="fixture",
            category=ModelCategory.SONG_FORGE,
            vram_gb=2.0,
            disk_gb=1.5,
            license="MIT",
            source="fixture/source",
            loader_module="fixture.engine",
            loader_fn="load_model",
            is_core=True,
        )
        readiness = ModelReadiness(
            "core-fixture", False, False, False, False, "not_downloaded"
        )
        manager = SimpleNamespace(
            get_core_models=lambda: [core_info],
            get_model_readiness=lambda _model_id: readiness,
            is_offline=False,
            _get_hf_token=lambda: None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self._isolated_settings(Path(tmp)), mock.patch(
                "ui.onboarding.ModelManager", return_value=manager
            ):
                settings = Settings()
                wizard = OnboardingWizard()
                wizard._models._model_action.setCurrentIndex(1)
                wizard._quickstart._output_dir.setText(str(Path(tmp) / "renders"))
                advanced = wizard._quickstart._experience.findData("advanced")
                wizard._quickstart._experience.setCurrentIndex(advanced)

                wizard._finish()

                self.assertEqual(
                    {"model_id": "core-fixture", "action": "download"},
                    wizard.model_handoff(),
                )
                self.assertEqual(str(Path(tmp) / "renders"), settings.get("general.output_dir"))
                self.assertEqual("advanced", settings.get("general.experience_level"))
                self.assertTrue(settings.get("general.onboarding_complete"))
                wizard.deleteLater()

    def test_failed_system_rows_offer_remediation_actions(self):
        app = QApplication.instance() or QApplication([])
        del app
        page = SystemCheckPage()
        requested = []
        page.remediation_requested.connect(requested.append)
        page._display_checks(
            {
                "python": "3.9.0",
                "python_ok": False,
                "deps_ok": False,
                "deps_missing": ["torch"],
                "setup_command": "python -m pip install -r requirements.txt",
                "os": "Windows",
                "arch": "AMD64",
                "cuda": False,
                "gpu_name": "None detected",
                "vram_gb": 0,
                "ram_gb": 4.0,
                "ram_ok": False,
                "disk_free_gb": 2.0,
                "disk_ok": False,
            }
        )
        buttons = page.findChildren(QPushButton, "remediationButton")
        self.assertEqual(5, len(buttons))
        gpu_button = next(button for button in buttons if button.text() == "Choose a model")
        gpu_button.click()
        self.assertEqual(["gpu"], requested)
        page.deleteLater()

    def test_dependency_setup_uses_the_project_requirements_file(self):
        result = SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch("ui.onboarding.subprocess.run", return_value=result) as run:
            message = run_dependency_setup()

        command = run.call_args.args[0]
        self.assertEqual(["-m", "pip", "install", "-r"], command[1:5])
        self.assertEqual(sys.executable, command[0])
        self.assertIn("installed", message.lower())

    def test_system_check_failures_do_not_report_false_green(self):
        with mock.patch(
            "core.deps.dependency_status",
            side_effect=RuntimeError("probe failed"),
        ):
            checks = check_system()

        self.assertFalse(checks["deps_ok"])
        self.assertTrue(checks["deps_missing"])

    def test_model_hub_accepts_an_onboarding_selection(self):
        view = ModelHubView.__new__(ModelHubView)
        card = mock.Mock()
        card.info = SimpleNamespace(name="Core Fixture")
        view._cards = {"core-fixture": card}
        view._search = mock.Mock()
        view._category_filter = mock.Mock()
        view._downloaded_only = mock.Mock()
        view._filter_cards = mock.Mock()

        self.assertTrue(view.prepare_onboarding_model("core-fixture", "open"))
        view._search.clear.assert_called_once_with()
        view._filter_cards.assert_called_once_with()
        card.setVisible.assert_called_once_with(True)
        self.assertFalse(view.prepare_onboarding_model("missing", "open"))


if __name__ == "__main__":
    unittest.main()
