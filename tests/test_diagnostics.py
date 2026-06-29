import os
import json
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.diagnostics import (
    collect_health_report,
    export_health_report,
    format_health_report_text,
    redact_text,
)
from core.job_state import JobStore
from core.model_manager import ModelManager
from core.settings import Settings
from ui.settings_view import SettingsView


class DiagnosticsTests(unittest.TestCase):
    def tearDown(self):
        Settings._instance = None
        ModelManager._instance = None

    def test_report_omits_private_job_inputs_and_redacts_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._patched_runtime(root):
                settings = Settings()
                settings.set("model_hub.hf_token", "hf_SECRET1234567890")
                store = JobStore(root / "jobs")
                record = store.create(
                    "song_generation",
                    "Failed render",
                    inputs={
                        "lyrics": "private chorus line",
                        "prompt": "trap metal song",
                        "token": "hf_INPUT1234567890",
                    },
                )
                store.mark_failed(
                    record.id,
                    "Decoder failed with hf_ERROR1234567890",
                )
                manager = ModelManager()

                report = collect_health_report(
                    settings=settings,
                    model_manager=manager,
                    job_store=store,
                )

            payload = json.dumps(report)
            self.assertIn("dependencies", report)
            self.assertIn("ffmpeg", report)
            self.assertIn("models", report)
            self.assertIn("recent_job_failures", report)
            self.assertIn("lyrics", report["recent_job_failures"][0]["input_keys"])
            self.assertTrue(report["recent_job_failures"][0]["error_present"])
            self.assertNotIn("private chorus line", payload)
            self.assertNotIn("trap metal song", payload)
            self.assertNotIn("hf_SECRET1234567890", payload)
            self.assertNotIn("hf_INPUT1234567890", payload)
            self.assertNotIn("hf_ERROR1234567890", payload)

    def test_private_opt_in_includes_redacted_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._patched_runtime(root):
                settings = Settings()
                store = JobStore(root / "jobs")
                record = store.create(
                    "song_generation",
                    "Failed render",
                    inputs={"lyrics": "private chorus line", "hf_token": "hf_INPUT1234567890"},
                )
                store.mark_failed(record.id, "Decoder failed with hf_ERROR1234567890")
                manager = ModelManager()

                report = collect_health_report(
                    include_private=True,
                    settings=settings,
                    model_manager=manager,
                    job_store=store,
                )

            payload = json.dumps(report)
            self.assertIn("private chorus line", payload)
            self.assertIn("[REDACTED]", payload)
            self.assertNotIn("hf_INPUT1234567890", payload)
            self.assertNotIn("hf_ERROR1234567890", payload)

    def test_export_health_report_writes_json_and_text_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._patched_runtime(root):
                settings = Settings()
                store = JobStore(root / "jobs")
                manager = ModelManager()
                output = export_health_report(
                    root / "support-bundle",
                    settings=settings,
                    model_manager=manager,
                    job_store=store,
                )

            self.assertEqual(output.suffix, ".zip")
            with zipfile.ZipFile(output) as bundle:
                names = set(bundle.namelist())
                self.assertEqual(names, {"health-report.json", "health-report.txt"})
                data = bundle.read("health-report.json").decode("utf-8")
                text = bundle.read("health-report.txt").decode("utf-8")
            self.assertIn('"schema_version"', data)
            self.assertIn("Slunder Studio Health Report", text)

    def test_text_formatter_and_redactor_are_stable(self):
        redacted = redact_text("token=hf_SECRET1234567890")
        self.assertNotIn("hf_", redacted)
        text = format_health_report_text({
            "schema_version": 1,
            "generated_at": "2026-06-29T00:00:00Z",
            "app": {"version": "0.1.17", "platform": "test", "python": "3.12", "frozen": False},
            "gpu": {"name": "No GPU detected", "free_gb": 0},
            "ffmpeg": {"available": False},
            "paths": {
                "config_dir": "<SLUNDER_CONFIG>",
                "output_dir": "<SLUNDER_OUTPUT>",
                "model_cache_dir": "<SLUNDER_MODEL_CACHE>",
            },
            "models": [],
            "recent_job_failures": [],
            "dependencies": {"PySide6": {"version": "6.test"}},
        })
        self.assertIn("Recent Job Failures", text)
        self.assertIn("- None", text)

    def test_settings_export_handler_uses_private_opt_in(self):
        app = QApplication.instance() or QApplication([])
        del app
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._patched_runtime(root):
                toast = _ToastProbe()
                view = SettingsView(toast_mgr=toast)
                target = root / "settings-report.zip"
                try:
                    view._health_private_inputs.setChecked(True)
                    with mock.patch(
                        "ui.settings_view.QFileDialog.getSaveFileName",
                        return_value=(str(target), "Health Report (*.zip)"),
                    ), mock.patch(
                        "ui.settings_view.export_health_report",
                        return_value=target,
                    ) as export:
                        view._export_health_report()
                    export.assert_called_once_with(str(target), include_private=True)
                    self.assertIn("Health report exported", toast.successes[-1])
                finally:
                    view.deleteLater()

    def _patched_runtime(self, root: Path):
        config_dir = root / "config"
        output_dir = root / "renders"
        model_dir = root / "models"
        trash_dir = root / "trash"
        config_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        trash_dir.mkdir(parents=True, exist_ok=True)
        Settings._instance = None
        ModelManager._instance = None
        stack = ExitStack()
        stack.enter_context(mock.patch("core.settings.get_config_dir", return_value=config_dir))
        stack.enter_context(mock.patch("core.settings.get_default_output_dir", return_value=output_dir))
        stack.enter_context(mock.patch("core.settings.get_default_cache_dir", return_value=model_dir))
        stack.enter_context(mock.patch("core.settings.get_trash_dir", return_value=trash_dir))
        stack.enter_context(mock.patch("core.model_manager.get_config_dir", return_value=config_dir))
        stack.enter_context(mock.patch("core.diagnostics.get_config_dir", return_value=config_dir))
        stack.enter_context(mock.patch("core.diagnostics.get_trash_dir", return_value=trash_dir))
        return stack


class _ToastProbe:
    def __init__(self):
        self.successes: list[str] = []
        self.errors: list[str] = []

    def success(self, message: str):
        self.successes.append(message)

    def error(self, message: str):
        self.errors.append(message)


if __name__ == "__main__":
    unittest.main()
