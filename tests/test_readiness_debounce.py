import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.engine_contract import CAP_VOCAL_SYNTHESIZE
from core.model_manager import ModelManager, ModelStatus
from ui.vocal_suite_view import VocalSuiteView


class ReadinessDebounceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def _wait_for(cls, predicate, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            cls._app.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        cls._app.processEvents()
        if not predicate():
            raise AssertionError("Timed out waiting for debounced readiness")

    def test_text_keystrokes_schedule_one_readiness_refresh(self):
        view = VocalSuiteView()
        try:
            readiness = SimpleNamespace(
                can_run=False,
                remedy="Install the voice model.",
                output_summary="audio",
            )
            with patch.object(
                view._model_mgr,
                "get_capability_readiness",
                return_value=readiness,
            ) as resolve:
                view._sing_lyrics.clear()
                self._app.processEvents()
                resolve.reset_mock()

                for text in ("a", "ab", "abc", "abcd", "abcde"):
                    view._sing_lyrics.setPlainText(text)

                self.assertEqual(0, resolve.call_count)
                self._wait_for(lambda: resolve.call_count == 1)
                self.assertEqual(
                    CAP_VOCAL_SYNTHESIZE,
                    resolve.call_args.args[0],
                )
        finally:
            view.close()

    def test_empty_inputs_skip_readiness_resolution(self):
        view = VocalSuiteView()
        try:
            with patch.object(
                view._model_mgr,
                "get_capability_readiness",
                side_effect=AssertionError("empty input resolved readiness"),
            ):
                view._sing_lyrics.clear()
                view._clone_text.clear()
                view._rvc_input_label.setProperty("path", "")
                view._stem_input_label.setProperty("path", "")
                view._refresh_capability_states()
                self._app.processEvents()
        finally:
            view.close()

    def test_model_readiness_cache_invalidates_on_status_change(self):
        manager = ModelManager()
        manager._readiness_cache.clear()
        manager._readiness_cache_state = None
        model_id = next(
            model_id
            for model_id, info in manager.registry.items()
            if not info.pip_managed
        )
        original_status = manager.get_status(model_id)
        try:
            with patch.object(manager, "_is_model_cached", return_value=True) as cached:
                manager.get_model_readiness(model_id)
                manager.get_model_readiness(model_id)
                self.assertEqual(1, cached.call_count)

                manager._set_status(model_id, ModelStatus.DOWNLOADED)
                manager.get_model_readiness(model_id)
                self.assertEqual(2, cached.call_count)
        finally:
            manager._set_status(model_id, original_status)

    def test_disk_usage_cache_reuses_scan_until_status_changes(self):
        manager = ModelManager()
        old_settings = manager._settings
        model_id = next(iter(manager.registry))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "nested" / "weights.bin").write_bytes(b"weights")

            class SettingsStub:
                def get(self, key, default=None):
                    if key == "model_hub.cache_dir":
                        return str(root)
                    return default

            manager._settings = SettingsStub()
            manager._disk_usage_cache = None
            manager._disk_usage_cache_path = None
            try:
                original_rglob = Path.rglob
                scans = []

                def tracked_rglob(path, pattern):
                    scans.append((path, pattern))
                    return original_rglob(path, pattern)

                with patch.object(Path, "rglob", tracked_rglob):
                    first = manager.get_total_disk_usage()
                    second = manager.get_total_disk_usage()
                    self.assertEqual(first, second)
                    self.assertEqual(1, len(scans))

                    manager._set_status(model_id, manager.get_status(model_id))
                    manager.get_total_disk_usage()
                    self.assertEqual(2, len(scans))
            finally:
                manager._settings = old_settings


if __name__ == "__main__":
    unittest.main()
