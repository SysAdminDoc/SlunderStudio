import ast
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from core.credentials import MemoryCredentialStore
from core.settings import Settings
from core.trash import TrashError, TrashManager
from ui.batch_view import BatchCard, BatchView
from ui.mixer_view import MixerView
from ui.seed_explorer import SeedExplorer


class _ToastStub:
    def __init__(self):
        self.actions = []
        self.messages = []

    def info(self, message, **kwargs):
        self.messages.append(message)
        callback = kwargs.get("action_callback")
        if callback:
            self.actions.append(callback)

    def success(self, message, **_kwargs):
        self.messages.append(message)

    def warning(self, message, **_kwargs):
        self.messages.append(message)

    def error(self, message, **_kwargs):
        self.messages.append(message)


class RecoverableDestructiveActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_trash_batch_rolls_back_when_one_path_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.wav"
            second = root / "second.wav"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            trash = TrashManager(root / "trash")

            with self.assertRaises(TrashError):
                trash.trash_paths([
                    {"path": first, "category": "generated_asset", "label": "first"},
                    {"path": second, "category": "generated_asset", "label": "second"},
                    {"path": root / "missing.wav", "category": "generated_asset", "label": "missing"},
                ])

            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertEqual([], trash.list_entries())

    def test_settings_snapshot_restores_non_secret_and_os_secret_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            store = MemoryCredentialStore()
            with ExitStack() as stack:
                stack.enter_context(mock.patch("core.settings.get_config_dir", return_value=config_dir))
                stack.enter_context(mock.patch("core.settings.get_default_output_dir", return_value=root / "out"))
                stack.enter_context(mock.patch("core.settings.get_default_cache_dir", return_value=root / "models"))
                stack.enter_context(mock.patch("core.credentials.get_credential_store", return_value=store))
                Settings._instance = None
                settings = Settings()
                settings.set("general.bit_depth", 16)
                settings.set("model_hub.hf_token", "hf_snapshot_test")
                snapshot = settings.snapshot()

                settings.reset_all()
                self.assertEqual(24, settings.get("general.bit_depth"))
                self.assertEqual("", settings.get("model_hub.hf_token"))

                settings.restore_snapshot(snapshot)
                self.assertEqual(16, settings.get("general.bit_depth"))
                self.assertEqual("hf_snapshot_test", settings.get("model_hub.hf_token"))
                self.assertNotIn(
                    "hf_snapshot_test",
                    (config_dir / "config.json").read_text(encoding="utf-8"),
                )
            Settings._instance = None

    def test_mixer_remove_undo_restores_audio_and_strip_state(self):
        toast = _ToastStub()
        view = MixerView(toast_mgr=toast)
        try:
            view.add_track("Lead", np.ones((256, 2), dtype=np.float32), 22050)
            view._strips[0].set_mix_state(volume=0.65, pan=-0.25, muted=True, soloed=False)

            view._on_remove_track(0)
            self.assertEqual(0, len(view._tracks))
            self.assertTrue(toast.actions)
            toast.actions[-1]()

            self.assertEqual(1, len(view._tracks))
            self.assertEqual("Lead", view._tracks[0]["name"])
            self.assertAlmostEqual(0.65, view._strips[0].volume, places=2)
            self.assertAlmostEqual(-0.25, view._strips[0].pan, places=2)
            self.assertTrue(view._strips[0].is_muted)
        finally:
            view.deleteLater()

    def test_seed_explore_undo_restores_previous_grid(self):
        toast = _ToastStub()
        explorer = SeedExplorer(toast_mgr=toast)
        try:
            cell = explorer._cells[0][0]
            cell.set_result("previous.wav", 123)
            cell._toggle_star()

            explorer._start_exploration()
            self.assertTrue(cell._is_generating)
            self.assertTrue(toast.actions)
            toast.actions[-1]()

            self.assertEqual("previous.wav", cell.audio_path)
            self.assertEqual(123, cell.seed)
            self.assertTrue(cell.is_starred)
            self.assertTrue(cell._is_generated)
        finally:
            explorer.deleteLater()

    def test_batch_delete_undo_restores_file_and_card(self):
        toast = _ToastStub()
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "variation.wav"
            audio_path.write_bytes(b"audio")
            with mock.patch.object(BatchCard, "_start_quality_job"):
                view = BatchView(toast_mgr=toast)
                try:
                    view.add_result(str(audio_path), seed=77, gen_time=1.5)
                    view._on_delete(0)
                    self.assertFalse(audio_path.exists())
                    self.assertEqual(0, view.count)
                    self.assertTrue(toast.actions)

                    toast.actions[-1]()
                    self.assertTrue(audio_path.exists())
                    self.assertEqual(1, view.count)
                    self.assertEqual(77, view._cards[0].seed)
                finally:
                    view.deleteLater()

    def test_destructive_handlers_are_explicitly_recoverable(self):
        repo = Path(__file__).resolve().parents[1]
        required = {
            "ui/sfx_view.py": ("_on_delete_card", "_clear_results"),
            "ui/batch_view.py": ("_on_delete", "clear"),
            "ui/mixer_view.py": ("_on_remove_track",),
            "ui/settings_view.py": ("_reset_all",),
            "ui/seed_explorer.py": ("_start_exploration", "_rebuild_grid"),
        }
        recovery_markers = ("trash", "snapshot", "restore", "Undo")
        for relative_path, function_names in required.items():
            source = (repo / relative_path).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative_path)
            functions = {
                node.name: node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for function_name in function_names:
                self.assertIn(function_name, functions, relative_path)
                body = ast.get_source_segment(source, functions[function_name]) or ""
                self.assertTrue(
                    any(marker in body for marker in recovery_markers),
                    f"{relative_path}:{function_name} has no recovery path",
                )


if __name__ == "__main__":
    unittest.main()
