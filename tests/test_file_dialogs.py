import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.audio_export import CodecAvailability
from core.routing import is_audio_path, is_midi_path
from core.settings import Settings
from ui.file_dialogs import (
    audio_import_filter,
    delivery_filter,
    ensure_extension,
    open_audio_file,
    open_audio_files,
    save_audio_file,
)
from ui.main_window import MainWindow


class _DialogStub:
    open_calls = []
    multi_open_calls = []
    save_calls = []

    @classmethod
    def getOpenFileName(cls, parent, title, start_dir, file_filter):
        cls.open_calls.append((title, start_dir, file_filter))
        source = Path(start_dir) / "reference.wav"
        return str(source), file_filter

    @classmethod
    def getOpenFileNames(cls, parent, title, start_dir, file_filter):
        cls.multi_open_calls.append((title, start_dir, file_filter))
        source = Path(start_dir)
        return [str(source / "one.wav"), str(source / "two.ogg")], file_filter

    @classmethod
    def getSaveFileName(cls, parent, title, start_path, file_filter):
        cls.save_calls.append((title, start_path, file_filter))
        return str(Path(start_path).with_suffix("")), file_filter


class FileDialogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = self.root / "config"
        self.output = self.root / "output"
        self.models = self.root / "models"
        self.trash = self.root / "trash"
        for path in (self.config, self.output, self.models, self.trash):
            path.mkdir(parents=True, exist_ok=True)
        self._settings_stack = ExitStack()
        self._settings_stack.enter_context(
            mock.patch("core.settings.get_config_dir", return_value=self.config)
        )
        self._settings_stack.enter_context(
            mock.patch("core.settings.get_default_output_dir", return_value=self.output)
        )
        self._settings_stack.enter_context(
            mock.patch("core.settings.get_default_cache_dir", return_value=self.models)
        )
        self._settings_stack.enter_context(
            mock.patch("core.settings.get_trash_dir", return_value=self.trash)
        )
        Settings._instance = None
        _DialogStub.open_calls.clear()
        _DialogStub.multi_open_calls.clear()
        _DialogStub.save_calls.clear()

    def tearDown(self):
        Settings._instance = None
        self._settings_stack.close()
        self._tmp.cleanup()

    def test_audio_and_midi_extensions_are_shared_with_drop_routing(self):
        self.assertIn("*.aiff", audio_import_filter())
        self.assertIn("*.aif", audio_import_filter())
        self.assertTrue(is_audio_path("take.M4A"))
        self.assertTrue(is_audio_path("take.aif"))
        self.assertFalse(is_audio_path("notes.txt"))
        self.assertTrue(is_midi_path("composition.MIDI"))

    def test_open_and_multi_open_remember_operation_directories(self):
        source_dir = self.root / "sources"
        source_dir.mkdir()
        source = source_dir / "reference.wav"
        source.touch()
        settings = Settings()
        settings.set("general.output_dir", str(self.output))

        path, _ = open_audio_file(
            None,
            "Reference",
            operation_kind="reference_import",
            dialog=_DialogStub,
            fallback_dir=source_dir,
        )
        self.assertEqual(str(source), path)
        self.assertEqual(str(source_dir), settings.get("general.file_dialog_dirs.reference_import"))

        paths, _ = open_audio_files(
            None,
            "Stems",
            operation_kind="stem_import",
            dialog=_DialogStub,
            fallback_dir=source_dir,
        )
        self.assertEqual(2, len(paths))
        self.assertEqual(str(source_dir), _DialogStub.multi_open_calls[-1][1])
        self.assertEqual(str(source_dir), settings.get("general.file_dialog_dirs.stem_import"))

    def test_save_filter_uses_available_delivery_codecs_and_remembers_directory(self):
        formats = {
            name: CodecAvailability(name, name in {"wav", "flac"}, "test")
            for name in ("wav", "flac", "mp3", "ogg", "opus")
        }
        with mock.patch("ui.file_dialogs.probe_codecs", return_value=formats):
            path, selected = save_audio_file(
                None,
                "Export",
                "song.wav",
                operation_kind="audio_export",
                dialog=_DialogStub,
            )
        self.assertIn("WAV (*.wav)", _DialogStub.save_calls[-1][2])
        self.assertIn("FLAC (*.flac)", _DialogStub.save_calls[-1][2])
        self.assertNotIn("MP3 (*.mp3)", _DialogStub.save_calls[-1][2])
        self.assertEqual(str(self.output), Settings().get("general.file_dialog_dirs.audio_export"))
        self.assertEqual("song.wav", Path(ensure_extension(path, selected)).name)
        self.assertEqual("ogg", ensure_extension(str(self.root / "mix"), "OGG (*.ogg)").rsplit(".", 1)[-1])

    def test_dropped_audio_routes_without_playback(self):
        window = MainWindow.__new__(MainWindow)
        window._pages = SimpleNamespace(currentIndex=lambda: 5)
        window._route_to_mixer = mock.Mock()
        window._route_to_forge_reference = mock.Mock()
        window._on_send_to_vocals = mock.Mock()
        window.toast_mgr = mock.Mock()

        window._load_dropped_audio("take.ogg")

        window._route_to_mixer.assert_called_once_with("take.ogg", "drag_drop")
        window._route_to_forge_reference.assert_not_called()
        window._on_send_to_vocals.assert_not_called()


if __name__ == "__main__":
    unittest.main()
