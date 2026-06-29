import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.i18n import (
    DEFAULT_LOCALE,
    REQUIRED_I18N_KEYS,
    available_locales,
    language_code_from_label,
    missing_keys,
    tr,
)
from core.lyrics_db import LyricsDB
from core.settings import Settings
from core.voice_bank import VoiceBank
from engines.lyrics_engine import default_lyrics_language
from engines.lyrics_templates import build_quick_prompt
from ui.lyrics_view import LyricsView
from ui.settings_view import SettingsView
from ui.vocal_suite_view import VocalSuiteView


class I18nTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self):
        if LyricsDB._instance is not None:
            LyricsDB._instance.close()
        Settings._instance = None
        LyricsDB._instance = None
        VoiceBank._instance = None

    def test_english_catalog_covers_required_ui_keys(self):
        self.assertIn(DEFAULT_LOCALE, available_locales())
        self.assertEqual([], missing_keys(REQUIRED_I18N_KEYS))
        self.assertEqual("Lyrics", tr("nav.lyrics"))
        self.assertEqual("Slunder Studio v9.9.9", tr("app.window_title", version="9.9.9"))

    def test_language_labels_normalize_to_prompt_codes(self):
        self.assertEqual("en", language_code_from_label("English"))
        self.assertEqual("es", language_code_from_label("Spanish"))
        self.assertEqual("zh", language_code_from_label("Chinese (Mandarin)"))
        self.assertEqual("zh", language_code_from_label("zh-CN"))
        self.assertEqual("en", language_code_from_label(""))

    def test_quick_prompt_adds_non_english_language_instruction(self):
        english_system, _english_user = build_quick_prompt("dark trap metal", language="en")
        spanish_system, _spanish_user = build_quick_prompt("dark trap metal", language="es")

        self.assertNotIn("WRITE THE LYRICS IN", english_system)
        self.assertIn("WRITE THE LYRICS IN: es", spanish_system)

    def test_default_lyrics_language_reads_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_config(Path(tmp)):
                settings = Settings()
                settings.set("lyrics.default_language", "French")
                self.assertEqual("fr", default_lyrics_language(settings))

    def test_settings_view_saves_default_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_config(Path(tmp)):
                view = SettingsView()
                try:
                    idx = view._default_language.findText("Japanese")
                    self.assertGreaterEqual(idx, 0)
                    view._default_language.setCurrentIndex(idx)
                    self.assertEqual("ja", Settings().get("lyrics.default_language"))
                finally:
                    view.deleteLater()

    def test_lyrics_view_initializes_and_saves_default_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_config(Path(tmp)):
                Settings().set("lyrics.default_language", "Spanish")
                view = LyricsView()
                try:
                    self.assertEqual("es", view._selected_language_code())
                    idx = view._lang_combo.findText("French")
                    self.assertGreaterEqual(idx, 0)
                    view._lang_combo.setCurrentIndex(idx)
                    self.assertEqual("fr", Settings().get("lyrics.default_language"))
                finally:
                    view.deleteLater()
                    if LyricsDB._instance is not None:
                        LyricsDB._instance.close()

    def test_vocal_clone_language_uses_supported_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_config(Path(tmp)):
                Settings().set("lyrics.default_language", "zh")
                view = VocalSuiteView()
                try:
                    self.assertEqual("zh", view._clone_language_code())
                finally:
                    view.deleteLater()

    def test_vocal_clone_language_falls_back_for_unsupported_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_config(Path(tmp)):
                Settings().set("lyrics.default_language", "es")
                view = VocalSuiteView()
                try:
                    self.assertEqual("en", view._clone_language_code())
                finally:
                    view.deleteLater()

    def _patched_config(self, root: Path):
        config_dir = root / "config"
        output_dir = root / "renders"
        model_dir = root / "models"
        trash_dir = root / "trash"
        config_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        trash_dir.mkdir(parents=True, exist_ok=True)
        Settings._instance = None
        LyricsDB._instance = None
        VoiceBank._instance = None
        stack = ExitStack()
        stack.enter_context(mock.patch("core.settings.get_config_dir", return_value=config_dir))
        stack.enter_context(mock.patch("core.settings.get_default_output_dir", return_value=output_dir))
        stack.enter_context(mock.patch("core.settings.get_default_cache_dir", return_value=model_dir))
        stack.enter_context(mock.patch("core.settings.get_trash_dir", return_value=trash_dir))
        stack.enter_context(mock.patch("core.lyrics_db.get_config_dir", return_value=config_dir))
        stack.enter_context(mock.patch("core.voice_bank.get_config_dir", return_value=config_dir))
        return stack


if __name__ == "__main__":
    unittest.main()
