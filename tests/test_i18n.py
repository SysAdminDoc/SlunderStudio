import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QWidget,
)

from core.i18n import (
    DEFAULT_LOCALE,
    PSEUDO_LOCALE,
    REQUIRED_I18N_KEYS,
    available_locales,
    clear_missing_key_log,
    current_locale,
    extract_i18n_keys,
    get_missing_key_log,
    is_rtl,
    language_code_from_label,
    load_catalog,
    missing_keys,
    pseudolocalize,
    pseudolocale_overflow,
    set_locale,
    tr,
    ui_locale_options,
)
from core.lyrics_db import LyricsDB
from core.settings import Settings
from core.voice_bank import VoiceBank
from engines.lyrics_engine import default_lyrics_language
from engines.lyrics_templates import build_quick_prompt
from ui.lyrics_view import LyricsView
from ui.i18n_runtime import apply_pseudolocale
from ui.settings_view import SettingsView
from ui.vocal_suite_view import VocalSuiteView


class I18nTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self):
        set_locale(DEFAULT_LOCALE, persist=False)
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

    def test_builtin_rtl_and_pseudo_catalogs_cover_required_keys(self):
        self.assertIn("ar", available_locales())
        self.assertIn(PSEUDO_LOCALE, available_locales())
        self.assertEqual([], missing_keys(REQUIRED_I18N_KEYS, "ar"))
        self.assertEqual([], missing_keys(REQUIRED_I18N_KEYS, PSEUDO_LOCALE))

    def test_source_translation_keys_are_present_in_every_builtin_catalog(self):
        root = Path(__file__).resolve().parents[1]
        keys = extract_i18n_keys([root / "main.py", *(root / "ui").glob("*.py")])
        self.assertTrue(keys)
        self.assertEqual([], missing_keys(keys, DEFAULT_LOCALE))
        self.assertEqual([], missing_keys(keys, "ar"))

    def test_pseudolocale_preserves_placeholders_and_exposes_overflow_gate(self):
        translated = pseudolocalize("Open {label}")
        self.assertIn("{label}", translated)
        self.assertGreater(len(translated), len("Open {label}"))
        text = tr("settings.messages.locale_changed", locale=PSEUDO_LOCALE)
        width = QFontMetrics(self._app.font()).horizontalAdvance(text)
        self.assertTrue(pseudolocale_overflow(text, width, max(1, width - 8)))
        self.assertFalse(pseudolocale_overflow("", width, 1))

    def test_ui_locale_persists_and_applies_rtl_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_config(Path(tmp)):
                self.assertEqual("en", current_locale())
                self.assertEqual("ar", set_locale("ar"))
                self.assertEqual("ar", Settings().get("general.ui_locale"))
                self.assertTrue(is_rtl())
                self.assertEqual(
                    Qt.LayoutDirection.RightToLeft,
                    self._app.layoutDirection(),
                )
                Settings._instance = None
                self.assertEqual("ar", current_locale())

    def test_rtl_layout_keeps_keyboard_tab_order_predictable(self):
        set_locale("ar", persist=False)
        window = QWidget()
        layout = QHBoxLayout(window)
        first = QPushButton("الأول")
        second = QPushButton("الثاني")
        layout.addWidget(first)
        layout.addWidget(second)
        window.setTabOrder(first, second)
        window.show()
        try:
            self.assertEqual(Qt.LayoutDirection.RightToLeft, window.layoutDirection())
            first.setFocus()
            self.assertIs(window.focusWidget(), first)
            self.assertTrue(window.focusNextChild())
            self.assertIs(window.focusWidget(), second)
        finally:
            window.close()
            window.deleteLater()

    def test_pseudolocale_expands_static_widget_copy_for_layout_qa(self):
        set_locale(PSEUDO_LOCALE, persist=False)
        window = QWidget()
        layout = QHBoxLayout(window)
        label = QLabel("Static shell copy")
        button = QPushButton("Continue")
        layout.addWidget(label)
        layout.addWidget(button)
        try:
            self.assertGreaterEqual(apply_pseudolocale(window), 2)
            self.assertTrue(label.text().startswith("［"))
            self.assertTrue(button.text().startswith("［"))
            self.assertEqual(0, apply_pseudolocale(window))
        finally:
            window.deleteLater()

    def test_main_pages_pass_rtl_and_pseudo_locale_probes(self):
        probe = r'''
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QApplication, QComboBox, QGroupBox, QLabel, QTabWidget, QWidget
from core.i18n import PSEUDO_LOCALE, set_locale
from core.lyrics_db import LyricsDB
from ui.main_window import MainWindow

app = QApplication([])
set_locale("ar", persist=False, app=app)
arabic = MainWindow()
app.processEvents()
pages = [arabic._pages.widget(i) for i in range(arabic._pages.count())]
if len(pages) != 10 or app.layoutDirection() != Qt.LayoutDirection.RightToLeft:
    raise SystemExit(f"arabic pages={len(pages)} direction={app.layoutDirection()}")
if any(not page.accessibleName() or not page.accessibleDescription() for page in pages):
    raise SystemExit("arabic page accessibility metadata is incomplete")
arabic.close()
arabic.deleteLater()
LyricsDB().reopen()
app.processEvents()

set_locale(PSEUDO_LOCALE, persist=False, app=app)
pseudo = MainWindow()
app.processEvents()
pages = [pseudo._pages.widget(i) for i in range(pseudo._pages.count())]
if len(pages) != 10:
    raise SystemExit(f"pseudo pages={len(pages)}")
text_widgets = []
for widget in [pseudo, *pseudo.findChildren(QWidget)]:
    if not isinstance(widget, (QLabel, QAbstractButton, QGroupBox)):
        continue
    text = widget.title() if isinstance(widget, QGroupBox) else widget.text()
    if any(char.isalpha() for char in text):
        text_widgets.append((widget, text))
unexpanded = [text for _widget, text in text_widgets if not text.startswith("［")]
if unexpanded:
    raise SystemExit("pseudo visible copy not expanded: " + repr(unexpanded[:8]))
for combo in pseudo.findChildren(QComboBox):
    values = [combo.itemText(i) for i in range(combo.count()) if any(char.isalpha() for char in combo.itemText(i))]
    if any(not value.startswith("［") for value in values):
        raise SystemExit("pseudo combo item not expanded: " + repr(values[:8]))
for tabs in pseudo.findChildren(QTabWidget):
    values = [tabs.tabText(i) for i in range(tabs.count()) if tabs.tabText(i)]
    if any(not value.startswith("［") for value in values):
        raise SystemExit("pseudo tab not expanded: " + repr(values))
print(f"arabic pages={len(pages)} pseudo_text={len(text_widgets)}")
pseudo.close()
'''
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"locale page probe failed:\n{result.stdout}\n{result.stderr}",
        )
        self.assertIn("arabic pages=10", result.stdout)

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

    def test_settings_view_saves_ui_locale(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_config(Path(tmp)):
                view = SettingsView()
                try:
                    idx = view._ui_locale_combo.findData("ar")
                    self.assertGreaterEqual(idx, 0)
                    view._ui_locale_combo.setCurrentIndex(idx)
                    self.assertEqual("ar", Settings().get("general.ui_locale"))
                    self.assertEqual(Qt.LayoutDirection.RightToLeft, self._app.layoutDirection())
                finally:
                    view.deleteLater()

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

    def test_missing_key_returns_visible_bracket_marker(self):
        clear_missing_key_log()
        result = tr("nonexistent.key.xyz")
        self.assertEqual(result, "[nonexistent.key.xyz]")
        self.assertIn("nonexistent.key.xyz", get_missing_key_log())
        clear_missing_key_log()

    def test_external_locale_loads_from_config_dir(self):
        load_catalog.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_config(Path(tmp)):
                ext_dir = Path(tmp) / "config" / "locales"
                ext_dir.mkdir(parents=True)
                test_locale = {"app": {"window_title": "Custom Title v{version}"}}
                (ext_dir / "xx.json").write_text(json.dumps(test_locale), encoding="utf-8")

                locales = available_locales()
                self.assertIn("xx", locales)

                result = tr("app.window_title", locale="xx", version="1.0")
                self.assertEqual(result, "Custom Title v1.0")
        load_catalog.cache_clear()

    def test_external_locale_overrides_builtin(self):
        load_catalog.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_config(Path(tmp)):
                ext_dir = Path(tmp) / "config" / "locales"
                ext_dir.mkdir(parents=True)
                override = {"nav": {"lyrics": "Letras"}}
                (ext_dir / "en.json").write_text(json.dumps(override), encoding="utf-8")

                result = tr("nav.lyrics", locale="en")
                self.assertEqual(result, "Letras")
        load_catalog.cache_clear()

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
        stack.enter_context(mock.patch("core.i18n.get_config_dir", return_value=config_dir))
        return stack


if __name__ == "__main__":
    unittest.main()
