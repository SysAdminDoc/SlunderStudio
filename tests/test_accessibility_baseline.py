import os
import tempfile
import unittest
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.project import ProjectManager
from core.settings import Settings
from ui.batch_view import BatchCard, BatchView
from ui.main_window import Sidebar, TransportBar
from ui.mixer_view import MixerTrackStrip, MixerView
from ui.model_hub import ModelHubView
from ui.piano_roll import CCAutomationLane, PianoRollWidget
from ui.settings_view import SettingsView
from ui.song_forge_view import SongForgeView
from ui.theme import build_stylesheet
from ui.vocal_suite_view import VocalSuiteView


class AccessibilityBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self):
        Settings._instance = None
        ProjectManager._instance = None

    def test_theme_exposes_visible_focus_selectors(self):
        stylesheet = build_stylesheet()
        for selector in [
            "QPushButton:focus",
            "QLineEdit:focus",
            "QComboBox:focus",
            "QSlider:focus",
            "QCheckBox:focus",
            "QTabBar::tab:focus",
        ]:
            self.assertIn(selector, stylesheet)
        self.assertIn("#f9e2af", stylesheet.lower())

    def test_main_shell_navigation_and_transport_have_accessible_names(self):
        sidebar = Sidebar()
        transport = TransportBar()
        try:
            for button in sidebar._buttons:
                self.assert_accessible(button)
            for widget in [
                transport._play_btn,
                transport._stop_btn,
                transport._seek_slider,
                transport._loop_btn,
                transport._vol_slider,
            ]:
                self.assert_accessible(widget)
            self.assertIs(sidebar._buttons[0].nextInFocusChain(), sidebar._buttons[1])
            self.assertIs(transport._play_btn.nextInFocusChain(), transport._stop_btn)
        finally:
            sidebar.deleteLater()
            transport.deleteLater()

    def test_major_views_expose_accessible_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            with self._patched_config(config_dir, root):
                views = [
                    (
                        SongForgeView(),
                        [
                            "_mode_tabs",
                            "_quick_lyrics",
                            "_quick_tags",
                            "_duration_spin",
                            "_cfg_spin",
                            "_tag_browser",
                            "_generate_btn",
                            "_play_btn",
                            "_sub_tabs",
                            "_to_vocal_stem_btn",
                        ],
                    ),
                    (
                        VocalSuiteView(),
                        [
                            "_tabs",
                            "_sing_lyrics",
                            "_sing_voice",
                            "_melody_browse_btn",
                            "_melody_lyrics",
                            "_melody_tempo",
                            "_melody_render_diffsinger",
                            "_melody_generate_btn",
                            "_rvc_browse_btn",
                            "_rvc_voice",
                            "_clone_voice",
                            "_clone_ref_btn",
                            "_clone_text",
                            "_autotune_browse_btn",
                            "_autotune_strength",
                            "_autotune_apply_btn",
                            "_stem_browse_btn",
                            "_stem_model",
                            "_to_forge_btn",
                        ],
                    ),
                    (
                        ModelHubView(),
                        [
                            "_search",
                            "_category_filter",
                            "_downloaded_only",
                            "_gpu_label",
                            "_disk_label",
                        ],
                    ),
                    (
                        SettingsView(),
                        [
                            "_tabs",
                            "_output_dir",
                            "_browse_output_btn",
                            "_format_combo",
                            "_sample_rate_combo",
                            "_gpu_device",
                            "_offline_mode",
                            "_hf_token",
                            "_default_language",
                            "_health_private_inputs",
                            "_export_health_btn",
                            "_reset_btn",
                            "_open_dir_btn",
                        ],
                    ),
                ]

                try:
                    for view, attrs in views:
                        self.assert_accessible(view)
                        for attr in attrs:
                            self.assert_accessible(getattr(view, attr))
                finally:
                    for view, _attrs in views:
                        view.deleteLater()

    def test_mixer_view_exposes_accessible_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            with self._patched_config(config_dir, root):
                view = MixerView()
                try:
                    self.assert_accessible(view)
                    for attr in [
                        "_add_btn", "_dynamic_eq_btn", "_preset_combo",
                        "_target_combo", "_lufs_spin", "_mid_gain_spin",
                        "_side_gain_spin", "_ref_btn", "_master_btn",
                    ]:
                        self.assert_accessible(getattr(view, attr))
                finally:
                    view.deleteLater()

    def test_mixer_track_strip_controls_have_accessible_names(self):
        strip = MixerTrackStrip(0, "Drums")
        try:
            self.assert_accessible(strip)
            for attr in ["_vol_slider", "_pan_slider", "_mute_btn", "_solo_btn", "_remove_btn"]:
                self.assert_accessible(getattr(strip, attr))
            self.assertIn("Drums", strip._vol_slider.accessibleName())
        finally:
            strip.deleteLater()

    def test_batch_view_exposes_accessible_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            with self._patched_config(config_dir, root):
                view = BatchView()
                try:
                    self.assert_accessible(view)
                    self.assert_accessible(view._use_best_btn)
                    self.assert_accessible(view._clear_btn)
                finally:
                    view.deleteLater()

    def test_batch_card_controls_have_accessible_names(self):
        card = BatchCard(2)
        try:
            self.assert_accessible(card)
            self.assert_accessible(card._star_btn)
            self.assert_accessible(card._delete_btn)
            self.assertIn("3", card._star_btn.accessibleName())
        finally:
            card.deleteLater()

    def test_piano_roll_exposes_accessible_controls(self):
        widget = PianoRollWidget()
        try:
            self.assert_accessible(widget)
            for attr in [
                "_snap_combo", "_velocity_spin", "_swing_spin", "_humanize_spin",
                "_quantize_btn", "_swing_btn", "_humanize_btn",
                "_select_all_btn", "_delete_btn",
            ]:
                self.assert_accessible(getattr(widget, attr))
        finally:
            widget.deleteLater()

    def test_cc_automation_lane_controls_have_accessible_names(self):
        lane = CCAutomationLane()
        try:
            self.assert_accessible(lane)
            for attr in [
                "_controller_combo", "_beat_spin", "_value_spin",
                "_add_cc_btn", "_clear_lane_btn", "_table",
            ]:
                self.assert_accessible(getattr(lane, attr))
        finally:
            lane.deleteLater()

    def test_tab_order_matches_primary_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            config_dir.mkdir()
            with self._patched_config(config_dir, root):
                song = SongForgeView()
                settings = SettingsView()
                hub = ModelHubView()
                try:
                    self.assertIs(self._next_named_focus(song._quick_lyrics), song._quick_tags)
                    self.assertIs(self._next_named_focus(settings._output_dir), settings._browse_output_btn)
                    self.assertIs(self._next_named_focus(hub._search), hub._category_filter)
                finally:
                    song.deleteLater()
                    settings.deleteLater()
                    hub.deleteLater()

    def assert_accessible(self, widget):
        self.assertTrue(widget.accessibleName(), f"{widget} missing accessibleName")
        self.assertTrue(
            widget.accessibleDescription(),
            f"{widget.accessibleName() or widget} missing accessibleDescription",
        )

    def _next_named_focus(self, widget):
        current = widget.nextInFocusChain()
        for _ in range(50):
            if current.accessibleName():
                return current
            current = current.nextInFocusChain()
        return None

    @contextmanager
    def _patched_config(self, config_dir: Path, root: Path):
        Settings._instance = None
        ProjectManager._instance = None
        with ExitStack() as stack:
            stack.enter_context(mock.patch("core.settings.get_config_dir", return_value=config_dir))
            stack.enter_context(mock.patch("core.settings.get_default_output_dir", return_value=root / "renders"))
            stack.enter_context(mock.patch("core.settings.get_default_cache_dir", return_value=root / "models"))
            stack.enter_context(mock.patch("core.project.get_config_dir", return_value=config_dir))
            stack.enter_context(mock.patch("core.model_manager.get_config_dir", return_value=config_dir))
            stack.enter_context(mock.patch("core.voice_bank.get_config_dir", return_value=config_dir))
            yield
