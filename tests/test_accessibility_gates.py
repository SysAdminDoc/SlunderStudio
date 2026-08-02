import os
import re
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from ui.contrast import (
    NON_TEXT_RATIO,
    NORMAL_TEXT_RATIO,
    SURFACE_TOKENS,
    contrast_ratio,
    failing_text_pairs,
    relative_luminance,
    token,
)
from ui.accessibility import FOCUS_RING_COLOR
from ui.theme import Palette, build_stylesheet, rgba
from ui.toast import ToastManager
from ui.waveform_widget import WaveformWidget
from ui.stem_mixer import STEM_COLORS

ROOT = Path(__file__).resolve().parents[1]

# Views must fit a 1024x768 display, and remain usable when the system scales
# that by 200% (Qt reports the scaled, smaller logical size to the widget).
TARGET_WIDTH = 1024
TARGET_HEIGHT = 768


class ContrastGateTests(unittest.TestCase):
    def test_known_ratios(self):
        self.assertAlmostEqual(contrast_ratio("#ffffff", "#000000"), 21.0, places=2)
        self.assertAlmostEqual(contrast_ratio("#000000", "#000000"), 1.0, places=2)
        self.assertGreater(relative_luminance("#ffffff"), relative_luminance("#808080"))

    def test_every_text_token_meets_normal_text_contrast(self):
        failures = failing_text_pairs()
        self.assertEqual(
            failures, [],
            "palette pairs below "
            f"{NORMAL_TEXT_RATIO}:1 - " + ", ".join(
                f"{fg} on {bg}={ratio}" for fg, bg, ratio in failures
            ),
        )

    def test_focus_ring_is_visible_against_every_surface(self):
        for surface in SURFACE_TOKENS:
            ratio = contrast_ratio(FOCUS_RING_COLOR, token(surface))
            with self.subTest(surface=surface):
                self.assertGreaterEqual(ratio, NON_TEXT_RATIO, f"{ratio:.2f}:1")

    def test_stylesheet_never_suppresses_focus_without_a_replacement(self):
        sheet = build_stylesheet()
        # Every rule block that hides the outline must style :focus somewhere.
        self.assertIn(":focus", sheet)
        for match in re.finditer(r"([^{}]*)\{([^{}]*outline:\s*none[^{}]*)\}", sheet):
            selector = match.group(1).strip().splitlines()[-1].strip()
            with self.subTest(selector=selector):
                base = selector.split("::")[0].split(":")[0].strip()
                self.assertTrue(
                    f"{base}:focus" in sheet
                    or "selection-background-color" in match.group(2),
                    f"{selector} hides the focus outline with no replacement",
                )

    def test_accent_setting_default_matches_the_palette(self):
        from core.settings import DEFAULTS

        self.assertEqual(DEFAULTS["general"]["theme_accent"].lower(), Palette.BLUE.lower())

    def test_no_view_hardcodes_the_retired_accent(self):
        offenders = []
        for path in (ROOT / "ui").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "#9b8cff" in text or "#8795a5" in text:
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_dynamic_hex_alpha_uses_rgb_ordered_rgba(self):
        offenders = []
        suffix = re.compile(r"\{[^{}\n]+\}[0-9a-fA-F]{2}(?=[;}\s])")
        for path in (ROOT / "ui").glob("*.py"):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if suffix.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{line_number}")

        self.assertEqual(offenders, [])
        self.assertEqual(rgba("#f38ba8", 68), "rgba(243, 139, 168, 68)")


class InlineButtonContrastTests(unittest.TestCase):
    _STATE_RULE = re.compile(
        r"QPushButton(?P<state>:(?:checked|hover))\s*\{\{?"
        r"(?P<body>(?:[^{}]|\{[^{}]*\})*)\}\}?",
        re.S,
    )
    _COLOR = re.compile(
        r"\b(?P<property>background|color)\s*:\s*"
        r"(?P<value>#[0-9a-fA-F]{6}|white|\{Palette\.[A-Z0-9_]+\})"
    )

    @staticmethod
    def _resolve_color(value: str) -> str | None:
        if value.lower() == "white":
            return "#ffffff"
        if value.startswith("{Palette."):
            token_name = value[len("{Palette."):-1]
            return getattr(Palette, token_name)
        if value.startswith("#"):
            return value
        return None

    def test_inline_checked_and_hover_colors_meet_wcag(self):
        failures = []
        for path in (ROOT / "ui").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for match in self._STATE_RULE.finditer(source):
                declarations = {
                    item.group("property"): self._resolve_color(item.group("value"))
                    for item in self._COLOR.finditer(match.group("body"))
                }
                background = declarations.get("background")
                foreground = declarations.get("color")
                if not background or not foreground:
                    continue
                ratio = contrast_ratio(foreground, background)
                if ratio < NORMAL_TEXT_RATIO:
                    failures.append(
                        f"{path.relative_to(ROOT)} {match.group('state')}: "
                        f"{foreground} on {background}={ratio:.2f}"
                    )

        self.assertEqual(failures, [])

    def test_stem_checked_colors_meet_wcag(self):
        failures = [
            f"{name}: {contrast_ratio(Palette.CRUST, color):.2f}:1"
            for name, color in STEM_COLORS.items()
            if contrast_ratio(Palette.CRUST, color) < NORMAL_TEXT_RATIO
        ]
        self.assertEqual(failures, [])


class WaveformKeyboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.widget = WaveformWidget(show_controls=False)
        self.addCleanup(self.widget.deleteLater)
        self.widget._duration = 60.0
        self.widget._audio_data = object()

    def _press(self, key):
        event = QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier)
        self.widget.keyPressEvent(event)
        return event

    def test_widget_is_reachable_by_keyboard_and_named(self):
        self.assertNotEqual(self.widget.focusPolicy(), Qt.FocusPolicy.NoFocus)
        self.assertTrue(self.widget.accessibleName())
        self.assertIn("Home", self.widget.accessibleDescription())

    def test_arrow_keys_seek_and_emit_position(self):
        seen: list[float] = []
        self.widget.position_clicked.connect(seen.append)

        self._press(Qt.Key.Key_Right)
        self.assertAlmostEqual(self.widget.playback_position, 1.0)
        self._press(Qt.Key.Key_Left)
        self.assertAlmostEqual(self.widget.playback_position, 0.0)
        self._press(Qt.Key.Key_PageDown)
        self.assertAlmostEqual(self.widget.playback_position, 10.0)
        self._press(Qt.Key.Key_End)
        self.assertAlmostEqual(self.widget.playback_position, 60.0)
        self._press(Qt.Key.Key_Home)
        self.assertAlmostEqual(self.widget.playback_position, 0.0)

        self.assertEqual(len(seen), 5)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in seen))

    def test_seeking_is_clamped_to_the_audio(self):
        self._press(Qt.Key.Key_Left)
        self.assertAlmostEqual(self.widget.playback_position, 0.0)
        self.widget.set_playback_position(60.0)
        self._press(Qt.Key.Key_Right)
        self.assertAlmostEqual(self.widget.playback_position, 60.0)

    def test_state_is_announced_as_a_value(self):
        self.widget.set_playback_position(30.0)
        state = self.widget.accessible_state()
        self.assertIn("30.0", state)
        self.assertIn("60.0", state)
        self.assertIn("50%", state)
        self.assertEqual(self.widget.accessibleDescription(), state)

    def test_mode_switch_is_keyboard_reachable(self):
        self._press(Qt.Key.Key_M)
        self.assertEqual(self.widget._mode, "spectrogram")
        self._press(Qt.Key.Key_M)
        self.assertEqual(self.widget._mode, "waveform")

    def test_unhandled_keys_are_not_swallowed(self):
        event = self._press(Qt.Key.Key_Z)
        self.assertFalse(event.isAccepted())


class TimedMessageAlternativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_toasts_are_also_recorded_in_a_non_timed_log(self):
        host = QWidget()
        self.addCleanup(host.deleteLater)
        manager = ToastManager(host)
        seen: list[dict] = []
        manager.on_message(seen.append)

        manager.info("Render started")
        manager.error("Render failed")

        self.assertEqual([e["type"] for e in manager.history], ["info", "error"])
        self.assertEqual(manager.latest_message(), "Error: Render failed")
        self.assertEqual(len(seen), 2)

    def test_history_is_bounded(self):
        host = QWidget()
        self.addCleanup(host.deleteLater)
        manager = ToastManager(host)
        for i in range(manager.HISTORY_LIMIT + 25):
            manager._record(f"message {i}", "info")
        self.assertEqual(len(manager.history), manager.HISTORY_LIMIT)
        self.assertEqual(manager.history[-1]["message"], f"message {manager.HISTORY_LIMIT + 24}")


class ResponsiveLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_main_window_minimum_fits_a_1024x768_display(self):
        import ui.main_window as main_window

        source = Path(main_window.__file__).read_text(encoding="utf-8")
        match = re.search(r"setMinimumSize\((\d+),\s*(\d+)\)", source)
        self.assertIsNotNone(match, "main window declares no minimum size")
        width, height = int(match.group(1)), int(match.group(2))
        self.assertLessEqual(width, TARGET_WIDTH)
        self.assertLessEqual(height, TARGET_HEIGHT)

    def test_no_view_demands_more_width_than_the_target_display(self):
        offenders = []
        pattern = re.compile(r"setFixedWidth\((\d+)\)|setMinimumWidth\((\d+)\)")
        for path in (ROOT / "ui").rglob("*.py"):
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                match = pattern.search(line)
                if not match:
                    continue
                value = int(match.group(1) or match.group(2))
                if value > TARGET_WIDTH:
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}={value}")
        self.assertEqual(offenders, [], "controls wider than 1024px: " + ", ".join(offenders))

    def test_sidebar_wordmark_fits_its_brand_row(self):
        from ui.main_window import Sidebar

        previous_stylesheet = self._app.styleSheet()
        self._app.setStyleSheet(build_stylesheet())
        sidebar = Sidebar()
        try:
            brand = sidebar.findChild(QLabel, "brand")
            self.assertIsNotNone(brand)
            brand.ensurePolished()
            brand_row = sidebar.layout().itemAt(0).layout()
            brand_row_margins = brand_row.getContentsMargins()
            sidebar_margins = sidebar.layout().getContentsMargins()
            brand_mark = brand_row.itemAt(0).widget()
            available = (
                sidebar.width()
                - sidebar_margins[0]
                - sidebar_margins[2]
                - brand_row_margins[0]
                - brand_row_margins[2]
                - brand_mark.width()
                - brand_row.spacing()
            )

            self.assertGreaterEqual(available, brand.sizeHint().width())
        finally:
            sidebar.deleteLater()
            self._app.setStyleSheet(previous_stylesheet)


if __name__ == "__main__":
    unittest.main()
