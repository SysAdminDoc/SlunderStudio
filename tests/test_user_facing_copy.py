import unittest
from pathlib import Path
from types import SimpleNamespace

from core.i18n import set_locale, user_facing_readiness


class UserFacingCopyTests(unittest.TestCase):
    def setUp(self):
        set_locale("en", persist=False)

    def test_readiness_remedies_are_rewritten_at_the_ui_boundary(self):
        readiness = SimpleNamespace(
            model_id="ace-step-v1.5",
            missing_packages=(),
            capability=SimpleNamespace(label="Song production"),
            remedy="Re-download ACE-Step; its local cache is not verified.",
        )

        message = user_facing_readiness(readiness, model_name="ACE-Step")

        self.assertEqual(
            "Download ACE-Step again to repair its local files.",
            message,
        )
        self.assertNotIn("local cache is not verified", message)

    def test_known_internal_copy_leaks_are_absent_from_views(self):
        root = Path(__file__).resolve().parents[1]
        sources = "\n".join(
            (root / path).read_text(encoding="utf-8")
            for path in (
                "ui/ai_producer_view.py",
                "ui/main_window.py",
                "ui/midi_studio_view.py",
                "ui/model_hub.py",
                "ui/settings_view.py",
                "ui/sfx_view.py",
                "ui/vocal_suite_view.py",
            )
        )
        for leaked in (
            "Produces declared outputs",
            "Route cancelled",
            "uninstall via pip",
            "placeholder.coming_soon",
            "readiness.remedy",
        ):
            with self.subTest(leaked=leaked):
                self.assertNotIn(leaked, sources)

        main_window = (root / "ui/main_window.py").read_text(encoding="utf-8")
        self.assertNotIn("class PlaceholderPage", main_window)


if __name__ == "__main__":
    unittest.main()
