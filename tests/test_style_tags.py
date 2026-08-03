import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engines.style_tags import StyleTagDB


class StyleTagTests(unittest.TestCase):
    def test_search_category_and_favorites_are_case_insensitive_and_persistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch("engines.style_tags.get_config_dir", return_value=root):
                db = StyleTagDB()
                results = db.search("  SYNTH  ", category="instrument")
                self.assertTrue(results)
                self.assertTrue(all(item["category"] == "instrument" for item in results))
                self.assertTrue(all("is_favorite" in item for item in results))

                self.assertTrue(db.toggle_favorite("synthesizer"))
                self.assertTrue(db.is_favorite("synthesizer"))
                self.assertEqual([item["tag"] for item in db.get_favorites()], ["synthesizer"])

                reopened = StyleTagDB()
                self.assertTrue(reopened.is_favorite("synthesizer"))
                self.assertFalse(reopened.toggle_favorite("synthesizer"))
                self.assertEqual(reopened.get_favorites(), [])

    def test_categories_suggestions_and_malformed_favorites_fail_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            favorites = root / "style_tag_favorites.json"
            favorites.write_text("not json", encoding="utf-8")
            with mock.patch("engines.style_tags.get_config_dir", return_value=root):
                db = StyleTagDB()
                self.assertEqual(db.get_favorites(), [])
                self.assertIn("genre", db.get_categories())
                self.assertIn("pop", db.get_by_category("genre")[0]["tag"])
                self.assertEqual(db.get_suggested_tags("unknown-genre"), ["pop", "catchy"])

                suggested = db.get_suggested_tags("pop")
                self.assertTrue(suggested)
                self.assertTrue(all(isinstance(tag, str) for tag in suggested))
                self.assertEqual(db.total_count, sum(len(tags) for tags in __import__(
                    "engines.style_tags", fromlist=["CATEGORIES"]
                ).CATEGORIES.values()))


if __name__ == "__main__":
    unittest.main()
