import json
import shutil
import tempfile
import unittest
from pathlib import Path

from engines import lyrics_templates


class LyricsTemplateAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.asset_dir = Path(lyrics_templates.__file__).resolve().parents[1] / "assets" / "templates"

    def test_shipped_genre_assets_are_loaded(self):
        asset_ids = {
            path.stem for path in self.asset_dir.glob("*.json") if not path.name.startswith("_")
        }
        self.assertEqual(asset_ids, set(lyrics_templates.GENRE_TEMPLATES))
        self.assertEqual(
            lyrics_templates.MOODS,
            json.loads((self.asset_dir / "_moods.json").read_text(encoding="utf-8")),
        )
        self.assertEqual(
            lyrics_templates.STANDARD_STRUCTURES,
            json.loads((self.asset_dir / "_structures.json").read_text(encoding="utf-8")),
        )

    def test_editing_a_genre_asset_changes_generation_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            template_dir = Path(tmp) / "templates"
            template_dir.mkdir()
            for asset in self.asset_dir.glob("*.json"):
                shutil.copy2(asset, template_dir / asset.name)

            trap_path = template_dir / "trap.json"
            trap = json.loads(trap_path.read_text(encoding="utf-8"))
            trap["vocabulary_style"] = "CUSTOM ASSET STYLE FOR TESTING"
            trap_path.write_text(json.dumps(trap), encoding="utf-8")

            try:
                lyrics_templates.reload_template_bundle(template_dir)
                system, _user = lyrics_templates.build_generation_prompt(
                    "a night drive",
                    genre_id="trap",
                )
                self.assertIn("CUSTOM ASSET STYLE FOR TESTING", system)
            finally:
                lyrics_templates.reload_template_bundle()


if __name__ == "__main__":
    unittest.main()
