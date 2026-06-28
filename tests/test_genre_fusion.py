import unittest

from engines.lyrics_templates import blend_genre_style_tags


class GenreFusionTests(unittest.TestCase):
    def test_primary_weight_prefers_primary_tags(self):
        tags = blend_genre_style_tags("trap", "metal", secondary_weight=0.2)

        self.assertLess(tags.index("trap"), tags.index("metal"))
        self.assertIn("hybrid", tags)

    def test_secondary_weight_prefers_secondary_tags(self):
        tags = blend_genre_style_tags("trap", "metal", secondary_weight=0.8)

        self.assertLess(tags.index("metal"), tags.index("trap"))
        self.assertIn("trap metal fusion", tags)


if __name__ == "__main__":
    unittest.main()
