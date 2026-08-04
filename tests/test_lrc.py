import tempfile
import unittest
from pathlib import Path

from core.lrc import LRCValidationError, format_enhanced_lrc, write_enhanced_lrc


class EnhancedLRCTests(unittest.TestCase):
    def test_format_includes_line_and_word_timestamps(self):
        lyrics = "[Verse]\nWe are here\nUnder the sky"
        aligned = [
            {"text": "We", "start": 1.0, "end": 1.2},
            {"text": "are", "start": 1.2, "end": 1.4},
            {"text": "here", "start": 1.4, "end": 1.8},
            {"text": "Under", "start": 3.0, "end": 3.2},
            {"text": "the", "start": 3.2, "end": 3.4},
            {"text": "sky", "start": 3.4, "end": 3.8},
        ]

        self.assertEqual(
            "[Verse]\n"
            "[00:01.00]<00:01.00>We <00:01.20>are <00:01.40>here\n"
            "[00:03.00]<00:03.00>Under <00:03.20>the <00:03.40>sky\n",
            format_enhanced_lrc(lyrics, aligned),
        )

    def test_hyphenated_words_and_parentheticals_remain_aligned(self):
        lyrics = "broken-hearted (softly)\nwe're here"
        aligned = [
            {"text": "broken-hearted", "start": 0.0, "end": 0.5},
            {"text": "we're", "start": 0.6, "end": 0.9},
            {"text": "here", "start": 0.9, "end": 1.2},
        ]

        rendered = format_enhanced_lrc(lyrics, aligned)

        self.assertIn("<00:00.00>broken-hearted", rendered)
        self.assertIn("<00:00.60>we're <00:00.90>here", rendered)

    def test_incomplete_extra_mismatched_and_invalid_alignment_is_refused(self):
        base = [
            {"text": "hello", "start": 0.0, "end": 0.5},
            {"text": "world", "start": 0.5, "end": 1.0},
        ]
        cases = [
            base[:1],
            base + [{"text": "extra", "start": 1.0, "end": 1.2}],
            [{"text": "wrong", "start": 0.0, "end": 0.5}, base[1]],
            [{"text": "hello", "start": float("nan"), "end": 0.5}, base[1]],
            [{"text": "hello", "start": 0.8, "end": 0.5}, base[1]],
        ]

        for aligned in cases:
            with self.subTest(aligned=aligned):
                with self.assertRaises(LRCValidationError):
                    format_enhanced_lrc("hello world", aligned)

    def test_write_uses_utf8_and_returns_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "nested" / "melody.lrc"
            aligned = [{"text": "café", "start": 2.345, "end": 2.8}]

            written = write_enhanced_lrc(destination, "café", aligned)

            self.assertEqual(str(destination), written)
            self.assertEqual(
                "[00:02.35]<00:02.35>café\n",
                destination.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
