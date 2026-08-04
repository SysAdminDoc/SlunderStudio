import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.lyrics_analysis import analyze_lyrics, estimate_syllables
from ui.lyrics_editor import LyricsEditor


class LyricsAnalysisTests(unittest.TestCase):
    def test_analysis_ignores_structure_and_direction_lines(self):
        analysis = analyze_lyrics(
            "[Verse 1]\n"
            "I see the light\n"
            "(whispered)\n"
            "We own the night\n"
            "[Chorus]\n"
            "We spark a fire\n"
            "We chase desire\n"
        )

        self.assertEqual(analysis.line_count, 4)
        self.assertEqual(analysis.word_count, 15)
        self.assertEqual(analysis.rhyme_coverage, 100.0)
        self.assertEqual(analysis.rhyme_covered_lines, 4)
        self.assertGreater(analysis.cadence_consistency, 80)
        self.assertEqual(analysis.rhyme_families, (("ight", 2), ("ire", 2)))

    def test_analysis_surfaces_cadence_spread_without_calling_it_quality(self):
        analysis = analyze_lyrics("Short\nA much longer line with many syllables tonight")

        self.assertEqual(analysis.min_syllables, 1)
        self.assertGreater(analysis.cadence_range, 4)
        self.assertLess(analysis.cadence_consistency, 60)
        self.assertIn("vary widely", " ".join(analysis.advisory_notes))
        self.assertEqual(estimate_syllables("fire"), 1)

    def test_empty_and_non_latin_text_fail_soft(self):
        empty = analyze_lyrics("[Verse]\n\n(softly)")
        non_latin = analyze_lyrics("[Verse]\nПривет мир\nمرحبا بالعالم")

        self.assertEqual(empty.line_count, 0)
        self.assertTrue(empty.advisory_notes)
        self.assertEqual(non_latin.line_count, 2)
        self.assertGreater(non_latin.word_count, 0)
        self.assertGreaterEqual(non_latin.syllable_count, 2)


class LyricsFeedbackPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_editor_updates_advisory_panel_without_mutating_text(self):
        editor = LyricsEditor()
        try:
            draft = "[Verse]\nI see the light\nWe own the night"
            editor.text = draft
            self.assertEqual(editor.text, draft)
            self.assertEqual(editor._feedback.analysis.line_count, 2)
            self.assertIn("End-rhyme coverage", editor._feedback._rhyme_label.text())
            self.assertEqual(editor._feedback._disclaimer_label.text(), "No automatic rewrite")
        finally:
            editor.deleteLater()


if __name__ == "__main__":
    unittest.main()
