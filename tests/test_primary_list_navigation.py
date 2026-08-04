import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.lyrics_db import LyricsEntry
from core.model_manager import BUILTIN_MODELS
from ui.lyrics_view import HistoryPanel
from ui.model_hub import ModelHubView
from ui.project_manager import ProjectCard, ProjectManagerView


class PrimaryListNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_project_search_uses_name_and_notes_but_not_internal_id(self):
        card = ProjectCard({
            "id": "proj_internal_only",
            "name": "Sunset Demo",
            "notes": "Try a wider chorus in the second take.",
            "updated_at": 100,
        })
        try:
            self.assertTrue(card.matches_query("sunset"))
            self.assertTrue(card.matches_query("wider chorus"))
            self.assertFalse(card.matches_query("internal_only"))
        finally:
            card.deleteLater()

    def test_project_sorting_supports_name_and_modified_date(self):
        projects = [
            {"id": "b", "name": "Beta", "updated_at": 20},
            {"id": "a", "name": "Alpha", "updated_at": 10},
        ]
        self.assertEqual(
            ["a", "b"],
            [item["id"] for item in ProjectManagerView._sort_projects(projects, "name_asc")],
        )
        self.assertEqual(
            ["a", "b"],
            [item["id"] for item in ProjectManagerView._sort_projects(projects, "updated_asc")],
        )
        self.assertEqual(
            ["b", "a"],
            [item["id"] for item in ProjectManagerView._sort_projects(projects, "updated_desc")],
        )

    def test_lyrics_sorting_supports_date_genre_and_prompt(self):
        entries = [
            LyricsEntry(id=1, timestamp=20, genre="Pop", prompt="Zed"),
            LyricsEntry(id=2, timestamp=10, genre="Ambient", prompt="Alpha"),
        ]
        self.assertEqual(
            [2, 1],
            [entry.id for entry in HistoryPanel._sort_entries(entries, "date_asc")],
        )
        self.assertEqual(
            [2, 1],
            [entry.id for entry in HistoryPanel._sort_entries(entries, "genre_asc")],
        )
        self.assertEqual(
            [2, 1],
            [entry.id for entry in HistoryPanel._sort_entries(entries, "prompt_asc")],
        )

    def test_model_sorting_supports_name_and_measurement_date(self):
        registry = {
            "z": BUILTIN_MODELS["audio-separator"],
            "a": BUILTIN_MODELS["ace-step-v1.5"],
        }
        name_order = ModelHubView._sort_model_ids(registry, "name_asc")
        date_order = ModelHubView._sort_model_ids(registry, "date_desc")
        self.assertEqual(["a", "z"], name_order)
        self.assertEqual(2, len(date_order))
        self.assertEqual(set(registry), set(date_order))


if __name__ == "__main__":
    unittest.main()
