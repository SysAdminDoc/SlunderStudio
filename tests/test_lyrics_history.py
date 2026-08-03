import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from core.lyrics_db import LyricsDB, LyricsEntry
from ui.lyrics_view import HistoryPanel


class LyricsHistoryFavoriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._config_patcher = mock.patch(
            "core.lyrics_db.get_config_dir", return_value=Path(self._tmp.name)
        )
        self._config_patcher.start()
        self.addCleanup(self._config_patcher.stop)

        LyricsDB._instance = None
        self.addCleanup(setattr, LyricsDB, "_instance", None)
        self.db = LyricsDB()
        self.addCleanup(self._close_db)

        self.entry = LyricsEntry(
            prompt="favorite prompt",
            genre="pop",
            lyrics_original="A saved chorus",
        )
        self.db.save(self.entry)

    def _close_db(self):
        if LyricsDB._instance is not None:
            LyricsDB._instance.close()
        LyricsDB._instance = None

    def test_double_click_persists_and_populates_favorites_after_restart(self):
        panel = HistoryPanel()
        self.addCleanup(panel.deleteLater)
        self.assertEqual(panel._list.count(), 1)

        item = panel._list.item(0)
        panel._toggle_favorite(item)
        self.assertTrue(self.db.get(self.entry.id).is_favorite)
        self.assertTrue(panel._list.item(0).text().startswith("\u2605 "))

        panel._set_filter("favorites")
        self.assertEqual(panel._list.count(), 1)
        favorite_entry = panel._list.item(0).data(Qt.ItemDataRole.UserRole)
        self.assertEqual(favorite_entry.id, self.entry.id)

        panel.deleteLater()
        self._app.processEvents()
        self.db.close()
        LyricsDB._instance = None

        reopened = LyricsDB()
        self.db = reopened
        reopened_panel = HistoryPanel()
        self.addCleanup(reopened_panel.deleteLater)
        reopened_panel._set_filter("favorites")
        self.assertEqual(reopened_panel._list.count(), 1)
        self.assertTrue(reopened_panel._list.item(0).text().startswith("\u2605 "))


if __name__ == "__main__":
    unittest.main()
