import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.job_state import JobStore
from ui.batch_view import BatchView


class BatchFavoritePersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_star_survives_clear_and_view_recreation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            store = JobStore(root / "jobs", cleanup_roots=[root])
            with (
                mock.patch("ui.batch_view.get_config_dir", return_value=config_dir),
                mock.patch("ui.batch_view.JobStore", return_value=store),
            ):
                view = BatchView()
                view.add_result(str(root / "variation.wav"), seed=42)
                view._cards[0]._toggle_star()
                self.assertTrue(view._cards[0].is_starred)

                view.clear()
                view.add_result(str(root / "variation.wav"), seed=42)
                self.assertTrue(view._cards[0].is_starred)
                view.deleteLater()

                reopened = BatchView()
                reopened.add_result(str(root / "variation.wav"), seed=42)
                self.assertTrue(reopened._cards[0].is_starred)
                self.assertEqual(reopened.get_starred()[0]["seed"], 42)
                reopened.deleteLater()


if __name__ == "__main__":
    unittest.main()
