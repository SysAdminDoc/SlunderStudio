import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.lyrics_db import LyricsDB
from ui.lyrics_view import HistoryPanel
from ui.model_hub import ModelHubView
from ui.project_manager import ProjectDetailPanel
from ui.seed_explorer import SeedExplorer
from ui.waveform_widget import WaveformWidget
from ui.widgets import EmptyStateWidget, OperationProgressWidget


class EmptyStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_shared_state_exposes_kind_and_action(self):
        state = EmptyStateWidget("Nothing here", "Create the first item.", "Create")
        requested = []
        state.action_requested.connect(lambda: requested.append(True))

        self.assertEqual(state.state, "empty")
        self.assertEqual(state.title, "Nothing here")
        state.action_button.click()
        self.assertEqual(requested, [True])

        state.set_no_matches("Try another search.")
        self.assertEqual(state.state, "no_matches")
        self.assertEqual(state.action_button.text(), "Clear filters")
        state.deleteLater()

    def test_operation_progress_keeps_numeric_progress_separate_from_copy(self):
        progress = OperationProgressWidget()
        requested = []
        progress.cancel_requested.connect(lambda: requested.append(True))
        try:
            progress.start("Rendering", determinate=True)
            progress.set_progress(42)
            progress.set_step("Rendering pass 2")
            self.assertTrue(progress.isVisible())
            self.assertEqual(progress.progress_bar.value(), 42)
            self.assertEqual(progress.message_label.text(), "Rendering pass 2")

            progress.cancel_button.click()
            self.assertEqual(requested, [True])
            self.assertFalse(progress.cancel_button.isEnabled())
            self.assertEqual(progress.progress_bar.minimum(), 0)
            self.assertEqual(progress.progress_bar.maximum(), 0)

            progress.finish()
            self.assertFalse(progress.isVisible())
            self.assertEqual(progress.progress_bar.value(), 0)
        finally:
            progress.deleteLater()

    def test_lyrics_history_distinguishes_empty_from_no_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            with mock.patch("core.lyrics_db.get_config_dir", return_value=config_dir):
                LyricsDB._instance = None
                panel = HistoryPanel()
                try:
                    self.assertEqual(panel._empty.state, "empty")
                    self.assertIs(panel._list_stack.currentWidget(), panel._empty)

                    panel._search.setText("not saved")
                    self.assertEqual(panel._empty.state, "no_matches")
                    self.assertIn("not saved", panel._empty.message)

                    panel._empty.action_button.click()
                    self.assertEqual(panel._search.text(), "")
                    self.assertEqual(panel._empty.state, "empty")
                finally:
                    panel.deleteLater()
                    if LyricsDB._instance is not None:
                        LyricsDB._instance.close()
                    LyricsDB._instance = None

    def test_filterable_model_hub_has_no_match_state_and_reset_action(self):
        view = ModelHubView()
        try:
            view._search.setText("model-that-cannot-exist-999")
            self.assertEqual(view._grid_empty.state, "no_matches")
            self.assertIs(view._grid_stack.currentWidget(), view._grid_empty)

            view._grid_empty.action_button.click()
            self.assertEqual(view._search.text(), "")
            self.assertIs(view._grid_stack.currentWidget(), view._grid_container)
        finally:
            view.deleteLater()

    def test_first_use_states_are_present_on_outputs_and_grids(self):
        waveform = WaveformWidget()
        explorer = SeedExplorer()
        details = ProjectDetailPanel()
        try:
            self.assertFalse(waveform.has_audio)
            self.assertEqual(waveform.empty_state.state, "empty")
            self.assertIn("No audio", waveform.empty_state.title)
            self.assertIs(explorer._grid_stack.currentWidget(), explorer._grid_empty)
            self.assertIs(details._asset_stack.currentWidget(), details._asset_empty)
            self.assertIs(details._version_stack.currentWidget(), details._version_empty)
        finally:
            waveform.deleteLater()
            explorer.deleteLater()
            details.deleteLater()


if __name__ == "__main__":
    unittest.main()
