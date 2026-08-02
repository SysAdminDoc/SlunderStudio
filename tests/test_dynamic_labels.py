import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui.mixer_view import MixerTrackStrip
from ui.model_hub import HFTokenDialog
from ui.widgets import ElidedLabel


class DynamicLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_long_mixer_track_name_elides_with_full_tooltip(self):
        name = "vocals_recovered_from_long_filename_take_07"
        strip = MixerTrackStrip(0, name)
        strip._name_label.resize(80, 20)
        strip._name_label._update_elided_text()

        self.assertEqual(strip._name_label.full_text, name)
        self.assertEqual(strip._name_label.toolTip(), name)
        self.assertNotEqual(strip._name_label.text(), name)
        self.assertIn("…", strip._name_label.text())

    def test_elided_label_reflows_when_width_grows(self):
        label = ElidedLabel("A very long dynamic status value", minimum_width=80)
        label.resize(80, 20)
        label._update_elided_text()
        self.assertIn("…", label.text())

        label.resize(500, 20)
        label._update_elided_text()

        self.assertEqual(label.text(), label.full_text)
        self.assertEqual(label.toolTip(), label.full_text)

    def test_invalid_huggingface_token_grows_dialog_with_visible_error(self):
        dialog = HFTokenDialog("A model with a long display name", "org/model")
        initial_height = dialog.height()
        dialog._token_input.setText("not-a-token")

        dialog._accept()

        self.assertFalse(dialog._error_label.isHidden())
        self.assertIn("hf_", dialog._error_label.text())
        self.assertGreater(dialog.height(), initial_height)
        dialog.close()


if __name__ == "__main__":
    unittest.main()
