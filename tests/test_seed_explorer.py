import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from ui.seed_explorer import SeedExplorer


class SeedExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_distance_slider_syncs_seed_range(self):
        explorer = SeedExplorer()

        explorer._distance_slider.setValue(750)
        self.assertEqual(explorer._range_spin.value(), 750)

        explorer._range_spin.setValue(250)
        self.assertEqual(explorer._distance_slider.value(), 250)

    def test_explore_emits_seed_and_cfg_grid(self):
        explorer = SeedExplorer()
        emitted = []
        explorer.generate_requested.connect(emitted.append)
        explorer._grid_combo.setCurrentIndex(0)
        explorer._seed_spin.setValue(1000)
        explorer._range_spin.setValue(100)
        explorer._cfg_min_spin.setValue(3.0)
        explorer._cfg_max_spin.setValue(5.0)

        explorer._start_exploration()

        self.assertEqual(len(emitted), 1)
        params = emitted[0]
        self.assertEqual(len(params), 4)
        self.assertEqual(params[0]["seed"], 950)
        self.assertEqual(params[-1]["seed"], 1050)
        self.assertEqual(params[0]["cfg_scale"], 3.0)
        self.assertEqual(params[-1]["cfg_scale"], 5.0)


if __name__ == "__main__":
    unittest.main()
