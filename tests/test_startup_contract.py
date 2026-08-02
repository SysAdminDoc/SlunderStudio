import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StartupContractTests(unittest.TestCase):
    def test_startup_does_not_purge_bytecode_caches(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("_clean_pycache", source)
        self.assertNotIn("shutil.rmtree", source)

    def test_gpu_monitor_defers_first_probe_to_its_timer(self):
        source = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
        self.assertIn("self._gpu_timer.start(2000)", source)
        self.assertNotIn("self._update_gpu_status()", source)

        model_hub_source = (ROOT / "ui" / "model_hub.py").read_text(encoding="utf-8")
        self.assertIn("QTimer.singleShot(0, self._update_gpu_display)", model_hub_source)


if __name__ == "__main__":
    unittest.main()
