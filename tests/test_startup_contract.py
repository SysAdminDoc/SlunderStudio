import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StartupContractTests(unittest.TestCase):
    def test_startup_does_not_purge_bytecode_caches(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("_clean_pycache", source)
        self.assertNotIn("shutil.rmtree", source)


if __name__ == "__main__":
    unittest.main()
