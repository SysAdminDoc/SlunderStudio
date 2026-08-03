import tempfile
import unittest
from pathlib import Path

from core.model_security import ModelSecurityError, assert_safe_transformers_snapshot


class ModelSecurityTests(unittest.TestCase):
    def test_snapshot_scan_accepts_nested_public_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                '{"model_type": "whisper", "layers": [{"width": 128}]}',
                encoding="utf-8",
            )

            checked = assert_safe_transformers_snapshot(root)

            self.assertEqual(checked, (root / "config.json",))

    def test_snapshot_scan_rejects_private_configuration_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                '{"model_type": "ace", "nested": {"_attn_implementation_internal": "evil.module"}}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ModelSecurityError, "_attn_implementation_internal"):
                assert_safe_transformers_snapshot(root)

    def test_snapshot_scan_rejects_malformed_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("not json", encoding="utf-8")

            with self.assertRaisesRegex(ModelSecurityError, "valid UTF-8 JSON"):
                assert_safe_transformers_snapshot(root)

    def test_snapshot_scan_requires_an_absolute_directory(self):
        with self.assertRaisesRegex(ModelSecurityError, "absolute"):
            assert_safe_transformers_snapshot(Path("relative-model"))


if __name__ == "__main__":
    unittest.main()
