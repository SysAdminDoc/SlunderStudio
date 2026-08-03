import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import main


class SingleInstanceLockTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(main._release_lock)

    def _write_lock(self, root: str, payload):
        config_dir = os.path.join(root, "SlunderStudio")
        os.makedirs(config_dir)
        lock_file = os.path.join(config_dir, "studio.lock")
        with open(lock_file, "w", encoding="utf-8") as handle:
            if isinstance(payload, dict):
                json.dump(payload, handle)
            else:
                handle.write(str(payload))
        return lock_file

    def test_stale_metadata_is_reclaimed_without_pid_liveness_guessing(self):
        with tempfile.TemporaryDirectory() as root:
            lock_file = self._write_lock(
                root,
                {
                    "schema": 2,
                    "pid": os.getpid(),
                    "executable": "C:\\other\\SlunderStudio.exe",
                },
            )
            with (
                mock.patch.dict(os.environ, {"APPDATA": root}),
            ):
                self.assertTrue(main._acquire_lock())

            main._release_lock()
            record = json.loads(Path(lock_file).read_text(encoding="utf-8"))
            self.assertEqual(record["pid"], os.getpid())
            self.assertEqual(record["executable_name"], Path(sys.executable).name)

    def test_active_file_lock_blocks_and_names_the_lock_path(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_lock(root, {"schema": 2, "pid": 1234})
            with (
                mock.patch.dict(os.environ, {"APPDATA": root}),
                mock.patch(
                    "main._acquire_file_lock",
                    side_effect=BlockingIOError("lock held"),
                ),
            ):
                self.assertFalse(main._acquire_lock())

            self.assertTrue(main._lock_failure_message().startswith("Another"))
            self.assertIn(os.path.join(root, "SlunderStudio", "studio.lock"),
                          main._lock_failure_message())

    def test_missing_psutil_does_not_disable_the_os_lock(self):
        with tempfile.TemporaryDirectory() as root:
            with (
                mock.patch.dict(os.environ, {"APPDATA": root}),
                mock.patch.dict(sys.modules, {"psutil": None}),
            ):
                self.assertTrue(main._acquire_lock())
            main._release_lock()

    def test_unwritable_lock_directory_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            with (
                mock.patch.dict(os.environ, {"APPDATA": root}),
                mock.patch.object(Path, "mkdir", side_effect=PermissionError("read only")),
            ):
                self.assertFalse(main._acquire_lock())

            message = main._lock_failure_message()
            self.assertIn("could not acquire", message)
            self.assertIn(os.path.join(root, "SlunderStudio", "studio.lock"), message)
            self.assertNotIn("Another Slunder Studio", message)

    def test_lock_source_uses_os_lock_and_fails_closed(self):
        source = Path(main.__file__).read_text(encoding="utf-8")
        lock_section = source[
            source.index("def _acquire_lock"):source.index("# ── Application Launch")
        ]
        self.assertIn("_acquire_file_lock(handle)", lock_section)
        self.assertIn("return False", lock_section)
        self.assertNotIn("pid_exists", lock_section)


if __name__ == "__main__":
    unittest.main()
