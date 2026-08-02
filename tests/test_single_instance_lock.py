import os
import tempfile
import time
import unittest
from unittest import mock

import main


class SingleInstanceLockTests(unittest.TestCase):
    def _write_lock(self, root: str, pid: int):
        config_dir = os.path.join(root, "SlunderStudio")
        os.makedirs(config_dir)
        lock_file = os.path.join(config_dir, "studio.lock")
        with open(lock_file, "w", encoding="utf-8") as handle:
            handle.write(str(pid))
        old = time.time() - 3600
        os.utime(lock_file, (old, old))
        return lock_file

    def test_dead_pid_reclaims_even_when_lockfile_is_old(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_lock(root, 987654321)
            with (
                mock.patch.dict(os.environ, {"APPDATA": root}),
                mock.patch("psutil.pid_exists", return_value=False) as pid_exists,
                mock.patch("main.os.kill", side_effect=AssertionError("os.kill used")),
                mock.patch("atexit.register"),
            ):
                self.assertTrue(main._acquire_lock())

            pid_exists.assert_called_once_with(987654321)

    def test_current_pid_blocks_even_when_lockfile_is_old(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_lock(root, os.getpid())
            with (
                mock.patch.dict(os.environ, {"APPDATA": root}),
                mock.patch("psutil.pid_exists", return_value=True) as pid_exists,
                mock.patch("main.os.kill", side_effect=AssertionError("os.kill used")),
            ):
                self.assertFalse(main._acquire_lock())

            pid_exists.assert_called_once_with(os.getpid())


if __name__ == "__main__":
    unittest.main()
