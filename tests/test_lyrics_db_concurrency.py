import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from core.lyrics_db import LyricsDB, LyricsEntry


def _make_entry(index: int) -> LyricsEntry:
    return LyricsEntry(
        prompt=f"prompt {index}",
        genre="trap metal",
        mood="dark",
        lyrics_original=f"line {index}",
        notes=f"note {index}",
    )


class LyricsDBConcurrencyTests(unittest.TestCase):
    """LyricsDB shares one connection across GUI and worker threads."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        config_dir = Path(self._tmp.name)
        patcher = mock.patch("core.lyrics_db.get_config_dir", return_value=config_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        LyricsDB._instance = None
        self.addCleanup(setattr, LyricsDB, "_instance", None)
        self.db = LyricsDB()
        self.addCleanup(self.db.close)

    def test_busy_timeout_and_wal_are_configured(self):
        conn = self.db._require_conn()
        self.assertEqual(
            conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal"
        )
        self.assertGreaterEqual(conn.execute("PRAGMA busy_timeout").fetchone()[0], 1000)

    def test_concurrent_writers_and_readers_produce_no_lock_errors(self):
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def writer(offset: int):
            try:
                barrier.wait(timeout=10)
                for i in range(25):
                    entry = _make_entry(offset * 100 + i)
                    entry_id = self.db.save(entry)
                    self.db.set_rating(entry_id, i % 6)
                    self.db.toggle_favorite(entry_id)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        def reader():
            try:
                barrier.wait(timeout=10)
                for _ in range(25):
                    self.db.get_recent(limit=20)
                    self.db.get_favorites(limit=20)
                    self.db.search("prompt")
                    self.db.count()
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        self.assertEqual([repr(e) for e in errors], [])
        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(self.db.count(), 100)

    def test_rows_are_never_partially_written(self):
        errors: list[BaseException] = []

        def worker(index: int):
            try:
                self.db.save(_make_entry(index))
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(30)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        self.assertEqual([repr(e) for e in errors], [])
        entries = self.db.get_recent(limit=100)
        self.assertEqual(len(entries), 30)
        self.assertEqual(len({e.id for e in entries}), 30)
        for entry in entries:
            self.assertTrue(entry.prompt.startswith("prompt "))
            self.assertTrue(entry.lyrics_original.startswith("line "))
            self.assertEqual(entry.genre, "trap metal")

    def test_import_batch_is_atomic(self):
        good = [_make_entry(i) for i in range(5)]
        self.db.save_many(good)
        self.assertEqual(self.db.count(), 5)
        self.assertTrue(all(entry.id > 0 for entry in good))

        broken = [_make_entry(100), _make_entry(101)]
        broken[1].timestamp = object()  # not a SQLite-bindable type
        with self.assertRaises(Exception):
            self.db.save_many(broken)
        self.assertEqual(self.db.count(), 5)

        # The failed import must not leave a transaction open.
        self.db.save(_make_entry(200))
        self.assertEqual(self.db.count(), 6)

    def test_locked_write_is_retried_then_succeeds(self):
        attempts = {"begin": 0}
        real_conn = self.db._require_conn()

        class FlakyConnection:
            """Fails the first BEGIN the way a competing writer would."""

            def __init__(self, inner):
                self._inner = inner

            def execute(self, sql, *args):
                if sql == "BEGIN IMMEDIATE":
                    attempts["begin"] += 1
                    if attempts["begin"] == 1:
                        raise sqlite3.OperationalError("database is locked")
                return self._inner.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        self.db._conn = FlakyConnection(real_conn)
        try:
            entry_id = self.db.save(_make_entry(1))
        finally:
            self.db._conn = real_conn

        self.assertEqual(attempts["begin"], 2)
        self.assertIsNotNone(self.db.get(entry_id))

    def test_competing_writer_is_waited_out_not_failed(self):
        # sqlite3 connections are thread-bound, so the blocker stays on this thread.
        blocker = sqlite3.connect(str(self.db._db_path), isolation_level=None)
        self.addCleanup(blocker.close)
        blocker.execute("BEGIN IMMEDIATE")

        started = threading.Event()
        result: dict[str, object] = {}

        def writer():
            started.set()
            try:
                result["id"] = self.db.save(_make_entry(7))
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                result["error"] = exc

        thread = threading.Thread(target=writer)
        thread.start()
        self.assertTrue(started.wait(timeout=5))
        time.sleep(0.3)
        self.assertEqual(result, {}, "write should still be waiting on the busy lock")

        blocker.execute("ROLLBACK")
        thread.join(timeout=30)

        self.assertNotIn("error", result, repr(result.get("error")))
        self.assertIn("id", result)
        self.assertEqual(self.db.count(), 1)

    def test_non_lock_errors_are_not_retried(self):
        with self.assertRaises(sqlite3.OperationalError):
            self.db._read("SELECT * FROM does_not_exist")

    def test_closed_database_reports_clearly_and_can_reopen(self):
        entry_id = self.db.save(_make_entry(1))
        self.db.close()
        with self.assertRaises(RuntimeError):
            self.db.get(entry_id)
        self.db.reopen()
        self.assertIsNotNone(self.db.get(entry_id))


if __name__ == "__main__":
    unittest.main()
