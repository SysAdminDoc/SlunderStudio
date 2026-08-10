"""Small cross-platform advisory locks for shared local state files."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path


_LOCK_REGISTRY_GUARD = threading.Lock()
_LOCK_REGISTRY: dict[str, threading.RLock] = {}


def _thread_lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCK_REGISTRY_GUARD:
        lock = _LOCK_REGISTRY.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCK_REGISTRY[key] = lock
        return lock


class InterProcessFileLock:
    """Serialize short read-modify-write sections across processes and threads."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._thread_lock = _thread_lock_for(self.path)
        self._handle = None

    def __enter__(self) -> "InterProcessFileLock":
        self._thread_lock.acquire()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            self._lock_handle(handle)
            self._handle = handle
            return self
        except Exception:
            self._thread_lock.release()
            raise

    def __exit__(self, *_exc) -> None:
        handle = self._handle
        self._handle = None
        try:
            if handle is not None:
                try:
                    self._unlock_handle(handle)
                finally:
                    handle.close()
        finally:
            self._thread_lock.release()

    @staticmethod
    def _lock_handle(handle) -> None:
        if os.name == "nt":
            import msvcrt

            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    return
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _unlock_handle(handle) -> None:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
