"""Central, cancellation-aware admission control for model work."""
from __future__ import annotations

import threading
import time
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from core.process_lock import InterProcessFileLock


logger = logging.getLogger(__name__)

ADMISSION_STATE_SCHEMA_VERSION = 1
ADMISSION_STATE_FILENAME = ".admission_state.json"
ADMISSION_HEARTBEAT_SECONDS = 5.0
ADMISSION_STALE_AFTER_SECONDS = 30.0


def _windows_process_token(pid: int) -> str | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class FileTime(ctypes.Structure):
            _fields_ = [
                ("low", ctypes.c_uint32),
                ("high", ctypes.c_uint32),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return None
        created = FileTime()
        exited = FileTime()
        kernel = FileTime()
        user = FileTime()
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return None
            if int(exit_code.value) != 259:  # STILL_ACTIVE
                return None
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return None
            value = (int(created.high) << 32) | int(created.low)
            return f"win:{value}"
        finally:
            kernel32.CloseHandle(handle)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _process_start_token(pid: int) -> str | None:
    token = _windows_process_token(pid)
    if token:
        return token
    proc_stat = Path(f"/proc/{int(pid)}/stat")
    try:
        fields = proc_stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return f"proc:{fields[19]}"
    except (IndexError, OSError, UnicodeError):
        return None


_PROCESS_PID = os.getpid()
_PROCESS_START_TOKEN = _process_start_token(_PROCESS_PID) or f"fallback:{time.time_ns()}"


def _process_is_alive(record: dict[str, object]) -> bool:
    try:
        pid = int(record.get("pid", 0) or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False

    token = str(record.get("process_token", "") or "")
    if pid == _PROCESS_PID:
        return not token or token == _PROCESS_START_TOKEN
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    actual = _process_start_token(pid)
    if os.name == "nt" and not actual:
        return False
    return not token or not actual or token == actual


class AdmissionError(RuntimeError):
    """Base error for a resource admission decision."""


class AdmissionBusyError(AdmissionError):
    """Raised when the same keyed resource is already queued or active."""


class AdmissionCancelledError(AdmissionError):
    """Raised when a queued request is cancelled before acquiring capacity."""


class AdmissionTimeoutError(AdmissionError):
    """Raised when a queued request exceeds its caller-supplied wait limit."""


@dataclass(frozen=True)
class AdmissionSnapshot:
    """Serializable capacity state for diagnostics and support reports."""

    limits: dict[str, int]
    active: dict[str, int]
    queued: dict[str, int]
    active_download_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "limits": dict(self.limits),
            "active": dict(self.active),
            "queued": dict(self.queued),
            "active_download_keys": list(self.active_download_keys),
        }


class AdmissionLease:
    """One acquired slot; release is idempotent and context-manager friendly."""

    def __init__(
        self,
        controller: "AdmissionController",
        kind: str,
        key: str,
        token: str,
    ):
        self._controller = controller
        self.kind = kind
        self.key = key
        self.token = token
        self._released = False
        self._lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"slunder-admission-{kind}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._heartbeat_stop.set()
        try:
            self._controller._release(self.kind, self.key, self.token)
        finally:
            if threading.current_thread() is not self._heartbeat_thread:
                self._heartbeat_thread.join(
                    timeout=max(0.1, self._controller.heartbeat_interval * 2)
                )

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._controller.heartbeat_interval):
            try:
                self._controller._heartbeat(self.token)
            except Exception:  # noqa: BLE001 - a later sweep can recover a dead lease
                logger.exception("Admission lease heartbeat failed")

    def __enter__(self) -> "AdmissionLease":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


class AdmissionController:
    """Bound model downloads and inference without busy-waiting.

    Downloads are limited globally and keyed by model ID so two requests for
    the same cache can never both enter. Inference is a separate capacity pool;
    its default of one protects the one-large-model GPU policy while still
    allowing downloads to resume independently.
    """

    DOWNLOAD = "download"
    INFERENCE = "inference"
    _KINDS = (DOWNLOAD, INFERENCE)

    def __init__(
        self,
        *,
        max_downloads: int = 2,
        max_inference: int = 1,
        shared_state_path: Path | str | None = None,
        heartbeat_interval: float = ADMISSION_HEARTBEAT_SECONDS,
        stale_after_seconds: float = ADMISSION_STALE_AFTER_SECONDS,
    ):
        self._lock = threading.RLock()
        self._limits = {
            self.DOWNLOAD: self._positive_limit(max_downloads, self.DOWNLOAD),
            self.INFERENCE: self._positive_limit(max_inference, self.INFERENCE),
        }
        if shared_state_path is None:
            from core.settings import get_config_dir

            shared_state_path = get_config_dir() / ADMISSION_STATE_FILENAME
        self._state_path = Path(shared_state_path)
        self._state_lock = InterProcessFileLock(
            self._state_path.with_name(self._state_path.name + ".lock")
        )
        try:
            self._heartbeat_interval = max(0.05, float(heartbeat_interval))
            self._stale_after_seconds = max(
                self._heartbeat_interval * 2,
                float(stale_after_seconds),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Admission heartbeat settings must be numeric") from exc

    @staticmethod
    def _positive_limit(value: int, kind: str) -> int:
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{kind} admission limit must be an integer") from exc
        if limit < 1:
            raise ValueError(f"{kind} admission limit must be at least one")
        return limit

    @property
    def limits(self) -> dict[str, int]:
        with self._lock:
            return dict(self._limits)

    @property
    def heartbeat_interval(self) -> float:
        return self._heartbeat_interval

    def snapshot(self) -> AdmissionSnapshot:
        with self._lock:
            with self._state_lock:
                state = self._read_state_unlocked()
                if self._prune_stale(state):
                    self._write_state_unlocked(state)
                active = self._counts(state["leases"])
                queued = self._counts(state["queued"])
                active_download_keys = tuple(
                    sorted(
                        str(record.get("key", ""))
                        for record in state["leases"]
                        if record.get("kind") == self.DOWNLOAD and record.get("key")
                    )
                )
            return AdmissionSnapshot(
                limits=dict(self._limits),
                active=active,
                queued=queued,
                active_download_keys=active_download_keys,
            )

    def acquire(
        self,
        kind: str,
        *,
        key: str = "",
        cancel_event: Optional[threading.Event] = None,
        timeout: Optional[float] = None,
        wait_cb: Optional[Callable[[], None]] = None,
    ) -> AdmissionLease:
        """Wait for one machine-wide slot, checking cancellation frequently."""
        if kind not in self._KINDS:
            raise ValueError(f"Unknown admission kind: {kind}")
        normalized_key = str(key or "")
        if cancel_event is not None and cancel_event.is_set():
            raise AdmissionCancelledError(
                f"{kind.title()} admission cancelled while waiting"
            )
        request_token = uuid.uuid4().hex
        request = self._new_record(request_token, kind, normalized_key)
        with self._lock:
            with self._state_lock:
                state = self._read_state_unlocked()
                dirty = self._prune_stale(state)
                if kind == self.DOWNLOAD and normalized_key:
                    duplicate = any(
                        record.get("kind") == kind
                        and str(record.get("key", "")) == normalized_key
                        for records in (state["leases"], state["queued"])
                        for record in records
                    )
                    if duplicate:
                        if dirty:
                            self._write_state_unlocked(state)
                        raise AdmissionBusyError(
                            f"A download for {normalized_key} is already queued or active"
                        )
                state["queued"].append(request)
                self._write_state_unlocked(state)

        started = time.monotonic()
        notified = False
        acquired = False
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise AdmissionCancelledError(
                        f"{kind.title()} admission cancelled while waiting"
                    )
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise AdmissionTimeoutError(
                        f"Timed out waiting for {kind} admission capacity"
                    )
                promoted = False
                with self._lock:
                    with self._state_lock:
                        state = self._read_state_unlocked()
                        dirty = self._prune_stale(state)
                        available = (
                            len(
                                [
                                    record
                                    for record in state["leases"]
                                    if record.get("kind") == kind
                                ]
                            )
                            < self._limits[kind]
                        )
                        if available:
                            state["queued"] = [
                                record
                                for record in state["queued"]
                                if record.get("token") != request_token
                            ]
                            request["acquired_at"] = time.time()
                            request["heartbeat_at"] = time.time()
                            state["leases"].append(request)
                            self._write_state_unlocked(state)
                            promoted = True
                        elif dirty:
                            self._write_state_unlocked(state)
                if promoted:
                    acquired = True
                    try:
                        return AdmissionLease(
                            self,
                            kind,
                            normalized_key,
                            request_token,
                        )
                    except Exception:
                        self._release(kind, normalized_key, request_token)
                        raise
                if not notified and wait_cb is not None:
                    wait_cb()
                    notified = True
                if cancel_event is not None:
                    cancel_event.wait(0.1)
                else:
                    time.sleep(0.1)
        finally:
            if not acquired:
                with self._lock:
                    with self._state_lock:
                        state = self._read_state_unlocked()
                        dirty = self._prune_stale(state)
                        retained = [
                            record
                            for record in state["queued"]
                            if record.get("token") != request_token
                        ]
                        if len(retained) != len(state["queued"]):
                            state["queued"] = retained
                            dirty = True
                        if dirty:
                            self._write_state_unlocked(state)

    def _release(self, kind: str, key: str, token: str) -> None:
        del kind, key
        with self._lock:
            with self._state_lock:
                state = self._read_state_unlocked()
                dirty = self._prune_stale(state)
                retained = [
                    record
                    for record in state["leases"]
                    if record.get("token") != token
                ]
                if len(retained) != len(state["leases"]):
                    state["leases"] = retained
                    dirty = True
                if dirty:
                    self._write_state_unlocked(state)

    def _heartbeat(self, token: str) -> None:
        with self._lock:
            with self._state_lock:
                state = self._read_state_unlocked()
                dirty = self._prune_stale(state)
                for record in state["leases"]:
                    if record.get("token") != token:
                        continue
                    record["heartbeat_at"] = time.time()
                    dirty = True
                    break
                if dirty:
                    self._write_state_unlocked(state)

    @staticmethod
    def _counts(records: list[dict[str, object]]) -> dict[str, int]:
        return {
            kind: sum(1 for record in records if record.get("kind") == kind)
            for kind in AdmissionController._KINDS
        }

    @staticmethod
    def _new_record(token: str, kind: str, key: str) -> dict[str, object]:
        now = time.time()
        return {
            "token": token,
            "kind": kind,
            "key": key,
            "pid": _PROCESS_PID,
            "process_token": _PROCESS_START_TOKEN,
            "created_at": now,
            "heartbeat_at": now,
        }

    def _prune_stale(self, state: dict[str, list[dict[str, object]]]) -> bool:
        changed = False
        for field in ("leases", "queued"):
            retained = []
            for record in state[field]:
                heartbeat = record.get("heartbeat_at", record.get("created_at", 0.0))
                try:
                    heartbeat_age = time.time() - float(heartbeat or 0.0)
                except (TypeError, ValueError):
                    heartbeat_age = self._stale_after_seconds + 1
                missing_identity = not str(record.get("process_token", "") or "")
                if _process_is_alive(record) and not (
                    missing_identity and heartbeat_age > self._stale_after_seconds
                ):
                    retained.append(record)
                else:
                    changed = True
            if len(retained) != len(state[field]):
                state[field] = retained
        return changed

    def _read_state_unlocked(self) -> dict[str, list[dict[str, object]]]:
        if not self._state_path.exists():
            return {"leases": [], "queued": []}
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AdmissionError(
                f"Admission state is unreadable; refusing to admit work: {exc}"
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != ADMISSION_STATE_SCHEMA_VERSION:
            raise AdmissionError(
                "Admission state schema is unsupported; refusing to admit work"
            )
        leases = payload.get("leases")
        queued = payload.get("queued")
        if not isinstance(leases, list) or not isinstance(queued, list):
            raise AdmissionError(
                "Admission state is malformed; refusing to admit work"
            )
        if not all(isinstance(record, dict) for record in (*leases, *queued)):
            raise AdmissionError(
                "Admission state contains malformed records; refusing to admit work"
            )
        return {"leases": leases, "queued": queued}

    def _write_state_unlocked(self, state: dict[str, list[dict[str, object]]]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._state_path.parent,
                prefix=f".{self._state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(
                    {
                        "schema_version": ADMISSION_STATE_SCHEMA_VERSION,
                        "updated_at": time.time(),
                        "leases": state["leases"],
                        "queued": state["queued"],
                    },
                    handle,
                    indent=2,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._state_path)
            temporary = None
        except (OSError, TypeError, ValueError) as exc:
            raise AdmissionError(
                f"Admission state could not be persisted; refusing to admit work: {exc}"
            ) from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass


_global_lock = threading.Lock()
_global_controller: Optional[AdmissionController] = None


def global_admission_controller() -> AdmissionController:
    """Return the process-wide controller configured at first use."""
    global _global_controller
    if _global_controller is None:
        with _global_lock:
            if _global_controller is None:
                try:
                    from core.settings import Settings

                    settings = Settings()
                    max_downloads = settings.get("model_hub.max_concurrent_downloads", 2)
                    max_inference = settings.get("model_hub.max_concurrent_inference", 1)
                except Exception:  # noqa: BLE001 - safe defaults are the admission boundary
                    max_downloads, max_inference = 2, 1
                try:
                    _global_controller = AdmissionController(
                        max_downloads=max_downloads,
                        max_inference=max_inference,
                    )
                except ValueError:
                    _global_controller = AdmissionController()
    return _global_controller


def reset_global_admission_controller() -> None:
    """Reset the lazy singleton for tests and controlled application restarts."""
    global _global_controller
    with _global_lock:
        if _global_controller is not None:
            snapshot = _global_controller.snapshot()
            if any(snapshot.active.values()) or any(snapshot.queued.values()):
                raise RuntimeError("Cannot reset admission control while work is active")
        _global_controller = None


MODEL_INFERENCE_JOB_KINDS = frozenset(
    {
        "lyrics_generation",
        "song_generation",
        "midi_generation",
        "ai_producer",
        "vocal_synthesis",
        "vocal_pronunciation_correction",
        "lyric_melody",
        "voice_conversion",
        "voice_clone",
        "vocal_autotune",
        "stem_separation",
        "sfx_generation",
        "reference_analysis",
        "model_activation",
        "model_update",
        "model_rollback",
    }
)


def admission_kind_for_job(job_kind: str) -> Optional[str]:
    """Map an explicit durable job kind to a model resource pool."""
    return (
        AdmissionController.INFERENCE
        if str(job_kind or "") in MODEL_INFERENCE_JOB_KINDS
        else None
    )
