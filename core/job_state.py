"""
Slunder Studio - Durable job state and recovery records.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from core.settings import APP_NAME, get_config_dir, get_default_output_dir


JOB_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    RECOVERABLE = "recoverable"


ACTIVE_STATUSES = {
    JobStatus.QUEUED,
    JobStatus.RUNNING,
    JobStatus.CANCEL_REQUESTED,
}

# All JobStore instances in this process may point at the same ledger.  A
# process-wide re-entrant lock keeps read-modify-write updates atomic across
# workers and views that each construct their own store instance.
_JOB_STORE_LOCK = threading.RLock()


@dataclass
class JobRecord:
    id: str
    kind: str
    label: str
    status: str = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: int = 0
    message: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    recoverable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    _extra_fields: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        if not isinstance(data, dict):
            raise ValueError("Job record must be an object")
        allowed = {
            item.name for item in fields(cls)
            if item.name != "_extra_fields"
        }
        values = {key: value for key, value in data.items() if key in allowed}
        extras = {key: value for key, value in data.items() if key not in allowed}
        return cls(**values, _extra_fields=extras)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self._extra_fields)
        payload.update({
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "_extra_fields"
        })
        return payload


class JobStore:
    """JSON-backed job ledger for restart recovery and cancellation cleanup."""

    def __init__(
        self,
        root: Optional[Path] = None,
        cleanup_roots: Optional[Iterable[Path | str]] = None,
    ):
        self._lock = _JOB_STORE_LOCK
        self.root = Path(root) if root is not None else get_config_dir() / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "jobs.json"
        roots = cleanup_roots if cleanup_roots is not None else _default_cleanup_roots()
        self._cleanup_roots = _canonical_cleanup_roots(roots)

    def create(
        self,
        kind: str,
        label: str,
        inputs: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> JobRecord:
        record = JobRecord(
            id=f"job_{int(time.time() * 1000)}_{uuid.uuid4().hex[:12]}",
            kind=kind,
            label=label,
            inputs=_jsonable(inputs or {}),
            metadata=_jsonable(metadata or {}),
        )
        with self._lock:
            records = self._read()
            records.append(record)
            self._write(records)
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        for record in self.list_records():
            if record.id == job_id:
                return record
        return None

    def list_records(
        self,
        status: Optional[str | Iterable[str]] = None,
        kind: Optional[str] = None,
    ) -> list[JobRecord]:
        with self._lock:
            records = self._read()
            if status is not None:
                statuses = {status} if isinstance(status, str) else set(status)
                records = [record for record in records if record.status in statuses]
            if kind is not None:
                records = [record for record in records if record.kind == kind]
            return sorted(records, key=lambda record: record.updated_at, reverse=True)

    def mark_running(self, job_id: str, message: str = "") -> Optional[JobRecord]:
        return self._update(
            job_id,
            status=JobStatus.RUNNING,
            started_at=time.time(),
            message=message,
            recoverable=True,
        )

    def update_progress(self, job_id: str, progress: int, message: str = "") -> Optional[JobRecord]:
        changes: dict[str, Any] = {"progress": max(0, min(100, int(progress)))}
        if message:
            changes["message"] = message
        return self._update(job_id, **changes)

    def update_message(self, job_id: str, message: str) -> Optional[JobRecord]:
        return self._update(job_id, message=message)

    def request_cancel(self, job_id: str) -> Optional[JobRecord]:
        return self._update(
            job_id,
            status=JobStatus.CANCEL_REQUESTED,
            message="Cancellation requested",
            recoverable=True,
        )

    def requeue(self, job_id: str, *, resume: bool = False) -> Optional[JobRecord]:
        """Create a fresh queued record for a failed or recoverable job.

        The original record is retained as an audit trail.  A worker can adopt
        the returned queued record by id, so retry/resume never mutates a
        finished record back into an active state.
        """
        with self._lock:
            records = self._read()
            source = next((record for record in records if record.id == job_id), None)
            if source is None:
                return None
            allowed = {JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.RECOVERABLE}
            if source.status not in allowed:
                return None
            if resume and not (
                source.status == JobStatus.RECOVERABLE
                or (source.status == JobStatus.CANCELLED and source.recoverable)
            ):
                return None
            metadata = dict(source.metadata) if isinstance(source.metadata, dict) else {}
            metadata.update({
                "retry_of": source.id,
                "resume": bool(resume),
            })
            queued = JobRecord(
                id=f"job_{int(time.time() * 1000)}_{uuid.uuid4().hex[:12]}",
                kind=source.kind,
                label=(f"Resume: {source.label}" if resume else f"Retry: {source.label}"),
                inputs=_jsonable(source.inputs),
                metadata=_jsonable(metadata),
            )
            records.append(queued)
            self._write(records)
            return queued

    def mark_completed(
        self,
        job_id: str,
        outputs: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[JobRecord]:
        return self._update(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            outputs=_jsonable(outputs or {}),
            metadata=_jsonable(metadata or {}),
            finished_at=time.time(),
            recoverable=False,
        )

    def mark_failed(
        self,
        job_id: str,
        error: str,
        outputs: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[JobRecord]:
        return self._update(
            job_id,
            status=JobStatus.FAILED,
            error=error,
            outputs=_jsonable(outputs or {}),
            metadata=_jsonable(metadata or {}),
            finished_at=time.time(),
            recoverable=True,
        )

    def mark_cancelled(
        self,
        job_id: str,
        outputs: Optional[dict[str, Any]] = None,
        recoverable: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[JobRecord]:
        message = "Cancelled; resume is available" if recoverable else "Cancelled"
        return self._update(
            job_id,
            status=JobStatus.CANCELLED,
            message=message,
            outputs=_jsonable(outputs or {}),
            metadata=_jsonable(metadata or {}),
            finished_at=time.time(),
            recoverable=recoverable,
        )

    def recover_stale_jobs(self) -> list[JobRecord]:
        recovered: list[JobRecord] = []
        with self._lock:
            records = self._read()
            changed = False
            for record in records:
                if record.status not in ACTIVE_STATUSES:
                    continue
                record.status = JobStatus.RECOVERABLE
                record.message = "Interrupted before completion; review or resume."
                record.recoverable = True
                record.updated_at = time.time()
                recovered.append(record)
                changed = True
            if changed:
                self._write(records)
        return recovered

    def cleanup_outputs(self, record_or_outputs: JobRecord | dict[str, Any] | list[str]) -> list[str]:
        paths = extract_output_paths(
            record_or_outputs,
            invalid_cb=lambda value: self._log_cleanup_rejection(
                value, "malformed output record"
            ),
        )
        removed: list[str] = []
        for raw_path in paths:
            try:
                if not raw_path or "\0" in raw_path:
                    self._log_cleanup_rejection(raw_path, "malformed path")
                    continue
                path = Path(raw_path)
                if not path.is_absolute():
                    self._log_cleanup_rejection(raw_path, "path is not absolute")
                    continue
                if path.is_symlink():
                    self._log_cleanup_rejection(raw_path, "symbolic links are not cleanup targets")
                    continue

                resolved = path.resolve(strict=True)
                if not self._is_approved_cleanup_path(resolved):
                    self._log_cleanup_rejection(raw_path, "path is outside approved roots")
                    continue
                if not resolved.is_file():
                    self._log_cleanup_rejection(raw_path, "path is not a regular file")
                    continue

                # Resolve again immediately before unlinking so a replaced symlink or
                # moved parent cannot redirect cleanup after the boundary check.
                current = path.resolve(strict=True)
                if current != resolved or not self._is_approved_cleanup_path(current):
                    self._log_cleanup_rejection(raw_path, "path changed during cleanup")
                    continue
                current.unlink()
                removed.append(str(current))
            except (OSError, RuntimeError, ValueError) as exc:
                self._log_cleanup_rejection(raw_path, f"unusable path: {type(exc).__name__}")
                continue
        return removed

    @property
    def cleanup_roots(self) -> tuple[Path, ...]:
        """Canonical directories that may contain cancellable job artifacts."""
        return self._cleanup_roots

    def _is_approved_cleanup_path(self, candidate: Path) -> bool:
        return any(
            candidate != root and candidate.is_relative_to(root)
            for root in self._cleanup_roots
        )

    @staticmethod
    def _log_cleanup_rejection(candidate: Any, reason: str) -> None:
        logger.warning("Skipped unsafe job output cleanup (%s): %r", reason, candidate)

    def delete(self, job_id: str) -> bool:
        """Remove one finished job record. Active jobs are never deleted."""
        with self._lock:
            records = self._read()
            keep = []
            removed = False
            for record in records:
                if record.id == job_id and record.status not in ACTIVE_STATUSES:
                    removed = True
                    continue
                keep.append(record)
            if removed:
                self._write(keep)
            return removed

    def _update(self, job_id: str, **changes: Any) -> Optional[JobRecord]:
        with self._lock:
            records = self._read()
            found: Optional[JobRecord] = None
            for record in records:
                if record.id != job_id:
                    continue
                for key, value in changes.items():
                    if key == "metadata" and value:
                        merged = dict(record.metadata)
                        merged.update(value)
                        setattr(record, key, merged)
                    else:
                        setattr(record, key, value)
                record.updated_at = time.time()
                found = record
                break
            if found:
                self._write(records)
            return found

    def _read(self) -> list[JobRecord]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Job ledger root must be an object")
            if payload.get("schema_version") != JOB_SCHEMA_VERSION:
                return []
            raw_jobs = payload.get("jobs", [])
            if not isinstance(raw_jobs, list):
                raise ValueError("Job ledger jobs must be a list")
            records: list[JobRecord] = []
            for item in raw_jobs:
                if not isinstance(item, dict):
                    logger.warning("Skipped malformed job record in %s", self.path)
                    continue
                try:
                    records.append(JobRecord.from_dict(item))
                except (TypeError, ValueError) as exc:
                    logger.warning("Skipped malformed job record in %s: %s", self.path, exc)
            return records
        except (json.JSONDecodeError, OSError, UnicodeError, TypeError, ValueError, AttributeError) as exc:
            logger.warning("Job ledger was unreadable and was quarantined: %s", exc)
            self._quarantine_corrupt_file()
            return []

    def _write(self, records: list[JobRecord]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": JOB_SCHEMA_VERSION,
            "updated_at": time.time(),
            "jobs": [record.to_dict() for record in records],
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def _quarantine_corrupt_file(self) -> None:
        if not self.path.exists():
            return
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        target = self.path.with_suffix(f".{stamp}.corrupt")
        try:
            self.path.replace(target)
        except OSError:
            pass


MAX_LOG_ENTRIES = 200
MAX_LOG_MESSAGE_LEN = 500


@dataclass
class JobLogEntry:
    timestamp: float
    level: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"t": self.timestamp, "l": self.level, "m": self.message}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobLogEntry":
        return cls(
            timestamp=data.get("t", 0.0),
            level=data.get("l", "info"),
            message=data.get("m", ""),
        )


class JobLog:
    """Bounded redacted log artifact for a single job."""

    def __init__(self, job_id: str, root: Optional[Path] = None):
        self._job_id = job_id
        self._root = Path(root) if root else get_config_dir() / "jobs" / "logs"
        self._root.mkdir(parents=True, exist_ok=True)
        self._path = self._root / f"{job_id}.log.json"
        self._entries: list[JobLogEntry] = []
        self._device_info: dict[str, Any] = {}
        self._model_info: dict[str, Any] = {}
        self._redact_patterns: list[str] = []

    def set_device_info(self, info: dict[str, Any]) -> None:
        self._device_info = dict(info)

    def set_model_info(self, info: dict[str, Any]) -> None:
        self._model_info = dict(info)

    def add_redact_pattern(self, pattern: str) -> None:
        if pattern and pattern not in self._redact_patterns:
            self._redact_patterns.append(pattern)

    def info(self, message: str) -> None:
        self._append("info", message)

    def warn(self, message: str) -> None:
        self._append("warn", message)

    def error(self, message: str) -> None:
        self._append("error", message)

    def _append(self, level: str, message: str) -> None:
        redacted = self._redact(message[:MAX_LOG_MESSAGE_LEN])
        self._entries.append(JobLogEntry(time.time(), level, redacted))
        if len(self._entries) > MAX_LOG_ENTRIES:
            self._entries = self._entries[-MAX_LOG_ENTRIES:]

    def _redact(self, text: str) -> str:
        import re
        result = text
        for pattern in self._redact_patterns:
            if pattern:
                result = result.replace(pattern, "[REDACTED]")
        result = re.sub(r"hf_[A-Za-z0-9][A-Za-z0-9_-]{7,}", "[REDACTED_HF_TOKEN]", result)
        return result

    def save(self) -> Path:
        payload = {
            "job_id": self._job_id,
            "device": self._device_info,
            "model": self._model_info,
            "entry_count": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self._path

    def summary(self, limit: int = 10) -> list[dict[str, Any]]:
        recent = self._entries[-limit:]
        return [e.to_dict() for e in recent]

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entry_count(self) -> int:
        return len(self._entries)


OUTPUT_KEYS = {
    "audio_path",
    "artifact_paths",
    "final_audio_path",
    "mastered_audio_path",
    "mix_path",
    "master_path",
    "output_paths",
    "provenance_path",
    "output_path",
    "path",
    "sfx_audio_path",
    "song_audio_path",
    "file_path",
    "vocal_audio_path",
    "vocal_stem_path",
    "vocal_stem_provenance_path",
}


def extract_output_paths(
    payload: Any,
    *,
    invalid_cb: Optional[Callable[[Any], None]] = None,
) -> list[str]:
    paths: list[str] = []

    def visit(value: Any, candidate_expected: bool = False) -> None:
        if value is None:
            return
        if isinstance(value, (str, Path)):
            paths.append(str(value))
            return
        if isinstance(value, JobRecord):
            visit(value.outputs)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if key in OUTPUT_KEYS:
                    if isinstance(item, (str, Path)):
                        paths.append(str(item))
                    elif isinstance(item, (list, tuple, set)):
                        visit(item, candidate_expected=True)
                    elif item is not None and invalid_cb:
                        invalid_cb(item)
                elif key in {"results", "outputs", "sections", "paths"}:
                    visit(item, candidate_expected=True)
                elif isinstance(item, (dict, list, tuple)):
                    visit(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item, candidate_expected=candidate_expected)
            return
        found_path = False
        for attr in (
            "audio_path",
            "artifact_paths",
            "final_audio_path",
            "mastered_audio_path",
            "mix_path",
            "master_path",
            "output_path",
            "output_paths",
            "provenance_path",
            "sfx_audio_path",
            "song_audio_path",
            "vocal_audio_path",
            "vocal_stem_path",
            "vocal_stem_provenance_path",
        ):
            item = getattr(value, attr, None)
            if item:
                if isinstance(item, (list, tuple, set)):
                    visit(item, candidate_expected=True)
                else:
                    paths.append(str(item))
                found_path = True
        if candidate_expected and not found_path and invalid_cb:
            invalid_cb(value)

    visit(payload)
    return list(dict.fromkeys(paths))


def _default_cleanup_roots() -> list[Path]:
    config_dir = get_config_dir()
    roots = [
        config_dir / "generations",
        config_dir / "projects",
        config_dir / "temp",
        Path(tempfile.gettempdir()) / APP_NAME,
        get_default_output_dir(),
    ]
    settings_path = config_dir / "config.json"
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
        configured = payload.get("general", {}).get("output_dir")
        if configured:
            roots.append(Path(configured))
    except (json.JSONDecodeError, OSError, TypeError, AttributeError):
        pass
    return roots


def _canonical_cleanup_roots(roots: Iterable[Path | str]) -> tuple[Path, ...]:
    canonical: list[Path] = []
    for raw_root in roots:
        try:
            root = Path(raw_root)
            if not root.is_absolute():
                logger.warning("Ignored relative job cleanup root: %r", raw_root)
                continue
            resolved = root.resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            logger.warning("Ignored malformed job cleanup root: %r", raw_root)
            continue
        if resolved not in canonical:
            canonical.append(resolved)
    return tuple(canonical)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return str(value)
