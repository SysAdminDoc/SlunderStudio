"""
Slunder Studio - Recoverable trash/quarantine support.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Optional

from core.settings import Settings, get_trash_dir


class TrashError(RuntimeError):
    """Raised when a recoverable delete or restore cannot be completed."""


logger = logging.getLogger(__name__)


@dataclass
class TrashEntry:
    id: str
    category: str
    label: str
    original_path: str
    trash_path: str
    manifest_path: str
    deleted_at: float
    expires_at: float
    is_dir: bool
    size_bytes: int
    file_count: int
    metadata: dict[str, Any]
    _extra_fields: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrashEntry":
        if not isinstance(data, dict):
            raise ValueError("Trash manifest root must be an object")
        known = {
            item.name for item in fields(cls)
            if item.name != "_extra_fields"
        }
        values = {key: value for key, value in data.items() if key in known}
        extras = {key: value for key, value in data.items() if key not in known}
        return cls(**values, _extra_fields=extras)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self._extra_fields)
        payload.update({
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "_extra_fields"
        })
        return payload


class TrashManager:
    """Moves expensive local artifacts into app trash and restores by manifest."""

    MANIFEST_NAME = "manifest.json"

    def __init__(
        self,
        trash_dir: Optional[Path | str] = None,
        retention_days: Optional[float] = None,
    ):
        settings = Settings()
        self.trash_dir = Path(trash_dir) if trash_dir else get_trash_dir()
        self.retention_days = (
            float(retention_days)
            if retention_days is not None
            else float(settings.get("general.trash_retention_days", 30))
        )
        self.trash_dir.mkdir(parents=True, exist_ok=True)

    def trash_path(
        self,
        path: Path | str,
        *,
        category: str,
        label: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TrashEntry:
        self.cleanup_expired()

        source = Path(path).resolve()
        if not source.exists():
            raise TrashError(f"Cannot delete missing path: {source}")

        entry_id = self._new_entry_id(category, label)
        entry_dir = self.trash_dir / entry_id
        trash_path = entry_dir / source.name
        manifest_path = entry_dir / self.MANIFEST_NAME
        size_bytes, file_count = self._summarize(source)
        deleted_at = time.time()
        expires_at = deleted_at + max(self.retention_days, 0) * 86400

        try:
            entry_dir.mkdir(parents=True, exist_ok=False)
            shutil.move(str(source), str(trash_path))
            entry = TrashEntry(
                id=entry_id,
                category=category,
                label=label,
                original_path=str(source),
                trash_path=str(trash_path),
                manifest_path=str(manifest_path),
                deleted_at=deleted_at,
                expires_at=expires_at,
                is_dir=trash_path.is_dir(),
                size_bytes=size_bytes,
                file_count=file_count,
                metadata=metadata or {},
            )
            manifest_path.write_text(
                json.dumps(entry.to_dict(), indent=2),
                encoding="utf-8",
            )
            return entry
        except Exception as exc:
            if trash_path.exists() and not source.exists():
                try:
                    source.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(trash_path), str(source))
                except Exception:
                    pass
            if entry_dir.exists():
                shutil.rmtree(entry_dir, ignore_errors=True)
            raise TrashError(f"Failed to move {source} to trash: {exc}") from exc

    def trash_paths(
        self,
        requests: Iterable[dict[str, Any]],
    ) -> list[TrashEntry]:
        """Move several paths transactionally, rolling back on any failure.

        Each request must contain ``path``, ``category`` and ``label``.  The
        optional ``metadata`` value is copied into that entry's manifest.  UI
        batch actions use this helper so a partial clear can never leave the
        visible collection out of sync with the files on disk.
        """
        entries: list[TrashEntry] = []
        try:
            for request in requests:
                entries.append(
                    self.trash_path(
                        request["path"],
                        category=str(request["category"]),
                        label=str(request["label"]),
                        metadata=request.get("metadata"),
                    )
                )
        except Exception:
            for entry in reversed(entries):
                try:
                    self.restore(entry.id)
                except TrashError:
                    # Preserve the original failure.  The manifest remains
                    # available for the recovery UI if rollback itself fails.
                    pass
            raise
        return entries

    def restore(self, entry_id: str) -> TrashEntry:
        entry = self.get_entry(entry_id)
        if entry is None:
            raise TrashError(f"Trash entry not found: {entry_id}")

        self._validate_entry(entry, self.trash_dir / entry_id / self.MANIFEST_NAME)

        source = Path(entry.trash_path)
        dest = Path(entry.original_path)
        if not source.exists():
            raise TrashError(f"Trash payload missing: {source}")
        if source.is_symlink():
            raise TrashError(f"Trash payload cannot be a symbolic link: {source}")
        if dest.exists():
            raise TrashError(f"Restore target already exists: {dest}")

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
            entry_dir = Path(entry.manifest_path).parent
            shutil.rmtree(entry_dir)
            return entry
        except Exception as exc:
            raise TrashError(f"Failed to restore {entry_id}: {exc}") from exc

    def get_entry(self, entry_id: str) -> Optional[TrashEntry]:
        entry_dir = self._entry_dir(entry_id)
        manifest = entry_dir / self.MANIFEST_NAME
        if not manifest.exists():
            return None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            entry = TrashEntry.from_dict(data)
            self._validate_entry(entry, manifest)
            return entry
        except TrashError:
            raise
        except (OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TrashError(f"Trash manifest is corrupt: {manifest}: {exc}") from exc

    def list_entries(self, category: Optional[str] = None) -> list[TrashEntry]:
        entries: list[TrashEntry] = []
        for manifest in self.trash_dir.glob(f"*/{self.MANIFEST_NAME}"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                entry = TrashEntry.from_dict(data)
                self._validate_entry(entry, manifest)
                if category is None or entry.category == category:
                    entries.append(entry)
            except (OSError, UnicodeError, TypeError, ValueError, TrashError) as exc:
                logger.warning("Skipped unsafe trash manifest %s: %s", manifest, exc)
                continue
        return sorted(entries, key=lambda e: e.deleted_at, reverse=True)

    def cleanup_expired(self, now: Optional[float] = None) -> list[str]:
        current = time.time() if now is None else now
        removed: list[str] = []
        for entry in self.list_entries():
            if entry.expires_at <= current:
                entry_dir = Path(entry.manifest_path).parent
                try:
                    shutil.rmtree(entry_dir)
                    removed.append(entry.id)
                except OSError as exc:
                    raise TrashError(
                        f"Failed to remove expired trash entry {entry.id}: {exc}"
                    ) from exc
        return removed

    def _new_entry_id(self, category: str, label: str) -> str:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
        safe_label = safe_label[:48] or "item"
        return f"{category}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:10]}_{safe_label}"

    def _entry_dir(self, entry_id: str) -> Path:
        if not isinstance(entry_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", entry_id):
            raise TrashError(f"Invalid trash entry id: {entry_id!r}")
        if entry_id in {".", ".."}:
            raise TrashError(f"Invalid trash entry id: {entry_id!r}")
        root = self.trash_dir.resolve(strict=False)
        entry_dir = self.trash_dir / entry_id
        if entry_dir.resolve(strict=False).parent != root:
            raise TrashError(f"Trash entry escapes the trash root: {entry_id!r}")
        return entry_dir

    def _validate_entry(self, entry: TrashEntry, manifest: Path) -> None:
        """Validate manifest paths before any manifest-directed filesystem move."""
        entry_dir = self._entry_dir(entry.id)
        root = self.trash_dir.resolve(strict=False)
        expected_manifest = entry_dir.resolve(strict=False) / self.MANIFEST_NAME
        if manifest.resolve(strict=False) != expected_manifest:
            raise TrashError("Trash manifest path does not match its entry directory")

        payload = Path(entry.trash_path)
        if payload.name in {"", ".", ".."}:
            raise TrashError("Trash payload name is invalid")
        if payload.parent.resolve(strict=False) != entry_dir.resolve(strict=False):
            raise TrashError("Trash payload escapes its entry directory")
        if payload.is_symlink() or payload.resolve(strict=False).parent != entry_dir.resolve(strict=False):
            raise TrashError("Trash payload is a symbolic-link escape")

        destination = Path(entry.original_path)
        if not destination.is_absolute():
            raise TrashError("Trash restore target must be absolute")
        if any(part in {".", ".."} for part in destination.parts):
            raise TrashError("Trash restore target contains traversal components")
        if destination.name != payload.name:
            raise TrashError("Trash restore target name does not match the payload")
        if destination.resolve(strict=False).is_relative_to(root):
            raise TrashError("Trash restore target cannot be inside the trash root")

    @staticmethod
    def _summarize(path: Path) -> tuple[int, int]:
        if path.is_file():
            return path.stat().st_size, 1
        size = 0
        count = 0
        for child in path.rglob("*"):
            if child.is_file():
                try:
                    size += child.stat().st_size
                    count += 1
                except OSError:
                    pass
        return size, count
