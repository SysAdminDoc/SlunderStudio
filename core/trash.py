"""
Slunder Studio v0.1.28 - Recoverable trash/quarantine support.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from core.settings import Settings, get_trash_dir


class TrashError(RuntimeError):
    """Raised when a recoverable delete or restore cannot be completed."""


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
                json.dumps(asdict(entry), indent=2),
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

    def restore(self, entry_id: str) -> TrashEntry:
        entry = self.get_entry(entry_id)
        if entry is None:
            raise TrashError(f"Trash entry not found: {entry_id}")

        source = Path(entry.trash_path)
        dest = Path(entry.original_path)
        if not source.exists():
            raise TrashError(f"Trash payload missing: {source}")
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
        manifest = self.trash_dir / entry_id / self.MANIFEST_NAME
        if not manifest.exists():
            return None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            return TrashEntry(**data)
        except Exception as exc:
            raise TrashError(f"Trash manifest is corrupt: {manifest}: {exc}") from exc

    def list_entries(self, category: Optional[str] = None) -> list[TrashEntry]:
        entries: list[TrashEntry] = []
        for manifest in self.trash_dir.glob(f"*/{self.MANIFEST_NAME}"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                entry = TrashEntry(**data)
                if category is None or entry.category == category:
                    entries.append(entry)
            except Exception:
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
        return f"{category}_{int(time.time() * 1000)}_{safe_label}"

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
