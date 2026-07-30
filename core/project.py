"""
Slunder Studio v0.1.29 — Project Management
Save, load, and manage music projects with auto-save, version history,
and asset tracking across all modules.
"""
import os
import json
import re
import threading
import time
import shutil
import uuid
from typing import Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

from core.provenance import (
    find_provenance_sidecar,
    project_metadata_from_provenance,
    read_provenance_sidecar,
    sidecar_path_for,
)
from core.settings import APP_VERSION, get_config_dir
from core.trash import TrashEntry, TrashError, TrashManager

PROJECT_SCHEMA_VERSION = 2


@dataclass
class ProjectRepairStatus:
    """Persistence repair or migration status for project JSON."""
    status: str = "ok"  # ok | migrated | repaired | error
    messages: list[str] = field(default_factory=list)
    backup_paths: list[str] = field(default_factory=list)


@dataclass
class ProjectAsset:
    """A single asset (audio, MIDI, lyrics, etc.) within a project."""
    id: str = ""
    name: str = ""
    asset_type: str = ""  # "audio" | "midi" | "lyrics" | "stems" | "sfx" | "export"
    file_path: str = ""
    module: str = ""  # which module created it
    provenance_path: str = ""
    created_at: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"asset_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class ProjectVersion:
    """A saved snapshot of the project state."""
    version: int = 1
    timestamp: float = 0.0
    description: str = ""
    auto_save: bool = False


@dataclass
class Project:
    """Complete project with all metadata and assets."""
    schema_version: int = PROJECT_SCHEMA_VERSION
    app_version: str = APP_VERSION
    id: str = ""
    name: str = "Untitled Project"
    description: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    tempo: float = 120.0
    key: str = "C major"
    time_signature: tuple = (4, 4)
    tags: list[str] = field(default_factory=list)
    assets: list[ProjectAsset] = field(default_factory=list)
    versions: list[ProjectVersion] = field(default_factory=list)
    mixer_state: dict = field(default_factory=dict)
    lyrics_text: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"proj_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = time.time()

    @property
    def asset_count(self) -> int:
        return len(self.assets)

    @property
    def version_count(self) -> int:
        return len(self.versions)

    def add_asset(self, asset: ProjectAsset) -> str:
        self.assets.append(asset)
        self.updated_at = time.time()
        return asset.id

    def remove_asset(self, asset_id: str) -> bool:
        for i, a in enumerate(self.assets):
            if a.id == asset_id:
                self.assets.pop(i)
                self.updated_at = time.time()
                return True
        return False

    def get_assets_by_type(self, asset_type: str) -> list[ProjectAsset]:
        return [a for a in self.assets if a.asset_type == asset_type]

    def get_assets_by_module(self, module: str) -> list[ProjectAsset]:
        return [a for a in self.assets if a.module == module]


# ── Project Manager ────────────────────────────────────────────────────────────

class ProjectManager:
    """Manages project persistence, auto-save, and version history."""

    _instance: Optional["ProjectManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._projects_dir = os.path.join(get_config_dir(), "projects")
        self._index_path = os.path.join(self._projects_dir, "index.json")
        self._current: Optional[Project] = None
        self._index: dict[str, dict] = {}  # id -> {name, path, updated_at}
        self._trash = TrashManager()
        self._repair_status: dict[str, ProjectRepairStatus] = {}
        self._last_repair_status = ProjectRepairStatus()
        os.makedirs(self._projects_dir, exist_ok=True)
        self._load_index()

    def _load_index(self):
        index_path = Path(self._index_path)
        loaded_index: dict[str, dict] = {}
        messages: list[str] = []
        backup_paths: list[str] = []
        index_failed = False

        if index_path.is_file():
            try:
                loaded_index = self._read_json_object(index_path)
            except (json.JSONDecodeError, OSError) as exc:
                index_failed = True
                backup = self._backup_file(index_path, "corrupt")
                if backup:
                    backup_paths.append(str(backup))
                messages.append(f"Project index was unreadable: {exc}")
                loaded_index, recovered_backup = self._load_latest_index_backup()
                if recovered_backup:
                    backup_paths.append(str(recovered_backup))
                    messages.append(
                        f"Recovered index seed from backup {recovered_backup.name}."
                    )

        rebuilt, scan_messages, scan_backups = self._reconstruct_index(loaded_index)
        messages.extend(scan_messages)
        backup_paths.extend(str(path) for path in scan_backups)
        changed = rebuilt != loaded_index
        self._index = rebuilt

        if index_failed or changed or messages:
            if rebuilt or index_path.exists():
                self._save_index(create_backup=False)
            self._last_repair_status = ProjectRepairStatus(
                status="repaired",
                messages=messages or ["Rebuilt the project index from project files."],
                backup_paths=list(dict.fromkeys(backup_paths)),
            )

    def _save_index(self, create_backup: bool = True):
        if create_backup:
            self._backup_file(Path(self._index_path), "pre-save")
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)
        os.replace(tmp, self._index_path)

    @staticmethod
    def _read_json_object(path: Path) -> dict:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise json.JSONDecodeError("JSON root is not an object", "", 0)
        return payload

    def _load_latest_index_backup(self) -> tuple[dict[str, dict], Optional[Path]]:
        backup_dir = Path(self._projects_dir) / "backups"
        candidates = sorted(
            backup_dir.glob("index.json.*.bak"),
            key=lambda path: path.name,
            reverse=True,
        )
        for candidate in candidates:
            try:
                return self._read_json_object(candidate), candidate
            except (json.JSONDecodeError, OSError):
                continue
        return {}, None

    @staticmethod
    def _safe_project_id(project_id: object) -> bool:
        if not isinstance(project_id, str) or not project_id:
            return False
        return (
            project_id not in {".", ".."}
            and "/" not in project_id
            and "\\" not in project_id
            and bool(re.fullmatch(r"[A-Za-z0-9._-]+", project_id))
        )

    def _reconstruct_index(
        self,
        seed: dict,
    ) -> tuple[dict[str, dict], list[str], list[Path]]:
        """Rebuild canonical index entries from internal project directories."""
        root = Path(self._projects_dir).resolve()
        rebuilt: dict[str, dict] = {}
        messages: list[str] = []
        backup_paths: list[Path] = []

        for project_id, entry in seed.items():
            if not self._safe_project_id(project_id) or not isinstance(entry, dict):
                messages.append(f"Ignored invalid index entry {project_id!r}.")
                continue
            project_dir = root / project_id
            if not project_dir.is_dir():
                messages.append(f"Removed missing project {project_id} from the index.")
                continue
            rebuilt[project_id] = {
                "name": str(entry.get("name") or project_id),
                "path": str(project_dir),
                "updated_at": self._safe_timestamp(entry.get("updated_at")),
            }

        for project_dir in sorted(root.iterdir(), key=lambda path: path.name):
            if (
                not project_dir.is_dir()
                or project_dir.name == "backups"
                or not self._safe_project_id(project_dir.name)
            ):
                continue
            data, recovery_messages, recovery_backups = (
                self._load_recoverable_project_metadata(project_dir)
            )
            messages.extend(recovery_messages)
            backup_paths.extend(recovery_backups)
            if data is None:
                continue

            data_id = data.get("id") or project_dir.name
            if data_id != project_dir.name:
                messages.append(
                    f"Ignored project directory {project_dir.name}: "
                    f"metadata id {data_id!r} does not match."
                )
                continue
            entry = {
                "name": str(data.get("name") or project_dir.name),
                "path": str(project_dir),
                "updated_at": self._safe_timestamp(data.get("updated_at")),
            }
            if project_dir.name not in rebuilt:
                messages.append(
                    f"Recovered project {entry['name']} ({project_dir.name}) into the index."
                )
            rebuilt[project_dir.name] = entry

        return rebuilt, messages, backup_paths

    def _load_recoverable_project_metadata(
        self,
        project_dir: Path,
    ) -> tuple[Optional[dict], list[str], list[Path]]:
        meta_path = project_dir / "project.json"
        try:
            return self._read_json_object(meta_path), [], []
        except (json.JSONDecodeError, OSError):
            pass

        backup_dir = project_dir / "backups"
        candidates = sorted(
            backup_dir.glob("project.json.*.bak"),
            key=lambda path: path.name,
            reverse=True,
        )
        for candidate in candidates:
            try:
                data = self._read_json_object(candidate)
            except (json.JSONDecodeError, OSError):
                continue
            backup_paths = [candidate]
            if meta_path.exists():
                corrupt_backup = self._backup_file(meta_path, "corrupt")
                if corrupt_backup:
                    backup_paths.append(corrupt_backup)
            tmp = meta_path.with_name(meta_path.name + ".recovery.tmp")
            try:
                shutil.copy2(candidate, tmp)
                os.replace(tmp, meta_path)
            except OSError:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                return None, [
                    f"Found a backup for {project_dir.name} but could not restore it."
                ], backup_paths
            return data, [
                f"Restored project {project_dir.name} from backup {candidate.name}."
            ], backup_paths
        return None, [], []

    @staticmethod
    def _safe_timestamp(value: object) -> float:
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return 0.0
        return timestamp if timestamp >= 0 else 0.0

    def rebuild_index(self) -> int:
        """Rescan project storage and persist a canonical internal index."""
        rebuilt, messages, backups = self._reconstruct_index(self._index)
        self._index = rebuilt
        self._save_index()
        self._last_repair_status = ProjectRepairStatus(
            status="repaired" if messages else "ok",
            messages=messages,
            backup_paths=[str(path) for path in backups],
        )
        return len(rebuilt)

    # ── CRUD ───────────────────────────────────────────────────────────────────

    def create(self, name: str = "Untitled Project", **kwargs) -> Project:
        project = Project(name=name, **kwargs)
        project_dir = os.path.join(self._projects_dir, project.id)
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "assets"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "versions"), exist_ok=True)

        self._save_project(project, create_backup=False)
        self._index[project.id] = {
            "name": project.name,
            "path": project_dir,
            "updated_at": project.updated_at,
        }
        self._save_index()
        self._current = project
        return project

    def open(self, project_id: str) -> Optional[Project]:
        self._last_repair_status = ProjectRepairStatus()
        if not self._safe_project_id(project_id):
            return None
        if project_id not in self._index:
            self.rebuild_index()
            if project_id not in self._index:
                return None

        project_dir = os.path.join(self._projects_dir, project_id)
        meta_path = os.path.join(project_dir, "project.json")

        try:
            data, recovery_messages, recovery_backups = (
                self._load_recoverable_project_metadata(Path(project_dir))
            )
            if data is None:
                raise json.JSONDecodeError(
                    "No readable project metadata or backup",
                    "",
                    0,
                )
            data, migrated, messages = self._migrate_project_data(data, project_id)
            if migrated or recovery_messages:
                backup = self._backup_file(Path(meta_path), "pre-migration")
                self._last_repair_status = ProjectRepairStatus(
                    status="migrated" if migrated else "repaired",
                    messages=[*recovery_messages, *messages],
                    backup_paths=list(dict.fromkeys([
                        *(str(path) for path in recovery_backups),
                        *([str(backup)] if backup else []),
                    ])),
                )
                self._repair_status[project_id] = self._last_repair_status

            project = Project(
                schema_version=data.get("schema_version", PROJECT_SCHEMA_VERSION),
                app_version=data.get("app_version", APP_VERSION),
                id=data.get("id", project_id),
                name=data.get("name", ""),
                description=data.get("description", ""),
                created_at=data.get("created_at", 0),
                updated_at=data.get("updated_at", 0),
                tempo=data.get("tempo", 120),
                key=data.get("key", "C major"),
                tags=data.get("tags", []),
                lyrics_text=data.get("lyrics_text", ""),
                notes=data.get("notes", ""),
                mixer_state=data.get("mixer_state", {}),
            )

            # Reconstruct time_signature from list
            ts = data.get("time_signature", [4, 4])
            project.time_signature = tuple(ts) if isinstance(ts, list) else ts

            # Load assets
            for a_data in data.get("assets", []):
                project.assets.append(ProjectAsset(**{
                    k: v for k, v in a_data.items()
                    if k in ProjectAsset.__dataclass_fields__
                }))

            # Load versions
            for v_data in data.get("versions", []):
                project.versions.append(ProjectVersion(**{
                    k: v for k, v in v_data.items()
                    if k in ProjectVersion.__dataclass_fields__
                }))

            self._current = project
            if migrated or recovery_messages:
                self._save_project(project, create_backup=False)
            return project

        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            backup = self._backup_file(Path(meta_path), "corrupt")
            self._last_repair_status = ProjectRepairStatus(
                status="repaired",
                messages=[f"Project JSON was unreadable and left closed: {exc}"],
                backup_paths=[str(backup)] if backup else [],
            )
            self._repair_status[project_id] = self._last_repair_status
            return None

    def save(self, project: Optional[Project] = None) -> bool:
        project = project or self._current
        if project is None:
            return False

        try:
            project.updated_at = time.time()
            self._save_project(project)

            self._index[project.id] = {
                "name": project.name,
                "path": os.path.join(self._projects_dir, project.id),
                "updated_at": project.updated_at,
            }
            self._save_index()
            return True
        except (IOError, OSError, json.JSONDecodeError) as e:
            print(f"[Slunder Studio] Failed to save project: {e}")
            return False

    def delete(self, project_id: str) -> Optional[TrashEntry]:
        if project_id not in self._index:
            return None

        index_entry = dict(self._index[project_id])
        project_dir = self._index[project_id]["path"]
        try:
            entry = self._trash.trash_path(
                project_dir,
                category="project",
                label=index_entry.get("name") or project_id,
                metadata={
                    "project_id": project_id,
                    "index_entry": index_entry,
                },
            )
        except TrashError as e:
            print(f"[Slunder Studio] Failed to delete project: {e}")
            return None

        del self._index[project_id]
        self._save_index()

        if self._current and self._current.id == project_id:
            self._current = None
        return entry

    def restore_deleted_project(self, trash_entry_id: str) -> bool:
        try:
            entry = self._trash.restore(trash_entry_id)
        except TrashError as e:
            print(f"[Slunder Studio] Failed to restore project: {e}")
            return False

        project_id = entry.metadata.get("project_id")
        index_entry = entry.metadata.get("index_entry") or {}
        if not project_id:
            return False

        index_entry["path"] = str(Path(entry.original_path))
        if "name" not in index_entry:
            index_entry["name"] = Path(entry.original_path).name
        if "updated_at" not in index_entry:
            index_entry["updated_at"] = time.time()
        self._index[project_id] = index_entry
        self._save_index()
        return True

    def _save_project(self, project: Project, create_backup: bool = True):
        project_dir = os.path.join(self._projects_dir, project.id)
        os.makedirs(project_dir, exist_ok=True)
        project.schema_version = PROJECT_SCHEMA_VERSION
        project.app_version = APP_VERSION

        data = {
            "schema_version": project.schema_version,
            "app_version": project.app_version,
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "updated_at": project.updated_at,
            "tempo": project.tempo,
            "key": project.key,
            "time_signature": list(project.time_signature),
            "tags": project.tags,
            "lyrics_text": project.lyrics_text,
            "notes": project.notes,
            "mixer_state": project.mixer_state,
            "assets": [asdict(a) for a in project.assets],
            "versions": [asdict(v) for v in project.versions],
        }

        meta_path = os.path.join(project_dir, "project.json")
        if create_backup:
            backup = self._backup_file(Path(meta_path), "pre-save")
            if backup:
                status = self._repair_status.get(project.id, ProjectRepairStatus())
                status.backup_paths.append(str(backup))
                self._repair_status[project.id] = status
        # Write to temp file first, then rename for atomicity
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, meta_path)

    # ── Listing ────────────────────────────────────────────────────────────────

    def list_projects(self) -> list[dict]:
        return sorted(
            [{"id": k, **v} for k, v in self._index.items()],
            key=lambda x: x.get("updated_at", 0),
            reverse=True,
        )

    @property
    def current(self) -> Optional[Project]:
        return self._current

    @property
    def project_count(self) -> int:
        return len(self._index)

    @property
    def last_repair_status(self) -> dict:
        return self._status_to_dict(self._last_repair_status)

    def repair_status(self, project_id: str) -> dict:
        return self._status_to_dict(
            self._repair_status.get(project_id, ProjectRepairStatus())
        )

    def _migrate_project_data(self, data: dict, project_id: str) -> tuple[dict, bool, list[str]]:
        if not isinstance(data, dict):
            raise json.JSONDecodeError("Project root is not an object", "", 0)

        migrated = False
        messages: list[str] = []
        updated = dict(data)

        try:
            schema_version = int(updated.get("schema_version", 1) or 1)
        except (TypeError, ValueError):
            schema_version = 1

        if schema_version < 2:
            updated["schema_version"] = PROJECT_SCHEMA_VERSION
            updated.setdefault("assets", [])
            updated.setdefault("versions", [])
            updated.setdefault("mixer_state", {})
            updated.setdefault("lyrics_text", "")
            updated.setdefault("notes", "")
            messages.append("Migrated project schema from v1 to v2.")
            migrated = True
        elif schema_version > PROJECT_SCHEMA_VERSION:
            messages.append(
                f"Project schema v{schema_version} is newer than supported v{PROJECT_SCHEMA_VERSION}; preserved compatible keys."
            )

        if updated.get("schema_version") != PROJECT_SCHEMA_VERSION:
            updated["schema_version"] = PROJECT_SCHEMA_VERSION
            migrated = True
        if updated.get("app_version") != APP_VERSION:
            updated["app_version"] = APP_VERSION
            messages.append(f"Updated project app version to {APP_VERSION}.")
            migrated = True
        if not updated.get("id"):
            updated["id"] = project_id
            messages.append("Restored missing project id from the project index.")
            migrated = True

        return updated, migrated, messages

    def _backup_file(self, path: Path, reason: str) -> Optional[Path]:
        if not path.exists():
            return None
        try:
            backup_dir = path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
            backup_path = backup_dir / f"{path.name}.{stamp}.{reason}.bak"
            shutil.copy2(path, backup_path)
            return backup_path
        except OSError as exc:
            self._last_repair_status = ProjectRepairStatus(
                status="error",
                messages=[f"Backup failed for {path}: {exc}"],
            )
            return None

    @staticmethod
    def _status_to_dict(status: ProjectRepairStatus) -> dict:
        return {
            "status": status.status,
            "messages": list(status.messages),
            "backup_paths": list(status.backup_paths),
        }

    # ── Version History ────────────────────────────────────────────────────────

    def create_version(self, description: str = "", auto_save: bool = False) -> bool:
        if self._current is None:
            return False

        ver = ProjectVersion(
            version=self._current.version_count + 1,
            timestamp=time.time(),
            description=description or f"Version {self._current.version_count + 1}",
            auto_save=auto_save,
        )
        self._current.versions.append(ver)

        # Save snapshot
        project_dir = os.path.join(self._projects_dir, self._current.id)
        ver_dir = os.path.join(project_dir, "versions", f"v{ver.version}")
        os.makedirs(ver_dir, exist_ok=True)

        # Copy current project.json as snapshot
        src = os.path.join(project_dir, "project.json")
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(ver_dir, "project.json"))

        self.save()
        return True

    # ── Asset Management ───────────────────────────────────────────────────────

    def import_asset(self, file_path: str, asset_type: str,
                     module: str, name: Optional[str] = None,
                     provenance_path: Optional[str] = None) -> Optional[str]:
        """Import a file as a project asset (copies to project directory)."""
        if self._current is None:
            return None

        source = Path(file_path).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Project asset is not a regular file: {source}")

        display_name = str(name or source.name).strip() or source.name
        display_name = Path(display_name).name or source.name
        asset = ProjectAsset(
            name=display_name,
            asset_type=asset_type,
            module=module,
        )

        assets_dir = Path(
            self._projects_dir,
            self._current.id,
            "assets",
        ).resolve()
        assets_dir.mkdir(parents=True, exist_ok=True)
        storage_name = self._asset_storage_name(asset.id, display_name, source.suffix)
        dest = assets_dir / storage_name
        if dest.parent != assets_dir:
            raise ValueError("Asset destination escaped the project assets directory")

        explicit_sidecar = Path(provenance_path) if provenance_path else None
        if explicit_sidecar is not None and not explicit_sidecar.is_file():
            raise FileNotFoundError(
                f"Provenance sidecar not found: {explicit_sidecar}"
            )
        sidecar = explicit_sidecar or find_provenance_sidecar(source)
        sidecar_dest = sidecar_path_for(dest) if sidecar else None
        copied_paths: list[Path] = []

        try:
            self._copy_file_exclusive(source, dest)
            copied_paths.append(dest)
            provenance_metadata = {}
            if sidecar and sidecar.is_file() and sidecar_dest is not None:
                self._copy_file_exclusive(sidecar.resolve(strict=True), sidecar_dest)
                copied_paths.append(sidecar_dest)
                provenance = read_provenance_sidecar(sidecar_dest)
                provenance_metadata = project_metadata_from_provenance(
                    provenance,
                    sidecar_dest,
                )

            asset.file_path = str(dest)
            asset.provenance_path = str(sidecar_dest) if sidecar_dest else ""
            asset.metadata = {
                "storage": {
                    "original_filename": source.name,
                    "stored_filename": dest.name,
                },
                **provenance_metadata,
            }
            self._current.add_asset(asset)
            if not self.save():
                self._current.remove_asset(asset.id)
                raise OSError("Project metadata could not be saved after asset import")
            return asset.id
        except Exception:
            removed = self._current.remove_asset(asset.id)
            if removed:
                try:
                    self._save_project(self._current, create_backup=False)
                except (OSError, TypeError, ValueError):
                    pass
            for copied_path in reversed(copied_paths):
                try:
                    copied_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    @staticmethod
    def _asset_storage_name(asset_id: str, display_name: str, source_suffix: str) -> str:
        safe_name = Path(display_name).name
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", safe_name)
        safe_name = safe_name.strip(" .") or "asset"
        suffix = Path(safe_name).suffix or source_suffix
        stem = Path(safe_name).stem if Path(safe_name).suffix else safe_name
        suffix = re.sub(r"[^A-Za-z0-9._-]", "_", suffix)[:16]
        stem = re.sub(r"\s+", " ", stem).strip()[:96] or "asset"
        return f"{asset_id}__{stem}{suffix}"

    @staticmethod
    def _copy_file_exclusive(source: Path, destination: Path) -> None:
        """Copy without any overwrite window, preserving source timestamps."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with source.open("rb") as source_handle:
                destination_handle = destination.open("xb")
                created = True
                with destination_handle:
                    shutil.copyfileobj(
                        source_handle,
                        destination_handle,
                        length=1024 * 1024,
                    )
            shutil.copystat(source, destination)
        except Exception:
            if created:
                try:
                    destination.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def delete_asset(self, asset_id: str) -> Optional[TrashEntry]:
        """Move a project asset file to trash and remove it from the project."""
        if self._current is None:
            return None

        asset = next((a for a in self._current.assets if a.id == asset_id), None)
        if asset is None:
            return None

        try:
            entry = self._trash.trash_path(
                asset.file_path,
                category="project_asset",
                label=asset.name or asset.id,
                metadata={
                    "project_id": self._current.id,
                    "asset": asdict(asset),
                },
            )
        except TrashError as e:
            print(f"[Slunder Studio] Failed to delete project asset: {e}")
            return None

        self._current.remove_asset(asset_id)
        self.save()
        return entry

    def restore_deleted_asset(self, trash_entry_id: str) -> bool:
        try:
            entry = self._trash.restore(trash_entry_id)
        except TrashError as e:
            print(f"[Slunder Studio] Failed to restore project asset: {e}")
            return False

        project_id = entry.metadata.get("project_id")
        asset_data = entry.metadata.get("asset") or {}
        if not project_id or not asset_data:
            return False

        project = self._current if self._current and self._current.id == project_id else self.open(project_id)
        if project is None:
            return False
        asset_data["file_path"] = str(Path(entry.original_path))
        project.assets.append(ProjectAsset(**{
            k: v for k, v in asset_data.items()
            if k in ProjectAsset.__dataclass_fields__
        }))
        self.save(project)
        return True


def get_project_manager() -> ProjectManager:
    return ProjectManager()
