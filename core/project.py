"""
Slunder Studio v0.1.22 — Project Management
Save, load, and manage music projects with auto-save, version history,
and asset tracking across all modules.
"""
import os
import json
import time
import shutil
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
            self.id = f"asset_{int(time.time() * 1000)}"
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
            self.id = f"proj_{int(time.time() * 1000)}"
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

    def __new__(cls):
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
        if os.path.isfile(self._index_path):
            try:
                with open(self._index_path, encoding="utf-8") as f:
                    self._index = json.load(f)
                if not isinstance(self._index, dict):
                    raise json.JSONDecodeError("Project index root is not an object", "", 0)
            except (json.JSONDecodeError, OSError) as exc:
                backup = self._backup_file(Path(self._index_path), "corrupt")
                self._last_repair_status = ProjectRepairStatus(
                    status="repaired",
                    messages=[f"Project index was unreadable and reset: {exc}"],
                    backup_paths=[str(backup)] if backup else [],
                )
                self._index = {}

    def _save_index(self):
        self._backup_file(Path(self._index_path), "pre-save")
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)
        os.replace(tmp, self._index_path)

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
        if project_id not in self._index:
            return None

        project_dir = self._index[project_id]["path"]
        meta_path = os.path.join(project_dir, "project.json")

        if not os.path.isfile(meta_path):
            return None

        try:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            data, migrated, messages = self._migrate_project_data(data, project_id)
            if migrated:
                backup = self._backup_file(Path(meta_path), "pre-migration")
                self._last_repair_status = ProjectRepairStatus(
                    status="migrated",
                    messages=messages,
                    backup_paths=[str(backup)] if backup else [],
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
            if migrated:
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

        if name is None:
            name = os.path.basename(file_path)

        project_dir = os.path.join(self._projects_dir, self._current.id, "assets")
        os.makedirs(project_dir, exist_ok=True)

        dest = os.path.join(project_dir, name)
        if os.path.abspath(file_path) != os.path.abspath(dest):
            shutil.copy2(file_path, dest)

        sidecar = Path(provenance_path) if provenance_path else find_provenance_sidecar(file_path)
        dest_sidecar = ""
        provenance_metadata = {}
        if sidecar and sidecar.is_file():
            sidecar_dest = sidecar_path_for(dest)
            if os.path.abspath(sidecar) != os.path.abspath(sidecar_dest):
                shutil.copy2(sidecar, sidecar_dest)
            dest_sidecar = str(sidecar_dest)
            provenance = read_provenance_sidecar(dest_sidecar)
            provenance_metadata = project_metadata_from_provenance(provenance, dest_sidecar)

        asset = ProjectAsset(
            name=name, asset_type=asset_type,
            file_path=dest, module=module,
            provenance_path=dest_sidecar,
            metadata=provenance_metadata,
        )
        self._current.add_asset(asset)
        self.save()
        return asset.id

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
