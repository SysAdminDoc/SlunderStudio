"""
Slunder Studio v0.0.2 — Project Management
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

from core.settings import get_config_dir


@dataclass
class ProjectAsset:
    """A single asset (audio, MIDI, lyrics, etc.) within a project."""
    id: str = ""
    name: str = ""
    asset_type: str = ""  # "audio" | "midi" | "lyrics" | "stems" | "sfx" | "export"
    file_path: str = ""
    module: str = ""  # which module created it
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
        os.makedirs(self._projects_dir, exist_ok=True)
        self._load_index()

    def _load_index(self):
        if os.path.isfile(self._index_path):
            try:
                with open(self._index_path) as f:
                    self._index = json.load(f)
            except Exception:
                self._index = {}

    def _save_index(self):
        tmp = self._index_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._index, f, indent=2)
        os.replace(tmp, self._index_path)

    # ── CRUD ───────────────────────────────────────────────────────────────────

    def create(self, name: str = "Untitled Project", **kwargs) -> Project:
        project = Project(name=name, **kwargs)
        project_dir = os.path.join(self._projects_dir, project.id)
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, "assets"), exist_ok=True)
        os.makedirs(os.path.join(project_dir, "versions"), exist_ok=True)

        self._save_project(project)
        self._index[project.id] = {
            "name": project.name,
            "path": project_dir,
            "updated_at": project.updated_at,
        }
        self._save_index()
        self._current = project
        return project

    def open(self, project_id: str) -> Optional[Project]:
        if project_id not in self._index:
            return None

        project_dir = self._index[project_id]["path"]
        meta_path = os.path.join(project_dir, "project.json")

        if not os.path.isfile(meta_path):
            return None

        try:
            with open(meta_path) as f:
                data = json.load(f)

            project = Project(
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
            return project

        except Exception:
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

    def delete(self, project_id: str) -> bool:
        if project_id not in self._index:
            return False

        project_dir = self._index[project_id]["path"]
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)

        del self._index[project_id]
        self._save_index()

        if self._current and self._current.id == project_id:
            self._current = None
        return True

    def _save_project(self, project: Project):
        project_dir = os.path.join(self._projects_dir, project.id)
        os.makedirs(project_dir, exist_ok=True)

        data = {
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
        # Write to temp file first, then rename for atomicity
        tmp_path = meta_path + ".tmp"
        with open(tmp_path, "w") as f:
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
                     module: str, name: Optional[str] = None) -> Optional[str]:
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

        asset = ProjectAsset(
            name=name, asset_type=asset_type,
            file_path=dest, module=module,
        )
        self._current.add_asset(asset)
        self.save()
        return asset.id


def get_project_manager() -> ProjectManager:
    return ProjectManager()
