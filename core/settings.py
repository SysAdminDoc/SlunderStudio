"""
Slunder Studio — Settings System
JSON config in %APPDATA%/SlunderStudio with presets, reactive updates, and two-tier mode.
"""
import json
import os
import copy
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.version import APP_NAME, APP_VERSION  # noqa: F401 - re-exported

SETTINGS_SCHEMA_VERSION = 3

# Settings keys whose values are secrets. They are never written to config
# JSON; reads and writes go to the OS credential service instead.
SECRET_SETTING_KEYS: dict[str, str] = {
    "model_hub.hf_token": "huggingface-token",
}


@dataclass
class RepairStatus:
    """Persistence repair or migration status for diagnostics."""
    status: str = "ok"  # ok | migrated | repaired | error
    messages: list[str] = field(default_factory=list)
    backup_paths: list[str] = field(default_factory=list)

# ── Default Configuration ──────────────────────────────────────────────────────

DEFAULTS = {
    "schema_version": SETTINGS_SCHEMA_VERSION,
    "version": APP_VERSION,
    "general": {
        "output_dir": "",
        "audio_format": "wav",
        "sample_rate": 48000,
        # Empty means PortAudio's system default. Otherwise this is the
        # host-api/name identity exposed by core.audio_engine.
        "audio_output_device": "",
        "bit_depth": 24,
        "gpu_device": 0,
        "theme_accent": "#a293ff",
        "ui_mode": "simple",
        "ui_locale": "en",
        "experience_level": "beginner",
        "onboarding_complete": False,
        "onboarding_skipped": False,
        "reduced_motion": False,
        "file_dialog_dirs": {},
        "auto_save_interval": 60,
        "auto_save_enabled": True,
        "max_project_versions": 20,
        "max_cache_gb": 20.0,
        "trash_retention_days": 30,
    },
    "retention": {
        # Age/count/size caps for recovery artifacts. 0 means no limit.
        "jobs": {"max_age_days": 30, "max_count": 500, "max_total_mb": 0},
        "job_logs": {"max_age_days": 30, "max_count": 500, "max_total_mb": 200},
        "crash_logs": {"max_age_days": 90, "max_count": 50, "max_total_mb": 50},
        "settings_backups": {"max_age_days": 90, "max_count": 40, "max_total_mb": 20},
        "project_versions": {"max_age_days": 0, "max_count": 20, "max_total_mb": 0},
    },
    "model_hub": {
        "cache_dir": "",
        "offline_mode": False,
        "auto_download_core": True,
        "show_experimental": False,
        "execution_consents": {},
    },
    "lyrics": {
        "model_id": "llama-3.1-8b-q4",
        "temperature": 0.8,
        "top_p": 0.92,
        "top_k": 50,
        "repeat_penalty": 1.1,
        "max_tokens": 2048,
        "default_genre": "pop",
        "default_language": "en",
    },
    "song_forge": {
        "model_id": "ace-step-v1.5",
        "timestep_shift": 3.0,
        "inference_steps": 8,
        "default_duration": 180,
        "batch_count": 4,
        "seed": -1,
        "scheduler": "flow_match_euler",
    },
    "midi_studio": {
        "model_id": "midi-llm-1b",
        "soundfont": "GeneralUser_GS.sf2",
        "default_bpm": 120,
        "quantize_grid": "1/8",
        "batch_count": 4,
    },
    "vocal_suite": {
        "rvc_pitch_shift": 0,
        "rvc_index_ratio": 0.75,
        "rvc_filter_radius": 3,
        "rvc_protect": 0.33,
        "diffsinger_model": "default",
        "sovits_reference_path": "",
        "autotune_strength": 0.75,
    },
    "production": {
        "mastering_target": "streaming",
        "mastering_auto_eq": True,
        "mastering_auto_compress": True,
        "effects_presets_dir": "",
    },
    "ai_producer": {
        "auto_master": True,
        "auto_rank": True,
        "batch_count": 4,
        "surprise_me_genres": ["pop", "rock", "hip-hop", "electronic", "jazz", "lo-fi"],
    },
    "seed_explorer": {
        "grid_size": 4,
        "seed_range": 1000,
        "param_axis": "cfg_scale",
    },
    "mood_curve": {
        "default_preset": "classic_pop_build",
    },
    "reference_analysis": {
        "auto_populate_tags": True,
        "auto_match_duration": True,
    },
}


# ── Config Directory ───────────────────────────────────────────────────────────

def get_config_dir() -> Path:
    """Get or create the app config directory."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path.home() / ".config"
    config_dir = base / APP_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_default_output_dir() -> Path:
    """Get the default output directory (~/Music/SlunderStudio)."""
    music_dir = Path.home() / "Music" / APP_NAME
    music_dir.mkdir(parents=True, exist_ok=True)
    return music_dir


def get_configured_output_dir() -> Path:
    """Return the user's configured render root, creating it when needed."""
    configured = str(Settings().get("general.output_dir", "") or "").strip()
    output_dir = Path(configured) if configured else get_default_output_dir()
    if not output_dir.is_absolute():
        output_dir = get_default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_default_cache_dir() -> Path:
    """Get the default model cache directory."""
    cache_dir = get_config_dir() / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_trash_dir() -> Path:
    """Get the app trash/quarantine directory."""
    trash_dir = get_config_dir() / "trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    return trash_dir


# ── Settings Manager ───────────────────────────────────────────────────────────

class Settings:
    """
    Reactive settings manager with JSON persistence.
    Supports nested key access (e.g., 'lyrics.temperature'),
    and change callbacks.
    """

    _instance: Optional["Settings"] = None
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
        self._data: dict = {}
        self._callbacks: list = []
        self._config_path = get_config_dir() / "config.json"
        self._repair_status = RepairStatus()
        # Secret keys whose plaintext could not be moved into a credential
        # service; their JSON copy is preserved so the user can still act on it.
        self._unmigrated_secrets: set[str] = set()
        self.load()

    def load(self):
        """Load config from disk, merging with defaults for any missing keys."""
        self._data = copy.deepcopy(DEFAULTS)
        self._repair_status = RepairStatus()
        should_save = False
        if self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                saved, migrated, messages = self._migrate(saved)
                if migrated:
                    backup = self._backup_file(self._config_path, "pre-migration")
                    self._repair_status.status = "migrated"
                    self._repair_status.messages.extend(messages)
                    if backup:
                        self._repair_status.backup_paths.append(str(backup))
                    should_save = True
                self._deep_merge(self._data, saved)
            except (json.JSONDecodeError, IOError, OSError) as exc:
                backup = self._backup_file(self._config_path, "corrupt")
                self._repair_status.status = "repaired"
                self._repair_status.messages.append(
                    f"Config JSON was unreadable and defaults were restored: {exc}"
                )
                if backup:
                    self._repair_status.backup_paths.append(str(backup))
                should_save = True

        # Fill empty paths with platform defaults
        if not self._data["general"]["output_dir"]:
            self._data["general"]["output_dir"] = str(get_default_output_dir())
            should_save = True
        if not self._data["model_hub"]["cache_dir"]:
            self._data["model_hub"]["cache_dir"] = str(get_default_cache_dir())
            should_save = True

        if self._migrate_secrets():
            should_save = True

        self._data["schema_version"] = SETTINGS_SCHEMA_VERSION
        self._data["version"] = APP_VERSION
        if should_save:
            self.save(create_backup=False)

    # ── Secret migration ───────────────────────────────────────────────────────

    def _migrate_secrets(self) -> bool:
        """Move plaintext secrets out of config JSON into the OS credential store.

        The plaintext copy — and the copies inside timestamped backups — are only
        removed once the credential service confirms the value can be read back.
        """
        changed = False
        store = self.credential_store
        self._unmigrated_secrets = set()
        for key, account in SECRET_SETTING_KEYS.items():
            keys = key.split(".")
            target = self._data
            for part in keys[:-1]:
                if not isinstance(target, dict) or part not in target:
                    target = None
                    break
                target = target[part]
            if not isinstance(target, dict):
                continue
            plaintext = target.get(keys[-1])
            if not isinstance(plaintext, str) or not plaintext.strip():
                # An empty placeholder is still a key we do not want in JSON.
                if keys[-1] in target:
                    target.pop(keys[-1], None)
                    changed = True
                continue

            plaintext = plaintext.strip()
            try:
                store.set_secret(account, plaintext)
                confirmed = store.get_secret(account) == plaintext
            except Exception as exc:
                confirmed = False
                self._repair_status.messages.append(
                    f"Could not move {key} into the OS credential service: {exc}"
                )

            if not confirmed:
                self._unmigrated_secrets.add(key)
                self._repair_status.status = "error"
                self._repair_status.messages.append(
                    f"{key} is still stored in plaintext because no OS credential "
                    f"service is available ({store.status().detail}). Clear it in "
                    "Settings > GPU and Models, or install a credential service."
                )
                continue

            target.pop(keys[-1], None)
            changed = True
            self._repair_status.status = (
                "migrated" if self._repair_status.status == "ok"
                else self._repair_status.status
            )
            self._repair_status.messages.append(
                f"Moved {key} into {store.backend_name} and removed the plaintext copy."
            )
            removed = self._purge_secret_from_backups(key)
            if removed:
                self._repair_status.messages.append(
                    f"Removed {key} from {removed} settings backup(s)."
                )
        return changed

    def _purge_secret_from_backups(self, key: str) -> int:
        """Strip a secret key from existing timestamped config backups."""
        backup_dir = self._config_path.parent / "backups"
        if not backup_dir.is_dir():
            return 0
        keys = key.split(".")
        cleaned = 0
        for path in sorted(backup_dir.glob(f"{self._config_path.name}.*")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError, OSError, UnicodeDecodeError):
                self._repair_status.messages.append(
                    f"Backup {path.name} could not be read; delete it manually if it "
                    f"may contain {key}."
                )
                continue
            target = data
            for part in keys[:-1]:
                if not isinstance(target, dict) or part not in target:
                    target = None
                    break
                target = target[part]
            if not isinstance(target, dict) or keys[-1] not in target:
                continue
            target.pop(keys[-1], None)
            try:
                tmp = path.with_name(path.name + ".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(tmp, path)
                cleaned += 1
            except (IOError, OSError) as exc:
                self._repair_status.messages.append(
                    f"Could not rewrite backup {path.name}: {exc}"
                )
        return cleaned

    @property
    def repair_status(self) -> dict:
        """Return the last load/save migration or repair status."""
        return {
            "status": self._repair_status.status,
            "messages": list(self._repair_status.messages),
            "backup_paths": list(self._repair_status.backup_paths),
        }

    def save(self, create_backup: bool = True):
        """Persist current settings to disk (atomic write)."""
        try:
            self._data["schema_version"] = SETTINGS_SCHEMA_VERSION
            self._data["version"] = APP_VERSION
            # Belt and braces: no secret reaches the JSON file or a backup unless
            # it is a legacy value we could not move into a credential service.
            for secret_key in SECRET_SETTING_KEYS:
                if secret_key not in self._unmigrated_secrets:
                    self._strip_secret_key(secret_key)
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            if create_backup:
                backup = self._backup_file(self._config_path, "pre-save")
                if backup:
                    self._repair_status.backup_paths.append(str(backup))
            tmp = self._config_path.with_name(self._config_path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._config_path)
        except (IOError, OSError) as exc:
            self._repair_status.status = "error"
            self._repair_status.messages.append(f"Config save failed: {exc}")

    # ── Secrets ────────────────────────────────────────────────────────────────

    @property
    def credential_store(self):
        from core.credentials import get_credential_store
        return get_credential_store()

    def credential_backend_status(self) -> dict:
        """Report which OS credential service is in use, or why none is."""
        return self.credential_store.status().as_dict()

    def get_secret(self, key: str, default: str = "") -> str:
        account = SECRET_SETTING_KEYS.get(key)
        if account is None:
            raise KeyError(f"{key} is not a registered secret setting")
        stored = self.credential_store.get_secret(account)
        if stored:
            return stored
        if key in self._unmigrated_secrets:
            # Migration failed for lack of a credential service. The plaintext
            # copy stays readable so the user can use or clear it.
            return self._read_plain_key(key) or default
        return default

    def _read_plain_key(self, key: str) -> str:
        value = self._data
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return ""
            value = value[part]
        return value if isinstance(value, str) else ""

    def set_secret(self, key: str, value: str):
        """Store a secret in the OS credential service. Never touches JSON."""
        account = SECRET_SETTING_KEYS.get(key)
        if account is None:
            raise KeyError(f"{key} is not a registered secret setting")
        store = self.credential_store
        old = self.get_secret(key, "")
        value = (value or "").strip()
        if value:
            store.set_secret(account, value)
        else:
            store.delete_secret(account)
        # Any stale plaintext copy is authoritative no longer.
        self._unmigrated_secrets.discard(key)
        self._strip_secret_key(key)
        if old != value:
            self._notify(key, value, old)

    def _strip_secret_key(self, key: str) -> bool:
        """Remove a secret key from in-memory config. Returns True if present."""
        keys = key.split(".")
        target = self._data
        for k in keys[:-1]:
            if not isinstance(target, dict) or k not in target:
                return False
            target = target[k]
        if isinstance(target, dict) and keys[-1] in target:
            target.pop(keys[-1], None)
            return True
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting by dotted key path.
        Example: settings.get('lyrics.temperature') -> 0.8
        """
        if key in SECRET_SETTING_KEYS:
            return self.get_secret(key, default if isinstance(default, str) else "")
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any, save: bool = True):
        """
        Set a setting by dotted key path. Auto-saves and fires callbacks.
        Example: settings.set('lyrics.temperature', 0.9)
        """
        if key in SECRET_SETTING_KEYS:
            self.set_secret(key, value)
            if save:
                self.save()
            return
        keys = key.split(".")
        target = self._data
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        old_value = target.get(keys[-1])
        target[keys[-1]] = value
        if save:
            self.save()
        if old_value != value:
            self._notify(key, value, old_value)

    def get_section(self, section: str) -> dict:
        """Get an entire settings section as a dict."""
        return copy.deepcopy(self._data.get(section, {}))

    def set_section(self, section: str, data: dict, save: bool = True):
        """Replace an entire settings section."""
        self._data[section] = data
        if save:
            self.save()
        self._notify(section, data, None)

    def reset_section(self, section: str):
        """Reset a section to defaults."""
        if section in DEFAULTS:
            self._data[section] = copy.deepcopy(DEFAULTS[section])
            self.save()
            self._notify(section, self._data[section], None)

    def reset_all(self):
        """Reset all settings to defaults, including stored secrets."""
        self._data = copy.deepcopy(DEFAULTS)
        self._data["general"]["output_dir"] = str(get_default_output_dir())
        self._data["model_hub"]["cache_dir"] = str(get_default_cache_dir())
        store = self.credential_store
        for account in SECRET_SETTING_KEYS.values():
            store.delete_secret(account)
        self.save()
        self._notify("*", self._data, None)

    def snapshot(self) -> dict[str, Any]:
        """Capture settings for an in-memory, user-requested undo.

        Secret values are read from the credential service and kept only in
        the returned Python object.  They are never serialized with the
        settings data, written to a toast, or placed in the trash manifest.
        """
        return {
            "data": copy.deepcopy(self._data),
            "secrets": {
                key: self.get_secret(key, "")
                for key in SECRET_SETTING_KEYS
            },
        }

    def restore_snapshot(self, snapshot: dict[str, Any]):
        """Restore a snapshot produced by :meth:`snapshot`."""
        data = snapshot.get("data")
        secrets = snapshot.get("secrets", {})
        if not isinstance(data, dict) or not isinstance(secrets, dict):
            raise ValueError("Invalid settings snapshot")

        restored_data = copy.deepcopy(data)
        for key in SECRET_SETTING_KEYS:
            target = restored_data
            parts = key.split(".")
            for part in parts[:-1]:
                target = target.get(part, {}) if isinstance(target, dict) else {}
            if isinstance(target, dict):
                target.pop(parts[-1], None)

        store = self.credential_store
        for key, account in SECRET_SETTING_KEYS.items():
            value = secrets.get(key, "")
            if value:
                store.set_secret(account, str(value))
            else:
                store.delete_secret(account)

        self._data = restored_data
        self._unmigrated_secrets.clear()
        self.save()
        self._notify("*", self._data, None)

    def on_change(self, callback):
        """Register a callback: callback(key, new_value, old_value)."""
        self._callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove a registered callback."""
        self._callbacks = [cb for cb in self._callbacks if cb is not callback]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _deep_merge(self, base: dict, override: dict):
        """Recursively merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _migrate(self, saved: dict) -> tuple[dict, bool, list[str]]:
        """Migrate settings data to the current schema."""
        if not isinstance(saved, dict):
            raise json.JSONDecodeError("Settings root is not an object", "", 0)

        migrated = False
        messages: list[str] = []
        data = copy.deepcopy(saved)

        try:
            schema_version = int(data.get("schema_version", 1) or 1)
        except (TypeError, ValueError):
            schema_version = 1

        if schema_version < 2:
            data.setdefault("general", {})
            data["general"].setdefault(
                "trash_retention_days",
                DEFAULTS["general"]["trash_retention_days"],
            )
            messages.append("Migrated settings schema from v1 to v2.")
            migrated = True
        if schema_version < 3:
            song_forge = data.setdefault("song_forge", {})
            song_forge.pop("cfg_scale", None)
            song_forge["timestep_shift"] = DEFAULTS["song_forge"]["timestep_shift"]
            song_forge["inference_steps"] = DEFAULTS["song_forge"]["inference_steps"]
            song_forge["scheduler"] = DEFAULTS["song_forge"]["scheduler"]
            messages.append(
                "Migrated Song Forge controls to the ACE-Step 1.5 XL Turbo schedule."
            )
            migrated = True
        elif schema_version > SETTINGS_SCHEMA_VERSION:
            messages.append(
                f"Settings schema v{schema_version} is newer than supported v{SETTINGS_SCHEMA_VERSION}; preserved compatible keys."
            )
        production = data.get("production")
        if isinstance(production, dict) and production.get("mastering_target") == "spotify":
            production["mastering_target"] = "streaming"
            messages.append("Migrated the legacy Spotify mastering target key to streaming.")
            migrated = True

        if data.get("schema_version") != SETTINGS_SCHEMA_VERSION:
            data["schema_version"] = SETTINGS_SCHEMA_VERSION
            migrated = True
        if data.get("version") != APP_VERSION:
            data["version"] = APP_VERSION
            migrated = True
            messages.append(f"Updated settings app version to {APP_VERSION}.")

        return data, migrated, messages

    def _backup_file(self, path: Path, reason: str) -> Optional[Path]:
        """Create a timestamped backup beside the config file."""
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
            self._repair_status.messages.append(f"Backup failed for {path}: {exc}")
            return None

    def _notify(self, key: str, new_value: Any, old_value: Any):
        """Fire all registered change callbacks."""
        for cb in self._callbacks:
            try:
                cb(key, new_value, old_value)
            except Exception:
                pass  # don't let a bad callback break settings
