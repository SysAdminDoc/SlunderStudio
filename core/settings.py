"""
Slunder Studio v0.0.2 — Settings System
JSON config in %APPDATA%/SlunderStudio with presets, reactive updates, and two-tier mode.
"""
import json
import os
import copy
from pathlib import Path
from typing import Any, Optional

APP_NAME = "SlunderStudio"
APP_VERSION = "0.0.2"

# ── Default Configuration ──────────────────────────────────────────────────────

DEFAULTS = {
    "version": APP_VERSION,
    "general": {
        "output_dir": "",
        "audio_format": "wav",
        "sample_rate": 48000,
        "bit_depth": 24,
        "gpu_device": 0,
        "theme_accent": "#89b4fa",
        "ui_mode": "simple",
        "experience_level": "beginner",
        "onboarding_complete": False,
        "auto_save_interval": 60,
        "max_cache_gb": 20.0,
    },
    "model_hub": {
        "cache_dir": "",
        "offline_mode": False,
        "auto_download_core": True,
        "show_experimental": False,
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
        "cfg_scale": 7.0,
        "inference_steps": 50,
        "default_duration": 180,
        "batch_count": 4,
        "seed": -1,
        "scheduler": "default",
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
    },
    "production": {
        "mastering_target": "spotify",
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


def get_default_cache_dir() -> Path:
    """Get the default model cache directory."""
    cache_dir = get_config_dir() / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_presets_dir() -> Path:
    """Get or create the presets directory."""
    presets_dir = get_config_dir() / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir


# ── Settings Manager ───────────────────────────────────────────────────────────

class Settings:
    """
    Reactive settings manager with JSON persistence.
    Supports nested key access (e.g., 'lyrics.temperature'),
    presets, and change callbacks.
    """

    _instance: Optional["Settings"] = None

    def __new__(cls):
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
        self.load()

    def load(self):
        """Load config from disk, merging with defaults for any missing keys."""
        self._data = copy.deepcopy(DEFAULTS)
        if self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._deep_merge(self._data, saved)
            except (json.JSONDecodeError, IOError):
                pass  # corrupt file — use defaults

        # Fill empty paths with platform defaults
        if not self._data["general"]["output_dir"]:
            self._data["general"]["output_dir"] = str(get_default_output_dir())
        if not self._data["model_hub"]["cache_dir"]:
            self._data["model_hub"]["cache_dir"] = str(get_default_cache_dir())

    def save(self):
        """Persist current settings to disk (atomic write)."""
        try:
            tmp = self._config_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, self._config_path)
        except (IOError, OSError):
            pass  # silently fail — toast will inform user

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting by dotted key path.
        Example: settings.get('lyrics.temperature') -> 0.8
        """
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
        """Reset all settings to defaults."""
        self._data = copy.deepcopy(DEFAULTS)
        self._data["general"]["output_dir"] = str(get_default_output_dir())
        self._data["model_hub"]["cache_dir"] = str(get_default_cache_dir())
        self.save()
        self._notify("*", self._data, None)

    def on_change(self, callback):
        """Register a callback: callback(key, new_value, old_value)."""
        self._callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove a registered callback."""
        self._callbacks = [cb for cb in self._callbacks if cb is not callback]

    # ── Presets ────────────────────────────────────────────────────────────────

    def save_preset(self, name: str, section: str):
        """Save current section settings as a named preset."""
        presets_dir = get_presets_dir() / section
        presets_dir.mkdir(parents=True, exist_ok=True)
        preset_path = presets_dir / f"{name}.json"
        tmp_path = preset_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.get_section(section), f, indent=2)
        tmp_path.replace(preset_path)

    def load_preset(self, name: str, section: str) -> bool:
        """Load a named preset into a section. Returns True on success."""
        preset_path = get_presets_dir() / section / f"{name}.json"
        if not preset_path.exists():
            return False
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.set_section(section, data)
            return True
        except (json.JSONDecodeError, IOError):
            return False

    def list_presets(self, section: str) -> list[str]:
        """List available preset names for a section."""
        presets_dir = get_presets_dir() / section
        if not presets_dir.exists():
            return []
        return [p.stem for p in presets_dir.glob("*.json")]

    def delete_preset(self, name: str, section: str) -> bool:
        """Delete a named preset. Returns True on success."""
        preset_path = get_presets_dir() / section / f"{name}.json"
        if preset_path.exists():
            preset_path.unlink()
            return True
        return False

    # ── Internal ───────────────────────────────────────────────────────────────

    def _deep_merge(self, base: dict, override: dict):
        """Recursively merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _notify(self, key: str, new_value: Any, old_value: Any):
        """Fire all registered change callbacks."""
        for cb in self._callbacks:
            try:
                cb(key, new_value, old_value)
            except Exception:
                pass  # don't let a bad callback break settings
