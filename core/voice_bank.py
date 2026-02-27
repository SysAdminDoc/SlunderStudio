"""
Slunder Studio v0.0.2 — Voice Bank
Voice profile management for RVC and GPT-SoVITS models.
Handles model discovery, metadata, favorites, and preset management.
"""
import os
import json
import time
from typing import Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

from core.settings import get_config_dir


@dataclass
class VoiceProfile:
    """A single voice model profile."""
    id: str = ""
    name: str = "Unnamed Voice"
    engine: str = "rvc"  # "rvc" | "gpt_sovits" | "diffsinger"
    model_path: str = ""
    index_path: str = ""  # RVC feature index
    config_path: str = ""  # GPT-SoVITS config
    ref_audio_path: str = ""  # GPT-SoVITS reference audio
    ref_text: str = ""  # GPT-SoVITS reference transcript
    speaker_id: int = 0  # DiffSinger speaker ID
    pitch_shift: int = 0  # semitones
    tags: list[str] = field(default_factory=list)  # e.g. ["male", "deep", "warm"]
    notes: str = ""
    created_at: float = 0.0
    is_favorite: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = f"voice_{int(time.time() * 1000)}"
        if self.created_at == 0.0:
            self.created_at = time.time()


# ── Voice Bank ─────────────────────────────────────────────────────────────────

class VoiceBank:
    """Manages voice profiles with persistence."""

    _instance: Optional["VoiceBank"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._profiles: dict[str, VoiceProfile] = {}
        self._db_path = os.path.join(get_config_dir(), "voice_bank.json")
        self._voice_dir = os.path.join(get_config_dir(), "voices")
        os.makedirs(self._voice_dir, exist_ok=True)
        self._load()

    def _load(self):
        """Load profiles from disk."""
        if os.path.isfile(self._db_path):
            try:
                with open(self._db_path, "r") as f:
                    data = json.load(f)
                for item in data.get("profiles", []):
                    profile = VoiceProfile(**{
                        k: v for k, v in item.items()
                        if k in VoiceProfile.__dataclass_fields__
                    })
                    self._profiles[profile.id] = profile
            except Exception:
                pass

    def _save(self):
        """Persist profiles to disk."""
        data = {"profiles": [asdict(p) for p in self._profiles.values()]}
        with open(self._db_path, "w") as f:
            json.dump(data, f, indent=2)

    # ── CRUD ───────────────────────────────────────────────────────────────────

    def add(self, profile: VoiceProfile) -> str:
        self._profiles[profile.id] = profile
        self._save()
        return profile.id

    def get(self, profile_id: str) -> Optional[VoiceProfile]:
        return self._profiles.get(profile_id)

    def remove(self, profile_id: str) -> bool:
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            self._save()
            return True
        return False

    def update(self, profile: VoiceProfile):
        self._profiles[profile.id] = profile
        self._save()

    def list_all(self) -> list[VoiceProfile]:
        return sorted(self._profiles.values(), key=lambda p: p.name.lower())

    def list_by_engine(self, engine: str) -> list[VoiceProfile]:
        return [p for p in self.list_all() if p.engine == engine]

    def list_favorites(self) -> list[VoiceProfile]:
        return [p for p in self.list_all() if p.is_favorite]

    def search(self, query: str) -> list[VoiceProfile]:
        q = query.lower()
        return [
            p for p in self.list_all()
            if q in p.name.lower() or any(q in t.lower() for t in p.tags)
        ]

    def toggle_favorite(self, profile_id: str) -> bool:
        profile = self.get(profile_id)
        if profile:
            profile.is_favorite = not profile.is_favorite
            self._save()
            return profile.is_favorite
        return False

    @property
    def count(self) -> int:
        return len(self._profiles)

    @property
    def voice_dir(self) -> str:
        return self._voice_dir

    # ── Discovery ──────────────────────────────────────────────────────────────

    def scan_directory(self, directory: str) -> list[VoiceProfile]:
        """Scan a directory for voice models and auto-create profiles."""
        found = []
        if not os.path.isdir(directory):
            return found

        for root, dirs, files in os.walk(directory):
            for f in files:
                path = os.path.join(root, f)
                ext = f.lower().rsplit(".", 1)[-1] if "." in f else ""

                # RVC models (.pth)
                if ext == "pth" and "rvc" in root.lower():
                    name = f.rsplit(".", 1)[0]
                    # Look for matching index file
                    index_path = ""
                    for idx_f in files:
                        if idx_f.endswith(".index"):
                            index_path = os.path.join(root, idx_f)
                            break

                    profile = VoiceProfile(
                        name=name, engine="rvc",
                        model_path=path, index_path=index_path,
                        tags=["imported"],
                    )
                    if profile.id not in self._profiles:
                        self.add(profile)
                        found.append(profile)

                # GPT-SoVITS models
                elif ext in ("ckpt", "safetensors") and "sovits" in root.lower():
                    name = f.rsplit(".", 1)[0]
                    profile = VoiceProfile(
                        name=name, engine="gpt_sovits",
                        model_path=path, tags=["imported"],
                    )
                    if profile.id not in self._profiles:
                        self.add(profile)
                        found.append(profile)

                # DiffSinger models
                elif ext == "onnx" and "diffsinger" in root.lower():
                    name = f.rsplit(".", 1)[0]
                    profile = VoiceProfile(
                        name=name, engine="diffsinger",
                        model_path=path, tags=["imported"],
                    )
                    if profile.id not in self._profiles:
                        self.add(profile)
                        found.append(profile)

        return found


# ── Convenience ────────────────────────────────────────────────────────────────

def get_voice_bank() -> VoiceBank:
    return VoiceBank()
