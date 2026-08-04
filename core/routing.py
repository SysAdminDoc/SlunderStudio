"""
Slunder Studio - Cross-module routing contract.
A route moves a real artifact plus its musical context between modules. It
never just switches pages: the destination receives a typed payload, selects
it, and can register it against the active project.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.provenance import find_provenance_sidecar, read_provenance_sidecar

# What kind of thing is being routed.
ARTIFACT_AUDIO = "audio"
ARTIFACT_MIDI = "midi"
ARTIFACT_STEMS = "stems"
ARTIFACT_LYRICS = "lyrics"

ARTIFACT_KINDS = (ARTIFACT_AUDIO, ARTIFACT_MIDI, ARTIFACT_STEMS, ARTIFACT_LYRICS)

# Keep these ordered tuples as the UI's import-format source of truth.  The
# sets below remain available for fast routing checks.
AUDIO_EXTENSIONS = (
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".aiff",
    ".aif",
)
MIDI_EXTENSIONS = (".mid", ".midi")
AUDIO_SUFFIXES = set(AUDIO_EXTENSIONS)
MIDI_SUFFIXES = set(MIDI_EXTENSIONS)


class RouteError(RuntimeError):
    """Raised when a route cannot carry a real artifact."""


@dataclass(frozen=True)
class RoutedArtifact:
    """One artifact plus the context a destination needs to use it."""
    path: str
    kind: str
    source_module: str
    label: str = ""
    tempo: float = 0.0
    musical_key: str = ""
    lyrics: str = ""
    duration_sec: float = 0.0
    provenance_path: str = ""
    provenance: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return bool(self.path) and os.path.isfile(self.path)

    @property
    def name(self) -> str:
        return self.label or (Path(self.path).stem if self.path else "artifact")

    def context_summary(self) -> str:
        """Short human description of the context travelling with the artifact."""
        parts = [self.name]
        if self.tempo:
            parts.append(f"{self.tempo:g} BPM")
        if self.musical_key:
            parts.append(self.musical_key)
        if self.duration_sec:
            parts.append(f"{self.duration_sec:.1f}s")
        return " - ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "source_module": self.source_module,
            "label": self.label,
            "tempo": self.tempo,
            "musical_key": self.musical_key,
            "duration_sec": self.duration_sec,
            "provenance_path": self.provenance_path,
            "has_lyrics": bool(self.lyrics),
        }


def infer_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in MIDI_SUFFIXES:
        return ARTIFACT_MIDI
    if suffix in AUDIO_SUFFIXES:
        return ARTIFACT_AUDIO
    return ARTIFACT_AUDIO


def is_audio_path(path: str) -> bool:
    """Return whether a path uses one of the supported audio suffixes."""
    return Path(path).suffix.lower() in AUDIO_SUFFIXES


def is_midi_path(path: str) -> bool:
    """Return whether a path uses one of the supported MIDI suffixes."""
    return Path(path).suffix.lower() in MIDI_SUFFIXES


def _audio_duration(path: str) -> float:
    try:
        import soundfile as sf

        info = sf.info(path)
        return round(info.frames / info.samplerate, 4) if info.samplerate else 0.0
    except Exception:
        return 0.0


def build_routed_artifact(
    path: str,
    *,
    source_module: str,
    kind: Optional[str] = None,
    label: str = "",
    tempo: float = 0.0,
    musical_key: str = "",
    lyrics: str = "",
    metadata: Optional[dict] = None,
) -> RoutedArtifact:
    """Build a route payload, filling context from the provenance sidecar.

    Raises RouteError when the file does not exist, so a route can never
    advertise a transfer it did not make.
    """
    if not path or not os.path.isfile(path):
        raise RouteError(f"Routed artifact is missing: {path or '(no path)'}")

    kind = kind or infer_kind(path)
    if kind not in ARTIFACT_KINDS:
        raise RouteError(f"Unknown artifact kind: {kind}")

    sidecar_path = str(find_provenance_sidecar(path) or "")
    provenance = {}
    if sidecar_path:
        try:
            provenance = read_provenance_sidecar(sidecar_path) or {}
        except Exception:
            provenance = {}

    parameters = provenance.get("parameters") or {}
    extra = provenance.get("extra") or {}

    def _first(*candidates):
        for value in candidates:
            if value not in (None, "", 0):
                return value
        return None

    tempo = float(tempo or _first(
        parameters.get("bpm"), parameters.get("tempo"),
        extra.get("bpm"), extra.get("tempo"),
    ) or 0.0)
    musical_key = musical_key or str(_first(
        parameters.get("key"), parameters.get("musical_key"), extra.get("key"),
    ) or "")
    lyrics = lyrics or str(_first(
        parameters.get("lyrics"), extra.get("lyrics"),
    ) or "")

    duration = _audio_duration(path) if kind in (ARTIFACT_AUDIO, ARTIFACT_STEMS) else 0.0

    return RoutedArtifact(
        path=os.path.abspath(path),
        kind=kind,
        source_module=source_module,
        label=label or Path(path).stem,
        tempo=tempo,
        musical_key=musical_key,
        lyrics=lyrics,
        duration_sec=duration,
        provenance_path=sidecar_path,
        provenance=provenance,
        metadata=dict(metadata or {}),
    )


def register_with_project(artifact: RoutedArtifact, *, module: str,
                          project_manager=None) -> Optional[str]:
    """Import a routed artifact into the open project. Returns the asset ID."""
    if project_manager is None:
        from core.project import get_project_manager

        project_manager = get_project_manager()
    if project_manager.current is None:
        return None
    asset_type = "midi" if artifact.kind == ARTIFACT_MIDI else "audio"
    return project_manager.import_asset(
        artifact.path,
        asset_type,
        module,
        name=artifact.name,
        provenance_path=artifact.provenance_path or None,
    )
