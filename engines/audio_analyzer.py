"""
Slunder Studio — Audio Analyzer
Reference track analysis: BPM, key, energy envelope, spectral features,
genre estimation, and song structure detection via librosa.
"""
import hashlib
import copy
import math
import threading
from collections import OrderedDict

import numpy as np
from typing import Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field

from core.model_security import assert_safe_transformers_snapshot


ANALYSIS_CONSTRAINTS_VERSION = 1


@dataclass
class AudioAnalysis:
    """Complete analysis of a reference track."""
    file_path: str = ""
    duration: float = 0.0
    sample_rate: int = 0
    # Rhythm
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    beat_times: list = field(default_factory=list)
    bpm_alternatives: list = field(default_factory=list)
    # Key
    key: str = ""  # e.g., "C major", "A minor"
    key_confidence: float = 0.0
    key_alternatives: list = field(default_factory=list)
    # Energy
    energy_mean: float = 0.0
    energy_std: float = 0.0
    energy_curve: list = field(default_factory=list)  # normalized 0-1 over time
    energy_times: list = field(default_factory=list)  # timestamps for curve
    # Spectral
    brightness_mean: float = 0.0  # spectral centroid
    brightness_std: float = 0.0
    onset_density: float = 0.0  # onsets per second
    # Structure
    sections: list = field(default_factory=list)  # [{"start": s, "end": s, "label": "verse"}]
    # Suggested tags
    suggested_tags: list = field(default_factory=list)
    suggested_tempo_tag: str = ""
    clap_backend: str = ""
    clap_embedding: list = field(default_factory=list)
    clap_style_tags: list = field(default_factory=list)
    clap_similarity: dict = field(default_factory=dict)
    # Explicit user corrections.  The fields above remain raw analyzer output
    # so cached measurements and generated-artifact provenance are never
    # silently overwritten by an editor action.
    corrected_bpm: Optional[float] = None
    corrected_key: Optional[str] = None
    corrected_sections: Optional[list] = None

    @property
    def effective_bpm(self) -> float:
        """Return the BPM that downstream generation should trust."""
        return float(self.corrected_bpm if self.corrected_bpm is not None else self.bpm)

    @property
    def effective_key(self) -> str:
        """Return the key that downstream generation should trust."""
        return str(self.corrected_key if self.corrected_key is not None else self.key)

    @property
    def effective_sections(self) -> list:
        """Return corrected section boundaries when present, otherwise raw."""
        sections = self.corrected_sections if self.corrected_sections is not None else self.sections
        return _copy_sections(sections)

    @property
    def has_corrections(self) -> bool:
        return any(
            value is not None
            for value in (self.corrected_bpm, self.corrected_key, self.corrected_sections)
        )

    def apply_corrections(
        self,
        *,
        bpm: Optional[float] = None,
        key: Optional[str] = None,
        sections: Optional[list] = None,
    ) -> None:
        """Apply validated user constraints without changing raw measurements.

        Omitted values leave an existing correction untouched.  Call
        ``clear_corrections`` first when a caller wants to replace the complete
        correction set, as the reference editor does.
        """
        if bpm is not None:
            try:
                parsed_bpm = float(bpm)
            except (TypeError, ValueError) as exc:
                raise ValueError("BPM must be a number") from exc
            if not math.isfinite(parsed_bpm) or not 20.0 <= parsed_bpm <= 300.0:
                raise ValueError("BPM must be between 20 and 300")
            self.corrected_bpm = parsed_bpm

        if key is not None:
            parsed_key = str(key).strip()
            if parsed_key not in _valid_key_values():
                raise ValueError("Key must use a supported major or minor value")
            self.corrected_key = parsed_key

        if sections is not None:
            self.corrected_sections = _normalize_sections(sections, self.duration)

    def clear_corrections(self) -> None:
        """Remove all user constraints while retaining raw analysis output."""
        self.corrected_bpm = None
        self.corrected_key = None
        self.corrected_sections = None

    def to_generation_constraints(self) -> dict:
        """Serialize the effective constraints with raw and correction lineage."""
        return {
            "schema_version": ANALYSIS_CONSTRAINTS_VERSION,
            "bpm": self.effective_bpm,
            "key": self.effective_key,
            "sections": self.effective_sections,
            "effective": {
                "bpm": self.effective_bpm,
                "key": self.effective_key,
                "sections": self.effective_sections,
            },
            "raw": {
                "bpm": float(self.bpm),
                "bpm_confidence": float(self.bpm_confidence),
                "bpm_alternatives": copy.deepcopy(self.bpm_alternatives),
                "key": self.key,
                "key_confidence": float(self.key_confidence),
                "key_alternatives": copy.deepcopy(self.key_alternatives),
                "sections": _copy_sections(self.sections),
            },
            "corrections": {
                "bpm": self.corrected_bpm,
                "key": self.corrected_key,
                "sections": (
                    None
                    if self.corrected_sections is None
                    else _copy_sections(self.corrected_sections)
                ),
            },
            "confidence": {
                "bpm": float(self.bpm_confidence),
                "key": float(self.key_confidence),
            },
            "alternatives": {
                "bpm": copy.deepcopy(self.bpm_alternatives),
                "key": copy.deepcopy(self.key_alternatives),
            },
            "provenance": {
                "source": "reference_analysis",
                "analyzer_version": int(globals().get("ANALYZER_VERSION", 0)),
                "heuristic_tags": True,
            },
        }

    def to_dict(self) -> dict:
        constraints = self.to_generation_constraints()
        return {
            "file_path": self.file_path,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "bpm": self.bpm,
            "bpm_confidence": self.bpm_confidence,
            "beat_times": self.beat_times,
            "bpm_alternatives": self.bpm_alternatives,
            "key": self.key,
            "key_confidence": self.key_confidence,
            "key_alternatives": self.key_alternatives,
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "energy_curve": self.energy_curve,
            "energy_times": self.energy_times,
            "brightness_mean": self.brightness_mean,
            "brightness_std": self.brightness_std,
            "onset_density": self.onset_density,
            "suggested_tags": self.suggested_tags,
            "suggested_tempo_tag": self.suggested_tempo_tag,
            "clap_backend": self.clap_backend,
            "clap_style_tags": self.clap_style_tags,
            "clap_similarity": self.clap_similarity,
            "sections": self.sections,
            "raw": constraints["raw"],
            "corrections": constraints["corrections"],
            "effective": {
                "bpm": constraints["bpm"],
                "key": constraints["key"],
                "sections": constraints["sections"],
            },
            "generation_constraints": constraints,
        }

    def to_ace_step_tags(self) -> str:
        """Convert analysis to ACE-Step compatible tag string."""
        tags = _dedupe_tags([*self.suggested_tags, *self.clap_style_tags])
        if self.corrected_bpm is not None and self.effective_bpm > 0:
            tags.append(_bpm_to_tag(self.effective_bpm))
        elif self.suggested_tempo_tag:
            tags.append(self.suggested_tempo_tag)
        if self.corrected_key:
            tags.append(self.effective_key)
        return ", ".join(_dedupe_tags(tags))

    def clone(self) -> "AudioAnalysis":
        """Return an independent analysis for an editable UI session."""
        return copy.deepcopy(self)


def _valid_key_values() -> set[str]:
    return {f"{name} {mode}" for name in KEY_NAMES for mode in ("major", "minor")}


def _copy_sections(sections) -> list:
    if not sections:
        return []
    return [
        {
            "start": float(section.get("start", 0.0)),
            "end": float(section.get("end", 0.0)),
            "label": str(section.get("label", "Section")),
        }
        for section in sections
        if isinstance(section, dict)
    ]


def _normalize_sections(sections, duration: float) -> list:
    """Validate and normalize editable section boundaries."""
    if not isinstance(sections, list) or not sections:
        raise ValueError("At least one section is required")

    normalized = []
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            raise ValueError("Each section must be an object")
        try:
            start = float(section.get("start"))
            end = float(section.get("end"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Section boundaries must be numbers") from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Section boundaries must be finite")
        if start < 0 or end <= start:
            raise ValueError("Section end must be after its start")
        if duration > 0 and end > float(duration) + 1e-3:
            raise ValueError("Section end cannot exceed the track duration")
        label = str(section.get("label", "")).strip() or f"Section {index + 1}"
        normalized.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "label": label,
        })

    normalized.sort(key=lambda section: section["start"])
    for previous, current in zip(normalized, normalized[1:]):
        if current["start"] < previous["end"] - 1e-3:
            raise ValueError("Sections cannot overlap")
    return normalized


# ── Key Detection ──────────────────────────────────────────────────────────────

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _detect_key(y, sr, *, include_alternatives: bool = False):
    """Detect musical key using chroma features."""
    from core.deps import ensure
    ensure("librosa")
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Major and minor profiles (Krumhansl-Kessler)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    candidates = []

    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        maj_corr = np.corrcoef(rolled, major_profile)[0, 1]
        min_corr = np.corrcoef(rolled, minor_profile)[0, 1]
        for mode, correlation in (("major", maj_corr), ("minor", min_corr)):
            if np.isfinite(correlation):
                candidates.append((float(correlation), f"{KEY_NAMES[i]} {mode}"))

    ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
    if ranked:
        best_corr, best_key = ranked[0]
    else:
        best_corr, best_key = -1.0, "C major"
    alternatives = [
        {
            "value": key,
            "confidence": round(max(0.0, correlation), 4),
            "rank": rank,
        }
        for rank, (correlation, key) in enumerate(ranked[1:5], start=2)
    ]

    result = (best_key, max(0.0, best_corr))
    if include_alternatives:
        return (*result, alternatives)
    return result


def _bpm_alternatives(bpm: float, confidence: float) -> list[dict]:
    """Expose common half-time/double-time interpretations of beat tracking."""
    alternatives = []
    for value, reason in ((bpm / 2.0, "half-time"), (bpm * 2.0, "double-time")):
        if 20.0 <= value <= 300.0 and abs(value - bpm) >= 1.0:
            alternatives.append({
                "value": round(float(value), 2),
                "confidence": round(max(0.0, min(1.0, confidence * 0.65)), 4),
                "reason": reason,
            })
    return alternatives


# ── Tempo Tag Mapping ──────────────────────────────────────────────────────────

def _bpm_to_tag(bpm: float) -> str:
    """Map BPM to a descriptive tempo tag."""
    if bpm < 70:
        return "very slow"
    elif bpm < 90:
        return "slow"
    elif bpm < 110:
        return "mid-tempo"
    elif bpm < 130:
        return "moderate"
    elif bpm < 150:
        return "fast"
    elif bpm < 170:
        return "very fast"
    else:
        return "extremely fast"


# ── Genre Estimation ───────────────────────────────────────────────────────────

def _estimate_genre_tags(bpm, brightness, onset_density, energy_mean, key) -> list[str]:
    """Heuristic genre estimation from audio features."""
    tags = []

    # BPM-based suggestions
    if 120 <= bpm <= 135:
        tags.append("house")
    elif 140 <= bpm <= 160:
        tags.append("drum and bass")
    elif 60 <= bpm <= 85:
        tags.append("hip hop")
    elif 85 <= bpm <= 105:
        tags.append("r&b")

    # Brightness/energy-based
    if brightness > 3500 and energy_mean > 0.3:
        tags.append("rock")
    elif brightness < 2000 and energy_mean < 0.15:
        tags.append("ambient")
    elif energy_mean > 0.25 and onset_density > 4:
        tags.append("energetic")
    elif energy_mean < 0.1:
        tags.append("calm")

    # Key-based mood hint
    if "minor" in key:
        tags.append("melancholic")
    else:
        tags.append("bright")

    return tags[:5]  # Limit suggestions


def _dedupe_tags(tags: list[str]) -> list[str]:
    """Keep tag order while dropping case-insensitive duplicates."""
    seen = set()
    result = []
    for tag in tags:
        clean = str(tag).strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


CLAP_STYLE_PROTOTYPES = {
    "ambient": [0.25, 0.12, 0.30, 0.12, 0.25, 0.20, 0.80, 0.90],
    "cinematic": [0.45, 0.45, 0.55, 0.28, 0.55, 0.75, 0.55, 0.65],
    "lo-fi": [0.42, 0.22, 0.25, 0.35, 0.45, 0.35, 0.85, 0.55],
    "hip hop": [0.42, 0.45, 0.48, 0.62, 0.65, 0.55, 0.55, 0.35],
    "trap": [0.72, 0.52, 0.60, 0.75, 0.75, 0.55, 0.42, 0.25],
    "r&b": [0.48, 0.32, 0.42, 0.42, 0.35, 0.45, 0.60, 0.50],
    "rock": [0.62, 0.72, 0.78, 0.72, 0.55, 0.65, 0.25, 0.20],
    "metal": [0.78, 0.85, 0.86, 0.82, 0.85, 0.70, 0.15, 0.15],
    "dance": [0.64, 0.62, 0.62, 0.78, 0.25, 0.35, 0.35, 0.18],
    "acoustic": [0.45, 0.24, 0.38, 0.22, 0.20, 0.42, 0.70, 0.70],
    "dark": [0.45, 0.45, 0.38, 0.45, 0.95, 0.60, 0.70, 0.35],
    "bright": [0.56, 0.42, 0.82, 0.45, 0.05, 0.45, 0.20, 0.38],
    "bass-heavy": [0.50, 0.62, 0.35, 0.60, 0.65, 0.50, 0.82, 0.35],
    "energetic": [0.70, 0.78, 0.70, 0.82, 0.45, 0.55, 0.25, 0.15],
}


def _audio_clap_lite_embedding(analysis: AudioAnalysis) -> list[float]:
    """
    Build a compact local audio embedding for CLAP-style tag matching.
    The vector tracks tempo, energy, brightness, density, tonality, dynamics,
    warmth, and sparseness. A real CLAP backend can replace this vector without
    changing downstream matching.
    """
    tempo = min(max(analysis.bpm, 0.0), 200.0) / 200.0
    energy = min(max(analysis.energy_mean * 8.0, 0.0), 1.0)
    brightness = min(max(analysis.brightness_mean / 6000.0, 0.0), 1.0)
    onset = min(max(analysis.onset_density / 8.0, 0.0), 1.0)
    minor = 1.0 if "minor" in analysis.key.lower() else 0.0
    dynamics = min(max(analysis.energy_std * 10.0, 0.0), 1.0)
    warmth = 1.0 - brightness
    sparseness = 1.0 - onset
    return [tempo, energy, brightness, onset, minor, dynamics, warmth, sparseness]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_vec = np.asarray(left, dtype=np.float32)
    right_vec = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(left_vec) * np.linalg.norm(right_vec))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left_vec, right_vec) / denom)


def _match_clap_style_tags(embedding: list[float], limit: int = 5) -> tuple[list[str], dict]:
    scores = {
        tag: round(_cosine_similarity(embedding, prototype), 4)
        for tag, prototype in CLAP_STYLE_PROTOTYPES.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [tag for tag, _ in ranked[:limit]], dict(ranked[:limit])


def infer_clap_style_tags(analysis: AudioAnalysis, limit: int = 5) -> tuple[list[str], str, dict, list[float]]:
    """Infer style tags from a reference-track audio embedding."""
    embedding = _audio_clap_lite_embedding(analysis)
    tags, scores = _match_clap_style_tags(embedding, limit=limit)
    return tags, "audio-clap-lite", scores, embedding


# ── Main Analysis Function ─────────────────────────────────────────────────────

# Bump when the analysis changes shape or meaning; cached results keyed on an
# older version are ignored rather than silently reused.
ANALYZER_VERSION = 3
_ANALYSIS_CACHE: "OrderedDict[str, AudioAnalysis]" = OrderedDict()
_ANALYSIS_CACHE_LIMIT = 32
_ANALYSIS_CACHE_LOCK = threading.Lock()


def _raise_if_cancelled(cancel_event, file_path: str):
    """Cancellation is a distinct outcome, not a partial result."""
    if cancel_event is not None and cancel_event.is_set():
        from core.workers import CancelledJobError

        raise CancelledJobError(
            "Reference analysis cancelled",
            outputs={"file_path": file_path},
        )


def analysis_cache_key(file_path: str) -> str:
    """Content hash plus analyzer version, so edits and upgrades both miss."""
    digest = hashlib.sha256()
    digest.update(str(ANALYZER_VERSION).encode("utf-8"))
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_analysis(cache_key: str) -> Optional["AudioAnalysis"]:
    with _ANALYSIS_CACHE_LOCK:
        analysis = _ANALYSIS_CACHE.get(cache_key)
        if analysis is not None:
            _ANALYSIS_CACHE.move_to_end(cache_key)
        return analysis


def store_analysis(cache_key: str, analysis: "AudioAnalysis") -> None:
    with _ANALYSIS_CACHE_LOCK:
        _ANALYSIS_CACHE[cache_key] = analysis
        _ANALYSIS_CACHE.move_to_end(cache_key)
        while len(_ANALYSIS_CACHE) > _ANALYSIS_CACHE_LIMIT:
            _ANALYSIS_CACHE.popitem(last=False)


def clear_analysis_cache() -> None:
    with _ANALYSIS_CACHE_LOCK:
        _ANALYSIS_CACHE.clear()


def analyze_track(
    file_path: str,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event=None,
    use_cache: bool = True,
    **kwargs,
) -> AudioAnalysis:
    """
    Analyze an audio file and extract production fingerprint.
    Returns AudioAnalysis with all features.

    Results are cached by content hash plus analyzer version, so re-selecting
    the same file is instant while an edited file or an analyzer upgrade both
    force a fresh analysis. Cancellation raises CancelledJobError.
    """
    cache_key = ""
    if use_cache:
        try:
            cache_key = analysis_cache_key(file_path)
        except OSError:
            cache_key = ""
        if cache_key:
            hit = cached_analysis(cache_key)
            if hit is not None:
                if step_cb:
                    step_cb("Using cached analysis")
                if progress_cb:
                    progress_cb(100)
                return hit

    _raise_if_cancelled(cancel_event, file_path)

    from core.deps import ensure
    ensure("librosa")
    import librosa
    import soundfile as sf

    analysis = AudioAnalysis(file_path=file_path)

    if step_cb:
        step_cb("Loading audio...")
    if progress_cb:
        progress_cb(5)

    # Load audio
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    analysis.sample_rate = sr
    analysis.duration = librosa.get_duration(y=y, sr=sr)

    _raise_if_cancelled(cancel_event, file_path)

    # BPM detection
    if step_cb:
        step_cb("Detecting tempo...")
    if progress_cb:
        progress_cb(15)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        analysis.bpm = float(tempo[0]) if len(tempo) > 0 else 120.0
    else:
        analysis.bpm = float(tempo)
    analysis.beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
    analysis.bpm_confidence = min(1.0, len(analysis.beat_times) / (analysis.duration / 2))
    analysis.bpm_alternatives = _bpm_alternatives(
        analysis.bpm, analysis.bpm_confidence
    )
    analysis.suggested_tempo_tag = _bpm_to_tag(analysis.bpm)

    _raise_if_cancelled(cancel_event, file_path)

    # Key detection
    if step_cb:
        step_cb("Detecting key...")
    if progress_cb:
        progress_cb(30)

    (
        analysis.key,
        analysis.key_confidence,
        analysis.key_alternatives,
    ) = _detect_key(y, sr, include_alternatives=True)

    _raise_if_cancelled(cancel_event, file_path)

    # Energy envelope
    if step_cb:
        step_cb("Analyzing energy...")
    if progress_cb:
        progress_cb(45)

    rms = librosa.feature.rms(y=y)[0]
    rms_normalized = rms / (rms.max() + 1e-8)
    # Downsample to ~100 points for curve editor
    n_points = min(100, len(rms_normalized))
    indices = np.linspace(0, len(rms_normalized) - 1, n_points, dtype=int)
    analysis.energy_curve = rms_normalized[indices].tolist()
    analysis.energy_times = np.linspace(0, analysis.duration, n_points).tolist()
    analysis.energy_mean = float(np.mean(rms))
    analysis.energy_std = float(np.std(rms))

    _raise_if_cancelled(cancel_event, file_path)

    # Spectral centroid (brightness)
    if step_cb:
        step_cb("Analyzing spectral features...")
    if progress_cb:
        progress_cb(60)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    analysis.brightness_mean = float(np.mean(centroid))
    analysis.brightness_std = float(np.std(centroid))

    # Onset density
    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    analysis.onset_density = len(onset_times) / max(1.0, analysis.duration)

    _raise_if_cancelled(cancel_event, file_path)

    # Structure detection via self-similarity
    if step_cb:
        step_cb("Detecting song structure...")
    if progress_cb:
        progress_cb(75)

    try:
        # Use recurrence matrix for structure
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Simple segment detection via novelty curve
        novelty = librosa.onset.onset_strength(y=y, sr=sr)
        # Peak-pick for section boundaries
        peaks = librosa.util.peak_pick(novelty, pre_max=30, post_max=30, pre_avg=30, post_avg=30, delta=0.1, wait=50)
        boundary_times = librosa.frames_to_time(peaks, sr=sr)

        # Label sections heuristically
        section_labels = ["Intro", "Verse", "Chorus", "Verse", "Chorus", "Bridge", "Chorus", "Outro"]
        sections = []
        prev_time = 0.0
        for i, t in enumerate(boundary_times[:8]):  # Max 8 sections
            label = section_labels[i] if i < len(section_labels) else f"Section {i+1}"
            sections.append({
                "start": round(prev_time, 2),
                "end": round(float(t), 2),
                "label": label,
            })
            prev_time = float(t)
        # Final section
        if prev_time < analysis.duration - 1:
            sections.append({
                "start": round(prev_time, 2),
                "end": round(analysis.duration, 2),
                "label": "Outro" if len(sections) > 2 else "Section",
            })
        analysis.sections = sections
    except Exception:
        # Structure detection can fail on very short or unusual audio
        analysis.sections = [{"start": 0, "end": analysis.duration, "label": "Full Track"}]

    _raise_if_cancelled(cancel_event, file_path)

    # Genre estimation
    if step_cb:
        step_cb("Estimating style tags...")
    if progress_cb:
        progress_cb(90)

    analysis.suggested_tags = _estimate_genre_tags(
        analysis.bpm, analysis.brightness_mean,
        analysis.onset_density, analysis.energy_mean, analysis.key,
    )
    clap_tags, clap_backend, clap_scores, clap_embedding = infer_clap_style_tags(analysis)
    analysis.clap_backend = clap_backend
    analysis.clap_embedding = clap_embedding
    analysis.clap_style_tags = clap_tags
    analysis.suggested_tags = _dedupe_tags([*analysis.suggested_tags, *clap_tags])[:8]
    analysis.clap_similarity = clap_scores

    if progress_cb:
        progress_cb(100)

    if cache_key:
        store_analysis(cache_key, analysis)
    return analysis


# ── Generation Quality Scoring ────────────────────────────────────────────────


@dataclass
class QualityScore:
    """Deterministic quality score for a generated audio file."""
    total: float = 0.0
    silence: float = 0.0
    clipping: float = 0.0
    duration: float = 0.0
    loudness: float = 0.0
    spectral_balance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 2),
            "silence": round(self.silence, 2),
            "clipping": round(self.clipping, 2),
            "duration": round(self.duration, 2),
            "loudness": round(self.loudness, 2),
            "spectral_balance": round(self.spectral_balance, 2),
        }


def score_generation_quality(
    audio_path: str,
    expected_duration: float = 0.0,
    progress_cb=None,
    cancel_event=None,
) -> QualityScore:
    """
    Score the quality of a generated audio file on a 0-100 scale.
    Components: silence (20), clipping (20), duration (20), loudness (20),
    spectral balance (20).
    """
    try:
        from core.audio_buffers import decode_audio_file

        audio, sr = decode_audio_file(
            audio_path,
            target_channels=2,
            progress_cb=(
                (lambda value: progress_cb(int(value * 0.6)))
                if progress_cb else None
            ),
            cancel_event=cancel_event,
        )
    except Exception as exc:
        from core.workers import CancelledJobError

        if isinstance(exc, CancelledJobError):
            raise
        return QualityScore()

    return score_audio_buffer(
        audio,
        sr,
        expected_duration=expected_duration,
        progress_cb=(
            (lambda value: progress_cb(60 + int(value * 0.4)))
            if progress_cb else None
        ),
        cancel_event=cancel_event,
    )


def score_audio_buffer(
    audio: np.ndarray,
    sample_rate: int,
    expected_duration: float = 0.0,
    progress_cb=None,
    cancel_event=None,
) -> QualityScore:
    """Score an already-decoded buffer without performing file I/O.

    The separate buffer entry point lets batch previews share one decode with
    their waveform while keeping the legacy path-based API intact.
    """
    from core.workers import CancelledJobError

    def _raise_if_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledJobError("Quality scoring cancelled")

    _raise_if_cancelled()
    score = QualityScore()

    if audio.ndim == 2:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    if len(mono) == 0:
        return score

    actual_duration = len(mono) / sample_rate
    if progress_cb:
        progress_cb(10)

    rms = np.sqrt(np.mean(mono ** 2))

    silent_frames = np.sum(np.abs(mono) < 1e-5)
    silence_ratio = silent_frames / len(mono)
    score.silence = max(0.0, 20.0 * (1.0 - silence_ratio * 2.0))

    clip_frames = np.sum(np.abs(mono) >= 0.999)
    clip_ratio = clip_frames / len(mono)
    score.clipping = max(0.0, 20.0 * (1.0 - clip_ratio * 50.0))

    if expected_duration > 0:
        ratio = actual_duration / expected_duration
        deviation = abs(1.0 - ratio)
        score.duration = max(0.0, 20.0 * (1.0 - deviation * 2.0))
    else:
        score.duration = 20.0 if actual_duration > 1.0 else 5.0

    _raise_if_cancelled()
    if progress_cb:
        progress_cb(45)

    if rms < 1e-6:
        score.loudness = 0.0
    else:
        rms_db = 20.0 * np.log10(rms + 1e-10)
        if -30.0 <= rms_db <= -6.0:
            score.loudness = 20.0
        elif rms_db < -30.0:
            score.loudness = max(0.0, 20.0 * (1.0 - (-30.0 - rms_db) / 30.0))
        else:
            score.loudness = max(0.0, 20.0 * (1.0 - (rms_db + 6.0) / 6.0))

    try:
        n_fft = min(2048, len(mono))
        if n_fft >= 64:
            spectrum = np.abs(np.fft.rfft(mono[:n_fft]))
            if spectrum.sum() > 0:
                freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
                if 500.0 <= centroid <= 4000.0:
                    score.spectral_balance = 20.0
                elif centroid < 500.0:
                    score.spectral_balance = max(0.0, 20.0 * centroid / 500.0)
                else:
                    score.spectral_balance = max(0.0, 20.0 * (1.0 - (centroid - 4000.0) / 4000.0))
            else:
                score.spectral_balance = 0.0
        else:
            score.spectral_balance = 10.0
    except Exception:
        score.spectral_balance = 10.0

    _raise_if_cancelled()
    score.total = score.silence + score.clipping + score.duration + score.loudness + score.spectral_balance
    if progress_cb:
        progress_cb(100)
    return score


# ── Whisper Integration ───────────────────────────────────────────────────────

_whisper_model = None
_whisper_processor = None
_whisper_device = "cpu"


def load_model(cache_dir: str = None, model_path: str = None, **kwargs):
    """Load Whisper model for transcription/alignment. Called by ModelManager._dynamic_load()."""
    global _whisper_model, _whisper_processor, _whisper_device
    from core.deps import ensure
    ensure("torch")
    ensure("transformers")

    from core.model_manager import ModelSecurityError
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    local = Path(model_path or cache_dir or "")
    safe_weights = tuple(local.glob("*.safetensors")) if local.is_dir() else ()
    if not local.is_absolute() or not local.is_dir() or not safe_weights:
        raise ModelSecurityError(
            "Whisper loading will not download model data during inference. "
            "A verified local Transformers snapshot with safetensors weights is required."
        )
    assert_safe_transformers_snapshot(local)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(
        str(local),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        str(local),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    _whisper_model = model.to(device)
    _whisper_model.eval()
    _whisper_processor = processor
    _whisper_device = device
    return _whisper_model


def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """Transcribe audio using loaded Whisper model."""
    global _whisper_model, _whisper_processor, _whisper_device
    if _whisper_model is None:
        from core.model_manager import ModelManager
        load_model(cache_dir=str(ModelManager().get_cache_dir("whisper-tiny")))

    from core.deps import ensure
    ensure("librosa")
    import librosa
    import torch

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = _whisper_processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = (
        inputs["input_features"]
        if isinstance(inputs, dict)
        else inputs.input_features
    ).to(_whisper_device)

    generate_kwargs = {}
    if language:
        generate_kwargs["forced_decoder_ids"] = (
            _whisper_processor.get_decoder_prompt_ids(
                language=language,
                task="transcribe",
            )
        )

    with torch.no_grad():
        generated_ids = _whisper_model.generate(input_features, **generate_kwargs)
    text = _whisper_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

    return {
        "text": text,
        "segments": [],
        "language": language or "",
    }
