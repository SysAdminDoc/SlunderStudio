"""
Slunder Studio v0.0.2 — Audio Analyzer
Reference track analysis: BPM, key, energy envelope, spectral features,
genre estimation, and song structure detection via librosa.
"""
import json
import numpy as np
from typing import Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field

from core.settings import get_config_dir


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
    # Key
    key: str = ""  # e.g., "C major", "A minor"
    key_confidence: float = 0.0
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

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "duration": self.duration,
            "bpm": self.bpm,
            "key": self.key,
            "energy_mean": self.energy_mean,
            "brightness_mean": self.brightness_mean,
            "onset_density": self.onset_density,
            "suggested_tags": self.suggested_tags,
            "suggested_tempo_tag": self.suggested_tempo_tag,
            "sections": self.sections,
        }

    def to_ace_step_tags(self) -> str:
        """Convert analysis to ACE-Step compatible tag string."""
        tags = list(self.suggested_tags)
        if self.suggested_tempo_tag:
            tags.append(self.suggested_tempo_tag)
        return ", ".join(tags)


# ── Key Detection ──────────────────────────────────────────────────────────────

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _detect_key(y, sr) -> tuple[str, float]:
    """Detect musical key using chroma features."""
    from core.deps import ensure
    ensure("librosa")
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Major and minor profiles (Krumhansl-Kessler)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_corr = -1.0
    best_key = "C major"

    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        maj_corr = np.corrcoef(rolled, major_profile)[0, 1]
        min_corr = np.corrcoef(rolled, minor_profile)[0, 1]

        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = f"{KEY_NAMES[i]} major"
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = f"{KEY_NAMES[i]} minor"

    return best_key, max(0.0, best_corr)


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


# ── Main Analysis Function ─────────────────────────────────────────────────────

def analyze_track(
    file_path: str,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event=None,
    **kwargs,
) -> AudioAnalysis:
    """
    Analyze an audio file and extract production fingerprint.
    Returns AudioAnalysis with all features.
    """
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

    if cancel_event and cancel_event.is_set():
        return analysis

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
    analysis.suggested_tempo_tag = _bpm_to_tag(analysis.bpm)

    if cancel_event and cancel_event.is_set():
        return analysis

    # Key detection
    if step_cb:
        step_cb("Detecting key...")
    if progress_cb:
        progress_cb(30)

    analysis.key, analysis.key_confidence = _detect_key(y, sr)

    if cancel_event and cancel_event.is_set():
        return analysis

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

    if cancel_event and cancel_event.is_set():
        return analysis

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

    if cancel_event and cancel_event.is_set():
        return analysis

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

    if cancel_event and cancel_event.is_set():
        return analysis

    # Genre estimation
    if step_cb:
        step_cb("Estimating style tags...")
    if progress_cb:
        progress_cb(90)

    analysis.suggested_tags = _estimate_genre_tags(
        analysis.bpm, analysis.brightness_mean,
        analysis.onset_density, analysis.energy_mean, analysis.key,
    )

    if progress_cb:
        progress_cb(100)

    return analysis


# ── Reference Library ──────────────────────────────────────────────────────────

class ReferenceLibrary:
    """Saves analyzed tracks as reusable style profiles."""

    def __init__(self):
        self._lib_path = get_config_dir() / "reference_library.json"
        self._profiles: list[dict] = []
        self._load()

    def _load(self):
        if self._lib_path.exists():
            try:
                self._profiles = json.loads(self._lib_path.read_text())
            except Exception:
                self._profiles = []

    def _save(self):
        self._lib_path.write_text(json.dumps(self._profiles, indent=2))

    def add(self, analysis: AudioAnalysis, name: str = ""):
        """Save an analysis as a reusable profile."""
        profile = analysis.to_dict()
        profile["name"] = name or Path(analysis.file_path).stem
        profile["saved_at"] = __import__("time").time()
        self._profiles.append(profile)
        self._save()

    def get_all(self) -> list[dict]:
        return list(self._profiles)

    def get(self, index: int) -> Optional[dict]:
        if 0 <= index < len(self._profiles):
            return self._profiles[index]
        return None

    def delete(self, index: int):
        if 0 <= index < len(self._profiles):
            self._profiles.pop(index)
            self._save()

    @property
    def count(self) -> int:
        return len(self._profiles)


# ── Whisper Integration ───────────────────────────────────────────────────────

_whisper_model = None


def load_model(cache_dir: str = None, **kwargs):
    """Load Whisper model for transcription/alignment. Called by ModelManager._dynamic_load()."""
    global _whisper_model
    from core.deps import ensure
    ensure("torch")
    ensure("whisper")  # maps to openai-whisper via _PIP_NAMES

    import whisper

    model_size = "tiny"
    download_root = None
    if cache_dir:
        from pathlib import Path
        local = Path(cache_dir) / "whisper-tiny"
        if local.exists():
            download_root = str(local)

    _whisper_model = whisper.load_model(model_size, download_root=download_root)
    return _whisper_model


def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """Transcribe audio using loaded Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        load_model()

    options = {}
    if language:
        options["language"] = language

    result = _whisper_model.transcribe(audio_path, **options)
    return {
        "text": result.get("text", ""),
        "segments": result.get("segments", []),
        "language": result.get("language", ""),
    }
