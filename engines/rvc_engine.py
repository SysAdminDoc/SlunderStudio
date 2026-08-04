"""
Slunder Studio — RVC / GPT-SoVITS Engine
Voice conversion (RVC v2) and voice cloning (GPT-SoVITS) for transforming
existing vocals or cloning a target voice from reference audio.
"""
import os
import time
import json
from typing import Optional, Callable
from dataclasses import dataclass, field

import numpy as np

from core.audio_export import write_audio_file
from core.provenance import write_provenance_sidecar
from core.settings import get_configured_output_dir
from core.voice_bank import (
    SAFER_CHECKPOINT_EXTENSIONS,
    VOICE_OPERATION_CLONE,
    VOICE_OPERATION_CONVERSION,
    VoiceProfile,
    ensure_voice_profile_allowed,
    voice_profile_provenance,
)


RVC_UNSUPPORTED_ERROR = (
    "RVC conversion is unavailable: Slunder Studio does not bundle a verified "
    "local RVC inference adapter yet. No placeholder audio will be generated."
)
GPT_SOVITS_UNSUPPORTED_ERROR = (
    "GPT-SoVITS cloning is unavailable: Slunder Studio does not bundle a verified "
    "local GPT-SoVITS inference adapter yet. No placeholder audio will be generated."
)


@dataclass
class VoiceConvertParams:
    """Parameters for RVC voice conversion."""
    input_audio: Optional[np.ndarray] = None
    input_path: str = ""
    sample_rate: int = 44100
    pitch_shift: int = 0  # semitones
    f0_method: str = "rmvpe"  # "pm" | "harvest" | "crepe" | "rmvpe"
    index_rate: float = 0.75  # 0.0-1.0, feature retrieval blend
    filter_radius: int = 3  # median filter for pitch
    rms_mix_rate: float = 0.25  # envelope mix
    protect: float = 0.33  # consonant protection
    # Retained for job/schema compatibility; it can never enable placeholder audio.
    allow_demo_output: bool = False


@dataclass
class VoiceCloneParams:
    """Parameters for GPT-SoVITS voice cloning."""
    text: str = ""  # text to speak/sing
    ref_audio_path: str = ""  # reference audio for voice cloning
    ref_text: str = ""  # transcript of reference audio
    language: str = "en"  # "en" | "zh" | "ja"
    speed: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.9
    sample_rate: int = 32000
    # Retained for job/schema compatibility; it can never enable placeholder audio.
    allow_demo_output: bool = False


@dataclass
class CloneReferenceQuality:
    """Quality report for a GPT-SoVITS reference sample."""
    path: str = ""
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 1
    peak_dbfs: float = -120.0
    rms_dbfs: float = -120.0
    silence_percent: float = 100.0
    clipped_percent: float = 0.0
    score: int = 0
    status: str = "fail"  # "pass" | "warn" | "fail"
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def can_onboard(self) -> bool:
        return self.status in ("pass", "warn")

    def metrics_summary(self) -> str:
        return (
            f"{self.duration:.1f}s, {self.sample_rate / 1000:.1f} kHz, "
            f"RMS {self.rms_dbfs:.1f} dBFS, silence {self.silence_percent:.0f}%"
        )


@dataclass
class VoiceResult:
    """Result from voice conversion or cloning."""
    audio: Optional[np.ndarray] = None
    sample_rate: int = 44100
    duration: float = 0.0
    generation_time: float = 0.0
    error: Optional[str] = None
    is_demo: bool = False
    output_kind: str = "model"  # "model" | "demo" | "error"
    can_route: bool = True
    provenance: dict = field(default_factory=dict)
    provenance_path: str = ""

    @property
    def is_success(self) -> bool:
        return self.error is None and self.audio is not None


def _dbfs(value: float) -> float:
    return 20.0 * np.log10(max(float(value), 1e-8))


def _load_audio_for_quality(path: str) -> tuple[np.ndarray, int, int]:
    """Load reference audio without forcing optional dependencies."""
    try:
        import soundfile as sf
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        channels = audio.shape[1]
        return audio.mean(axis=1).astype(np.float32), int(sample_rate), int(channels)
    except Exception:
        pass

    if path.lower().endswith(".wav"):
        import wave
        with wave.open(path, "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if width == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 4:
            audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV bit depth: {width * 8}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio.astype(np.float32), int(sample_rate), int(channels)

    try:
        import librosa
        loaded = librosa.load(path, sr=None, mono=False)
        audio, sample_rate = loaded
        channels = 1
        if audio.ndim > 1:
            channels = int(audio.shape[0])
            audio = audio.mean(axis=0)
        return audio.astype(np.float32), int(sample_rate), channels
    except Exception as exc:
        raise ValueError(f"Could not read reference audio: {exc}") from exc


def assess_clone_reference(
    path: str,
    min_duration: float = 10.0,
    max_duration: float = 30.0,
) -> CloneReferenceQuality:
    """Assess whether a 10-30s reference sample is suitable for GPT-SoVITS."""
    report = CloneReferenceQuality(path=path)
    if not path or not os.path.isfile(path):
        report.issues.append("Reference audio file is missing.")
        report.suggestions.append("Choose a WAV, FLAC, MP3, or OGG file before onboarding.")
        return report

    try:
        audio, sample_rate, channels = _load_audio_for_quality(path)
    except Exception as exc:
        report.issues.append(str(exc))
        report.suggestions.append("Use a readable WAV file if the decoder for this format is unavailable.")
        return report

    if audio.size == 0 or sample_rate <= 0:
        report.issues.append("Reference audio is empty.")
        report.suggestions.append("Record a clean 10-30 second phrase with audible voice.")
        return report

    audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    duration = len(audio) / sample_rate
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    clipped = float(np.mean(np.abs(audio) >= 0.995) * 100.0) if audio.size else 0.0

    frame_len = max(1, int(sample_rate * 0.05))
    frame_count = max(1, len(audio) // frame_len)
    frames = audio[:frame_count * frame_len].reshape(frame_count, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    silence_threshold = 10 ** (-45.0 / 20.0)
    silence_percent = float(np.mean(frame_rms < silence_threshold) * 100.0)

    report.duration = duration
    report.sample_rate = sample_rate
    report.channels = channels
    report.peak_dbfs = _dbfs(peak)
    report.rms_dbfs = _dbfs(rms)
    report.silence_percent = silence_percent
    report.clipped_percent = clipped

    score = 100
    fatal = False

    if duration < min_duration:
        fatal = True
        score -= 55
        report.issues.append(f"Sample is too short ({duration:.1f}s).")
        report.suggestions.append("Use a continuous 10-30 second reference take.")
    elif duration > max_duration:
        fatal = True
        score -= 45
        report.issues.append(f"Sample is too long ({duration:.1f}s).")
        report.suggestions.append("Trim the reference to the strongest 10-30 seconds.")

    if report.rms_dbfs < -38.0:
        fatal = True
        score -= 35
        report.issues.append("Voice level is too quiet.")
        report.suggestions.append("Record closer to the mic or normalize the sample.")
    elif report.rms_dbfs < -30.0:
        score -= 15
        report.issues.append("Voice level is low.")
        report.suggestions.append("Normalize the sample before onboarding.")
    elif report.rms_dbfs > -8.0:
        score -= 12
        report.issues.append("Voice level is very hot.")
        report.suggestions.append("Lower input gain to leave headroom.")

    if clipped > 1.0:
        fatal = True
        score -= 30
        report.issues.append(f"Sample is clipping ({clipped:.1f}% clipped samples).")
        report.suggestions.append("Re-record without clipping or use a cleaner take.")
    elif clipped > 0.0:
        score -= 10
        report.issues.append("Sample has clipped peaks.")
        report.suggestions.append("Use a take with clean peaks below 0 dBFS.")

    if silence_percent > 60.0:
        fatal = True
        score -= 30
        report.issues.append("Sample contains too much silence.")
        report.suggestions.append("Trim leading gaps and long pauses.")
    elif silence_percent > 35.0:
        score -= 12
        report.issues.append("Sample has long quiet gaps.")
        report.suggestions.append("Trim pauses so the model hears mostly voice.")

    if sample_rate < 16000:
        score -= 10
        report.issues.append("Sample rate is below 16 kHz.")
        report.suggestions.append("Use a 24 kHz or higher reference when possible.")

    report.score = max(0, min(100, int(round(score))))
    if fatal:
        report.status = "fail"
    elif report.issues:
        report.status = "warn"
    else:
        report.status = "pass"

    if not report.suggestions:
        report.suggestions.append("Ready for GPT-SoVITS onboarding.")
    return report


def _ensure_checkpoint_trust(profile: VoiceProfile, engine_name: str):
    """Apply the checkpoint trust gate without executing an unavailable adapter."""
    path = profile.model_path or ""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SAFER_CHECKPOINT_EXTENSIONS and not profile.trusted:
        raise RuntimeError(
            f"{os.path.basename(path)} is an unsafe local checkpoint format. "
            f"Open Vocal Suite > {engine_name} and click 'Trust unsafe checkpoint' "
            "for this profile before loading it, or use a safetensors/ONNX model."
        )


# ── RVC Engine ─────────────────────────────────────────────────────────────────

class RVCEngine:
    """
    RVC v2 voice conversion engine.
    Converts input vocals to match a target voice model.
    """

    def __init__(self):
        self._model = None
        self._index = None
        self._model_path: Optional[str] = None
        self._base_model_path: Optional[str] = None
        self._profile: Optional[VoiceProfile] = None
        self._device = "cpu"
        self._output_dir = os.path.join(get_configured_output_dir(), "generations", "voice_convert")
        os.makedirs(self._output_dir, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def activate_base_model(self, model_path: str):
        """Reject activation until a real verified RVC adapter is bundled."""
        raise RuntimeError(RVC_UNSUPPORTED_ERROR)

    def prepare_demo_profile(self, profile: VoiceProfile):
        """Reject the removed placeholder pipeline without producing audio."""
        ensure_voice_profile_allowed(profile, VOICE_OPERATION_CONVERSION)
        raise RuntimeError(RVC_UNSUPPORTED_ERROR)

    def load_model(self, profile: VoiceProfile,
                   device: str = "cuda",
                   progress_callback: Optional[Callable] = None):
        """Fail closed until a verified local RVC inference adapter is available."""
        try:
            ensure_voice_profile_allowed(profile, VOICE_OPERATION_CONVERSION)
            _ensure_checkpoint_trust(profile, "Voice Conversion")
            raise RuntimeError(RVC_UNSUPPORTED_ERROR)

        except Exception as e:
            self._model = None
            raise RuntimeError(f"Failed to load RVC model: {e}") from e

    def unload_model(self):
        """Release model resources."""
        self._model = None
        self._index = None
        self._model_path = None
        self._base_model_path = None
        self._profile = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def convert(self, params: VoiceConvertParams,
                progress_callback: Optional[Callable] = None) -> VoiceResult:
        """Convert voice only through a verified adapter; never synthesize a fallback."""
        if not self.is_loaded:
            return VoiceResult(
                error="RVC model not loaded",
                output_kind="error",
                can_route=False,
            )

        t0 = time.time()
        return VoiceResult(
            error=RVC_UNSUPPORTED_ERROR,
            generation_time=time.time() - t0,
            output_kind="error",
            can_route=False,
        )

    def save_output(
        self,
        result: VoiceResult,
        name: Optional[str] = None,
        profile: Optional[VoiceProfile] = None,
    ) -> Optional[str]:
        """Save conversion result to WAV."""
        if result.audio is None or not result.can_route:
            return None

        if name is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = f"rvc_{ts}"

        path = os.path.join(self._output_dir, f"{name}.wav")
        write_audio_file(
            path,
            result.audio,
            result.sample_rate,
            file_format="wav",
            channels=1,
        )

        prov = result.provenance or {}
        active_profile = profile or self._profile
        extra = dict(prov.get("extra", {}))
        if active_profile:
            extra["voice_profile"] = voice_profile_provenance(active_profile)
        sidecar = write_provenance_sidecar(
            path,
            module=prov.get("module", "vocal_suite"),
            operation=prov.get("operation", "rvc_convert"),
            model_id=prov.get("model_id", "rvc-v2"),
            model_name=prov.get("model_name", active_profile.name if active_profile else ""),
            model_source=prov.get("model_source", active_profile.source if active_profile else ""),
            model_revision=prov.get("model_revision", active_profile.source_revision if active_profile else ""),
            model_hash=prov.get("model_hash", ""),
            model_license=prov.get("model_license", active_profile.license if active_profile else ""),
            parameters=prov.get("parameters", {}),
            source_asset_ids=prov.get("source_asset_ids") or ([active_profile.id] if active_profile else []),
            source_paths=prov.get("source_paths", []),
            export_format="wav",
            output_kind=prov.get("output_kind", result.output_kind),
            extra=extra,
        )
        result.provenance_path = str(sidecar)
        return path


# ── GPT-SoVITS Engine ──────────────────────────────────────────────────────────

class GPTSoVITSEngine:
    """
    GPT-SoVITS voice cloning engine.
    Zero-shot/few-shot voice cloning from reference audio.
    """

    def __init__(self):
        self._gpt_model = None
        self._sovits_model = None
        self._model_path: Optional[str] = None
        self._base_model_path: Optional[str] = None
        self._profile: Optional[VoiceProfile] = None
        self._device = "cpu"
        self._output_dir = os.path.join(get_configured_output_dir(), "generations", "voice_clone")
        os.makedirs(self._output_dir, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self._sovits_model is not None

    def activate_base_model(self, model_path: str):
        """Reject activation until a real verified GPT-SoVITS adapter is bundled."""
        raise RuntimeError(GPT_SOVITS_UNSUPPORTED_ERROR)

    def prepare_demo_profile(self, profile: VoiceProfile):
        """Reject the removed placeholder pipeline without producing audio."""
        ensure_voice_profile_allowed(profile, VOICE_OPERATION_CLONE)
        raise RuntimeError(GPT_SOVITS_UNSUPPORTED_ERROR)

    def load_model(self, profile: VoiceProfile, device: str = "cuda",
                   progress_callback: Optional[Callable] = None):
        """Fail closed until a verified local GPT-SoVITS adapter is available."""
        try:
            ensure_voice_profile_allowed(profile, VOICE_OPERATION_CLONE)
            _ensure_checkpoint_trust(profile, "Voice Cloning")
            raise RuntimeError(GPT_SOVITS_UNSUPPORTED_ERROR)

        except Exception as e:
            self._sovits_model = None
            self._gpt_model = None
            raise RuntimeError(f"Failed to load GPT-SoVITS: {e}") from e

    def unload_model(self):
        self._sovits_model = None
        self._gpt_model = None
        self._model_path = None
        self._base_model_path = None
        self._profile = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def clone(self, params: VoiceCloneParams,
              progress_callback: Optional[Callable] = None) -> VoiceResult:
        """Clone only through a verified adapter; never synthesize a fallback."""
        if not self.is_loaded:
            return VoiceResult(
                error="GPT-SoVITS model not loaded",
                output_kind="error",
                can_route=False,
            )

        t0 = time.time()

        try:
            if progress_callback:
                progress_callback(0.1, "Processing reference audio...")

            quality = assess_clone_reference(params.ref_audio_path)
            if not quality.can_onboard:
                return VoiceResult(
                    error="Reference audio failed guardrails: " + "; ".join(quality.issues),
                    generation_time=time.time() - t0,
                    output_kind="error",
                    can_route=False,
                )

            return VoiceResult(
                error=GPT_SOVITS_UNSUPPORTED_ERROR,
                generation_time=time.time() - t0,
                output_kind="error",
                can_route=False,
            )

        except Exception as e:
            return VoiceResult(
                error=str(e),
                generation_time=time.time() - t0,
                output_kind="error",
                can_route=False,
            )

    def save_output(
        self,
        result: VoiceResult,
        name: Optional[str] = None,
        profile: Optional[VoiceProfile] = None,
    ) -> Optional[str]:
        if result.audio is None or not result.can_route:
            return None
        if name is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = f"clone_{ts}"
        path = os.path.join(self._output_dir, f"{name}.wav")
        write_audio_file(
            path,
            result.audio,
            result.sample_rate,
            file_format="wav",
            channels=1,
        )
        prov = result.provenance or {}
        active_profile = profile or self._profile
        extra = dict(prov.get("extra", {}))
        if active_profile:
            extra["voice_profile"] = voice_profile_provenance(active_profile)
        sidecar = write_provenance_sidecar(
            path,
            module=prov.get("module", "vocal_suite"),
            operation=prov.get("operation", "gpt_sovits_clone"),
            model_id=prov.get("model_id", "gpt-sovits-v2"),
            model_name=prov.get("model_name", active_profile.name if active_profile else ""),
            model_source=prov.get("model_source", active_profile.source if active_profile else ""),
            model_revision=prov.get("model_revision", active_profile.source_revision if active_profile else ""),
            model_hash=prov.get("model_hash", ""),
            model_license=prov.get("model_license", active_profile.license if active_profile else ""),
            prompt=prov.get("prompt", ""),
            parameters=prov.get("parameters", {}),
            source_asset_ids=prov.get("source_asset_ids") or ([active_profile.id] if active_profile else []),
            source_paths=prov.get("source_paths", []),
            export_format="wav",
            output_kind=prov.get("output_kind", result.output_kind),
            extra=extra,
        )
        result.provenance_path = str(sidecar)
        return path


# ── High-Level ─────────────────────────────────────────────────────────────────

_rvc: Optional[RVCEngine] = None
_sovits: Optional[GPTSoVITSEngine] = None


def get_rvc() -> RVCEngine:
    global _rvc
    if _rvc is None:
        _rvc = RVCEngine()
    return _rvc


def get_sovits() -> GPTSoVITSEngine:
    global _sovits
    if _sovits is None:
        _sovits = GPTSoVITSEngine()
    return _sovits


def load_model(cache_dir: str = None, model_id: str = "", **kwargs):
    """Activate the requested voice engine; profiles are loaded in Vocal Suite."""
    from core.deps import ensure
    ensure("torch")
    if model_id == "gpt-sovits-v2":
        engine = get_sovits()
    else:
        engine = get_rvc()
    engine.activate_base_model(cache_dir or "")
    return engine
