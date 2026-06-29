"""
Slunder Studio v0.1.20 — RVC / GPT-SoVITS Engine
Voice conversion (RVC v2) and voice cloning (GPT-SoVITS) for transforming
existing vocals or cloning a target voice from reference audio.
"""
import os
import time
import json
from typing import Optional, Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from core.provenance import file_sha256, write_provenance_sidecar
from core.settings import get_config_dir
from core.voice_bank import (
    SAFER_CHECKPOINT_EXTENSIONS,
    UNSAFE_CHECKPOINT_EXTENSIONS,
    VOICE_OPERATION_CLONE,
    VOICE_OPERATION_CONVERSION,
    VoiceProfile,
    ensure_voice_profile_allowed,
    voice_profile_provenance,
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


def load_voice_checkpoint(profile: VoiceProfile, path: str, device: str):
    """
    Load a voice checkpoint while enforcing local trust for pickle-backed formats.
    PyTorch pickle checkpoints can execute code during deserialization, so local
    .pth/.pt/.ckpt/.bin files must be explicitly trusted in the voice profile.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in UNSAFE_CHECKPOINT_EXTENSIONS and not profile.trusted:
        raise RuntimeError(
            f"{os.path.basename(path)} is an unsafe local checkpoint format. "
            "Mark this voice profile as trusted before loading it, or use a safetensors/ONNX model."
        )

    if ext == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required to load safetensors voice checkpoints."
            ) from exc
        return load_file(path, device=device)

    if ext in SAFER_CHECKPOINT_EXTENSIONS and ext != ".safetensors":
        raise RuntimeError(f"Unsupported safer checkpoint format: {ext}")

    import torch
    return torch.load(path, map_location=device, weights_only=False)


def _safe_file_hash(path: Optional[str]) -> str:
    try:
        return file_sha256(path) if path and os.path.isfile(path) else ""
    except Exception:
        return ""


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
        self._profile: Optional[VoiceProfile] = None
        self._device = "cpu"
        self._output_dir = os.path.join(get_config_dir(), "generations", "voice_convert")
        os.makedirs(self._output_dir, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, profile: VoiceProfile,
                   device: str = "cuda",
                   progress_callback: Optional[Callable] = None):
        """Load an RVC voice model."""
        try:
            if progress_callback:
                progress_callback(0.1, "Loading RVC model...")

            ensure_voice_profile_allowed(profile, VOICE_OPERATION_CONVERSION)

            # Load the model checkpoint
            checkpoint = load_voice_checkpoint(profile, profile.model_path, device)

            self._model = checkpoint
            self._model_path = profile.model_path
            self._profile = profile
            self._device = device

            # Load feature index if available
            if profile.index_path and os.path.isfile(profile.index_path):
                if progress_callback:
                    progress_callback(0.6, "Loading feature index...")
                try:
                    import faiss
                    self._index = faiss.read_index(profile.index_path)
                except ImportError:
                    self._index = None

            if progress_callback:
                progress_callback(1.0, "RVC model loaded")

        except Exception as e:
            self._model = None
            raise RuntimeError(f"Failed to load RVC model: {e}") from e

    def unload_model(self):
        """Release model resources."""
        self._model = None
        self._index = None
        self._model_path = None
        self._profile = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def convert(self, params: VoiceConvertParams,
                progress_callback: Optional[Callable] = None) -> VoiceResult:
        """Convert voice using loaded RVC model."""
        if not self.is_loaded:
            return VoiceResult(
                error="RVC model not loaded",
                output_kind="error",
                can_route=False,
            )

        t0 = time.time()

        try:
            if progress_callback:
                progress_callback(0.1, "Loading audio...")

            # Get input audio
            audio = params.input_audio
            if audio is None and params.input_path:
                audio = self._load_audio(params.input_path, params.sample_rate)

            if audio is None:
                return VoiceResult(
                    error="No input audio provided",
                    output_kind="error",
                    can_route=False,
                )

            if not params.allow_demo_output:
                return VoiceResult(
                    error=(
                        "RVC inference pipeline is not available yet. "
                        "Demo spectral conversion must be explicitly enabled."
                    ),
                    output_kind="error",
                    can_route=False,
                    generation_time=time.time() - t0,
                )

            if progress_callback:
                progress_callback(0.2, f"Extracting pitch ({params.f0_method})...")

            # Extract F0 (pitch)
            f0 = self._extract_f0(audio, params.sample_rate, params.f0_method)

            # Apply pitch shift
            if params.pitch_shift != 0:
                f0 = f0 * (2.0 ** (params.pitch_shift / 12.0))

            if progress_callback:
                progress_callback(0.5, "Running voice conversion...")

            # Run conversion pipeline
            converted = self._run_conversion(audio, f0, params)

            # Apply RMS envelope mixing
            if params.rms_mix_rate > 0:
                converted = self._mix_rms(audio, converted, params.rms_mix_rate)

            # Normalize
            peak = np.max(np.abs(converted))
            if peak > 0:
                converted = converted / peak * 0.95

            gen_time = time.time() - t0
            duration = len(converted) / params.sample_rate
            param_meta = {
                k: v for k, v in asdict(params).items()
                if k != "input_audio"
            }
            if params.input_audio is not None:
                param_meta["input_audio_shape"] = list(params.input_audio.shape)

            if progress_callback:
                progress_callback(1.0, "Done")

            return VoiceResult(
                audio=converted,
                sample_rate=params.sample_rate,
                duration=duration,
                generation_time=gen_time,
                is_demo=True,
                output_kind="demo",
                can_route=True,
                provenance={
                    "module": "vocal_suite",
                    "operation": "rvc_convert",
                    "model_id": "rvc-v2",
                    "model_name": self._profile.name if self._profile else "",
                    "model_source": self._profile.source if self._profile else "",
                    "model_revision": self._profile.source_revision if self._profile else "",
                    "model_hash": _safe_file_hash(self._model_path),
                    "model_license": self._profile.license if self._profile else "",
                    "parameters": param_meta,
                    "source_asset_ids": [self._profile.id] if self._profile else [],
                    "source_paths": [params.input_path] if params.input_path else [],
                    "output_kind": "demo",
                    "extra": {
                        "model_path": self._model_path or "",
                        "voice_profile": voice_profile_provenance(self._profile),
                    },
                },
            )

        except Exception as e:
            return VoiceResult(
                error=str(e),
                generation_time=time.time() - t0,
                output_kind="error",
                can_route=False,
            )

    def _load_audio(self, path: str, target_sr: int) -> np.ndarray:
        """Load audio file to numpy array."""
        try:
            import librosa
            audio, _ = librosa.load(path, sr=target_sr, mono=True)
            return audio
        except ImportError:
            import wave
            with wave.open(path, "r") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if wf.getnchannels() == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)
                return audio

    def _extract_f0(self, audio: np.ndarray, sr: int, method: str) -> np.ndarray:
        """Extract fundamental frequency contour."""
        hop_size = 160
        n_frames = len(audio) // hop_size

        if method == "rmvpe":
            # RMVPE is the preferred method but requires its own model
            # Fallback to simple autocorrelation
            pass

        # Simple zero-crossing rate based F0 estimation (fallback)
        f0 = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_size
            end = min(start + hop_size * 4, len(audio))
            frame = audio[start:end]
            if len(frame) < hop_size:
                continue
            # Autocorrelation
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2:]
            # Find first peak after initial decay
            d = np.diff(corr)
            start_idx = max(int(sr / 500), 1)  # min ~500Hz
            end_idx = min(int(sr / 50), len(d) - 1)  # max ~50Hz
            if start_idx < end_idx:
                peaks = []
                for j in range(start_idx, end_idx):
                    if d[j - 1] > 0 and d[j] <= 0:
                        peaks.append(j)
                if peaks:
                    f0[i] = sr / peaks[0]

        return f0

    def _run_conversion(self, audio: np.ndarray, f0: np.ndarray,
                        params: VoiceConvertParams) -> np.ndarray:
        """
        Run the actual voice conversion inference.
        This is a placeholder for the full RVC pipeline which requires:
        - Feature extraction (HuBERT/ContentVec)
        - Optional feature index retrieval (FAISS)
        - Synthesis network forward pass
        """
        # In production, this would run the full RVC inference pipeline
        # For now, apply basic spectral envelope transfer as placeholder
        try:
            import torch
            import librosa

            # Extract spectral features
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            mag, phase = np.abs(stft), np.angle(stft)

            # Simple spectral shaping based on pitch shift
            if params.pitch_shift != 0:
                shift_ratio = 2.0 ** (params.pitch_shift / 12.0)
                n_bins = mag.shape[0]
                new_mag = np.zeros_like(mag)
                for i in range(n_bins):
                    src_bin = int(i / shift_ratio)
                    if 0 <= src_bin < n_bins:
                        new_mag[i] = mag[src_bin]
                mag = new_mag

            # Reconstruct
            stft_modified = mag * np.exp(1j * phase)
            converted = librosa.istft(stft_modified, hop_length=512, length=len(audio))
            return converted.astype(np.float32)

        except ImportError:
            # Without librosa, just return pitch-shifted audio
            return audio

    def _mix_rms(self, original: np.ndarray, converted: np.ndarray,
                 rate: float) -> np.ndarray:
        """Mix RMS envelope from original onto converted audio."""
        hop = 512
        n_frames = min(len(original), len(converted)) // hop

        for i in range(n_frames):
            s, e = i * hop, (i + 1) * hop
            if e > len(original) or e > len(converted):
                break
            rms_orig = np.sqrt(np.mean(original[s:e] ** 2) + 1e-8)
            rms_conv = np.sqrt(np.mean(converted[s:e] ** 2) + 1e-8)
            target_rms = rms_orig * rate + rms_conv * (1 - rate)
            if rms_conv > 0:
                converted[s:e] *= target_rms / rms_conv

        return converted

    def save_output(
        self,
        result: VoiceResult,
        name: Optional[str] = None,
        profile: Optional[VoiceProfile] = None,
    ) -> Optional[str]:
        """Save conversion result to WAV."""
        if result.audio is None or not result.can_route:
            return None

        import wave

        if name is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = f"rvc_{ts}"

        path = os.path.join(self._output_dir, f"{name}.wav")
        int_audio = (result.audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(result.sample_rate)
            wf.writeframes(int_audio.tobytes())

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
        self._profile: Optional[VoiceProfile] = None
        self._device = "cpu"
        self._output_dir = os.path.join(get_config_dir(), "generations", "voice_clone")
        os.makedirs(self._output_dir, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self._sovits_model is not None

    def load_model(self, profile: VoiceProfile, device: str = "cuda",
                   progress_callback: Optional[Callable] = None):
        """Load GPT-SoVITS model pair."""
        try:
            if progress_callback:
                progress_callback(0.1, "Loading SoVITS model...")

            ensure_voice_profile_allowed(profile, VOICE_OPERATION_CLONE)

            # Load SoVITS model
            self._sovits_model = load_voice_checkpoint(profile, profile.model_path, device)

            # Look for corresponding GPT model
            gpt_path = profile.config_path
            if gpt_path and os.path.isfile(gpt_path):
                if progress_callback:
                    progress_callback(0.5, "Loading GPT model...")
                self._gpt_model = load_voice_checkpoint(profile, gpt_path, device)

            self._model_path = profile.model_path
            self._profile = profile
            self._device = device

            if progress_callback:
                progress_callback(1.0, "GPT-SoVITS loaded")

        except Exception as e:
            self._sovits_model = None
            self._gpt_model = None
            raise RuntimeError(f"Failed to load GPT-SoVITS: {e}") from e

    def unload_model(self):
        self._sovits_model = None
        self._gpt_model = None
        self._model_path = None
        self._profile = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def clone(self, params: VoiceCloneParams,
              progress_callback: Optional[Callable] = None) -> VoiceResult:
        """Generate speech/singing in the cloned voice."""
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

            # Load reference audio
            ref_audio = self._load_reference(params.ref_audio_path, params.sample_rate)
            if ref_audio is None:
                return VoiceResult(
                    error="Failed to load reference audio",
                    output_kind="error",
                    can_route=False,
                )

            if not params.allow_demo_output:
                return VoiceResult(
                    error=(
                        "GPT-SoVITS inference pipeline is not available yet. "
                        "Demo voice synthesis must be explicitly enabled."
                    ),
                    generation_time=time.time() - t0,
                    output_kind="error",
                    can_route=False,
                )

            if progress_callback:
                progress_callback(0.3, "Extracting voice features...")

            # Extract features from reference
            ref_features = self._extract_reference_features(ref_audio, params)

            if progress_callback:
                progress_callback(0.5, "Generating speech...")

            # Run GPT for semantic tokens
            semantic_tokens = self._run_gpt(params.text, ref_features, params)

            if progress_callback:
                progress_callback(0.7, "Synthesizing audio...")

            # Run SoVITS for audio generation
            audio = self._run_sovits(semantic_tokens, ref_features, params)

            # Speed adjustment
            if params.speed != 1.0 and audio is not None:
                audio = self._change_speed(audio, params.speed, params.sample_rate)

            # Normalize
            if audio is not None:
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak * 0.95

            gen_time = time.time() - t0
            duration = len(audio) / params.sample_rate if audio is not None else 0

            if progress_callback:
                progress_callback(1.0, "Done")

            return VoiceResult(
                audio=audio,
                sample_rate=params.sample_rate,
                duration=duration,
                generation_time=gen_time,
                is_demo=True,
                output_kind="demo",
                can_route=True,
                provenance={
                    "module": "vocal_suite",
                    "operation": "gpt_sovits_clone",
                    "model_id": "gpt-sovits-v2",
                    "model_name": self._profile.name if self._profile else "",
                    "model_source": self._profile.source if self._profile else "",
                    "model_revision": self._profile.source_revision if self._profile else "",
                    "model_hash": _safe_file_hash(self._model_path),
                    "model_license": self._profile.license if self._profile else "",
                    "prompt": params.text,
                    "parameters": asdict(params),
                    "source_asset_ids": [self._profile.id] if self._profile else [],
                    "source_paths": [params.ref_audio_path] if params.ref_audio_path else [],
                    "output_kind": "demo",
                    "extra": {
                        "model_path": self._model_path or "",
                        "loaded_voice_profile": voice_profile_provenance(self._profile),
                    },
                },
            )

        except Exception as e:
            return VoiceResult(
                error=str(e),
                generation_time=time.time() - t0,
                output_kind="error",
                can_route=False,
            )

    def _load_reference(self, path: str, sr: int) -> Optional[np.ndarray]:
        if not path or not os.path.isfile(path):
            return None
        try:
            import librosa
            audio, _ = librosa.load(path, sr=sr, mono=True)
            return audio
        except ImportError:
            return None

    def _extract_reference_features(self, audio: np.ndarray,
                                    params: VoiceCloneParams) -> dict:
        """Extract voice characteristics from reference audio."""
        return {
            "audio": audio,
            "text": params.ref_text,
            "language": params.language,
        }

    def _run_gpt(self, text: str, ref_features: dict,
                 params: VoiceCloneParams) -> np.ndarray:
        """Run GPT model to generate semantic tokens."""
        # Placeholder: return random tokens
        # Real implementation passes text + ref through GPT for token prediction
        n_tokens = len(text.split()) * 20  # rough estimate
        return np.random.randn(n_tokens).astype(np.float32)

    def _run_sovits(self, semantic_tokens: np.ndarray, ref_features: dict,
                    params: VoiceCloneParams) -> np.ndarray:
        """Run SoVITS model to synthesize audio from semantic tokens."""
        # Placeholder: generate simple sine wave based on token count
        # Real implementation runs the SoVITS decoder
        duration = len(semantic_tokens) * 0.02  # ~20ms per token
        n_samples = int(duration * params.sample_rate)
        t = np.arange(n_samples) / params.sample_rate
        # Simple placeholder audio
        audio = 0.3 * np.sin(2 * np.pi * 220 * t) * np.exp(-t / duration)
        return audio.astype(np.float32)

    def _change_speed(self, audio: np.ndarray, speed: float,
                      sr: int) -> np.ndarray:
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=speed)
        except ImportError:
            return audio

    def save_output(
        self,
        result: VoiceResult,
        name: Optional[str] = None,
        profile: Optional[VoiceProfile] = None,
    ) -> Optional[str]:
        if result.audio is None or not result.can_route:
            return None
        import wave
        if name is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = f"clone_{ts}"
        path = os.path.join(self._output_dir, f"{name}.wav")
        int_audio = (result.audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(result.sample_rate)
            wf.writeframes(int_audio.tobytes())
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


def load_model(cache_dir: str = None, **kwargs) -> RVCEngine:
    """Initialize RVC engine. Called by ModelManager._dynamic_load().
    Note: actual voice profiles are loaded separately via engine.load_model(profile)."""
    from core.deps import ensure
    ensure("torch")
    engine = get_rvc()
    return engine
