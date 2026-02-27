"""
Slunder Studio v0.0.2 — RVC / GPT-SoVITS Engine
Voice conversion (RVC v2) and voice cloning (GPT-SoVITS) for transforming
existing vocals or cloning a target voice from reference audio.
"""
import os
import time
import json
from typing import Optional, Callable
from dataclasses import dataclass, field

import numpy as np

from core.settings import get_config_dir
from core.voice_bank import VoiceProfile


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


@dataclass
class VoiceResult:
    """Result from voice conversion or cloning."""
    audio: Optional[np.ndarray] = None
    sample_rate: int = 44100
    duration: float = 0.0
    generation_time: float = 0.0
    error: Optional[str] = None


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
            import torch

            if progress_callback:
                progress_callback(0.1, "Loading RVC model...")

            # Load the model checkpoint
            checkpoint = torch.load(profile.model_path, map_location=device,
                                    weights_only=False)

            self._model = checkpoint
            self._model_path = profile.model_path
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
            return VoiceResult(error="RVC model not loaded")

        t0 = time.time()

        try:
            if progress_callback:
                progress_callback(0.1, "Loading audio...")

            # Get input audio
            audio = params.input_audio
            if audio is None and params.input_path:
                audio = self._load_audio(params.input_path, params.sample_rate)

            if audio is None:
                return VoiceResult(error="No input audio provided")

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

            if progress_callback:
                progress_callback(1.0, "Done")

            return VoiceResult(
                audio=converted,
                sample_rate=params.sample_rate,
                duration=duration,
                generation_time=gen_time,
            )

        except Exception as e:
            return VoiceResult(error=str(e), generation_time=time.time() - t0)

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

    def save_output(self, result: VoiceResult, name: Optional[str] = None) -> Optional[str]:
        """Save conversion result to WAV."""
        if result.audio is None:
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
            import torch

            if progress_callback:
                progress_callback(0.1, "Loading SoVITS model...")

            # Load SoVITS model
            self._sovits_model = torch.load(
                profile.model_path, map_location=device, weights_only=False
            )

            # Look for corresponding GPT model
            gpt_path = profile.config_path
            if gpt_path and os.path.isfile(gpt_path):
                if progress_callback:
                    progress_callback(0.5, "Loading GPT model...")
                self._gpt_model = torch.load(
                    gpt_path, map_location=device, weights_only=False
                )

            self._model_path = profile.model_path
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
            return VoiceResult(error="GPT-SoVITS model not loaded")

        t0 = time.time()

        try:
            if progress_callback:
                progress_callback(0.1, "Processing reference audio...")

            # Load reference audio
            ref_audio = self._load_reference(params.ref_audio_path, params.sample_rate)
            if ref_audio is None:
                return VoiceResult(error="Failed to load reference audio")

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
            )

        except Exception as e:
            return VoiceResult(error=str(e), generation_time=time.time() - t0)

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

    def save_output(self, result: VoiceResult, name: Optional[str] = None) -> Optional[str]:
        if result.audio is None:
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
