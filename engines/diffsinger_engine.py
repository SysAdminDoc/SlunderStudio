"""
Slunder Studio — DiffSinger Engine
Singing voice synthesis from lyrics + MIDI using DiffSinger/ONNX models.
Converts phoneme-aligned lyrics into natural singing audio.
"""
import os
import time
import json
from typing import Optional, Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from core.provenance import file_sha256, write_provenance_sidecar
from core.settings import get_config_dir
from core.voice_bank import VoiceProfile


@dataclass
class SingParams:
    """Parameters for singing synthesis."""
    lyrics: str = ""
    notes: list[dict] = field(default_factory=list)  # [{pitch, start, end, text}]
    tempo: float = 120.0
    key: str = "C4"
    speaker_id: int = 0
    pitch_shift: int = 0  # semitones
    breathiness: float = 0.0  # 0.0 - 1.0
    voicing: float = 1.0  # 0.0 - 1.0
    tension: float = 0.5  # 0.0 - 1.0
    gender: float = 0.0  # -1.0 (feminine) to 1.0 (masculine)
    velocity: float = 1.0  # dynamics
    vibrato_depth: float = 0.5
    vibrato_rate: float = 5.5  # Hz
    # Retained for compatibility with callers that serialize SingParams. The
    # loaded model's declared rate is authoritative for generated audio.
    sample_rate: int = 44100


@dataclass
class SingResult:
    """Result from singing synthesis."""
    audio: Optional[np.ndarray] = None  # float32 mono
    sample_rate: int = 44100
    duration: float = 0.0
    generation_time: float = 0.0
    error: Optional[str] = None
    provenance: dict = field(default_factory=dict)
    provenance_path: str = ""


def _safe_file_hash(path: Optional[str]) -> str:
    try:
        return file_sha256(path) if path and os.path.isfile(path) else ""
    except Exception:
        return ""


class DiffSingerEngine:
    """
    DiffSinger singing synthesis engine.
    Uses ONNX runtime for inference with pre-trained vocal models.
    """

    def __init__(self):
        self._session = None
        self._config = None
        self._model_path: Optional[str] = None
        self._sample_rate = 0
        self._hop_size = 0
        self._phonemizer = None
        self._output_dir = os.path.join(get_config_dir(), "generations", "vocals")
        os.makedirs(self._output_dir, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def load_model(self, model_path: str,
                   progress_callback: Optional[Callable] = None):
        """Load a DiffSinger ONNX model."""
        try:
            import onnxruntime as ort

            if progress_callback:
                progress_callback(0.1, "Loading DiffSinger model...")

            self._config = self._read_model_config(model_path)
            # Frame timing must come from the model, not an assumption.
            self._sample_rate, self._hop_size = self._resolve_frame_timing(
                self._config, model_path
            )

            if progress_callback:
                progress_callback(0.4, "Creating inference session...")

            # Create ONNX session
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(model_path, providers=providers)
            self._model_path = model_path

            if progress_callback:
                progress_callback(0.8, "Loading phonemizer...")

            # Initialize phonemizer
            self._init_phonemizer()

            if progress_callback:
                progress_callback(1.0, "DiffSinger ready")

        except Exception as e:
            self._session = None
            raise RuntimeError(f"Failed to load DiffSinger: {e}") from e

    # ── Model frame timing ─────────────────────────────────────────────────────

    # Keys used by DiffSinger / OpenUtau ONNX model configs.
    SAMPLE_RATE_KEYS = ("audio_sample_rate", "sample_rate", "sampling_rate")
    HOP_SIZE_KEYS = ("hop_size", "hop_length", "hop")

    @staticmethod
    def _read_model_config(model_path: str) -> dict:
        """Read the model's config from any of its documented filenames."""
        directory = os.path.dirname(model_path)
        for name in ("dsconfig.yaml", "dsconfig.yml", "config.yaml", "config.json"):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            try:
                if name.endswith(".json"):
                    import json

                    with open(path, encoding="utf-8") as handle:
                        data = json.load(handle)
                else:
                    import yaml

                    with open(path, encoding="utf-8") as handle:
                        data = yaml.safe_load(handle)
            except Exception:
                continue
            if isinstance(data, dict):
                return data
        return {}

    @classmethod
    def _resolve_frame_timing(cls, config: dict, model_path: str) -> tuple[int, int]:
        """Return (sample_rate, hop_size) from the model config.

        Fails explicitly rather than guessing: a wrong hop size silently
        misplaces every pitch event.
        """
        def _lookup(keys):
            for key in keys:
                if key in config:
                    return config[key]
            return None

        raw_rate = _lookup(cls.SAMPLE_RATE_KEYS)
        raw_hop = _lookup(cls.HOP_SIZE_KEYS)
        if raw_rate is None or raw_hop is None:
            raise RuntimeError(
                "DiffSinger model config is missing frame timing "
                f"({', '.join(cls.SAMPLE_RATE_KEYS)} and "
                f"{', '.join(cls.HOP_SIZE_KEYS)}) for {os.path.basename(model_path)}. "
                "Reinstall the model with its dsconfig."
            )
        try:
            sample_rate = int(raw_rate)
            hop_size = int(raw_hop)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"DiffSinger model config has non-numeric frame timing: {exc}"
            ) from exc
        if sample_rate <= 0 or hop_size <= 0:
            raise RuntimeError(
                "DiffSinger model config declares an invalid frame timing "
                f"(sample_rate={sample_rate}, hop_size={hop_size})."
            )
        return sample_rate, hop_size

    @property
    def frame_period_sec(self) -> float:
        """Seconds per model frame, from the loaded model's own configuration."""
        sample_rate = getattr(self, "_sample_rate", 0)
        hop_size = getattr(self, "_hop_size", 0)
        if not sample_rate or not hop_size:
            raise RuntimeError(
                "DiffSinger frame timing is unavailable; load a model first."
            )
        return hop_size / sample_rate

    def time_to_frame(self, seconds: float) -> int:
        """Convert a time in seconds to the model's frame index."""
        return int(round(max(0.0, float(seconds)) / self.frame_period_sec))

    def _init_phonemizer(self):
        """Initialize text-to-phoneme converter."""
        try:
            from pypinyin import pinyin, Style
            self._phonemizer = "pypinyin"
        except ImportError:
            try:
                import g2p_en
                self._phonemizer = g2p_en.G2p()
            except ImportError:
                self._phonemizer = None

    def unload_model(self):
        """Release model resources."""
        self._session = None
        self._config = None
        self._model_path = None
        self._sample_rate = 0
        self._hop_size = 0

    def synthesize(self, params: SingParams,
                   progress_callback: Optional[Callable] = None) -> SingResult:
        """Synthesize singing voice from parameters."""
        if not self.is_loaded:
            return SingResult(error="DiffSinger model not loaded")

        t0 = time.time()

        try:
            if progress_callback:
                progress_callback(0.1, "Preparing phonemes...")

            # Convert lyrics to phoneme sequence
            phonemes = self._lyrics_to_phonemes(params.lyrics)

            # Build note sequence from params
            note_seq = self._build_note_sequence(params)

            if progress_callback:
                progress_callback(0.3, "Running inference...")

            # Prepare model inputs
            inputs = self._prepare_inputs(phonemes, note_seq, params)

            # Run inference
            output = self._session.run(None, inputs)
            audio = output[0].squeeze().astype(np.float32)
            output_sample_rate = self._sample_rate

            # Post-process
            if params.pitch_shift != 0:
                audio = self._pitch_shift(audio, params.pitch_shift, output_sample_rate)

            # Apply gender shift
            if params.gender != 0.0:
                audio = self._apply_gender(audio, params.gender, output_sample_rate)

            if progress_callback:
                progress_callback(0.9, "Finalizing...")

            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.95

            duration = len(audio) / output_sample_rate
            gen_time = time.time() - t0

            if progress_callback:
                progress_callback(1.0, "Done")

            parameters = asdict(params)
            parameters["sample_rate"] = output_sample_rate
            return SingResult(
                audio=audio,
                sample_rate=output_sample_rate,
                duration=duration,
                generation_time=gen_time,
                provenance={
                    "module": "vocal_suite",
                    "operation": "diffsinger_synthesize",
                    "model_id": "diffsinger",
                    "model_hash": _safe_file_hash(self._model_path),
                    "lyrics": params.lyrics,
                    "parameters": parameters,
                    "output_kind": "model",
                    "extra": {"model_path": self._model_path or ""},
                },
            )

        except Exception as e:
            return SingResult(error=str(e), generation_time=time.time() - t0)

    def _lyrics_to_phonemes(self, lyrics: str) -> list[str]:
        """Convert lyrics text to phoneme sequence."""
        if self._phonemizer is None:
            # Simple fallback: split by characters/syllables
            return list(lyrics.replace(" ", " SP ").replace("\n", " SP "))

        if self._phonemizer == "pypinyin":
            from pypinyin import pinyin, Style
            result = []
            for char in lyrics:
                if char.strip():
                    py = pinyin(char, style=Style.TONE3)
                    result.extend([p[0] for p in py])
                else:
                    result.append("SP")
            return result
        else:
            # g2p_en
            return self._phonemizer(lyrics)

    def _build_note_sequence(self, params: SingParams) -> list[dict]:
        """Build note sequence from params or generate from lyrics length."""
        if params.notes:
            return params.notes

        # Auto-generate simple melody from lyrics
        words = params.lyrics.split()
        beat_dur = 60.0 / params.tempo
        notes = []
        t = 0.0

        base_pitch = 60  # C4
        for i, word in enumerate(words):
            pitch = base_pitch + [0, 2, 4, 5, 7, 5, 4, 2][i % 8]
            dur = beat_dur * (1.0 if len(word) <= 3 else 1.5)
            notes.append({
                "pitch": pitch,
                "start": t,
                "end": t + dur,
                "text": word,
            })
            t += dur

        return notes

    def build_f0_curve(self, notes: list[dict], n_frames: int) -> np.ndarray:
        """Frame-aligned F0 curve for the model's own frame rate."""
        n_frames = max(int(n_frames), 1)
        f0 = np.zeros((1, n_frames), dtype=np.float32)
        for note in notes:
            freq = 440.0 * (2.0 ** ((note["pitch"] - 69) / 12.0))
            start_frame = self.time_to_frame(note["start"])
            end_frame = self.time_to_frame(note["end"])
            start_frame = max(0, min(n_frames - 1, start_frame))
            end_frame = max(start_frame + 1, min(n_frames, end_frame))
            f0[0, start_frame:end_frame] = freq
        return f0

    def _prepare_inputs(self, phonemes: list[str], notes: list[dict],
                        params: SingParams) -> dict:
        """Prepare ONNX model inputs."""
        # Frame count follows the model's own frame period, so a note at t
        # seconds lands on frame round(t / frame_period).
        total_seconds = max((note["end"] for note in notes), default=0.0)
        n_frames = max(self.time_to_frame(total_seconds), 1)

        inputs = {}
        input_names = [inp.name for inp in self._session.get_inputs()]

        # Common DiffSinger inputs
        if "tokens" in input_names:
            # Phoneme tokens (simplified mapping)
            import hashlib as _hl
            token_ids = [int.from_bytes(_hl.md5(p.encode()).digest()[:2], "big") % 256 for p in phonemes]
            inputs["tokens"] = np.array([token_ids], dtype=np.int64)

        if "durations" in input_names:
            dur_per_phone = n_frames // max(len(phonemes), 1)
            durations = [dur_per_phone] * len(phonemes)
            inputs["durations"] = np.array([durations], dtype=np.int64)

        if "f0" in input_names:
            inputs["f0"] = self.build_f0_curve(notes, n_frames)

        if "speedup" in input_names:
            inputs["speedup"] = np.array([10], dtype=np.int64)

        if "spk_id" in input_names:
            inputs["spk_id"] = np.array([params.speaker_id], dtype=np.int64)

        return inputs

    def _pitch_shift(self, audio: np.ndarray, semitones: int,
                     sr: int) -> np.ndarray:
        """Pitch shift audio by semitones."""
        try:
            import librosa
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
        except ImportError:
            return audio  # No librosa, return unshifted

    def _apply_gender(self, audio: np.ndarray, gender: float,
                      sr: int) -> np.ndarray:
        """Apply gender shift via formant manipulation."""
        # Simplified: gender > 0 deepens, gender < 0 brightens
        shift = int(gender * -4)  # map to semitones
        if shift != 0:
            return self._pitch_shift(audio, shift, sr)
        return audio

    def save_output(self, result: SingResult, name: Optional[str] = None) -> Optional[str]:
        """Save synthesis result to WAV."""
        if result.audio is None:
            return None

        import wave

        if name is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = f"vocal_{ts}"

        path = os.path.join(self._output_dir, f"{name}.wav")
        int_audio = (result.audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(result.sample_rate)
            wf.writeframes(int_audio.tobytes())

        prov = result.provenance or {}
        sidecar = write_provenance_sidecar(
            path,
            module=prov.get("module", "vocal_suite"),
            operation=prov.get("operation", "diffsinger_synthesize"),
            model_id=prov.get("model_id", "diffsinger"),
            model_hash=prov.get("model_hash", ""),
            lyrics=prov.get("lyrics", ""),
            parameters=prov.get("parameters", {}),
            source_asset_ids=prov.get("source_asset_ids", []),
            source_paths=prov.get("source_paths", []),
            export_format="wav",
            output_kind=prov.get("output_kind", "model"),
            extra=prov.get("extra", {}),
        )
        result.provenance_path = str(sidecar)
        return path


# ── High-Level ─────────────────────────────────────────────────────────────────

_engine: Optional[DiffSingerEngine] = None


def get_diffsinger() -> DiffSingerEngine:
    global _engine
    if _engine is None:
        _engine = DiffSingerEngine()
    return _engine


def synthesize_vocals(params: SingParams,
                      voice_profile: Optional[VoiceProfile] = None,
                      progress_callback: Optional[Callable] = None) -> SingResult:
    """
    Synthesize vocals. Uses DiffSinger if loaded, else returns error.
    Called by InferenceWorker.
    """
    engine = get_diffsinger()

    if voice_profile:
        params.speaker_id = voice_profile.speaker_id
        params.pitch_shift = voice_profile.pitch_shift

    if engine.is_loaded:
        return engine.synthesize(params, progress_callback)
    else:
        return SingResult(error="DiffSinger model not loaded. Load a model from Model Hub.")


def load_model(cache_dir: str = None, **kwargs) -> DiffSingerEngine:
    """Load DiffSinger engine. Called by ModelManager._dynamic_load().
    Note: DiffSinger is pip_managed — voice models are loaded separately."""
    from core.deps import ensure
    ensure("onnxruntime")
    engine = get_diffsinger()
    return engine
