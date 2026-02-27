"""
Slunder Studio v0.0.2 — Demucs Engine
Audio stem separation using Demucs (htdemucs) for isolating
vocals, drums, bass, and other instruments from mixed audio.
"""
import os
import time
from typing import Optional, Callable
from dataclasses import dataclass, field

import numpy as np

from core.settings import get_config_dir


STEM_NAMES = ["drums", "bass", "other", "vocals"]


@dataclass
class StemResult:
    """Individual stem from separation."""
    name: str = ""
    audio: Optional[np.ndarray] = None  # float32, shape (samples, channels)
    sample_rate: int = 44100
    file_path: Optional[str] = None


@dataclass
class SeparationResult:
    """Complete separation result with all stems."""
    stems: list[StemResult] = field(default_factory=list)
    sample_rate: int = 44100
    duration: float = 0.0
    separation_time: float = 0.0
    model_name: str = ""
    error: Optional[str] = None

    def get_stem(self, name: str) -> Optional[StemResult]:
        for s in self.stems:
            if s.name.lower() == name.lower():
                return s
        return None

    @property
    def vocals(self) -> Optional[StemResult]:
        return self.get_stem("vocals")

    @property
    def drums(self) -> Optional[StemResult]:
        return self.get_stem("drums")

    @property
    def bass(self) -> Optional[StemResult]:
        return self.get_stem("bass")

    @property
    def other(self) -> Optional[StemResult]:
        return self.get_stem("other")

    @property
    def instrumental(self) -> Optional[np.ndarray]:
        """Combine all non-vocal stems into instrumental."""
        non_vocal = [s.audio for s in self.stems
                     if s.name != "vocals" and s.audio is not None]
        if not non_vocal:
            return None
        result = np.zeros_like(non_vocal[0])
        for stem in non_vocal:
            result += stem[:len(result)]
        return result


class DemucsEngine:
    """
    Demucs stem separation engine.
    Supports htdemucs (default), htdemucs_ft (fine-tuned), and mdx variants.
    """

    MODELS = {
        "htdemucs": "Default 4-stem separator (vocals/drums/bass/other)",
        "htdemucs_ft": "Fine-tuned variant with better vocal isolation",
        "htdemucs_6s": "6-stem variant (adds piano and guitar)",
        "mdx_extra": "MDX-Net architecture for extra quality",
    }

    def __init__(self):
        self._model = None
        self._model_name: Optional[str] = None
        self._device = "cpu"
        self._output_dir = os.path.join(get_config_dir(), "generations", "stems")
        os.makedirs(self._output_dir, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, model_name: str = "htdemucs",
                   device: str = "cuda",
                   progress_callback: Optional[Callable] = None):
        """Load a Demucs model."""
        from core.deps import ensure
        ensure("torch", "torchaudio", "demucs")
        try:
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import BagOfModels

            if progress_callback:
                progress_callback(0.1, f"Loading {model_name}...")

            model = get_model(model_name)
            if isinstance(model, BagOfModels):
                for sub in model.models:
                    sub.to(device)
            else:
                model.to(device)

            self._model = model
            self._model_name = model_name
            self._device = device

            if progress_callback:
                progress_callback(1.0, f"{model_name} loaded")

        except ImportError as e:
            raise RuntimeError(f"Failed to install demucs: {e}") from e
        except Exception as e:
            self._model = None
            raise RuntimeError(f"Failed to load Demucs model: {e}") from e

    def unload_model(self):
        """Release model resources."""
        self._model = None
        self._model_name = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def separate(self, input_path: str,
                 progress_callback: Optional[Callable] = None) -> SeparationResult:
        """Separate audio file into stems."""
        if not self.is_loaded:
            return SeparationResult(error="Demucs model not loaded")

        t0 = time.time()

        try:
            import torch
            import torchaudio
            from demucs.apply import apply_model

            if progress_callback:
                progress_callback(0.05, "Loading audio...")

            # Load audio
            wav, sr = torchaudio.load(input_path)

            # Resample to model sample rate if needed
            model_sr = self._model.samplerate
            if sr != model_sr:
                wav = torchaudio.functional.resample(wav, sr, model_sr)
                sr = model_sr

            # Ensure stereo
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2]

            # Add batch dimension
            wav = wav.unsqueeze(0).to(self._device)

            if progress_callback:
                progress_callback(0.1, "Separating stems...")

            # Run separation
            with torch.no_grad():
                sources = apply_model(
                    self._model, wav,
                    device=self._device,
                    progress=True,
                    num_workers=0,
                )

            if progress_callback:
                progress_callback(0.85, "Saving stems...")

            # Build result
            source_names = self._model.sources
            stems = []

            for i, name in enumerate(source_names):
                stem_audio = sources[0, i].cpu().numpy().T  # (samples, channels)

                # Save to file
                stem_path = self._save_stem(stem_audio, sr, name, input_path)

                stems.append(StemResult(
                    name=name,
                    audio=stem_audio.astype(np.float32),
                    sample_rate=sr,
                    file_path=stem_path,
                ))

            sep_time = time.time() - t0
            duration = wav.shape[-1] / sr

            if progress_callback:
                progress_callback(1.0, f"Separation complete ({sep_time:.1f}s)")

            return SeparationResult(
                stems=stems,
                sample_rate=sr,
                duration=duration,
                separation_time=sep_time,
                model_name=self._model_name or "",
            )

        except Exception as e:
            return SeparationResult(
                error=str(e),
                separation_time=time.time() - t0,
            )

    def separate_numpy(self, audio: np.ndarray, sample_rate: int,
                       progress_callback: Optional[Callable] = None) -> SeparationResult:
        """Separate numpy audio array into stems."""
        if not self.is_loaded:
            return SeparationResult(error="Demucs model not loaded")

        t0 = time.time()

        try:
            import torch
            from demucs.apply import apply_model

            if progress_callback:
                progress_callback(0.05, "Preparing audio...")

            # Convert to torch tensor
            if audio.ndim == 1:
                wav = torch.from_numpy(audio).float().unsqueeze(0).repeat(2, 1)
            else:
                wav = torch.from_numpy(audio.T).float()
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)

            # Resample if needed
            model_sr = self._model.samplerate
            if sample_rate != model_sr:
                try:
                    import torchaudio
                    wav = torchaudio.functional.resample(wav, sample_rate, model_sr)
                except ImportError:
                    # torchaudio not available — skip resampling, model will handle mismatch
                    pass

            wav = wav.unsqueeze(0).to(self._device)

            if progress_callback:
                progress_callback(0.1, "Separating...")

            with torch.no_grad():
                sources = apply_model(self._model, wav, device=self._device,
                                      num_workers=0)

            source_names = self._model.sources
            stems = []

            for i, name in enumerate(source_names):
                stem_audio = sources[0, i].cpu().numpy().T
                stems.append(StemResult(
                    name=name,
                    audio=stem_audio.astype(np.float32),
                    sample_rate=model_sr,
                ))

            if progress_callback:
                progress_callback(1.0, "Done")

            return SeparationResult(
                stems=stems,
                sample_rate=model_sr,
                duration=wav.shape[-1] / model_sr,
                separation_time=time.time() - t0,
                model_name=self._model_name or "",
            )

        except Exception as e:
            return SeparationResult(error=str(e), separation_time=time.time() - t0)

    def _save_stem(self, audio: np.ndarray, sr: int, stem_name: str,
                   input_path: str) -> str:
        """Save a stem to WAV file."""
        import wave

        base = os.path.splitext(os.path.basename(input_path))[0]
        ts = time.strftime("%Y%m%d_%H%M%S")
        stem_dir = os.path.join(self._output_dir, f"{base}_{ts}")
        os.makedirs(stem_dir, exist_ok=True)
        path = os.path.join(stem_dir, f"{stem_name}.wav")

        # Convert to int16
        int_audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(path, "w") as wf:
            wf.setnchannels(audio.shape[1] if audio.ndim == 2 else 1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int_audio.tobytes())

        return path


# ── High-Level ─────────────────────────────────────────────────────────────────

_engine: Optional[DemucsEngine] = None


def get_demucs() -> DemucsEngine:
    global _engine
    if _engine is None:
        _engine = DemucsEngine()
    return _engine


def separate_stems(input_path: str,
                   model_name: str = "htdemucs",
                   progress_callback: Optional[Callable] = None) -> SeparationResult:
    """
    Separate audio into stems. Auto-loads model if needed.
    Called by InferenceWorker.
    """
    engine = get_demucs()

    if not engine.is_loaded or engine._model_name != model_name:
        try:
            if progress_callback:
                progress_callback(0.0, f"Loading {model_name}...")
            engine.load_model(model_name, progress_callback=progress_callback)
        except Exception as e:
            return SeparationResult(error=str(e))

    return engine.separate(input_path, progress_callback)


def load_model(cache_dir: str = None, **kwargs) -> DemucsEngine:
    """Load Demucs model. Called by ModelManager._dynamic_load()."""
    from core.deps import ensure
    ensure("torch")
    ensure("demucs")
    engine = get_demucs()
    engine.load_model()  # loads htdemucs by default
    return engine
