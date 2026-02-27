"""
Slunder Studio v0.0.2 — SFX Engine
Text-to-SFX generation using Stable Audio Open for creating sound effects,
ambient textures, and audio layers from text prompts.
"""
import os
import time
from typing import Optional, Callable
from dataclasses import dataclass, field

import numpy as np

from core.settings import get_config_dir


@dataclass
class SFXParams:
    """Parameters for SFX generation."""
    prompt: str = ""
    negative_prompt: str = ""
    duration: float = 5.0  # seconds (max ~47s for Stable Audio Open)
    cfg_scale: float = 7.0
    steps: int = 100
    seed: Optional[int] = None
    sample_rate: int = 44100
    batch_size: int = 1


@dataclass
class SFXResult:
    """Result from SFX generation."""
    audio: Optional[np.ndarray] = None  # float32 stereo
    sample_rate: int = 44100
    duration: float = 0.0
    generation_time: float = 0.0
    seed: int = 0
    file_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SFXBatchResult:
    """Batch generation results."""
    results: list[SFXResult] = field(default_factory=list)
    total_time: float = 0.0
    error: Optional[str] = None


# ── SFX Prompt Presets ─────────────────────────────────────────────────────────

SFX_CATEGORIES = {
    "Nature": [
        "rain falling on leaves", "thunder rumbling in distance",
        "ocean waves crashing on shore", "wind howling through trees",
        "birds singing in forest", "crackling campfire",
        "flowing river water", "rainforest ambient",
    ],
    "Urban": [
        "city traffic ambience", "subway train arriving",
        "coffee shop background noise", "construction site",
        "crowd cheering in stadium", "footsteps on concrete",
        "car engine starting", "sirens in the distance",
    ],
    "Sci-Fi": [
        "spaceship engine humming", "laser beam firing",
        "teleportation sound effect", "robot servo motors",
        "alien communication signal", "futuristic door opening",
        "energy shield activation", "warp drive engaging",
    ],
    "Musical": [
        "vinyl record crackle", "tape hiss noise",
        "808 bass hit", "orchestral hit",
        "reverse cymbal swell", "ambient pad texture",
        "glitch electronic texture", "lo-fi vinyl noise bed",
    ],
    "UI / Game": [
        "button click sound", "notification chime",
        "level up fanfare", "coin collect sound",
        "explosion boom", "sword slash whoosh",
        "magic spell casting", "menu select beep",
    ],
    "Foley": [
        "glass breaking", "door creaking open",
        "paper rustling", "keyboard typing",
        "zipper opening", "cloth rustling",
        "metal clang", "wooden floor creaking",
    ],
}


class SFXEngine:
    """
    Stable Audio Open engine for text-to-SFX generation.
    Uses diffusion-based audio generation from text prompts.
    """

    def __init__(self):
        self._model = None
        self._model_config = None
        self._device = "cpu"
        self._output_dir = os.path.join(get_config_dir(), "generations", "sfx")
        os.makedirs(self._output_dir, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self, model_path: str = "stabilityai/stable-audio-open-1.0",
                   device: str = "cuda",
                   progress_callback: Optional[Callable] = None):
        """Load Stable Audio Open model."""
        from core.deps import ensure
        ensure("torch")
        ensure("stable_audio_tools", pip_name="stable-audio-tools")
        try:
            import torch
            from stable_audio_tools import get_pretrained_model

            if progress_callback:
                progress_callback(0.1, "Loading Stable Audio Open...")

            model, model_config = get_pretrained_model(model_path)
            model = model.to(device)

            self._model = model
            self._model_config = model_config
            self._device = device

            if progress_callback:
                progress_callback(1.0, "Stable Audio Open loaded")

        except ImportError as e:
            raise RuntimeError(f"Failed to load Stable Audio dependencies: {e}") from e
        except Exception as e:
            self._model = None
            raise RuntimeError(f"Failed to load Stable Audio: {e}") from e

    def unload_model(self):
        self._model = None
        self._model_config = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def generate(self, params: SFXParams,
                 progress_callback: Optional[Callable] = None) -> SFXResult:
        """Generate a single SFX from text prompt."""
        if not self.is_loaded:
            # Fallback to noise-based generation
            return self._generate_fallback(params, progress_callback)

        t0 = time.time()

        try:
            import torch
            from stable_audio_tools.inference.generation import generate_diffusion_cond

            if progress_callback:
                progress_callback(0.1, "Preparing generation...")

            sample_rate = self._model_config.get("sample_rate", params.sample_rate)
            sample_size = self._model_config.get("sample_size", int(params.duration * sample_rate))

            conditioning = [{
                "prompt": params.prompt,
                "seconds_start": 0,
                "seconds_total": params.duration,
            }]

            if params.negative_prompt:
                negative_conditioning = [{
                    "prompt": params.negative_prompt,
                    "seconds_start": 0,
                    "seconds_total": params.duration,
                }]
            else:
                negative_conditioning = None

            seed = params.seed if params.seed is not None else int(time.time()) % (2**31)
            torch.manual_seed(seed)

            if progress_callback:
                progress_callback(0.2, "Running diffusion...")

            with torch.no_grad():
                output = generate_diffusion_cond(
                    self._model,
                    conditioning=conditioning,
                    negative_conditioning=negative_conditioning,
                    steps=params.steps,
                    cfg_scale=params.cfg_scale,
                    sample_size=sample_size,
                    sigma_min=0.3,
                    sigma_max=500,
                    sampler_type="dpmpp-3m-sde",
                    device=self._device,
                    seed=seed,
                )

            audio = output.squeeze().cpu().numpy()
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio])
            elif audio.shape[0] == 2:
                audio = audio.T

            # Normalize
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak * 0.95

            gen_time = time.time() - t0

            # Save
            file_path = self._save_sfx(audio, sample_rate, params.prompt)

            if progress_callback:
                progress_callback(1.0, "Done")

            return SFXResult(
                audio=audio.astype(np.float32),
                sample_rate=sample_rate,
                duration=len(audio) / sample_rate,
                generation_time=gen_time,
                seed=seed,
                file_path=file_path,
            )

        except Exception as e:
            return SFXResult(error=str(e), generation_time=time.time() - t0)

    def generate_batch(self, params: SFXParams,
                       progress_callback: Optional[Callable] = None) -> SFXBatchResult:
        """Generate multiple SFX variations."""
        import random
        t0 = time.time()
        results = []

        for i in range(params.batch_size):
            if progress_callback:
                progress_callback(i / params.batch_size, f"Generating {i + 1}/{params.batch_size}...")

            p = SFXParams(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                duration=params.duration,
                cfg_scale=params.cfg_scale,
                steps=params.steps,
                seed=params.seed + i if params.seed is not None else random.randint(0, 2**31 - 1),
                sample_rate=params.sample_rate,
                batch_size=1,
            )
            result = self.generate(p)
            results.append(result)

        if progress_callback:
            progress_callback(1.0, f"Generated {len(results)} SFX")

        return SFXBatchResult(results=results, total_time=time.time() - t0)

    def _generate_fallback(self, params: SFXParams,
                           progress_callback: Optional[Callable] = None) -> SFXResult:
        """Generate placeholder SFX using noise synthesis when model is unavailable."""
        import random
        t0 = time.time()

        if progress_callback:
            progress_callback(0.2, "Generating placeholder SFX...")

        seed = params.seed if params.seed is not None else int(time.time()) % (2**31)
        random.seed(seed)
        np.random.seed(seed % (2**31))

        sr = params.sample_rate
        n_samples = int(params.duration * sr)
        t = np.arange(n_samples) / sr

        prompt_lower = params.prompt.lower()

        # Generate different textures based on prompt keywords
        if any(w in prompt_lower for w in ["rain", "water", "ocean", "river"]):
            # Filtered noise for water sounds
            noise = np.random.randn(n_samples) * 0.3
            # Simple low-pass via running average
            kernel = int(sr * 0.002)
            if kernel > 1:
                noise = np.convolve(noise, np.ones(kernel) / kernel, mode="same")
            audio = noise * (1 + 0.3 * np.sin(2 * np.pi * 0.2 * t))
        elif any(w in prompt_lower for w in ["explosion", "boom", "hit", "impact"]):
            # Impact: short burst + decay
            audio = np.random.randn(n_samples) * np.exp(-t * 8) * 0.8
            audio += 0.5 * np.sin(2 * np.pi * 40 * t) * np.exp(-t * 5)
        elif any(w in prompt_lower for w in ["beep", "chime", "bell", "click"]):
            # Tonal beep
            freq = random.choice([440, 880, 1320, 1760])
            audio = np.sin(2 * np.pi * freq * t) * np.exp(-t * 4) * 0.6
        elif any(w in prompt_lower for w in ["wind", "air", "whoosh"]):
            # Modulated noise
            noise = np.random.randn(n_samples)
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
            kernel = int(sr * 0.005)
            if kernel > 1:
                noise = np.convolve(noise, np.ones(kernel) / kernel, mode="same")
            audio = noise * mod * 0.4
        elif any(w in prompt_lower for w in ["engine", "motor", "hum", "drone"]):
            # Harmonic drone
            freq = random.uniform(60, 120)
            audio = np.zeros(n_samples)
            for h in range(1, 8):
                amp = 0.3 / h
                audio += amp * np.sin(2 * np.pi * freq * h * t)
            audio *= 0.5 + 0.1 * np.sin(2 * np.pi * 0.3 * t)
        else:
            # Generic ambient texture
            audio = np.random.randn(n_samples) * 0.2
            # Add some tonal content
            freq = random.uniform(200, 800)
            audio += 0.15 * np.sin(2 * np.pi * freq * t) * np.exp(-t / params.duration)

        # Fade in/out
        fade_len = min(int(0.05 * sr), n_samples // 4)
        if fade_len > 0:
            audio[:fade_len] *= np.linspace(0, 1, fade_len)
            audio[-fade_len:] *= np.linspace(1, 0, fade_len)

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.85

        stereo = np.column_stack([audio, audio]).astype(np.float32)
        gen_time = time.time() - t0
        file_path = self._save_sfx(stereo, sr, params.prompt)

        if progress_callback:
            progress_callback(1.0, "Done (placeholder)")

        return SFXResult(
            audio=stereo, sample_rate=sr,
            duration=params.duration,
            generation_time=gen_time,
            seed=seed, file_path=file_path,
        )

    def _save_sfx(self, audio: np.ndarray, sr: int, prompt: str) -> str:
        """Save SFX to WAV."""
        import wave
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt[:40])
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._output_dir, f"sfx_{safe_name}_{ts}.wav")

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        int_audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int_audio.tobytes())

        return path


# ── High-Level ─────────────────────────────────────────────────────────────────

_engine: Optional[SFXEngine] = None


def get_sfx_engine() -> SFXEngine:
    global _engine
    if _engine is None:
        _engine = SFXEngine()
    return _engine


def generate_sfx(params: SFXParams,
                 progress_callback: Optional[Callable] = None) -> SFXResult:
    """Generate SFX. Uses model if loaded, falls back to synthesis."""
    engine = get_sfx_engine()
    return engine.generate(params, progress_callback)


def load_model(cache_dir: str = None, source: str = None, **kwargs) -> SFXEngine:
    """Load Stable Audio Open model. Called by ModelManager._dynamic_load()."""
    from core.deps import ensure
    ensure("torch")
    ensure("stable_audio_tools")
    engine = get_sfx_engine()
    model_path = source or "stabilityai/stable-audio-open-1.0"
    engine.load_model(model_path)
    return engine
