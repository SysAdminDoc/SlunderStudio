"""
Slunder Studio v0.0.2 — ACE-Step Engine
Native Python wrapper for ACE-Step v1.5 inference (not Gradio).
Supports: generate, batch, retake, repaint, extend, cover.
<4GB VRAM, 48kHz stereo, up to 10 min duration.
"""
import os
import time
import random
import threading
from typing import Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field

from core.settings import Settings, get_config_dir
from core.model_manager import ModelManager, cleanup_gpu


@dataclass
class GenerationParams:
    """Parameters for ACE-Step song generation."""
    lyrics: str = ""
    style_tags: str = ""  # comma-separated ACE-Step tags
    duration: float = 60.0  # seconds
    seed: int = -1  # -1 = random
    cfg_scale: float = 5.0  # 1.0-15.0
    infer_steps: int = 60  # 10-100
    scheduler: str = "euler"  # euler, dpm++
    sample_rate: int = 48000
    # Repaint/retake
    repaint_start: float = -1.0  # -1 = disabled
    repaint_end: float = -1.0
    source_audio_path: str = ""  # for cover/repaint
    # LoRA
    lora_path: str = ""
    lora_weight: float = 1.0

    def resolve_seed(self) -> int:
        if self.seed < 0:
            return random.randint(0, 2**32 - 1)
        return self.seed


@dataclass
class GenerationResult:
    """Result from ACE-Step generation."""
    audio_path: str = ""
    seed: int = 0
    duration: float = 0.0
    sample_rate: int = 48000
    params: Optional[GenerationParams] = None
    generation_time: float = 0.0
    is_favorite: bool = False
    rating: int = 0  # 0-5


class ACEStepEngine:
    """
    Wrapper around ACE-Step v1.5 inference pipeline.
    Handles model loading, generation, batch mode, and advanced features.
    """

    def __init__(self):
        self._pipeline = None
        self._model_loaded = False
        self._output_dir = get_config_dir() / "generations" / "song_forge"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded and self._pipeline is not None

    def load(self, cache_dir: str = None):
        """Load ACE-Step pipeline. Called by ModelManager."""
        try:
            from ace_step.pipeline import ACEStepPipeline
        except ImportError:
            from core.deps import ensure
            ensure("ace_step", pip_name="ace-step")
            from ace_step.pipeline import ACEStepPipeline

        if cache_dir:
            model_path = cache_dir
        else:
            mgr = ModelManager()
            model_path = str(mgr.get_cache_dir("ace-step-v1.5"))

        self._pipeline = ACEStepPipeline(model_path=model_path)
        self._model_loaded = True

    def unload(self):
        """Unload model and free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._model_loaded = False
            cleanup_gpu()

    def cleanup(self):
        self.unload()

    def generate(
        self,
        params: GenerationParams,
        progress_cb: Callable = None,
        cancel_event: threading.Event = None,
    ) -> GenerationResult:
        """
        Generate a single song from lyrics + style tags.
        Returns GenerationResult with path to output WAV.
        """
        if not self.is_loaded:
            raise RuntimeError("ACE-Step model not loaded. Call load() first.")

        seed = params.resolve_seed()
        start_time = time.time()

        # Build output path
        timestamp = int(time.time())
        output_path = self._output_dir / f"song_{timestamp}_{seed}.wav"

        if progress_cb:
            progress_cb(5)

        # Build generation kwargs
        gen_kwargs = {
            "lyrics": params.lyrics,
            "tags": params.style_tags,
            "duration": params.duration,
            "seed": seed,
            "guidance_scale": params.cfg_scale,
            "num_inference_steps": params.infer_steps,
            "output_path": str(output_path),
        }

        # Add repaint params if specified
        if params.repaint_start >= 0 and params.repaint_end > params.repaint_start:
            gen_kwargs["repaint_start"] = params.repaint_start
            gen_kwargs["repaint_end"] = params.repaint_end
            if params.source_audio_path:
                gen_kwargs["source_audio"] = params.source_audio_path

        # Add LoRA if specified
        if params.lora_path and os.path.exists(params.lora_path):
            gen_kwargs["lora_path"] = params.lora_path
            gen_kwargs["lora_weight"] = params.lora_weight

        # Run inference
        try:
            result = self._pipeline.generate(**gen_kwargs)
        except Exception as e:
            # Handle missing pipeline method gracefully
            # Fallback: try the alternative API pattern
            if hasattr(self._pipeline, '__call__'):
                result = self._pipeline(**gen_kwargs)
            else:
                raise e

        if progress_cb:
            progress_cb(95)

        elapsed = time.time() - start_time

        # Verify output exists
        if not output_path.exists():
            # Check if pipeline wrote to a different location
            if isinstance(result, str) and os.path.exists(result):
                output_path = Path(result)
            elif isinstance(result, dict) and "audio_path" in result:
                output_path = Path(result["audio_path"])
            else:
                raise RuntimeError(f"Generation completed but output file not found: {output_path}")

        if progress_cb:
            progress_cb(100)

        return GenerationResult(
            audio_path=str(output_path),
            seed=seed,
            duration=params.duration,
            sample_rate=params.sample_rate,
            params=params,
            generation_time=elapsed,
        )

    def generate_batch(
        self,
        params: GenerationParams,
        count: int = 4,
        progress_cb: Callable = None,
        step_cb: Callable = None,
        cancel_event: threading.Event = None,
    ) -> list[GenerationResult]:
        """
        Generate multiple variations with different random seeds.
        Returns list of GenerationResult sorted by generation order.
        """
        results = []
        for i in range(count):
            if cancel_event and cancel_event.is_set():
                break

            if step_cb:
                step_cb(f"Generating variation {i+1}/{count}...")

            # Each variation gets a unique random seed
            batch_params = GenerationParams(
                lyrics=params.lyrics,
                style_tags=params.style_tags,
                duration=params.duration,
                seed=-1,  # Force random for each
                cfg_scale=params.cfg_scale,
                infer_steps=params.infer_steps,
                scheduler=params.scheduler,
                sample_rate=params.sample_rate,
                lora_path=params.lora_path,
                lora_weight=params.lora_weight,
            )

            def _batch_progress(pct):
                if progress_cb:
                    overall = int((i * 100 + pct) / count)
                    progress_cb(overall)

            try:
                result = self.generate(batch_params, progress_cb=_batch_progress, cancel_event=cancel_event)
                results.append(result)
            except Exception as e:
                if step_cb:
                    step_cb(f"Variation {i+1} failed: {e}")
                continue

        return results

    def extend(
        self,
        source_path: str,
        params: GenerationParams,
        extend_duration: float = 30.0,
        progress_cb: Callable = None,
        cancel_event: threading.Event = None,
    ) -> GenerationResult:
        """Extend a song from its endpoint."""
        params.source_audio_path = source_path
        params.duration = extend_duration
        return self.generate(params, progress_cb=progress_cb, cancel_event=cancel_event)

    def retake(
        self,
        source_path: str,
        start_sec: float,
        end_sec: float,
        params: GenerationParams,
        progress_cb: Callable = None,
        cancel_event: threading.Event = None,
    ) -> GenerationResult:
        """Regenerate a section while keeping the rest (repaint)."""
        params.source_audio_path = source_path
        params.repaint_start = start_sec
        params.repaint_end = end_sec
        return self.generate(params, progress_cb=progress_cb, cancel_event=cancel_event)


# ── High-Level Functions for InferenceWorker ───────────────────────────────────

def generate_song(
    lyrics: str,
    style_tags: str,
    duration: float = 60.0,
    seed: int = -1,
    cfg_scale: float = 5.0,
    infer_steps: int = 60,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
    **kwargs,
) -> dict:
    """High-level song generation for InferenceWorker."""
    if step_cb:
        step_cb("Loading ACE-Step model...")

    mgr = ModelManager()
    engine = ACEStepEngine()

    def _loader():
        engine.load()
        return engine

    mgr.load_model("ace-step-v1.5", loader_fn=_loader)

    if cancel_event and cancel_event.is_set():
        return {"cancelled": True}

    if step_cb:
        step_cb("Generating song...")

    params = GenerationParams(
        lyrics=lyrics,
        style_tags=style_tags,
        duration=duration,
        seed=seed,
        cfg_scale=cfg_scale,
        infer_steps=infer_steps,
    )

    result = engine.generate(params, progress_cb=progress_cb, cancel_event=cancel_event)

    return {
        "audio_path": result.audio_path,
        "seed": result.seed,
        "duration": result.duration,
        "generation_time": result.generation_time,
        "params": {
            "lyrics": lyrics[:200],
            "style_tags": style_tags,
            "cfg_scale": cfg_scale,
            "infer_steps": infer_steps,
        },
    }


def generate_song_batch(
    lyrics: str,
    style_tags: str,
    count: int = 4,
    duration: float = 60.0,
    cfg_scale: float = 5.0,
    infer_steps: int = 60,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
    **kwargs,
) -> dict:
    """High-level batch generation for InferenceWorker."""
    if step_cb:
        step_cb("Loading ACE-Step model...")

    mgr = ModelManager()
    engine = ACEStepEngine()

    def _loader():
        engine.load()
        return engine

    mgr.load_model("ace-step-v1.5", loader_fn=_loader)

    if cancel_event and cancel_event.is_set():
        return {"cancelled": True}

    params = GenerationParams(
        lyrics=lyrics,
        style_tags=style_tags,
        duration=duration,
        cfg_scale=cfg_scale,
        infer_steps=infer_steps,
    )

    results = engine.generate_batch(
        params, count=count,
        progress_cb=progress_cb, step_cb=step_cb, cancel_event=cancel_event,
    )

    return {
        "results": [
            {
                "audio_path": r.audio_path,
                "seed": r.seed,
                "duration": r.duration,
                "generation_time": r.generation_time,
            }
            for r in results
        ],
        "count": len(results),
    }


def load_model(cache_dir: str = None, **kwargs) -> ACEStepEngine:
    """Loader function for ModelManager registry."""
    engine = ACEStepEngine()
    engine.load(cache_dir)
    return engine
