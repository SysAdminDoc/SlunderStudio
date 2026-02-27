"""
Slunder Studio v0.0.2 — ACE-Step Engine
Native Python wrapper for ACE-Step inference (not Gradio).
Supports: generate, batch, retake, repaint, extend.
<4GB VRAM, 48kHz stereo, up to 4 min duration.

Real upstream API (pip install ace-step):
  from acestep.pipeline_ace_step import ACEStepPipeline
  pipe = ACEStepPipeline(checkpoint_dir=path)
  result = pipe(prompt=..., lyrics=..., audio_duration=..., ...)
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
    style_tags: str = ""  # comma-separated ACE-Step tags (maps to 'prompt')
    duration: float = 60.0  # seconds (maps to 'audio_duration')
    seed: int = -1  # -1 = random (maps to 'manual_seeds')
    cfg_scale: float = 15.0  # 1.0-30.0 (maps to 'guidance_scale')
    infer_steps: int = 60  # 10-200 (maps to 'infer_step')
    scheduler: str = "euler"  # euler, heun, pingpong (maps to 'scheduler_type')
    sample_rate: int = 48000
    # Repaint/retake
    repaint_start: float = -1.0  # -1 = disabled
    repaint_end: float = -1.0
    source_audio_path: str = ""  # for cover/repaint
    # LoRA
    lora_path: str = ""  # maps to 'lora_name_or_path'
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
    Wrapper around ACE-Step inference pipeline.
    Uses the real acestep.pipeline_ace_step.ACEStepPipeline API.
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
            from acestep.pipeline_ace_step import ACEStepPipeline
        except ImportError:
            from core.deps import ensure
            ensure("acestep", pip_name="ace-step")
            from acestep.pipeline_ace_step import ACEStepPipeline

        if cache_dir:
            checkpoint_dir = cache_dir
        else:
            mgr = ModelManager()
            checkpoint_dir = str(mgr.get_cache_dir("ace-step-v1.5"))

        # ACEStepPipeline downloads from HuggingFace if checkpoint_dir is empty
        self._pipeline = ACEStepPipeline(checkpoint_dir=checkpoint_dir)
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

        save_dir = str(self._output_dir)

        if progress_cb:
            progress_cb(5)

        # Build kwargs matching the real ACEStepPipeline.__call__ signature
        gen_kwargs = {
            "prompt": params.style_tags,
            "lyrics": params.lyrics,
            "audio_duration": params.duration,
            "infer_step": params.infer_steps,
            "guidance_scale": params.cfg_scale,
            "scheduler_type": params.scheduler,
            "manual_seeds": str(seed),
            "save_path": save_dir,
            "format": "wav",
        }

        # Add repaint params if specified
        if params.repaint_start >= 0 and params.repaint_end > params.repaint_start:
            gen_kwargs["repaint_start"] = params.repaint_start
            gen_kwargs["repaint_end"] = params.repaint_end
            if params.source_audio_path:
                gen_kwargs["src_audio_path"] = params.source_audio_path

        # Add LoRA if specified
        if params.lora_path and os.path.exists(params.lora_path):
            gen_kwargs["lora_name_or_path"] = params.lora_path
            gen_kwargs["lora_weight"] = params.lora_weight

        if progress_cb:
            progress_cb(10)

        # ACEStepPipeline is callable
        result = self._pipeline(**gen_kwargs)

        if progress_cb:
            progress_cb(95)

        elapsed = time.time() - start_time

        output_path = self._find_output(save_dir, result)

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

    def _find_output(self, save_dir: str, pipeline_result) -> Path:
        """
        Locate the output file from pipeline result.
        Pipeline may return file paths, audio tensor, or save to save_path.
        """
        # If pipeline returned file path(s)
        if isinstance(pipeline_result, str) and os.path.isfile(pipeline_result):
            return Path(pipeline_result)
        if isinstance(pipeline_result, (list, tuple)):
            for item in pipeline_result:
                if isinstance(item, str) and os.path.isfile(item):
                    return Path(item)
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, str) and os.path.isfile(sub):
                            return Path(sub)
        if isinstance(pipeline_result, dict):
            for key in ("audio_path", "path", "output_path"):
                val = pipeline_result.get(key)
                if isinstance(val, str) and os.path.isfile(val):
                    return Path(val)

        # If pipeline returned audio tensor, save it ourselves
        try:
            import torch
            import numpy as np

            audio_data = None
            if isinstance(pipeline_result, torch.Tensor):
                audio_data = pipeline_result
            elif isinstance(pipeline_result, np.ndarray):
                audio_data = torch.from_numpy(pipeline_result)
            elif isinstance(pipeline_result, (list, tuple)):
                for item in pipeline_result:
                    if isinstance(item, (torch.Tensor, np.ndarray)):
                        audio_data = item if isinstance(item, torch.Tensor) else torch.from_numpy(item)
                        break

            if audio_data is not None:
                import torchaudio
                if audio_data.dim() == 1:
                    audio_data = audio_data.unsqueeze(0)
                if audio_data.dim() == 3:
                    audio_data = audio_data.squeeze(0)
                audio_data = audio_data.float().cpu()
                peak = audio_data.abs().max()
                if peak > 0:
                    audio_data = audio_data / peak * 0.95

                timestamp = int(time.time())
                out_path = Path(save_dir) / f"output_{timestamp}.wav"
                torchaudio.save(str(out_path), audio_data, 48000)
                return out_path
        except (ImportError, Exception):
            pass

        # Fallback: find most recent wav in save_dir
        wavs = sorted(Path(save_dir).glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        if wavs:
            return wavs[0]

        raise RuntimeError(f"Generation completed but no output file found in {save_dir}")

    def generate_batch(
        self,
        params: GenerationParams,
        count: int = 4,
        progress_cb: Callable = None,
        step_cb: Callable = None,
        cancel_event: threading.Event = None,
    ) -> list[GenerationResult]:
        """Generate multiple variations with different random seeds."""
        results = []
        for i in range(count):
            if cancel_event and cancel_event.is_set():
                break

            if step_cb:
                step_cb(f"Generating variation {i+1}/{count}...")

            batch_params = GenerationParams(
                lyrics=params.lyrics,
                style_tags=params.style_tags,
                duration=params.duration,
                seed=-1,
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


# -- High-Level Functions for InferenceWorker ----------------------------------

def generate_song(
    lyrics: str,
    style_tags: str,
    duration: float = 60.0,
    seed: int = -1,
    cfg_scale: float = 15.0,
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
    cfg_scale: float = 15.0,
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
