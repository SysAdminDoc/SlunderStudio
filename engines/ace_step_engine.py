"""
Slunder Studio v0.1.1 — ACE-Step Engine
Native Python wrapper for ACE-Step inference (not Gradio).
Supports: generate, batch, retake, repaint, extend.
<4GB VRAM, 48kHz stereo, up to 4 min duration.

Real upstream API (pip install ace-step):
  from acestep.pipeline_ace_step import ACEStepPipeline
  pipe = ACEStepPipeline(checkpoint_dir=path)
  result = pipe(prompt=..., lyrics=..., audio_duration=..., ...)
"""
import os
import re
import time
import random
import threading
from typing import Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field, replace

from core.settings import get_config_dir


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
    long_form: bool = False
    section_crossfade: float = 2.0

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
    sections: list[dict] = field(default_factory=list)


@dataclass
class LongFormSection:
    """A planned section for stitched long-form song generation."""
    label: str
    lyrics: str
    duration: float = 0.0


SECTION_HEADER_RE = re.compile(r"^\s*\[([A-Za-z][A-Za-z0-9 /_-]{0,60})\]\s*$")
SONG_SECTION_PREFIXES = (
    "intro",
    "verse",
    "pre-chorus",
    "pre chorus",
    "chorus",
    "hook",
    "bridge",
    "breakdown",
    "instrumental",
    "solo",
    "outro",
    "coda",
)
SECTION_DURATION_WEIGHTS = {
    "intro": 0.65,
    "verse": 1.0,
    "pre-chorus": 0.8,
    "pre chorus": 0.8,
    "chorus": 1.15,
    "hook": 1.1,
    "bridge": 0.9,
    "breakdown": 0.85,
    "instrumental": 0.95,
    "solo": 0.95,
    "outro": 0.7,
    "coda": 0.6,
}


def _canonical_label(raw_label: str) -> str:
    label = " ".join(raw_label.strip().split())
    return label or "Section"


def _section_key(label: str) -> str:
    lowered = label.lower()
    for prefix in SONG_SECTION_PREFIXES:
        if lowered.startswith(prefix):
            return prefix
    return "section"


def _is_song_section(label: str) -> bool:
    return _section_key(label) != "section"


def parse_lyric_sections(lyrics: str) -> list[LongFormSection]:
    """Parse ACE/Suno-style structure tags into song sections."""
    clean_lyrics = lyrics.strip()
    if not clean_lyrics:
        return [LongFormSection("Instrumental", "[Instrumental]")]

    sections: list[LongFormSection] = []
    current_label = "Full Track"
    current_lines: list[str] = []
    saw_song_section = False

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(line for line in current_lines).strip()
        if text:
            sections.append(LongFormSection(current_label, text))
        current_lines = []

    for line in clean_lyrics.splitlines():
        match = SECTION_HEADER_RE.match(line)
        if match and _is_song_section(match.group(1)):
            if current_lines:
                flush()
            current_label = _canonical_label(match.group(1))
            current_lines = [f"[{current_label}]"]
            saw_song_section = True
            continue
        current_lines.append(line)

    if current_lines:
        flush()

    if not saw_song_section:
        return [LongFormSection("Full Track", clean_lyrics)]

    return sections or [LongFormSection("Full Track", clean_lyrics)]


def _split_section_lines(section: LongFormSection, parts: int) -> list[LongFormSection]:
    if parts <= 1:
        return [section]

    lines = [line for line in section.lyrics.splitlines() if line.strip()]
    header = ""
    if lines and SECTION_HEADER_RE.match(lines[0]):
        header = lines.pop(0)

    if not lines:
        return [
            LongFormSection(f"{section.label} Part {i + 1}", section.lyrics)
            for i in range(parts)
        ]

    chunks: list[LongFormSection] = []
    chunk_size = max(1, (len(lines) + parts - 1) // parts)
    for i in range(parts):
        start = i * chunk_size
        chunk = lines[start:start + chunk_size]
        if not chunk:
            chunk = lines[-chunk_size:]
        label = section.label if parts == 1 else f"{section.label} Part {i + 1}"
        text_lines = [f"[{label}]"]
        if header and i == 0:
            text_lines[0] = header
        text_lines.extend(chunk)
        chunks.append(LongFormSection(label, "\n".join(text_lines)))
    return chunks


def _expand_sections_for_duration(
    sections: list[LongFormSection],
    target_duration: float,
    max_section_duration: float,
) -> list[LongFormSection]:
    if not sections:
        return [LongFormSection("Instrumental", "[Instrumental]")]

    min_count = max(1, int((target_duration + max_section_duration - 0.001) // max_section_duration))
    if len(sections) >= min_count:
        return sections

    expanded: list[LongFormSection] = []
    extra_needed = min_count - len(sections)
    weights = [_duration_weight(section.label) for section in sections]

    while extra_needed > 0:
        split_index = max(range(len(sections)), key=lambda i: weights[i])
        sections[split_index:split_index + 1] = _split_section_lines(sections[split_index], 2)
        weights[split_index:split_index + 1] = [
            _duration_weight(section.label) for section in sections[split_index:split_index + 2]
        ]
        extra_needed -= 1

    expanded.extend(sections)
    return expanded


def _duration_weight(label: str) -> float:
    return SECTION_DURATION_WEIGHTS.get(_section_key(label), 1.0)


def _allocate_durations(
    sections: list[LongFormSection],
    target_duration: float,
    min_section_duration: float,
    max_section_duration: float,
) -> list[float]:
    weights = [_duration_weight(section.label) for section in sections]
    total_weight = sum(weights) or 1.0
    durations = [
        max(min_section_duration, min(max_section_duration, target_duration * weight / total_weight))
        for weight in weights
    ]

    for _ in range(20):
        diff = target_duration - sum(durations)
        if abs(diff) < 0.01:
            break
        if diff > 0:
            candidates = [i for i, dur in enumerate(durations) if dur < max_section_duration]
        else:
            candidates = [i for i, dur in enumerate(durations) if dur > min_section_duration]
        if not candidates:
            break
        share = diff / len(candidates)
        for i in candidates:
            if diff > 0:
                durations[i] = min(max_section_duration, durations[i] + share)
            else:
                durations[i] = max(min_section_duration, durations[i] + share)

    return durations


def plan_long_form_sections(
    lyrics: str,
    target_duration: float,
    min_section_duration: float = 12.0,
    max_section_duration: float = 120.0,
) -> list[LongFormSection]:
    """Build a duration-balanced section plan for stitched generation."""
    target_duration = max(float(target_duration), min_section_duration)
    sections = parse_lyric_sections(lyrics)
    sections = _expand_sections_for_duration(sections, target_duration, max_section_duration)
    if len(sections) * min_section_duration > target_duration:
        min_section_duration = max(1.0, target_duration / len(sections))
    durations = _allocate_durations(
        sections,
        target_duration,
        min_section_duration,
        max_section_duration,
    )
    return [
        replace(section, duration=round(duration, 2))
        for section, duration in zip(sections, durations)
    ]


def _ensure_stereo(audio):
    import numpy as np

    if audio.ndim == 1:
        return np.column_stack([audio, audio])
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    if audio.shape[1] > 2:
        return audio[:, :2]
    return audio


def stitch_audio_files(
    audio_paths: list[str],
    output_path: str,
    target_sample_rate: int = 48000,
    crossfade_seconds: float = 2.0,
) -> tuple[str, float]:
    """Stitch rendered sections with an equal-power crossfade."""
    if not audio_paths:
        raise ValueError("No audio files to stitch")

    import numpy as np
    import soundfile as sf

    stitched = None
    crossfade_samples = max(0, int(target_sample_rate * crossfade_seconds))

    for path in audio_paths:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if sr != target_sample_rate:
            try:
                import librosa
                channels = [
                    librosa.resample(audio[:, ch], orig_sr=sr, target_sr=target_sample_rate)
                    for ch in range(audio.shape[1])
                ]
                audio = np.column_stack(channels)
            except ImportError as exc:
                raise RuntimeError("librosa is required to stitch mixed sample rates") from exc

        audio = _ensure_stereo(audio)
        if stitched is None:
            stitched = audio
            continue

        fade_len = min(crossfade_samples, len(stitched) - 1, len(audio) - 1)
        if fade_len <= 0:
            stitched = np.concatenate([stitched, audio], axis=0)
            continue

        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)[:, None]
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)[:, None]
        crossfade = stitched[-fade_len:] * fade_out + audio[:fade_len] * fade_in
        stitched = np.concatenate([stitched[:-fade_len], crossfade, audio[fade_len:]], axis=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, stitched, target_sample_rate, subtype="PCM_16")
    return output_path, len(stitched) / target_sample_rate


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
            from core.model_manager import ModelManager
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
            from core.model_manager import cleanup_gpu
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

    def generate_long_form(
        self,
        params: GenerationParams,
        progress_cb: Callable = None,
        step_cb: Callable = None,
        cancel_event: threading.Event = None,
    ) -> GenerationResult:
        """
        Generate a long song as section renders stitched with crossfades.
        This keeps each ACE-Step call focused on a musical section while the
        final output reaches the requested full-song duration.
        """
        if not self.is_loaded:
            raise RuntimeError("ACE-Step model not loaded. Call load() first.")

        start_time = time.time()
        base_seed = params.resolve_seed()
        render_duration = params.duration
        sections = plan_long_form_sections(params.lyrics, render_duration)
        for _ in range(3):
            if len(sections) <= 1 or params.section_crossfade <= 0:
                break
            adjusted_duration = params.duration + params.section_crossfade * (len(sections) - 1)
            if abs(adjusted_duration - render_duration) < 0.01:
                break
            render_duration = adjusted_duration
            sections = plan_long_form_sections(params.lyrics, render_duration)
        section_results: list[GenerationResult] = []
        section_paths: list[str] = []
        total = max(1, len(sections))

        if progress_cb:
            progress_cb(3)

        for i, section in enumerate(sections):
            if cancel_event and cancel_event.is_set():
                break

            section_seed = (base_seed + i) % (2**32 - 1)
            section_params = replace(
                params,
                lyrics=section.lyrics,
                duration=section.duration,
                seed=section_seed,
                repaint_start=-1.0,
                repaint_end=-1.0,
                source_audio_path="",
            )

            if step_cb:
                step_cb(f"Generating {section.label} ({i + 1}/{total})...")

            start_pct = 5 + int(i * 82 / total)
            end_pct = 5 + int((i + 1) * 82 / total)

            def _section_progress(pct, start=start_pct, end=end_pct):
                if progress_cb:
                    progress_cb(start + int((end - start) * pct / 100))

            result = self.generate(
                section_params,
                progress_cb=_section_progress,
                cancel_event=cancel_event,
            )
            section_results.append(result)
            section_paths.append(result.audio_path)

        if cancel_event and cancel_event.is_set():
            return GenerationResult(seed=base_seed, params=params, sections=[])

        if step_cb:
            step_cb("Stitching long-form sections...")
        if progress_cb:
            progress_cb(92)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self._output_dir / f"longform_{timestamp}_{base_seed}.wav"
        stitched_path, actual_duration = stitch_audio_files(
            section_paths,
            str(output_path),
            target_sample_rate=params.sample_rate,
            crossfade_seconds=params.section_crossfade,
        )

        if progress_cb:
            progress_cb(100)

        elapsed = time.time() - start_time
        section_payload = [
            {
                "label": section.label,
                "duration": section.duration,
                "audio_path": result.audio_path,
                "seed": result.seed,
            }
            for section, result in zip(sections, section_results)
        ]

        return GenerationResult(
            audio_path=stitched_path,
            seed=base_seed,
            duration=actual_duration,
            sample_rate=params.sample_rate,
            params=params,
            generation_time=elapsed,
            sections=section_payload,
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
                if batch_params.duration > 120 or batch_params.long_form:
                    result = self.generate_long_form(
                        batch_params,
                        progress_cb=_batch_progress,
                        step_cb=step_cb,
                        cancel_event=cancel_event,
                    )
                else:
                    result = self.generate(
                        batch_params,
                        progress_cb=_batch_progress,
                        cancel_event=cancel_event,
                    )
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
    style_tags: str = "",
    duration: float = 60.0,
    seed: int = -1,
    cfg_scale: float = 15.0,
    infer_steps: int = 60,
    long_form: bool = False,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
    **kwargs,
) -> dict:
    """High-level song generation for InferenceWorker."""
    style_tags = style_tags or kwargs.get("tags", "")
    if seed is None:
        seed = -1

    if step_cb:
        step_cb("Loading ACE-Step model...")

    from core.model_manager import ModelManager
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
        long_form=long_form,
    )

    if long_form or duration > 120:
        result = engine.generate_long_form(
            params,
            progress_cb=progress_cb,
            step_cb=step_cb,
            cancel_event=cancel_event,
        )
    else:
        result = engine.generate(params, progress_cb=progress_cb, cancel_event=cancel_event)

    return {
        "audio_path": result.audio_path,
        "seed": result.seed,
        "duration": result.duration,
        "generation_time": result.generation_time,
        "mode": "long_form" if result.sections else "single",
        "sections": result.sections,
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
    long_form: bool = False,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
    **kwargs,
) -> dict:
    """High-level batch generation for InferenceWorker."""
    if step_cb:
        step_cb("Loading ACE-Step model...")

    from core.model_manager import ModelManager
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
        long_form=long_form,
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
                "mode": "long_form" if r.sections else "single",
                "sections": r.sections,
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
