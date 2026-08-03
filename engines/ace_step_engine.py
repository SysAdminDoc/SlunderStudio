"""Native offline adapter for the official ACE-Step 1.5 Diffusers pipeline."""
import importlib.metadata
import os
import re
import sys
import time
import random
import threading
from typing import Optional, Callable
from pathlib import Path
from dataclasses import asdict, dataclass, field, replace

from core.provenance import write_provenance_sidecar
from core.settings import get_config_dir
from core.ace_step_contract import (
    ACE_STEP_ADAPTER,
    ACE_STEP_APP_TASKS,
    ACE_STEP_DEFAULT_SHIFT,
    ACE_STEP_DEFAULT_STEPS,
    ACE_STEP_DEPENDENCY_BOUNDS,
    ACE_STEP_MAX_DURATION,
    ACE_STEP_MIN_DURATION,
    ACE_STEP_MODEL_ID,
    ACE_STEP_PYTHON_VERSIONS,
    ACE_STEP_REVISION,
    ACE_STEP_SAMPLE_RATE,
    ACE_STEP_SOURCE,
    ACE_STEP_SOURCE_TASKS,
)
from core.dependency_profiles import version_at_least, version_less_than
from core.model_security import assert_safe_transformers_snapshot


def _cleanup_output_paths(paths: list[str | Path]) -> None:
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def _result_paths(results: list["GenerationResult"]) -> list[str]:
    paths: list[str] = []
    for result in results:
        if result.audio_path:
            paths.append(result.audio_path)
        if result.provenance_path:
            paths.append(result.provenance_path)
        if result.vocal_stem_path:
            paths.append(result.vocal_stem_path)
        if result.vocal_stem_provenance_path:
            paths.append(result.vocal_stem_provenance_path)
    return paths


def _verified_paths(paths: list[str | Path]) -> list[str]:
    """Return only output paths that are present as regular files."""
    return [str(path) for path in paths if path and Path(path).is_file()]


def _rethrow_with_preserved_results(
    error,
    completed_paths: list[str | Path],
    result=None,
):
    """Propagate cancellation while retaining verified earlier results."""
    from core.job_state import extract_output_paths
    from core.workers import CancelledJobError

    completed = [str(path) for path in completed_paths]
    preserved = _verified_paths(completed)
    preserved.extend(extract_output_paths(error.preserved))
    preserved = list(dict.fromkeys(preserved))
    outputs = extract_output_paths(error.outputs)
    outputs.extend(completed)
    outputs = list(dict.fromkeys(outputs))
    return CancelledJobError(
        str(error),
        outputs=outputs,
        preserved=preserved,
        result=result if result is not None else error.result,
    )


def _raise_if_cancelled(
    cancel_event: threading.Event = None,
    outputs: Optional[list[str | Path]] = None,
    preserved: Optional[list[str | Path]] = None,
    result=None,
) -> None:
    if cancel_event and cancel_event.is_set():
        output_paths = [str(path) for path in (outputs or []) if path]
        preserved_paths = _verified_paths(preserved or [])
        _cleanup_output_paths(
            [path for path in output_paths if path not in set(preserved_paths)]
        )
        from core.workers import CancelledJobError
        raise CancelledJobError(
            "Generation cancelled",
            outputs={"paths": output_paths},
            preserved={"paths": preserved_paths},
            result=result,
        )


def validate_ace_step_runtime() -> dict[str, str]:
    """Fail before model allocation when the optional runtime is incompatible."""
    python_version = sys.version_info[:2]
    if python_version not in ACE_STEP_PYTHON_VERSIONS:
        supported = ", ".join(
            f"{major}.{minor}" for major, minor in ACE_STEP_PYTHON_VERSIONS
        )
        raise RuntimeError(
            f"ACE-Step 1.5 requires Python {supported}; "
            f"found {python_version[0]}.{python_version[1]}."
        )

    versions: dict[str, str] = {}
    missing: list[str] = []
    for package, (minimum, maximum) in ACE_STEP_DEPENDENCY_BOUNDS.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
            continue
        versions[package] = installed
        if minimum and not version_at_least(installed, minimum):
            raise RuntimeError(
                f"ACE-Step 1.5 requires {package}>={minimum}; found {installed}."
            )
        if maximum and not version_less_than(installed, maximum):
            raise RuntimeError(
                f"ACE-Step 1.5 requires {package}<{maximum}; found {installed}."
            )
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"ACE-Step 1.5 optional runtime is incomplete ({names}). "
            "Install a verified profile with tools/dependency_profiles.py."
        )
    return versions


def _load_source_audio_tensor(
    source_path: str,
    *,
    sample_rate: int = ACE_STEP_SAMPLE_RATE,
    target_duration: Optional[float] = None,
):
    """Decode, resample, and normalize source audio to stereo [channels, samples]."""
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source audio not found: {source_path}")

    import librosa
    import numpy as np
    import soundfile as sf
    import torch

    audio, source_rate = sf.read(
        source,
        dtype="float32",
        always_2d=True,
    )
    if audio.size == 0 or source_rate <= 0:
        raise ValueError(f"Source audio is empty or invalid: {source}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Source audio contains non-finite samples: {source}")

    source_duration = len(audio) / source_rate
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        mono = audio.mean(axis=1, keepdims=True)
        audio = np.repeat(mono, 2, axis=1)

    if source_rate != sample_rate:
        channels = [
            librosa.resample(
                audio[:, channel],
                orig_sr=source_rate,
                target_sr=sample_rate,
            )
            for channel in range(2)
        ]
        audio = np.stack(channels, axis=1)

    if target_duration is not None:
        target_samples = max(1, int(round(float(target_duration) * sample_rate)))
        if len(audio) < target_samples:
            audio = np.pad(audio, ((0, target_samples - len(audio)), (0, 0)))
        else:
            audio = audio[:target_samples]

    return torch.from_numpy(np.ascontiguousarray(audio.T)), float(source_duration)


def recover_song_vocal_stem(
    audio_path: str,
    *,
    model_name: str = "htdemucs",
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
) -> dict:
    """Recover a vocals-only stem for a completed Song Forge render without failing the song."""
    if not audio_path:
        return {"vocal_stem_path": "", "vocal_stem_provenance_path": "", "vocal_stem_error": "Missing song render path"}
    if cancel_event and cancel_event.is_set():
        return {"vocal_stem_path": "", "vocal_stem_provenance_path": "", "vocal_stem_error": "Cancelled before vocal stem recovery"}

    if step_cb:
        step_cb("Recovering vocal stem...")

    try:
        from engines.demucs_engine import recover_vocal_stem

        recovery = recover_vocal_stem(
            audio_path,
            model_name=model_name,
            progress_callback=lambda _fraction, message: step_cb(message) if step_cb else None,
        )
    except Exception as exc:
        message = str(exc)
        if log_cb:
            log_cb(f"Vocal stem recovery failed: {message}")
        return {
            "vocal_stem_path": "",
            "vocal_stem_provenance_path": "",
            "vocal_stem_error": message,
        }

    if recovery.error:
        if log_cb:
            log_cb(f"Vocal stem recovery skipped: {recovery.error}")
        return {
            "vocal_stem_path": "",
            "vocal_stem_provenance_path": "",
            "vocal_stem_error": recovery.error,
        }

    if log_cb:
        log_cb(f"Recovered vocal stem: {recovery.path}")
    return {
        "vocal_stem_path": recovery.path,
        "vocal_stem_provenance_path": recovery.provenance_path,
        "vocal_stem_error": "",
    }


@dataclass
class GenerationParams:
    """Parameters for ACE-Step song generation."""
    lyrics: str = ""
    style_tags: str = ""  # comma-separated ACE-Step tags (maps to 'prompt')
    duration: float = 60.0  # seconds (maps to 'audio_duration')
    seed: int = -1  # -1 = random
    cfg_scale: float = 1.0  # XL Turbo is guidance-distilled; compatibility field
    infer_steps: int = ACE_STEP_DEFAULT_STEPS
    shift: float = ACE_STEP_DEFAULT_SHIFT
    scheduler: str = "flow_match_euler"
    sample_rate: int = ACE_STEP_SAMPLE_RATE
    task_type: str = "text2music"
    vocal_language: str = "en"
    audio_cover_strength: float = 1.0
    # Repaint/retake
    repaint_start: float = -1.0  # -1 = disabled
    repaint_end: float = -1.0
    source_audio_path: str = ""  # required by cover/repaint/extend
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
    provenance_path: str = ""
    vocal_stem_path: str = ""
    vocal_stem_provenance_path: str = ""
    vocal_stem_error: str = ""


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
    write_provenance_sidecar(
        output_path,
        module="song_forge",
        operation="stitch_audio_files",
        parameters={
            "target_sample_rate": target_sample_rate,
            "crossfade_seconds": crossfade_seconds,
        },
        source_paths=audio_paths,
        export_format="wav",
        output_kind="export",
    )
    return output_path, len(stitched) / target_sample_rate


class ACEStepEngine:
    """
    Wrapper around the official Diffusers ACE-Step 1.5 DiT pipeline.
    """

    def __init__(self):
        self._pipeline = None
        self._model_loaded = False
        self._device = "cpu"
        self._output_dir = get_config_dir() / "generations" / "song_forge"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded and self._pipeline is not None

    def load(self, cache_dir: str = None):
        """Load ACE-Step pipeline. Called by ModelManager."""
        validate_ace_step_runtime()
        import torch
        from diffusers import AceStepPipeline

        from core.model_manager import ModelManager, OfflineModeError

        mgr = ModelManager()

        if cache_dir:
            checkpoint_dir = cache_dir
        else:
            checkpoint_dir = str(mgr.get_cache_dir(ACE_STEP_MODEL_ID))

        if mgr.is_offline and not Path(checkpoint_dir).exists():
            raise OfflineModeError(
                "ACE-Step model not cached locally and Offline Mode is enabled. "
                "Download the model first, then enable Offline Mode."
            )

        assert_safe_transformers_snapshot(checkpoint_dir)

        if torch.cuda.is_available():
            self._device = "cuda"
            dtype = (
                torch.bfloat16
                if getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                else torch.float16
            )
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            self._device = "mps"
            dtype = torch.float16
        else:
            self._device = "cpu"
            dtype = torch.float32

        self._pipeline = AceStepPipeline.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=dtype,
        )
        if self._device == "cuda" and hasattr(
            self._pipeline, "enable_model_cpu_offload"
        ):
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline.to(self._device)
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
        if params.task_type not in ACE_STEP_APP_TASKS:
            options = ", ".join(ACE_STEP_APP_TASKS)
            raise ValueError(
                f"Unsupported ACE-Step task {params.task_type!r}; choose: {options}"
            )
        if not ACE_STEP_MIN_DURATION <= params.duration <= ACE_STEP_MAX_DURATION:
            raise ValueError(
                f"ACE-Step duration must be {ACE_STEP_MIN_DURATION:g}-"
                f"{ACE_STEP_MAX_DURATION:g} seconds."
            )
        if params.sample_rate != ACE_STEP_SAMPLE_RATE:
            raise ValueError(
                f"ACE-Step 1.5 renders at {ACE_STEP_SAMPLE_RATE} Hz only."
            )
        if params.infer_steps < 1:
            raise ValueError("ACE-Step inference steps must be at least 1.")
        if params.shift not in {1.0, 2.0, 3.0}:
            raise ValueError("ACE-Step timestep shift must be 1, 2, or 3.")
        if not 0.0 <= params.audio_cover_strength <= 1.0:
            raise ValueError("Cover strength must be between 0 and 1.")
        if params.task_type in ACE_STEP_SOURCE_TASKS:
            if not params.source_audio_path:
                raise ValueError(
                    f"{params.task_type.title()} requires a source audio file."
                )
            if not Path(params.source_audio_path).is_file():
                raise FileNotFoundError(
                    f"Source audio not found: {params.source_audio_path}"
                )
        if params.lora_path:
            raise RuntimeError(
                "ACE-Step 1.5 LoRA loading is unavailable until the versioned "
                "training and adapter contract is implemented."
            )

        _raise_if_cancelled(cancel_event)
        seed = params.resolve_seed()
        start_time = time.time()

        save_dir = str(self._output_dir)

        if progress_cb:
            progress_cb(5)

        pipeline_task = "repaint" if params.task_type == "extend" else params.task_type
        gen_kwargs = {
            "prompt": params.style_tags,
            "lyrics": params.lyrics,
            "audio_duration": params.duration,
            "vocal_language": params.vocal_language,
            "num_inference_steps": params.infer_steps,
            "guidance_scale": 1.0,
            "shift": params.shift,
            "task_type": pipeline_task,
            "output_type": "pt",
        }

        import torch
        gen_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)

        if params.task_type == "cover":
            reference_audio, _ = _load_source_audio_tensor(
                params.source_audio_path,
                sample_rate=ACE_STEP_SAMPLE_RATE,
            )
            gen_kwargs["reference_audio"] = reference_audio
            gen_kwargs["audio_cover_strength"] = params.audio_cover_strength
        elif params.task_type in {"repaint", "extend"}:
            source_audio, source_duration = _load_source_audio_tensor(
                params.source_audio_path,
                sample_rate=ACE_STEP_SAMPLE_RATE,
                target_duration=params.duration,
            )
            repaint_start = params.repaint_start
            repaint_end = params.repaint_end
            if params.task_type == "repaint":
                if abs(params.duration - source_duration) > 0.05:
                    raise ValueError(
                        "Repaint duration must match the source audio duration."
                    )
                if repaint_start < 0 or repaint_end <= repaint_start:
                    raise ValueError(
                        "Repaint requires an end time greater than its start time."
                    )
                if repaint_end > source_duration:
                    raise ValueError(
                        "Repaint end time exceeds the source audio duration."
                    )
            else:
                if abs(repaint_start - source_duration) > 0.05:
                    raise ValueError(
                        "Extend must start at the end of the source audio."
                    )
                if repaint_end <= repaint_start:
                    raise ValueError(
                        "Extend requires a positive continuation duration."
                    )
                if abs(repaint_end - params.duration) > 0.05:
                    raise ValueError(
                        "Extend must repaint through the requested output endpoint."
                    )
            gen_kwargs.update({
                "src_audio": source_audio,
                "repainting_start": repaint_start,
                "repainting_end": repaint_end,
            })

        if progress_cb:
            progress_cb(10)

        _raise_if_cancelled(cancel_event)

        def _on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
            _raise_if_cancelled(cancel_event)
            if progress_cb:
                completed = (step_index + 1) / max(1, params.infer_steps)
                progress_cb(10 + int(completed * 80))
            return callback_kwargs

        gen_kwargs["callback_on_step_end"] = _on_step_end
        result = self._pipeline(**gen_kwargs)

        elapsed = time.time() - start_time

        output_path = self._find_output(
            save_dir,
            result,
            sample_rate=getattr(
                self._pipeline,
                "sample_rate",
                ACE_STEP_SAMPLE_RATE,
            ),
            seed=seed,
        )
        _raise_if_cancelled(cancel_event, [output_path])

        if progress_cb:
            progress_cb(95)

        sidecar = write_provenance_sidecar(
            output_path,
            module="song_forge",
            operation="generate",
            model_id=ACE_STEP_MODEL_ID,
            model_source=ACE_STEP_SOURCE,
            model_revision=ACE_STEP_REVISION,
            seed=seed,
            prompt=params.style_tags,
            lyrics=params.lyrics,
            parameters=asdict(params),
            source_paths=[params.source_audio_path] if params.source_audio_path else [],
            export_format="wav",
            output_kind="model",
            extra={
                "adapter": ACE_STEP_ADAPTER,
                "task_type": pipeline_task,
                "requested_task": params.task_type,
            },
        )
        _raise_if_cancelled(cancel_event, [output_path, sidecar])

        if progress_cb:
            progress_cb(100)

        return GenerationResult(
            audio_path=str(output_path),
            seed=seed,
            duration=params.duration,
            sample_rate=ACE_STEP_SAMPLE_RATE,
            params=params,
            generation_time=elapsed,
            provenance_path=str(sidecar),
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
        if params.task_type != "text2music":
            raise ValueError(
                "Long-form stitching supports text-to-music only; source tasks "
                "must use one native ACE-Step render."
            )

        _raise_if_cancelled(cancel_event)
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
            _raise_if_cancelled(cancel_event, _result_paths(section_results))

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

            try:
                result = self.generate(
                    section_params,
                    progress_cb=_section_progress,
                    cancel_event=cancel_event,
                )
            except Exception as exc:
                from core.workers import CancelledJobError
                if isinstance(exc, CancelledJobError):
                    _cleanup_output_paths(_result_paths(section_results))
                raise
            section_results.append(result)
            section_paths.append(result.audio_path)
            _raise_if_cancelled(cancel_event, _result_paths(section_results))

        if step_cb:
            step_cb("Stitching long-form sections...")
        if progress_cb:
            progress_cb(92)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self._output_dir / f"longform_{timestamp}_{base_seed}.wav"
        stitched_path, actual_duration = stitch_audio_files(
            section_paths,
            str(output_path),
            target_sample_rate=ACE_STEP_SAMPLE_RATE,
            crossfade_seconds=params.section_crossfade,
        )
        stitched_outputs = _result_paths(section_results) + [stitched_path]
        _raise_if_cancelled(cancel_event, stitched_outputs)
        sidecar = write_provenance_sidecar(
            stitched_path,
            module="song_forge",
            operation="generate_long_form",
            model_id=ACE_STEP_MODEL_ID,
            model_source=ACE_STEP_SOURCE,
            model_revision=ACE_STEP_REVISION,
            seed=base_seed,
            prompt=params.style_tags,
            lyrics=params.lyrics,
            parameters=asdict(params),
            source_paths=section_paths,
            export_format="wav",
            output_kind="model",
            extra={
                "section_count": len(section_results),
                "section_crossfade": params.section_crossfade,
            },
        )
        _raise_if_cancelled(cancel_event, stitched_outputs + [sidecar])

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
            sample_rate=ACE_STEP_SAMPLE_RATE,
            params=params,
            generation_time=elapsed,
            sections=section_payload,
            provenance_path=str(sidecar),
        )

    def _find_output(
        self,
        save_dir: str,
        pipeline_result,
        *,
        sample_rate: int = ACE_STEP_SAMPLE_RATE,
        seed: int = 0,
    ) -> Path:
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

        import numpy as np
        import soundfile as sf
        import torch

        audio_data = getattr(pipeline_result, "audios", None)
        if audio_data is None and isinstance(pipeline_result, torch.Tensor):
            audio_data = pipeline_result
        elif audio_data is None and isinstance(pipeline_result, np.ndarray):
            audio_data = pipeline_result
        elif audio_data is None and isinstance(pipeline_result, (list, tuple)):
            for item in pipeline_result:
                if isinstance(item, (torch.Tensor, np.ndarray)):
                    audio_data = item
                    break

        if audio_data is not None:
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.detach().float().cpu().numpy()
            else:
                audio_data = np.asarray(audio_data, dtype=np.float32)
            if audio_data.ndim == 3:
                audio_data = audio_data[0]
            if audio_data.ndim == 1:
                audio_data = np.stack([audio_data, audio_data])
            if audio_data.ndim != 2:
                raise RuntimeError(
                    f"ACE-Step returned unsupported audio shape {audio_data.shape}"
                )
            if audio_data.shape[0] <= 8:
                audio_data = audio_data.T
            if audio_data.shape[1] == 1:
                audio_data = np.repeat(audio_data, 2, axis=1)
            elif audio_data.shape[1] > 2:
                audio_data = audio_data[:, :2]
            if not np.isfinite(audio_data).all():
                raise RuntimeError("ACE-Step returned non-finite audio samples")
            audio_data = np.clip(audio_data, -1.0, 1.0)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = (
                Path(save_dir)
                / f"ace_step_{timestamp}_{time.time_ns() % 1_000_000_000:09d}_{seed}.wav"
            )
            sf.write(out_path, audio_data, sample_rate, subtype="PCM_24")
            return out_path

        # Fallback: find most recent wav in save_dir created after generation started
        cutoff = time.time() - 300
        wavs = sorted(
            (p for p in Path(save_dir).glob("*.wav") if p.stat().st_mtime > cutoff),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
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
            paths = _result_paths(results)
            _raise_if_cancelled(
                cancel_event,
                paths,
                preserved=_verified_paths(paths),
                result=results,
            )

            if step_cb:
                step_cb(f"Generating variation {i+1}/{count}...")

            batch_params = replace(params, seed=-1)

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
                from core.workers import CancelledJobError
                if isinstance(e, CancelledJobError):
                    raise _rethrow_with_preserved_results(
                        e,
                        _result_paths(results),
                        result=results,
                    ) from e
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
        if not source_path or not Path(source_path).is_file():
            raise FileNotFoundError(f"Source audio not found: {source_path}")
        if extend_duration <= 0:
            raise ValueError("Extend duration must be positive.")
        import soundfile as sf

        source_duration = float(sf.info(source_path).duration)
        target_duration = source_duration + float(extend_duration)
        extended_params = replace(
            params,
            task_type="extend",
            source_audio_path=source_path,
            duration=target_duration,
            repaint_start=source_duration,
            repaint_end=target_duration,
        )
        return self.generate(
            extended_params,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

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
        if not source_path or not Path(source_path).is_file():
            raise FileNotFoundError(f"Source audio not found: {source_path}")
        import soundfile as sf

        source_duration = float(sf.info(source_path).duration)
        repaint_params = replace(
            params,
            task_type="repaint",
            source_audio_path=source_path,
            duration=source_duration,
            repaint_start=start_sec,
            repaint_end=end_sec,
        )
        return self.generate(
            repaint_params,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )


# -- High-Level Functions for InferenceWorker ----------------------------------

def _load_managed_engine(model_manager) -> ACEStepEngine:
    """Load ACE-Step through ModelManager and use its canonical engine object."""
    requested_engine = ACEStepEngine()

    def _loader():
        requested_engine.load()
        return requested_engine

    engine = model_manager.load_model(ACE_STEP_MODEL_ID, loader_fn=_loader)
    if not isinstance(engine, ACEStepEngine):
        raise TypeError(
            f"ModelManager returned {type(engine).__name__} for {ACE_STEP_MODEL_ID}"
        )
    return engine

def generate_song(
    lyrics: str,
    style_tags: str = "",
    duration: float = 60.0,
    seed: int = -1,
    cfg_scale: float = 1.0,
    infer_steps: int = ACE_STEP_DEFAULT_STEPS,
    shift: float = ACE_STEP_DEFAULT_SHIFT,
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
    engine = _load_managed_engine(mgr)

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
        shift=shift,
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

    vocal_recovery = recover_song_vocal_stem(
        result.audio_path,
        step_cb=step_cb,
        log_cb=log_cb,
        cancel_event=cancel_event,
    )
    return {
        "audio_path": result.audio_path,
        "provenance_path": result.provenance_path,
        **vocal_recovery,
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
            "shift": shift,
        },
    }


def generate_song_batch(
    lyrics: str,
    style_tags: str,
    count: int = 4,
    duration: float = 60.0,
    cfg_scale: float = 1.0,
    infer_steps: int = ACE_STEP_DEFAULT_STEPS,
    shift: float = ACE_STEP_DEFAULT_SHIFT,
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
    engine = _load_managed_engine(mgr)

    if cancel_event and cancel_event.is_set():
        return {"cancelled": True}

    params = GenerationParams(
        lyrics=lyrics,
        style_tags=style_tags,
        duration=duration,
        cfg_scale=cfg_scale,
        infer_steps=infer_steps,
        shift=shift,
        long_form=long_form,
    )

    results = engine.generate_batch(
        params, count=count,
        progress_cb=progress_cb, step_cb=step_cb, cancel_event=cancel_event,
    )

    recovered = [
        recover_song_vocal_stem(
            item.audio_path,
            step_cb=step_cb,
            log_cb=log_cb,
            cancel_event=cancel_event,
        )
        for item in results
    ]
    return {
        "results": [
            {
                "audio_path": r.audio_path,
                "provenance_path": r.provenance_path,
                **recovered[index],
                "seed": r.seed,
                "duration": r.duration,
                "generation_time": r.generation_time,
                "mode": "long_form" if r.sections else "single",
                "sections": r.sections,
            }
            for index, r in enumerate(results)
        ],
        "count": len(results),
    }


def generate_seed_grid(
    lyrics: str,
    style_tags: str,
    params_list: list[dict],
    duration: float = 60.0,
    infer_steps: int = ACE_STEP_DEFAULT_STEPS,
    long_form: bool = False,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
    **kwargs,
) -> dict:
    """Generate a seed explorer grid with explicit seed/timestep-shift cells."""
    if step_cb:
        step_cb("Loading ACE-Step model...")

    from core.model_manager import ModelManager
    mgr = ModelManager()
    engine = _load_managed_engine(mgr)

    total = max(1, len(params_list))
    results = []

    for i, cell in enumerate(params_list):
        paths = [
            path
            for item in results
            for path in (item.get("audio_path", ""), item.get("provenance_path", ""))
            if path
        ]
        _raise_if_cancelled(
            cancel_event,
            paths,
            preserved=_verified_paths(paths),
            result={"results": results, "count": len(results)},
        )

        row = int(cell.get("row", 0))
        col = int(cell.get("col", 0))
        seed = int(cell.get("seed", -1))
        shift = float(
            cell.get("shift", kwargs.get("shift", ACE_STEP_DEFAULT_SHIFT))
        )

        if step_cb:
            step_cb(f"Generating seed cell {i + 1}/{total} (seed {seed})...")

        cell_params = GenerationParams(
            lyrics=lyrics,
            style_tags=style_tags,
            duration=duration,
            seed=seed,
            infer_steps=infer_steps,
            shift=shift,
            long_form=long_form,
        )

        start_pct = int(i * 100 / total)
        end_pct = int((i + 1) * 100 / total)

        def _cell_progress(pct, start=start_pct, end=end_pct):
            if progress_cb:
                progress_cb(start + int((end - start) * pct / 100))

        try:
            if long_form or duration > 120:
                result = engine.generate_long_form(
                    cell_params,
                    progress_cb=_cell_progress,
                    step_cb=step_cb,
                    cancel_event=cancel_event,
                )
            else:
                result = engine.generate(
                    cell_params,
                    progress_cb=_cell_progress,
                    cancel_event=cancel_event,
                )

            results.append({
                "row": row,
                "col": col,
                "audio_path": result.audio_path,
                "provenance_path": result.provenance_path,
                "seed": result.seed,
                "shift": shift,
                "duration": result.duration,
                "generation_time": result.generation_time,
                "mode": "long_form" if result.sections else "single",
                "sections": result.sections,
            })
        except Exception as exc:
            from core.workers import CancelledJobError
            if isinstance(exc, CancelledJobError):
                paths = [
                    path
                    for item in results
                    for path in (item.get("audio_path", ""), item.get("provenance_path", ""))
                    if path
                ]
                raise _rethrow_with_preserved_results(
                    exc,
                    paths,
                    result={"results": results, "count": len(results)},
                ) from exc
            message = f"{type(exc).__name__}: {exc}"
            if log_cb:
                log_cb(f"Seed cell {row},{col} failed: {message}")
            results.append({
                "row": row,
                "col": col,
                "seed": seed,
                "shift": shift,
                "error": message,
            })

    if progress_cb:
        progress_cb(100)

    return {"results": results, "count": len(results)}


def generate_cover(
    source_audio_path: str,
    style_tags: str = "",
    lyrics: str = "",
    duration: float = 0.0,
    seed: int = -1,
    cfg_scale: float = 1.0,
    infer_steps: int = ACE_STEP_DEFAULT_STEPS,
    shift: float = ACE_STEP_DEFAULT_SHIFT,
    audio_cover_strength: float = 1.0,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
    **kwargs,
) -> dict:
    """Generate a cover: re-render a source track with new style tags."""
    if not source_audio_path or not os.path.isfile(source_audio_path):
        raise FileNotFoundError(f"Source audio not found: {source_audio_path}")

    if step_cb:
        step_cb("Loading ACE-Step for cover...")

    from core.model_manager import ModelManager
    mgr = ModelManager()
    engine = _load_managed_engine(mgr)

    if cancel_event and cancel_event.is_set():
        return {"cancelled": True}

    if step_cb:
        step_cb("Generating cover...")

    import soundfile as sf
    info = sf.info(source_audio_path)
    actual_duration = duration or info.duration

    params = GenerationParams(
        lyrics=lyrics,
        style_tags=style_tags,
        duration=actual_duration,
        seed=seed,
        cfg_scale=cfg_scale,
        infer_steps=infer_steps,
        shift=shift,
        task_type="cover",
        audio_cover_strength=audio_cover_strength,
        source_audio_path=source_audio_path,
    )

    result = engine.generate(params, progress_cb=progress_cb, cancel_event=cancel_event)

    return {
        "audio_path": result.audio_path,
        "provenance_path": result.provenance_path,
        "seed": result.seed,
        "duration": result.duration,
        "generation_time": result.generation_time,
        "mode": "cover",
        "source_audio_path": source_audio_path,
    }


def generate_extend(
    source_audio_path: str,
    extend_duration: float = 30.0,
    style_tags: str = "",
    lyrics: str = "",
    seed: int = -1,
    cfg_scale: float = 1.0,
    infer_steps: int = ACE_STEP_DEFAULT_STEPS,
    shift: float = ACE_STEP_DEFAULT_SHIFT,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
    **kwargs,
) -> dict:
    """Extend source audio by repainting a zero-padded continuation region."""
    if not source_audio_path or not os.path.isfile(source_audio_path):
        raise FileNotFoundError(f"Source audio not found: {source_audio_path}")
    if extend_duration <= 0:
        raise ValueError("Extend duration must be positive.")

    if step_cb:
        step_cb("Loading ACE-Step for extension...")

    from core.model_manager import ModelManager
    mgr = ModelManager()
    engine = _load_managed_engine(mgr)

    if cancel_event and cancel_event.is_set():
        return {"cancelled": True}
    if step_cb:
        step_cb(f"Extending source by {extend_duration:.1f}s...")

    params = GenerationParams(
        lyrics=lyrics,
        style_tags=style_tags,
        seed=seed,
        cfg_scale=cfg_scale,
        infer_steps=infer_steps,
        shift=shift,
    )
    result = engine.extend(
        source_audio_path,
        params,
        extend_duration=extend_duration,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )
    return {
        "audio_path": result.audio_path,
        "provenance_path": result.provenance_path,
        "seed": result.seed,
        "duration": result.duration,
        "generation_time": result.generation_time,
        "mode": "extend",
        "source_audio_path": source_audio_path,
        "extend_duration": extend_duration,
    }


def generate_repaint(
    source_audio_path: str,
    start_sec: float,
    end_sec: float,
    style_tags: str = "",
    lyrics: str = "",
    seed: int = -1,
    cfg_scale: float = 1.0,
    infer_steps: int = ACE_STEP_DEFAULT_STEPS,
    shift: float = ACE_STEP_DEFAULT_SHIFT,
    progress_cb: Callable = None,
    step_cb: Callable = None,
    log_cb: Callable = None,
    cancel_event: threading.Event = None,
    **kwargs,
) -> dict:
    """Repaint: regenerate a region of a source track while keeping the rest."""
    if not source_audio_path or not os.path.isfile(source_audio_path):
        raise FileNotFoundError(f"Source audio not found: {source_audio_path}")
    if end_sec <= start_sec:
        raise ValueError(f"Invalid repaint region: {start_sec}s to {end_sec}s")

    if step_cb:
        step_cb("Loading ACE-Step for repaint...")

    from core.model_manager import ModelManager
    mgr = ModelManager()
    engine = _load_managed_engine(mgr)

    if cancel_event and cancel_event.is_set():
        return {"cancelled": True}

    if step_cb:
        step_cb(f"Repainting {start_sec:.1f}s – {end_sec:.1f}s...")

    import soundfile as sf
    info = sf.info(source_audio_path)

    params = GenerationParams(
        lyrics=lyrics,
        style_tags=style_tags,
        duration=info.duration,
        seed=seed,
        cfg_scale=cfg_scale,
        infer_steps=infer_steps,
        shift=shift,
        task_type="repaint",
        source_audio_path=source_audio_path,
        repaint_start=start_sec,
        repaint_end=end_sec,
    )

    result = engine.generate(params, progress_cb=progress_cb, cancel_event=cancel_event)

    return {
        "audio_path": result.audio_path,
        "provenance_path": result.provenance_path,
        "seed": result.seed,
        "duration": result.duration,
        "generation_time": result.generation_time,
        "mode": "repaint",
        "source_audio_path": source_audio_path,
        "repaint_start": start_sec,
        "repaint_end": end_sec,
    }


def load_model(cache_dir: str = None, **kwargs) -> ACEStepEngine:
    """Loader function for ModelManager registry."""
    engine = ACEStepEngine()
    engine.load(cache_dir)
    return engine
