"""
Slunder Studio v0.1.29 — AI Producer Engine
One-prompt-to-full-song orchestrator. Decomposes a high-level creative brief
into a multi-step pipeline: lyrics generation, style selection, song generation,
vocal synthesis, SFX layering, and mastering — all automated.
"""
import os
import time
import threading
import uuid
import wave
from pathlib import Path
from typing import Any, Optional, Callable
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from core.audio_buffers import normalize_channel_layout, resample_audio
from core.provenance import write_provenance_sidecar
from core.settings import get_config_dir


class PipelineStage(Enum):
    PLANNING = "planning"
    LYRICS = "lyrics"
    STYLE = "style"
    SONG_GEN = "song_generation"
    MIDI = "midi"
    VOCALS = "vocals"
    SFX = "sfx"
    MIXING = "mixing"
    MASTERING = "mastering"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProducerBrief:
    """High-level creative brief for AI Producer."""
    prompt: str = ""  # e.g. "A dreamy lo-fi hip-hop track about rainy nights"
    genre: str = ""
    mood: str = ""
    duration_seconds: float = 180.0  # target song length
    tempo: float = 0.0  # 0 = auto-detect from genre
    key: str = ""  # empty = auto-select
    vocal_style: str = ""  # "male", "female", "none"
    include_sfx: bool = True
    mastering_preset: str = "Balanced"
    seed: Optional[int] = None
    demo_fallback: bool = False


@dataclass
class PipelineStep:
    """Record of a single pipeline step execution."""
    stage: PipelineStage = PipelineStage.PLANNING
    status: str = "pending"  # pending | running | complete | skipped | failed | cancelled
    start_time: float = 0.0
    end_time: float = 0.0
    output_path: Optional[str] = None
    output_data: Optional[dict] = None
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        if self.end_time > 0 and self.start_time > 0:
            return self.end_time - self.start_time
        return 0.0

    def to_job_dict(self) -> dict[str, Any]:
        """Return the bounded, JSON-safe stage state stored in the job ledger."""
        return {
            "stage": self.stage.value,
            "status": self.status,
            "duration_seconds": round(self.duration, 3),
            "error": self.error or "",
        }


@dataclass
class ProducerResult:
    """Full pipeline execution result."""
    brief: Optional[ProducerBrief] = None
    steps: list[PipelineStep] = field(default_factory=list)
    final_audio_path: Optional[str] = None
    total_time: float = 0.0
    stage: PipelineStage = PipelineStage.PLANNING
    error: Optional[str] = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at_ns: int = field(default_factory=time.time_ns)
    output_kind: str = "model"
    degraded_reasons: list[str] = field(default_factory=list)
    cancelled: bool = False
    artifact_paths: list[str] = field(default_factory=list)

    # Intermediate outputs
    lyrics_text: str = ""
    style_tags: list[str] = field(default_factory=list)
    song_audio_path: Optional[str] = None
    vocal_audio_path: Optional[str] = None
    sfx_audio_path: Optional[str] = None
    mastered_audio_path: Optional[str] = None

    def get_step(self, stage: PipelineStage) -> Optional[PipelineStep]:
        for s in self.steps:
            if s.stage == stage:
                return s
        return None

    @property
    def completed_stages(self) -> list[PipelineStage]:
        return [s.stage for s in self.steps if s.status == "complete"]

    @property
    def progress(self) -> float:
        total = len(PIPELINE_ORDER)
        done = sum(
            step.status in {"complete", "skipped", "failed", "cancelled"}
            for step in self.steps
        )
        return done / total if total > 0 else 0.0

    @property
    def is_demo(self) -> bool:
        return self.output_kind == "demo"

    @property
    def is_degraded(self) -> bool:
        return bool(self.degraded_reasons)

    @property
    def is_success(self) -> bool:
        return (
            self.stage == PipelineStage.COMPLETE
            and not self.cancelled
            and not self.error
            and self.can_export
        )

    @property
    def can_export(self) -> bool:
        return (
            bool(self.final_audio_path)
            and self.final_audio_path in self.artifact_paths
            and _verify_audio_artifact(self.final_audio_path)
        )

    @property
    def output_paths(self) -> list[str]:
        return list(dict.fromkeys(self.artifact_paths))

    def add_artifact(self, *paths: str | Path | None) -> None:
        for path in paths:
            if path:
                value = str(Path(path))
                if value not in self.artifact_paths:
                    self.artifact_paths.append(value)

    def add_degraded_reason(self, reason: str) -> None:
        reason = reason.strip()
        if reason and reason not in self.degraded_reasons:
            self.degraded_reasons.append(reason)

    def job_metadata(self) -> dict[str, Any]:
        """Persist a truthful, bounded pipeline summary with the durable job."""
        if self.cancelled:
            outcome = "cancelled"
        elif self.stage == PipelineStage.FAILED:
            outcome = "failed"
        elif self.is_demo:
            outcome = "demo"
        elif self.is_degraded:
            outcome = "degraded"
        elif self.is_success:
            outcome = "complete"
        else:
            outcome = self.stage.value
        return {
            "pipeline": {
                "run_id": self.run_id,
                "outcome": outcome,
                "output_kind": self.output_kind,
                "degraded_reasons": self.degraded_reasons[:12],
                "stages": [step.to_job_dict() for step in self.steps],
            }
        }


# Pipeline execution order
PIPELINE_ORDER = [
    PipelineStage.PLANNING,
    PipelineStage.LYRICS,
    PipelineStage.STYLE,
    PipelineStage.SONG_GEN,
    PipelineStage.VOCALS,
    PipelineStage.SFX,
    PipelineStage.MIXING,
    PipelineStage.MASTERING,
]

REQUIRED_STAGES = {
    PipelineStage.PLANNING,
    PipelineStage.LYRICS,
    PipelineStage.STYLE,
    PipelineStage.SONG_GEN,
    PipelineStage.MIXING,
    PipelineStage.MASTERING,
}


def _verify_audio_artifact(path: str | Path | None) -> bool:
    """Verify that a pipeline artifact is a non-empty, readable audio file."""
    if not path:
        return False
    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.stat().st_size <= 44:
            return False
        import soundfile as sf

        info = sf.info(str(candidate))
        return info.frames > 0 and info.samplerate > 0 and info.channels > 0
    except (OSError, RuntimeError, ValueError):
        return False


# ── Genre Intelligence ─────────────────────────────────────────────────────────

GENRE_DEFAULTS = {
    "lo-fi": {"tempo": 80, "key": "D minor", "tags": ["lo-fi", "chill", "mellow", "vinyl crackle"]},
    "hip-hop": {"tempo": 90, "key": "C minor", "tags": ["hip-hop", "808", "trap", "bass heavy"]},
    "pop": {"tempo": 120, "key": "C major", "tags": ["pop", "catchy", "upbeat", "polished"]},
    "rock": {"tempo": 130, "key": "E minor", "tags": ["rock", "guitar", "drums", "energetic"]},
    "jazz": {"tempo": 110, "key": "Bb major", "tags": ["jazz", "swing", "piano", "smooth"]},
    "electronic": {"tempo": 128, "key": "A minor", "tags": ["electronic", "synth", "dance", "EDM"]},
    "r&b": {"tempo": 85, "key": "Ab major", "tags": ["r&b", "soul", "smooth", "groove"]},
    "classical": {"tempo": 100, "key": "D major", "tags": ["classical", "orchestral", "strings"]},
    "ambient": {"tempo": 70, "key": "F major", "tags": ["ambient", "atmospheric", "pad", "ethereal"]},
    "metal": {"tempo": 160, "key": "D minor", "tags": ["metal", "heavy", "distortion", "aggressive"]},
    "country": {"tempo": 115, "key": "G major", "tags": ["country", "acoustic guitar", "steel guitar"]},
    "reggae": {"tempo": 80, "key": "G major", "tags": ["reggae", "dub", "offbeat", "bass"]},
    "funk": {"tempo": 105, "key": "E minor", "tags": ["funk", "groove", "bass", "rhythmic"]},
    "indie": {"tempo": 118, "key": "A minor", "tags": ["indie", "alternative", "dreamy", "guitar"]},
    "latin": {"tempo": 100, "key": "A minor", "tags": ["latin", "percussion", "rhythm", "tropical"]},
}

MOOD_TAGS = {
    "happy": ["upbeat", "cheerful", "bright", "major key"],
    "sad": ["melancholy", "minor key", "slow", "emotional"],
    "energetic": ["high energy", "fast", "driving", "powerful"],
    "chill": ["relaxed", "mellow", "laid back", "smooth"],
    "dark": ["dark", "ominous", "minor key", "heavy"],
    "dreamy": ["ethereal", "atmospheric", "reverb", "ambient"],
    "aggressive": ["intense", "distorted", "loud", "fast"],
    "romantic": ["warm", "soft", "intimate", "gentle"],
    "nostalgic": ["vintage", "analog", "warm", "retro"],
    "epic": ["cinematic", "orchestral", "building", "powerful"],
}

SFX_SUGGESTIONS = {
    "rain": "gentle rain on window",
    "night": "crickets and night ambience",
    "city": "distant city traffic ambience",
    "ocean": "ocean waves softly crashing",
    "forest": "birds and forest ambience",
    "space": "deep space ambient hum",
    "fire": "crackling fireplace",
    "storm": "distant thunder rumble",
    "cafe": "coffee shop background chatter",
    "vinyl": "vinyl record crackle noise bed",
}


def analyze_brief(brief: ProducerBrief) -> dict:
    """Analyze the creative brief and fill in defaults intelligently."""
    plan = {
        "tempo": brief.tempo,
        "key": brief.key,
        "style_tags": [],
        "sfx_prompt": "",
        "lyrics_prompt": brief.prompt,
        "genre": brief.genre,
        "mood": brief.mood,
    }

    prompt_lower = brief.prompt.lower()

    # Auto-detect genre from prompt
    if not plan["genre"]:
        for genre, defaults in GENRE_DEFAULTS.items():
            if genre in prompt_lower:
                plan["genre"] = genre
                break

    # Apply genre defaults
    if plan["genre"] in GENRE_DEFAULTS:
        defaults = GENRE_DEFAULTS[plan["genre"]]
        if plan["tempo"] == 0:
            plan["tempo"] = defaults["tempo"]
        if not plan["key"]:
            plan["key"] = defaults["key"]
        plan["style_tags"].extend(defaults["tags"])

    # Auto-detect mood
    if not plan["mood"]:
        for mood in MOOD_TAGS:
            if mood in prompt_lower:
                plan["mood"] = mood
                break

    # Apply mood tags
    if plan["mood"] in MOOD_TAGS:
        plan["style_tags"].extend(MOOD_TAGS[plan["mood"]])

    # Fallback defaults
    if plan["tempo"] == 0:
        plan["tempo"] = 120
    if not plan["key"]:
        plan["key"] = "C minor" if any(w in prompt_lower for w in ["sad", "dark", "minor", "melancholy"]) else "C major"

    # SFX suggestion from prompt keywords
    if brief.include_sfx:
        for keyword, sfx_prompt in SFX_SUGGESTIONS.items():
            if keyword in prompt_lower:
                plan["sfx_prompt"] = sfx_prompt
                break

    # Deduplicate tags
    plan["style_tags"] = list(dict.fromkeys(plan["style_tags"]))

    return plan


# ── Pipeline Executor ──────────────────────────────────────────────────────────

class _ProducerCancelled(RuntimeError):
    """Internal sentinel used to stop the pipeline at a safe stage boundary."""


class AIProducer:
    """
    AI Producer: one-prompt-to-full-song pipeline orchestrator.
    Chains all Slunder Studio modules automatically.
    """

    def __init__(self):
        self._output_dir = os.path.join(get_config_dir(), "generations", "ai_producer")
        os.makedirs(self._output_dir, exist_ok=True)
        self._current_result: Optional[ProducerResult] = None

    def produce(
        self,
        brief: ProducerBrief,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        step_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> ProducerResult:
        """Execute the pipeline with cooperative cancellation and truthful outcomes."""
        t0 = time.time()
        result = ProducerResult(brief=brief)
        self._current_result = result

        try:
            # Stage 1: Planning
            step = self._run_stage(PipelineStage.PLANNING, result,
                                   lambda: self._plan(brief), progress_callback,
                                   step_callback, log_callback, cancel_event)
            if self._must_stop(step, result):
                return self._finish(result, t0)

            plan = step.output_data

            # Stage 2: Lyrics
            step = self._run_stage(PipelineStage.LYRICS, result,
                                   lambda: self._generate_lyrics(plan, brief),
                                   progress_callback, step_callback, log_callback,
                                   cancel_event)
            if self._must_stop(step, result):
                return self._finish(result, t0)

            # Stage 3: Style
            step = self._run_stage(PipelineStage.STYLE, result,
                                   lambda: self._select_style(plan, brief),
                                   progress_callback, step_callback, log_callback,
                                   cancel_event)
            if self._must_stop(step, result):
                return self._finish(result, t0)

            # Stage 4: Song Generation (must succeed to continue)
            step = self._run_stage(PipelineStage.SONG_GEN, result,
                                   lambda: self._generate_song(
                                       plan,
                                       result,
                                       brief,
                                       progress_callback=lambda pct, message: self._emit_nested_progress(
                                           PipelineStage.SONG_GEN,
                                           pct,
                                           message,
                                           progress_callback,
                                           step_callback,
                                       ),
                                       step_callback=step_callback,
                                       log_callback=log_callback,
                                       cancel_event=cancel_event,
                                   ),
                                   progress_callback, step_callback, log_callback,
                                   cancel_event)
            if self._must_stop(step, result):
                return self._finish(result, t0)

            # Stage 5: Vocals (if requested)
            if brief.vocal_style and brief.vocal_style != "none":
                step = self._run_stage(PipelineStage.VOCALS, result,
                                       lambda: self._add_vocals(plan, result, brief),
                                       progress_callback, step_callback, log_callback,
                                       cancel_event, required=False)
                if self._must_stop(step, result):
                    return self._finish(result, t0)
            else:
                self._record_skipped(
                    PipelineStage.VOCALS,
                    result,
                    "No vocal layer requested",
                    progress_callback,
                    step_callback,
                )

            # Stage 6: SFX (if requested)
            if brief.include_sfx and plan.get("sfx_prompt"):
                step = self._run_stage(PipelineStage.SFX, result,
                                       lambda: self._add_sfx(plan, result, brief),
                                       progress_callback, step_callback, log_callback,
                                       cancel_event, required=False)
                if self._must_stop(step, result):
                    return self._finish(result, t0)
            else:
                self._record_skipped(
                    PipelineStage.SFX,
                    result,
                    "No matching SFX layer requested",
                    progress_callback,
                    step_callback,
                )

            # Stage 7: Mixing
            step = self._run_stage(PipelineStage.MIXING, result,
                                   lambda: self._mix(result, brief),
                                   progress_callback, step_callback, log_callback,
                                   cancel_event)
            if self._must_stop(step, result):
                return self._finish(result, t0)

            # Stage 8: Mastering
            step = self._run_stage(PipelineStage.MASTERING, result,
                                   lambda: self._master(result, brief),
                                   progress_callback, step_callback, log_callback,
                                   cancel_event)
            if self._must_stop(step, result):
                return self._finish(result, t0)

            if not result.can_export:
                step.status = "failed"
                step.error = "Mastering did not produce a new, readable audio artifact"
                result.stage = PipelineStage.FAILED
                result.error = f"Failed at mastering: {step.error}"
                if log_callback:
                    log_callback(result.error)
                return self._finish(result, t0)

            result.stage = PipelineStage.COMPLETE

            if progress_callback:
                progress_callback(1.0, "Production complete!")
            if step_callback:
                status = "demo" if result.is_demo else (
                    "degraded" if result.is_degraded else "complete"
                )
                step_callback(f"Production {status}")

        except Exception as e:
            result.stage = PipelineStage.FAILED
            result.error = f"{type(e).__name__}: {e}"
            if log_callback:
                log_callback(result.error)

        return self._finish(result, t0)

    def _run_stage(
        self,
        stage: PipelineStage,
        result: ProducerResult,
        func: Callable,
        progress_callback: Optional[Callable[[float, str], None]],
        step_callback: Optional[Callable[[str], None]],
        log_callback: Optional[Callable[[str], None]],
        cancel_event: Optional[threading.Event],
        *,
        required: bool = True,
    ) -> PipelineStep:
        """Execute a single pipeline stage with timing and error handling."""
        step = PipelineStep(stage=stage, status="running", start_time=time.time())
        result.steps.append(step)
        result.stage = stage

        stage_idx = PIPELINE_ORDER.index(stage) if stage in PIPELINE_ORDER else 0
        base_progress = stage_idx / len(PIPELINE_ORDER)
        label = stage.value.replace("_", " ").title()

        if progress_callback:
            progress_callback(base_progress, f"{label}...")
        if step_callback:
            step_callback(f"{label}: running")
        if log_callback:
            log_callback(f"{label} started")

        try:
            if cancel_event and cancel_event.is_set():
                raise _ProducerCancelled()
            output = func()
            if cancel_event and cancel_event.is_set():
                raise _ProducerCancelled()
            step.output_data = output if isinstance(output, dict) else {"result": output}
            if step.output_data.get("cancelled"):
                raise _ProducerCancelled()
            if step.output_data.get("error"):
                raise RuntimeError(str(step.output_data["error"]))

            reported_status = str(step.output_data.get("status", ""))
            if reported_status.startswith("skipped"):
                step.status = "skipped"
                reason = str(
                    step.output_data.get("note")
                    or reported_status.replace("_", " ")
                )
                result.add_degraded_reason(f"{label}: {reason}")
            else:
                step.status = "complete"

            if step.output_data.get("demo") or step.output_data.get("fallback"):
                result.output_kind = "demo"
                result.add_degraded_reason(
                    f"{label}: explicit demo fallback was used"
                )
        except _ProducerCancelled:
            step.status = "cancelled"
            step.error = "Cancellation requested"
            result.cancelled = True
            result.stage = PipelineStage.CANCELLED
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            if required:
                result.error = f"Failed at {stage.value}: {e}"
                result.stage = PipelineStage.FAILED
            else:
                result.add_degraded_reason(f"{label} failed: {e}")

        step.end_time = time.time()
        terminal_progress = (stage_idx + 1) / len(PIPELINE_ORDER)
        if progress_callback:
            progress_callback(
                terminal_progress,
                f"{label}: {step.status}",
            )
        if step_callback:
            step_callback(f"{label}: {step.status}")
        if log_callback:
            detail = f" ({step.error})" if step.error else ""
            log_callback(f"{label} {step.status}{detail}")
        return step

    @staticmethod
    def _finish(result: ProducerResult, started_at: float) -> ProducerResult:
        result.total_time = time.time() - started_at
        return result

    @staticmethod
    def _must_stop(step: PipelineStep, result: ProducerResult) -> bool:
        if step.status == "cancelled":
            result.cancelled = True
            result.stage = PipelineStage.CANCELLED
            return True
        if step.status == "failed" and step.stage in REQUIRED_STAGES:
            result.stage = PipelineStage.FAILED
            return True
        return False

    @staticmethod
    def _record_skipped(
        stage: PipelineStage,
        result: ProducerResult,
        reason: str,
        progress_callback: Optional[Callable[[float, str], None]],
        step_callback: Optional[Callable[[str], None]],
    ) -> None:
        now = time.time()
        result.steps.append(PipelineStep(
            stage=stage,
            status="skipped",
            start_time=now,
            end_time=now,
            output_data={"reason": reason},
        ))
        stage_idx = PIPELINE_ORDER.index(stage)
        label = stage.value.replace("_", " ").title()
        if progress_callback:
            progress_callback(
                (stage_idx + 1) / len(PIPELINE_ORDER),
                f"{label}: skipped",
            )
        if step_callback:
            step_callback(f"{label}: skipped")

    @staticmethod
    def _emit_nested_progress(
        stage: PipelineStage,
        pct: int | float,
        message: str,
        progress_callback: Optional[Callable[[float, str], None]],
        step_callback: Optional[Callable[[str], None]],
    ) -> None:
        stage_idx = PIPELINE_ORDER.index(stage)
        normalized = max(0.0, min(1.0, float(pct) / 100.0))
        if progress_callback:
            progress_callback(
                (stage_idx + normalized) / len(PIPELINE_ORDER),
                message,
            )
        if step_callback and message:
            step_callback(message)

    # ── Pipeline Stage Implementations ─────────────────────────────────────────

    def _plan(self, brief: ProducerBrief) -> dict:
        """Analyze brief and create production plan."""
        return analyze_brief(brief)

    def _generate_lyrics(self, plan: dict, brief: ProducerBrief) -> dict:
        """Generate lyrics using the Lyrics Engine."""
        try:
            from engines.lyrics_engine import LyricsLLM
            engine = LyricsLLM()

            genre = plan.get("genre", "pop")
            mood = plan.get("mood", "")
            prompt = f"Write lyrics for a {genre} song. {brief.prompt}"
            if mood:
                prompt += f" The mood is {mood}."

            # If engine has a model loaded, use it
            if engine.is_loaded:
                result = engine.generate(prompt)
                lyrics = result.get("text", "")
            else:
                lyrics = f"[Verse 1]\n{brief.prompt}\n\n[Chorus]\n{brief.prompt}\n"

        except Exception:
            lyrics = f"[Verse 1]\n{brief.prompt}\n\n[Chorus]\n{brief.prompt}\n"

        self._current_result.lyrics_text = lyrics
        return {"lyrics": lyrics}

    def _select_style(self, plan: dict, brief: ProducerBrief) -> dict:
        """Select style tags for generation."""
        tags = plan.get("style_tags", [])
        self._current_result.style_tags = tags
        return {"tags": tags, "tempo": plan["tempo"], "key": plan["key"]}

    def _generate_song(
        self,
        plan: dict,
        result: ProducerResult,
        brief: ProducerBrief,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        step_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> dict:
        """Generate the instrumental track."""
        try:
            from engines.ace_step_engine import generate_song

            def _progress(pct: int) -> None:
                if progress_callback:
                    progress_callback(pct, f"Song generation: {pct}%")

            song_result = generate_song(
                lyrics=result.lyrics_text,
                tags=", ".join(result.style_tags),
                duration=brief.duration_seconds,
                seed=brief.seed,
                progress_cb=_progress,
                step_cb=step_callback,
                log_cb=log_callback,
                cancel_event=cancel_event,
            )
            if cancel_event and cancel_event.is_set():
                return {"cancelled": True}
            if isinstance(song_result, dict) and song_result.get("cancelled"):
                return {"cancelled": True}
            audio_path = (
                song_result.get("audio_path", "")
                if isinstance(song_result, dict)
                else str(song_result)
            )
            if not _verify_audio_artifact(audio_path):
                raise RuntimeError(
                    "Song generation completed without a readable audio artifact"
                )
            result.song_audio_path = audio_path
            result.add_artifact(
                audio_path,
                song_result.get("provenance_path")
                if isinstance(song_result, dict) else None,
                song_result.get("vocal_stem_path")
                if isinstance(song_result, dict) else None,
                song_result.get("vocal_stem_provenance_path")
                if isinstance(song_result, dict) else None,
            )
            if isinstance(song_result, dict):
                song_result["audio_path"] = audio_path
                return song_result
            return {"audio_path": audio_path}
        except Exception as exc:
            if cancel_event and cancel_event.is_set():
                return {"cancelled": True}
            if not brief.demo_fallback:
                raise RuntimeError(
                    f"Song generation failed: {exc}. "
                    "Enable 'Demo Fallback' to continue with a silent placeholder."
                ) from exc

            path = os.path.join(self._output_dir, f"song_demo_{result.run_id}.wav")
            sr = 44100
            n = int(brief.duration_seconds * sr)
            silence = np.zeros((n, 2), dtype=np.int16)
            with wave.open(path, "w") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(silence.tobytes())
            provenance_path = write_provenance_sidecar(
                path,
                module="ai_producer",
                operation="generate_song_fallback",
                seed=brief.seed,
                prompt=brief.prompt,
                lyrics=result.lyrics_text,
                parameters=asdict(brief),
                export_format="wav",
                output_kind="demo",
                extra={
                    "fallback": True,
                    "demo_fallback": True,
                    "reason": "song_generation_failed",
                    "original_error": str(exc),
                },
            )
            result.song_audio_path = path
            result.add_artifact(path, provenance_path)
            return {
                "audio_path": path,
                "provenance_path": str(provenance_path),
                "fallback": True,
                "demo": True,
            }

    def _add_vocals(self, plan: dict, result: ProducerResult,
                    brief: ProducerBrief) -> dict:
        """Add vocal synthesis."""
        # In production this would call DiffSinger or GPT-SoVITS
        return {"status": "skipped_no_model", "note": "Vocal model not loaded"}

    def _add_sfx(self, plan: dict, result: ProducerResult,
                 brief: ProducerBrief) -> dict:
        """Generate and add SFX layer."""
        from engines.sfx_engine import SFXParams, generate_sfx

        sfx_prompt = plan.get("sfx_prompt", "ambient texture")
        params = SFXParams(
            prompt=sfx_prompt,
            duration=min(brief.duration_seconds, 30.0),
            seed=brief.seed,
            allow_demo_output=brief.demo_fallback,
        )
        sfx_result = generate_sfx(params)
        if not sfx_result.is_success or not _verify_audio_artifact(sfx_result.file_path):
            raise RuntimeError(sfx_result.error or "SFX generation produced no readable audio")
        result.sfx_audio_path = sfx_result.file_path
        result.add_artifact(sfx_result.file_path, sfx_result.provenance_path)
        return {
            "sfx_path": sfx_result.file_path,
            "provenance_path": sfx_result.provenance_path,
            "prompt": sfx_prompt,
            "demo": sfx_result.is_demo,
            "output_kind": sfx_result.output_kind,
        }

    def _mix(self, result: ProducerResult, brief: ProducerBrief) -> dict:
        """Mix all layers together."""
        import soundfile as sf

        layers = []
        sr = 0

        # Load song
        if not _verify_audio_artifact(result.song_audio_path):
            raise RuntimeError("Generated song artifact is missing or unreadable")
        audio, sr = sf.read(
            result.song_audio_path,
            dtype="float32",
            always_2d=True,
        )
        audio = normalize_channel_layout(audio, target_channels=2)
        layers.append(("song", audio, 1.0))

        # Load SFX (at lower volume)
        if result.sfx_audio_path:
            if not _verify_audio_artifact(result.sfx_audio_path):
                raise RuntimeError("SFX artifact is missing or unreadable")
            sfx_audio, sfx_sr = sf.read(
                result.sfx_audio_path,
                dtype="float32",
                always_2d=True,
            )
            sfx_audio = normalize_channel_layout(sfx_audio, target_channels=2)
            if sfx_sr != sr:
                sfx_audio = resample_audio(sfx_audio, sfx_sr, sr)
            layers.append(("sfx", sfx_audio, 0.15))

        if not layers:
            raise RuntimeError("No audio layers to mix")

        # Mix
        max_len = max(len(a) for _, a, _ in layers)
        mixed = np.zeros((max_len, 2), dtype=np.float32)
        for name, audio, vol in layers:
            length = min(len(audio), max_len)
            mixed[:length] += audio[:length] * vol

        # Clip
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed /= peak

        # Save
        mix_path = os.path.join(self._output_dir, f"mix_{result.run_id}.wav")
        sf.write(mix_path, mixed, sr, subtype="PCM_24")
        provenance_path = write_provenance_sidecar(
            mix_path,
            module="ai_producer",
            operation="mix",
            seed=brief.seed,
            prompt=brief.prompt,
            lyrics=result.lyrics_text,
            parameters={"brief": asdict(brief), "layers": [name for name, _, _ in layers]},
            source_paths=[
                path for path in [result.song_audio_path, result.sfx_audio_path]
                if path
            ],
            export_format="wav",
            output_kind=result.output_kind,
        )
        result.add_artifact(mix_path, provenance_path)

        return {
            "mix_path": mix_path,
            "provenance_path": str(provenance_path),
            "layers": len(layers),
            "duration": max_len / sr,
            "sample_rate": sr,
        }

    def _master(self, result: ProducerResult, brief: ProducerBrief) -> dict:
        """Apply mastering to the final mix."""
        from core.mastering import master_audio, PRESETS

        mix_step = result.get_step(PipelineStage.MIXING)
        if not mix_step or mix_step.status != "complete" or not mix_step.output_data:
            raise RuntimeError("No verified mix to master")

        mix_path = mix_step.output_data.get("mix_path")
        if not _verify_audio_artifact(mix_path):
            raise RuntimeError("Mix file is missing or unreadable")

        # Load mix
        import soundfile as sf
        audio, sr = sf.read(mix_path, dtype="float32", always_2d=True)
        audio = normalize_channel_layout(audio, target_channels=2)

        preset = PRESETS.get(brief.mastering_preset, PRESETS["Balanced"])
        master_result = master_audio(audio, sr, preset)

        if master_result.error:
            raise RuntimeError(master_result.error)

        # Save mastered
        master_path = os.path.join(
            self._output_dir,
            f"mastered_{result.run_id}.wav",
        )
        sf.write(master_path, master_result.audio, sr, subtype="PCM_24")
        provenance_path = write_provenance_sidecar(
            master_path,
            module="ai_producer",
            operation="master",
            seed=brief.seed,
            prompt=brief.prompt,
            lyrics=result.lyrics_text,
            parameters={
                "brief": asdict(brief),
                "preset": brief.mastering_preset,
                "input_lufs": master_result.input_lufs,
                "output_lufs": master_result.output_lufs,
                "peak_db": master_result.peak_db,
            },
            source_paths=[mix_path],
            export_format="wav",
            output_kind=result.output_kind,
        )

        result.mastered_audio_path = master_path
        result.final_audio_path = master_path
        result.add_artifact(master_path, provenance_path)

        return {
            "master_path": master_path,
            "provenance_path": str(provenance_path),
            "input_lufs": master_result.input_lufs,
            "output_lufs": master_result.output_lufs,
            "peak_db": master_result.peak_db,
            "preset": brief.mastering_preset,
        }


# ── High-Level ─────────────────────────────────────────────────────────────────

_producer: Optional[AIProducer] = None


def get_producer() -> AIProducer:
    global _producer
    if _producer is None:
        _producer = AIProducer()
    return _producer


def produce_song(
    brief: ProducerBrief,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    progress_cb: Optional[Callable[[int], None]] = None,
    step_cb: Optional[Callable[[str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    **_kwargs,
) -> ProducerResult:
    """One-shot song production from brief. Called by InferenceWorker."""
    def _report(progress: float, message: str) -> None:
        if progress_callback:
            progress_callback(progress, message)
        if progress_cb:
            progress_cb(round(max(0.0, min(1.0, progress)) * 100))

    return get_producer().produce(
        brief,
        progress_callback=_report,
        step_callback=step_cb,
        log_callback=log_cb,
        cancel_event=cancel_event,
    )
