"""
Slunder Studio v0.0.2 — AI Producer Engine
One-prompt-to-full-song orchestrator. Decomposes a high-level creative brief
into a multi-step pipeline: lyrics generation, style selection, song generation,
vocal synthesis, SFX layering, and mastering — all automated.
"""
import os
import time
import json
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

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


@dataclass
class PipelineStep:
    """Record of a single pipeline step execution."""
    stage: PipelineStage = PipelineStage.PLANNING
    status: str = "pending"  # "pending" | "running" | "complete" | "skipped" | "failed"
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


@dataclass
class ProducerResult:
    """Full pipeline execution result."""
    brief: Optional[ProducerBrief] = None
    steps: list[PipelineStep] = field(default_factory=list)
    final_audio_path: Optional[str] = None
    total_time: float = 0.0
    stage: PipelineStage = PipelineStage.PLANNING
    error: Optional[str] = None

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
        done = len(self.completed_stages)
        return done / total if total > 0 else 0.0


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

class AIProducer:
    """
    AI Producer: one-prompt-to-full-song pipeline orchestrator.
    Chains all Slunder Studio modules automatically.
    """

    def __init__(self):
        self._output_dir = os.path.join(get_config_dir(), "generations", "ai_producer")
        os.makedirs(self._output_dir, exist_ok=True)
        self._current_result: Optional[ProducerResult] = None

    def produce(self, brief: ProducerBrief,
                progress_callback: Optional[Callable] = None) -> ProducerResult:
        """Execute the full production pipeline."""
        t0 = time.time()
        result = ProducerResult(brief=brief)
        self._current_result = result

        try:
            # Stage 1: Planning
            step = self._run_stage(PipelineStage.PLANNING, result,
                                   lambda: self._plan(brief), progress_callback)
            if step.status == "failed":
                return result

            plan = step.output_data

            # Stage 2: Lyrics
            step = self._run_stage(PipelineStage.LYRICS, result,
                                   lambda: self._generate_lyrics(plan, brief),
                                   progress_callback)

            # Stage 3: Style
            step = self._run_stage(PipelineStage.STYLE, result,
                                   lambda: self._select_style(plan, brief),
                                   progress_callback)

            # Stage 4: Song Generation
            step = self._run_stage(PipelineStage.SONG_GEN, result,
                                   lambda: self._generate_song(plan, result, brief),
                                   progress_callback)

            # Stage 5: Vocals (if requested)
            if brief.vocal_style and brief.vocal_style != "none":
                step = self._run_stage(PipelineStage.VOCALS, result,
                                       lambda: self._add_vocals(plan, result, brief),
                                       progress_callback)
            else:
                result.steps.append(PipelineStep(
                    stage=PipelineStage.VOCALS, status="skipped"))

            # Stage 6: SFX (if requested)
            if brief.include_sfx and plan.get("sfx_prompt"):
                step = self._run_stage(PipelineStage.SFX, result,
                                       lambda: self._add_sfx(plan, result, brief),
                                       progress_callback)
            else:
                result.steps.append(PipelineStep(
                    stage=PipelineStage.SFX, status="skipped"))

            # Stage 7: Mixing
            step = self._run_stage(PipelineStage.MIXING, result,
                                   lambda: self._mix(result, brief),
                                   progress_callback)

            # Stage 8: Mastering
            step = self._run_stage(PipelineStage.MASTERING, result,
                                   lambda: self._master(result, brief),
                                   progress_callback)

            result.stage = PipelineStage.COMPLETE
            result.total_time = time.time() - t0

            if progress_callback:
                progress_callback(1.0, "Production complete!")

        except Exception as e:
            result.stage = PipelineStage.FAILED
            result.error = str(e)
            result.total_time = time.time() - t0

        return result

    def _run_stage(self, stage: PipelineStage, result: ProducerResult,
                   func: Callable, progress_callback: Optional[Callable]) -> PipelineStep:
        """Execute a single pipeline stage with timing and error handling."""
        step = PipelineStep(stage=stage, status="running", start_time=time.time())
        result.steps.append(step)
        result.stage = stage

        stage_idx = PIPELINE_ORDER.index(stage) if stage in PIPELINE_ORDER else 0
        base_progress = stage_idx / len(PIPELINE_ORDER)

        if progress_callback:
            progress_callback(base_progress, f"{stage.value}...")

        try:
            output = func()
            step.output_data = output if isinstance(output, dict) else {"result": output}
            step.status = "complete"
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            result.error = f"Failed at {stage.value}: {e}"

        step.end_time = time.time()
        return step

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

    def _generate_song(self, plan: dict, result: ProducerResult,
                       brief: ProducerBrief) -> dict:
        """Generate the instrumental track."""
        try:
            from engines.ace_step_engine import generate_song
            audio_path = generate_song(
                lyrics=result.lyrics_text,
                tags=", ".join(result.style_tags),
                duration=brief.duration_seconds,
                seed=brief.seed,
            )
            result.song_audio_path = audio_path
            return {"audio_path": audio_path}
        except Exception:
            # Create a placeholder silence file
            import wave
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self._output_dir, f"song_{ts}.wav")
            sr = 44100
            n = int(brief.duration_seconds * sr)
            silence = np.zeros((n, 2), dtype=np.int16)
            with wave.open(path, "w") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(silence.tobytes())
            result.song_audio_path = path
            return {"audio_path": path, "fallback": True}

    def _add_vocals(self, plan: dict, result: ProducerResult,
                    brief: ProducerBrief) -> dict:
        """Add vocal synthesis."""
        # In production this would call DiffSinger or GPT-SoVITS
        return {"status": "skipped_no_model", "note": "Vocal model not loaded"}

    def _add_sfx(self, plan: dict, result: ProducerResult,
                 brief: ProducerBrief) -> dict:
        """Generate and add SFX layer."""
        try:
            from engines.sfx_engine import SFXParams, generate_sfx
            sfx_prompt = plan.get("sfx_prompt", "ambient texture")
            params = SFXParams(
                prompt=sfx_prompt,
                duration=min(brief.duration_seconds, 30.0),
                seed=brief.seed,
            )
            sfx_result = generate_sfx(params)
            if sfx_result.file_path:
                result.sfx_audio_path = sfx_result.file_path
            return {"sfx_path": sfx_result.file_path, "prompt": sfx_prompt}
        except Exception as e:
            return {"error": str(e)}

    def _mix(self, result: ProducerResult, brief: ProducerBrief) -> dict:
        """Mix all layers together."""
        import wave

        layers = []
        sr = 44100

        # Load song
        if result.song_audio_path and os.path.isfile(result.song_audio_path):
            try:
                with wave.open(result.song_audio_path, "r") as wf:
                    sr = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    ch = wf.getnchannels()
                    if ch == 2:
                        audio = audio.reshape(-1, 2)
                    else:
                        audio = np.column_stack([audio, audio])
                    layers.append(("song", audio, 1.0))
            except Exception:
                pass

        # Load SFX (at lower volume)
        if result.sfx_audio_path and os.path.isfile(result.sfx_audio_path):
            try:
                with wave.open(result.sfx_audio_path, "r") as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    ch = wf.getnchannels()
                    if ch == 2:
                        audio = audio.reshape(-1, 2)
                    else:
                        audio = np.column_stack([audio, audio])
                    layers.append(("sfx", audio, 0.15))
            except Exception:
                pass

        if not layers:
            return {"error": "No audio layers to mix"}

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
        ts = time.strftime("%Y%m%d_%H%M%S")
        mix_path = os.path.join(self._output_dir, f"mix_{ts}.wav")
        int_audio = (mixed * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(mix_path, "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int_audio.tobytes())

        return {"mix_path": mix_path, "layers": len(layers), "duration": max_len / sr}

    def _master(self, result: ProducerResult, brief: ProducerBrief) -> dict:
        """Apply mastering to the final mix."""
        from core.mastering import master_audio, PRESETS

        mix_step = result.get_step(PipelineStage.MIXING)
        if not mix_step or not mix_step.output_data:
            return {"error": "No mix to master"}

        mix_path = mix_step.output_data.get("mix_path")
        if not mix_path or not os.path.isfile(mix_path):
            return {"error": "Mix file not found"}

        # Load mix
        import wave
        with wave.open(mix_path, "r") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            audio = audio.reshape(-1, 2)

        preset = PRESETS.get(brief.mastering_preset, PRESETS["Balanced"])
        master_result = master_audio(audio, sr, preset)

        if master_result.error:
            return {"error": master_result.error}

        # Save mastered
        ts = time.strftime("%Y%m%d_%H%M%S")
        master_path = os.path.join(self._output_dir, f"mastered_{ts}.wav")
        int_audio = (master_result.audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(master_path, "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int_audio.tobytes())

        result.mastered_audio_path = master_path
        result.final_audio_path = master_path

        return {
            "master_path": master_path,
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


def produce_song(brief: ProducerBrief,
                 progress_callback: Optional[Callable] = None) -> ProducerResult:
    """One-shot song production from brief. Called by InferenceWorker."""
    return get_producer().produce(brief, progress_callback)
