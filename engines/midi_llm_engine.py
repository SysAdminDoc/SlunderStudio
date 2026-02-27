"""
Slunder Studio v0.0.2 — MIDI-LLM Engine
Text-to-MIDI generation using fine-tuned language models that output MIDI token sequences.
Supports prompt-based composition, continuation, and style-conditioned generation.
"""
import os
import json
import re
import time
from typing import Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

from core.settings import get_config_dir
from core.midi_utils import MidiData, TrackData, NoteData


# ── MIDI Token Vocabulary ──────────────────────────────────────────────────────

# Token format: "NOTE_ON p=60 v=100 t=0.000" / "NOTE_OFF p=60 t=0.500"
# Track delimiter: "TRACK name=Piano program=0"
# Tempo: "TEMPO bpm=120"
# Time sig: "TIMESIG 4/4"

TOKEN_PATTERNS = {
    "tempo": re.compile(r"TEMPO\s+bpm=(\d+(?:\.\d+)?)"),
    "timesig": re.compile(r"TIMESIG\s+(\d+)/(\d+)"),
    "track": re.compile(r"TRACK\s+name=(.+?)\s+program=(\d+)(?:\s+drum=(true|false))?"),
    "note_on": re.compile(r"NOTE_ON\s+p=(\d+)\s+v=(\d+)\s+t=(\d+(?:\.\d+)?)"),
    "note_off": re.compile(r"NOTE_OFF\s+p=(\d+)\s+t=(\d+(?:\.\d+)?)"),
    "end": re.compile(r"END_TRACK"),
}


@dataclass
class MidiGenParams:
    """Parameters for MIDI generation."""
    prompt: str = ""
    style: str = ""  # e.g. "jazz piano ballad", "rock drums aggressive"
    key: str = "C major"
    tempo: float = 120.0
    time_signature: tuple = (4, 4)
    duration_bars: int = 16
    instruments: list[str] = field(default_factory=lambda: ["Piano"])
    temperature: float = 0.85
    top_p: float = 0.92
    max_tokens: int = 4096
    seed: Optional[int] = None
    continuation_context: Optional[str] = None  # existing MIDI tokens to continue from


@dataclass
class MidiGenResult:
    """Result from MIDI generation."""
    midi_data: Optional[MidiData] = None
    raw_tokens: str = ""
    generation_time: float = 0.0
    token_count: int = 0
    error: Optional[str] = None


# ── Prompt Templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a MIDI composition engine. Generate MIDI token sequences for musical compositions.

Output format:
TEMPO bpm=<BPM>
TIMESIG <num>/<den>
TRACK name=<name> program=<GM_program> [drum=true]
NOTE_ON p=<pitch_0-127> v=<velocity_0-127> t=<time_seconds>
NOTE_OFF p=<pitch> t=<time_seconds>
END_TRACK

Rules:
- Every NOTE_ON must have a matching NOTE_OFF
- Notes must be in chronological order by start time within each track
- Use General MIDI program numbers
- Drum tracks use channel 9 (drum=true)
- Time values are in seconds
- Generate musically coherent compositions"""


def build_generation_prompt(params: MidiGenParams) -> str:
    """Build the LLM prompt for MIDI generation."""
    parts = [f"Compose a MIDI piece with these specifications:"]

    if params.prompt:
        parts.append(f"Description: {params.prompt}")
    if params.style:
        parts.append(f"Style: {params.style}")

    parts.append(f"Key: {params.key}")
    parts.append(f"Tempo: {params.tempo} BPM")
    parts.append(f"Time Signature: {params.time_signature[0]}/{params.time_signature[1]}")
    parts.append(f"Length: {params.duration_bars} bars")
    parts.append(f"Instruments: {', '.join(params.instruments)}")

    if params.continuation_context:
        parts.append(f"\nContinue from this existing sequence:\n{params.continuation_context}")
        parts.append("\nGenerate the next section:")
    else:
        parts.append("\nGenerate the MIDI token sequence:")

    return "\n".join(parts)


# ── Token Parser ───────────────────────────────────────────────────────────────

def parse_midi_tokens(token_text: str) -> MidiData:
    """Parse MIDI token sequence into MidiData."""
    midi_data = MidiData()
    current_track: Optional[TrackData] = None
    pending_notes: dict[int, NoteData] = {}  # pitch -> note (awaiting NOTE_OFF)

    for line in token_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Tempo
        m = TOKEN_PATTERNS["tempo"].match(line)
        if m:
            midi_data.tempo = float(m.group(1))
            continue

        # Time signature
        m = TOKEN_PATTERNS["timesig"].match(line)
        if m:
            midi_data.time_signature = (int(m.group(1)), int(m.group(2)))
            continue

        # Track header
        m = TOKEN_PATTERNS["track"].match(line)
        if m:
            # Finalize previous track
            if current_track is not None:
                _close_pending(current_track, pending_notes)
                midi_data.tracks.append(current_track)
            is_drum = m.group(3) == "true" if m.group(3) else False
            current_track = TrackData(
                name=m.group(1),
                program=int(m.group(2)),
                is_drum=is_drum,
                channel=9 if is_drum else len(midi_data.tracks),
            )
            pending_notes.clear()
            continue

        # Note on
        m = TOKEN_PATTERNS["note_on"].match(line)
        if m and current_track is not None:
            pitch = int(m.group(1))
            vel = int(m.group(2))
            t = float(m.group(3))
            note = NoteData(pitch=pitch, start=t, end=t + 0.5, velocity=vel,
                            channel=current_track.channel)
            pending_notes[pitch] = note
            continue

        # Note off
        m = TOKEN_PATTERNS["note_off"].match(line)
        if m and current_track is not None:
            pitch = int(m.group(1))
            t = float(m.group(2))
            if pitch in pending_notes:
                pending_notes[pitch].end = t
                current_track.notes.append(pending_notes.pop(pitch))
            continue

        # End track
        m = TOKEN_PATTERNS["end"].match(line)
        if m and current_track is not None:
            _close_pending(current_track, pending_notes)
            midi_data.tracks.append(current_track)
            current_track = None
            pending_notes.clear()
            continue

    # Handle unclosed final track
    if current_track is not None:
        _close_pending(current_track, pending_notes)
        midi_data.tracks.append(current_track)

    # Calculate total duration
    if midi_data.tracks:
        midi_data.duration = max(
            (t.duration for t in midi_data.tracks if t.notes), default=0.0
        )

    return midi_data


def _close_pending(track: TrackData, pending: dict[int, NoteData]):
    """Close any pending notes that never got NOTE_OFF."""
    for pitch, note in pending.items():
        note.end = note.start + 0.25  # default short duration
        track.notes.append(note)
    pending.clear()


def midi_data_to_tokens(midi_data: MidiData) -> str:
    """Convert MidiData back to token sequence (for continuation prompts)."""
    lines = []
    lines.append(f"TEMPO bpm={midi_data.tempo:.1f}")
    lines.append(f"TIMESIG {midi_data.time_signature[0]}/{midi_data.time_signature[1]}")

    for track in midi_data.tracks:
        drum_flag = " drum=true" if track.is_drum else ""
        lines.append(f"TRACK name={track.name} program={track.program}{drum_flag}")

        # Sort notes by start time
        sorted_notes = sorted(track.notes, key=lambda n: (n.start, n.pitch))

        # Build event list (note_on and note_off interleaved by time)
        events = []
        for note in sorted_notes:
            events.append(("on", note.start, note))
            events.append(("off", note.end, note))
        events.sort(key=lambda e: (e[1], 0 if e[0] == "off" else 1))

        for evt_type, t, note in events:
            if evt_type == "on":
                lines.append(f"NOTE_ON p={note.pitch} v={note.velocity} t={t:.3f}")
            else:
                lines.append(f"NOTE_OFF p={note.pitch} t={t:.3f}")

        lines.append("END_TRACK")

    return "\n".join(lines)


# ── Engine ─────────────────────────────────────────────────────────────────────

class MidiLLMEngine:
    """
    MIDI-LLM generation engine.
    Uses a fine-tuned language model to generate MIDI token sequences.
    Supports: text-to-MIDI, continuation, style transfer.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._model_id: Optional[str] = None
        self._device = "cpu"
        self._generation_dir = os.path.join(get_config_dir(), "generations", "midi_studio")
        os.makedirs(self._generation_dir, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self, model_path: str, device: str = "cuda",
                   progress_callback: Optional[Callable] = None):
        """Load MIDI-LLM model from local path or HuggingFace ID."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if progress_callback:
                progress_callback(0.1, "Loading tokenizer...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            if progress_callback:
                progress_callback(0.3, "Loading model...")

            dtype = torch.float16 if device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True,
            )

            if device == "cpu":
                self.model = self.model.to("cpu")

            self.model.eval()
            self._model_id = model_path
            self._device = device

            if progress_callback:
                progress_callback(1.0, "Model loaded")

        except Exception as e:
            self.model = None
            self.tokenizer = None
            raise RuntimeError(f"Failed to load MIDI-LLM model: {e}") from e

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._model_id = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def generate(self, params: MidiGenParams) -> MidiGenResult:
        """Generate MIDI from parameters."""
        if not self.is_loaded:
            return MidiGenResult(error="Model not loaded")

        t0 = time.time()
        try:
            import torch
            prompt = build_generation_prompt(params)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            # Format for chat model
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = f"{SYSTEM_PROMPT}\n\n{prompt}\n"

            inputs = self.tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self._device)

            gen_kwargs = {
                "max_new_tokens": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if params.seed is not None:
                torch.manual_seed(params.seed)

            with torch.no_grad():
                output = self.model.generate(input_ids, **gen_kwargs)

            # Decode only new tokens
            new_tokens = output[0][input_ids.shape[1]:]
            raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Parse tokens into MidiData
            midi_data = parse_midi_tokens(raw_text)

            # Override tempo/time sig from params if not in output
            if midi_data.tempo == 120.0 and params.tempo != 120.0:
                midi_data.tempo = params.tempo
            if midi_data.time_signature == (4, 4) and params.time_signature != (4, 4):
                midi_data.time_signature = params.time_signature

            gen_time = time.time() - t0

            return MidiGenResult(
                midi_data=midi_data,
                raw_tokens=raw_text,
                generation_time=gen_time,
                token_count=len(new_tokens),
            )

        except Exception as e:
            return MidiGenResult(error=str(e), generation_time=time.time() - t0)

    def continue_sequence(self, existing: MidiData, params: MidiGenParams) -> MidiGenResult:
        """Continue an existing MIDI sequence."""
        context_tokens = midi_data_to_tokens(existing)
        params.continuation_context = context_tokens
        return self.generate(params)

    def save_generation(self, result: MidiGenResult, name: Optional[str] = None) -> Optional[str]:
        """Save generation result to disk."""
        if result.midi_data is None:
            return None

        from core.midi_utils import save_midi

        if name is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = f"midi_gen_{ts}"

        midi_path = os.path.join(self._generation_dir, f"{name}.mid")
        save_midi(result.midi_data, midi_path)

        # Save metadata
        meta = {
            "generation_time": result.generation_time,
            "token_count": result.token_count,
            "tracks": result.midi_data.track_count,
            "total_notes": result.midi_data.total_notes,
            "duration": result.midi_data.duration,
            "tempo": result.midi_data.tempo,
        }
        meta_path = os.path.join(self._generation_dir, f"{name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return midi_path


# ── Demo / Fallback Generator ─────────────────────────────────────────────────

def generate_demo_midi(params: MidiGenParams) -> MidiData:
    """
    Generate a simple demo MIDI without a model (fallback/preview).
    Creates basic chord progressions and melodies algorithmically.
    """
    import random

    if params.seed is not None:
        random.seed(params.seed)

    midi_data = MidiData(
        tempo=params.tempo,
        time_signature=params.time_signature,
    )

    beat_dur = 60.0 / params.tempo
    bar_dur = beat_dur * params.time_signature[0]
    total_dur = bar_dur * params.duration_bars

    # Key root mapping
    key_roots = {
        "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
        "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
        "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71,
    }
    key_parts = params.key.split()
    root = key_roots.get(key_parts[0], 60)
    is_minor = "minor" in params.key.lower() or "min" in params.key.lower()

    scale = [0, 2, 3, 5, 7, 8, 10] if is_minor else [0, 2, 4, 5, 7, 9, 11]

    # Common chord progressions
    if is_minor:
        prog = random.choice([
            [0, 5, 3, 4],  # i-VI-iv-v
            [0, 3, 5, 4],  # i-iv-VI-v
            [0, 5, 6, 4],  # i-VI-VII-v
        ])
    else:
        prog = random.choice([
            [0, 4, 5, 3],  # I-V-vi-IV
            [0, 5, 3, 4],  # I-vi-IV-V
            [0, 3, 4, 4],  # I-IV-V-V
        ])

    def scale_note(degree: int, octave_offset: int = 0) -> int:
        oct = degree // 7
        deg = degree % 7
        return root + scale[deg] + (oct + octave_offset) * 12

    # Piano chords track
    piano = TrackData(name="Piano", program=0, channel=0)
    for bar in range(params.duration_bars):
        chord_deg = prog[bar % len(prog)]
        chord_root = scale_note(chord_deg, -1)
        # Triad
        chord_notes = [chord_root, chord_root + scale[2 % 7], chord_root + scale[4 % 7]]
        t = bar * bar_dur
        for p in chord_notes:
            piano.notes.append(NoteData(
                pitch=max(0, min(127, p)),
                start=t, end=t + bar_dur * 0.95,
                velocity=random.randint(70, 90),
            ))
    midi_data.tracks.append(piano)

    # Melody track
    melody = TrackData(name="Melody", program=0, channel=1)
    t = 0.0
    prev_deg = 0
    while t < total_dur:
        deg = prev_deg + random.choice([-2, -1, 0, 1, 1, 2])
        deg = max(-3, min(10, deg))
        pitch = scale_note(deg, 1)

        dur_choices = [beat_dur * 0.5, beat_dur, beat_dur * 1.5, beat_dur * 2]
        dur = random.choice(dur_choices)

        if random.random() > 0.15:  # 85% note density
            melody.notes.append(NoteData(
                pitch=max(0, min(127, pitch)),
                start=t, end=min(t + dur * 0.9, total_dur),
                velocity=random.randint(80, 110),
            ))

        prev_deg = deg
        t += dur
    midi_data.tracks.append(melody)

    # Bass track
    bass = TrackData(name="Bass", program=33, channel=2)
    for bar in range(params.duration_bars):
        chord_deg = prog[bar % len(prog)]
        bass_pitch = scale_note(chord_deg, -2)
        t = bar * bar_dur
        # Root on beats 1 and 3
        for beat in [0, 2]:
            bt = t + beat * beat_dur
            if bt < total_dur:
                bass.notes.append(NoteData(
                    pitch=max(0, min(127, bass_pitch)),
                    start=bt, end=bt + beat_dur * 0.8,
                    velocity=random.randint(80, 100),
                ))
    midi_data.tracks.append(bass)

    midi_data.duration = total_dur
    return midi_data


# ── High-Level Functions ───────────────────────────────────────────────────────

_engine: Optional[MidiLLMEngine] = None


def get_engine() -> MidiLLMEngine:
    global _engine
    if _engine is None:
        _engine = MidiLLMEngine()
    return _engine


def generate_midi(params: MidiGenParams,
                  progress_callback: Optional[Callable] = None) -> MidiGenResult:
    """
    Generate MIDI - uses model if loaded, falls back to demo generator.
    Called by InferenceWorker.
    """
    engine = get_engine()

    if engine.is_loaded:
        if progress_callback:
            progress_callback(0.1, "Generating MIDI sequence...")
        result = engine.generate(params)
        if progress_callback:
            progress_callback(1.0, "Done")
        return result
    else:
        # Fallback to algorithmic generation
        if progress_callback:
            progress_callback(0.2, "Generating demo MIDI (no model loaded)...")
        t0 = time.time()
        midi_data = generate_demo_midi(params)
        gen_time = time.time() - t0

        if progress_callback:
            progress_callback(1.0, "Done")

        return MidiGenResult(
            midi_data=midi_data,
            raw_tokens="[algorithmic demo - no model]",
            generation_time=gen_time,
            token_count=0,
        )


def load_model(cache_dir: str = None, model_id: str = None, source: str = None, **kwargs) -> MidiLLMEngine:
    """Load MIDI LLM model. Called by ModelManager._dynamic_load()."""
    from core.deps import ensure
    ensure("torch")
    ensure("transformers")

    engine = get_engine()

    # Determine model path: check cache first, then use HF source
    model_path = source or "slseanwu/MIDI-LLM_Llama-3.2-1B"
    if cache_dir and model_id:
        from pathlib import Path
        local = Path(cache_dir) / model_id
        if local.exists():
            model_path = str(local)

    engine.load_model(model_path)
    return engine
