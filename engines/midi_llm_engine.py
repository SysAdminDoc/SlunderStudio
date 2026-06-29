"""
Slunder Studio v0.1.23 — MIDI-LLM Engine
Text-to-MIDI generation using fine-tuned language models that output MIDI token sequences.
Supports prompt-based composition, continuation, and style-conditioned generation.
"""
import os
import json
import re
import time
from typing import Optional, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.provenance import sidecar_path_for
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
    chord_progression: str = "Auto"
    drum_groove: str = "Auto"
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
    provenance: dict = field(default_factory=dict)
    provenance_path: str = ""


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
- Honor requested chord progressions by repeating one chord per bar unless the user asks otherwise
- Honor requested drum groove templates with swing, ghost notes, and humanized velocities when drums are present
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
    if normalize_chord_progression(params.chord_progression):
        parts.append(
            f"Chord Progression: {params.chord_progression} "
            "(one chord per bar, repeat for the full length)"
        )
    else:
        parts.append("Chord Progression: choose a coherent progression for the style")
    if wants_drum_track(params):
        groove = normalize_drum_groove(params.drum_groove)
        if groove:
            parts.append(
                f"Drum Groove: {groove} "
                "(use matching kick/snare/hat placement, swing, ghost notes, and velocity humanize)"
            )
        else:
            parts.append("Drum Groove: choose a coherent groove template for the style")

    if params.continuation_context:
        parts.append(f"\nContinue from this existing sequence:\n{params.continuation_context}")
        parts.append("\nGenerate the next section:")
    else:
        parts.append("\nGenerate the MIDI token sequence:")

    return "\n".join(parts)


ROMAN_DEGREES = {
    "I": 0,
    "II": 1,
    "III": 2,
    "IV": 3,
    "V": 4,
    "VI": 5,
    "VII": 6,
}

MAJOR_PROGRESSIONS = {
    "I-V-vi-IV": [0, 4, 5, 3],
    "I-vi-IV-V": [0, 5, 3, 4],
    "I-IV-V-V": [0, 3, 4, 4],
    "ii-V-I": [1, 4, 0],
    "12-bar blues": [0, 0, 0, 0, 3, 3, 0, 0, 4, 3, 0, 4],
}

MINOR_PROGRESSIONS = {
    "i-VI-iv-v": [0, 5, 3, 4],
    "i-iv-VI-v": [0, 3, 5, 4],
    "i-VI-VII-v": [0, 5, 6, 4],
    "i-VI-III-VII": [0, 5, 2, 6],
    "ii-V-i": [1, 4, 0],
}


DRUM_KICK = 36
DRUM_SNARE = 38
DRUM_CLAP = 39
DRUM_CLOSED_HAT = 42
DRUM_OPEN_HAT = 46
DRUM_LOW_CONGA = 64


@dataclass(frozen=True)
class DrumHit:
    pitch: int
    step: int
    velocity: int
    duration_steps: float = 0.75
    probability: float = 1.0
    ghost: bool = False


@dataclass(frozen=True)
class DrumGrooveTemplate:
    name: str
    description: str
    hits: tuple[DrumHit, ...]
    steps_per_bar: int = 16
    swing: float = 0.0
    velocity_humanize: int = 0
    timing_humanize_ms: float = 0.0


DRUM_GROOVE_TEMPLATES = {
    "Straight Rock": DrumGrooveTemplate(
        name="Straight Rock",
        description="Backbeat rock groove with eighth-note hats and light snare ghosts.",
        swing=0.0,
        velocity_humanize=7,
        timing_humanize_ms=5,
        hits=tuple(
            [DrumHit(DRUM_CLOSED_HAT, step, 78) for step in range(0, 16, 2)]
            + [
                DrumHit(DRUM_KICK, 0, 104),
                DrumHit(DRUM_KICK, 8, 100),
                DrumHit(DRUM_KICK, 10, 88, probability=0.65),
                DrumHit(DRUM_SNARE, 4, 108),
                DrumHit(DRUM_SNARE, 12, 110),
                DrumHit(DRUM_SNARE, 7, 34, probability=0.6, ghost=True),
                DrumHit(DRUM_SNARE, 15, 36, probability=0.6, ghost=True),
            ]
        ),
    ),
    "Hip-Hop Half-Time": DrumGrooveTemplate(
        name="Hip-Hop Half-Time",
        description="Laid-back half-time pocket with swung hats and snare ghosts.",
        swing=0.18,
        velocity_humanize=13,
        timing_humanize_ms=8,
        hits=tuple(
            [DrumHit(DRUM_CLOSED_HAT, step, 70 if step % 4 else 82) for step in range(16)]
            + [
                DrumHit(DRUM_KICK, 0, 108),
                DrumHit(DRUM_KICK, 6, 86),
                DrumHit(DRUM_KICK, 10, 96),
                DrumHit(DRUM_SNARE, 8, 112),
                DrumHit(DRUM_SNARE, 7, 32, probability=0.65, ghost=True),
                DrumHit(DRUM_SNARE, 14, 34, probability=0.55, ghost=True),
            ]
        ),
    ),
    "Trap Hats": DrumGrooveTemplate(
        name="Trap Hats",
        description="Trap groove with busy hats, backbeat clap, and syncopated kicks.",
        swing=0.08,
        velocity_humanize=18,
        timing_humanize_ms=4,
        hits=tuple(
            [DrumHit(DRUM_CLOSED_HAT, step, 62 if step % 2 else 82) for step in range(16)]
            + [
                DrumHit(DRUM_KICK, 0, 112),
                DrumHit(DRUM_KICK, 7, 94),
                DrumHit(DRUM_KICK, 10, 106),
                DrumHit(DRUM_KICK, 14, 88),
                DrumHit(DRUM_SNARE, 8, 104),
                DrumHit(DRUM_CLAP, 8, 96),
                DrumHit(DRUM_OPEN_HAT, 6, 78, probability=0.75),
                DrumHit(DRUM_OPEN_HAT, 14, 72, probability=0.65),
                DrumHit(DRUM_SNARE, 15, 30, probability=0.55, ghost=True),
            ]
        ),
    ),
    "Swing Shuffle": DrumGrooveTemplate(
        name="Swing Shuffle",
        description="Triplet-feel shuffle with delayed offbeats and ghost snares.",
        swing=0.42,
        velocity_humanize=9,
        timing_humanize_ms=6,
        hits=tuple(
            [DrumHit(DRUM_CLOSED_HAT, step, 74 if step % 4 else 88) for step in range(0, 16, 2)]
            + [
                DrumHit(DRUM_KICK, 0, 102),
                DrumHit(DRUM_KICK, 8, 96),
                DrumHit(DRUM_KICK, 10, 82, probability=0.7),
                DrumHit(DRUM_SNARE, 4, 106),
                DrumHit(DRUM_SNARE, 12, 108),
                DrumHit(DRUM_SNARE, 3, 32, probability=0.8, ghost=True),
                DrumHit(DRUM_SNARE, 7, 34, probability=0.8, ghost=True),
                DrumHit(DRUM_SNARE, 11, 31, probability=0.8, ghost=True),
                DrumHit(DRUM_SNARE, 15, 35, probability=0.8, ghost=True),
            ]
        ),
    ),
    "Four-on-the-Floor": DrumGrooveTemplate(
        name="Four-on-the-Floor",
        description="Dance groove with quarter kicks, backbeat claps, and open-hat lift.",
        swing=0.0,
        velocity_humanize=10,
        timing_humanize_ms=3,
        hits=tuple(
            [DrumHit(DRUM_KICK, step, 112) for step in (0, 4, 8, 12)]
            + [DrumHit(DRUM_CLOSED_HAT, step, 72) for step in range(0, 16, 2)]
            + [
                DrumHit(DRUM_SNARE, 4, 96),
                DrumHit(DRUM_CLAP, 4, 100),
                DrumHit(DRUM_SNARE, 12, 98),
                DrumHit(DRUM_CLAP, 12, 102),
                DrumHit(DRUM_OPEN_HAT, 2, 82),
                DrumHit(DRUM_OPEN_HAT, 6, 82),
                DrumHit(DRUM_OPEN_HAT, 10, 82),
                DrumHit(DRUM_OPEN_HAT, 14, 82),
            ]
        ),
    ),
    "Latin Pop": DrumGrooveTemplate(
        name="Latin Pop",
        description="Syncopated pop groove with conga color and lighter backbeat.",
        swing=0.06,
        velocity_humanize=12,
        timing_humanize_ms=7,
        hits=tuple(
            [DrumHit(DRUM_CLOSED_HAT, step, 70 if step % 4 else 82) for step in range(0, 16, 2)]
            + [
                DrumHit(DRUM_KICK, 0, 104),
                DrumHit(DRUM_KICK, 6, 92),
                DrumHit(DRUM_KICK, 10, 98),
                DrumHit(DRUM_SNARE, 4, 92),
                DrumHit(DRUM_CLAP, 12, 96),
                DrumHit(DRUM_LOW_CONGA, 3, 74),
                DrumHit(DRUM_LOW_CONGA, 7, 68),
                DrumHit(DRUM_LOW_CONGA, 11, 72),
                DrumHit(DRUM_LOW_CONGA, 15, 66),
                DrumHit(DRUM_SNARE, 14, 34, probability=0.6, ghost=True),
            ]
        ),
    ),
}

DRUM_GROOVE_NAMES = ["Auto", *DRUM_GROOVE_TEMPLATES.keys(), "None"]


def normalize_chord_progression(value: str) -> str:
    normalized = (value or "").strip()
    if not normalized or normalized.lower() == "auto":
        return ""
    return normalized


def parse_chord_progression(value: str, *, is_minor: bool = False) -> list[int]:
    """Parse a Roman-numeral chord progression into zero-based scale degrees."""
    normalized = normalize_chord_progression(value)
    if not normalized:
        return []

    preset_map = MINOR_PROGRESSIONS if is_minor else MAJOR_PROGRESSIONS
    if normalized in preset_map:
        return list(preset_map[normalized])

    degrees: list[int] = []
    for raw in re.split(r"[-\s,|/]+", normalized):
        token = raw.strip()
        if not token:
            continue
        match = re.match(r"^[#b]*([ivIV]+)", token)
        if not match:
            continue
        roman = match.group(1).upper()
        if roman in ROMAN_DEGREES:
            degrees.append(ROMAN_DEGREES[roman])
    return degrees


def select_chord_progression(params: MidiGenParams, *, is_minor: bool, rng) -> tuple[str, list[int]]:
    requested = normalize_chord_progression(params.chord_progression)
    if requested:
        parsed = parse_chord_progression(requested, is_minor=is_minor)
        if parsed:
            return requested, parsed

    candidates = MINOR_PROGRESSIONS if is_minor else MAJOR_PROGRESSIONS
    label = rng.choice(list(candidates.keys()))
    return label, list(candidates[label])


def normalize_drum_groove(value: str) -> str:
    normalized = (value or "").strip()
    if not normalized or normalized.lower() == "auto":
        return ""
    if normalized.lower() in {"none", "no drums", "off"}:
        return "None"
    return normalized


def wants_drum_track(params: MidiGenParams) -> bool:
    groove = normalize_drum_groove(params.drum_groove)
    if groove == "None":
        return False
    if groove:
        return True
    return any(
        "drum" in instrument.lower() or "percussion" in instrument.lower()
        for instrument in params.instruments
    )


def select_drum_groove(params: MidiGenParams, rng) -> Optional[DrumGrooveTemplate]:
    if rng is None:
        import random
        rng = random

    requested = normalize_drum_groove(params.drum_groove)
    if requested == "None":
        return None
    if requested in DRUM_GROOVE_TEMPLATES:
        return DRUM_GROOVE_TEMPLATES[requested]

    style_text = " ".join([params.prompt, params.style, *params.instruments]).lower()
    style_map = [
        (("swing", "shuffle", "jazz", "blues"), "Swing Shuffle"),
        (("trap",), "Trap Hats"),
        (("hip-hop", "hip hop", "boom bap", "lo-fi", "lofi"), "Hip-Hop Half-Time"),
        (("house", "techno", "edm", "dance", "electronic"), "Four-on-the-Floor"),
        (("latin", "afro", "reggaeton", "dancehall"), "Latin Pop"),
        (("rock", "metal", "punk", "grunge"), "Straight Rock"),
    ]
    for keywords, label in style_map:
        if any(keyword in style_text for keyword in keywords):
            return DRUM_GROOVE_TEMPLATES[label]

    return DRUM_GROOVE_TEMPLATES[rng.choice(["Straight Rock", "Hip-Hop Half-Time", "Four-on-the-Floor"])]


def generate_drum_track(params: MidiGenParams, rng=None) -> Optional[TrackData]:
    """Generate a GM channel-9 drum track from the selected groove template."""
    if not wants_drum_track(params):
        return None

    import random

    rng = rng or (random.Random(params.seed) if params.seed is not None else random)
    template = select_drum_groove(params, rng)
    if template is None:
        return None

    beat_dur = 60.0 / params.tempo
    bar_dur = beat_dur * params.time_signature[0]
    bar_steps = max(1, params.time_signature[0] * 4)
    step_dur = bar_dur / bar_steps
    track = TrackData(
        name=f"Drums - {template.name}",
        program=0,
        channel=9,
        is_drum=True,
    )

    for bar in range(params.duration_bars):
        bar_start = bar * bar_dur
        for hit in template.hits:
            step = int(round(hit.step * bar_steps / template.steps_per_bar))
            if step >= bar_steps:
                continue
            if hit.probability < 1.0 and rng.random() > hit.probability:
                continue

            swing_delay = step_dur * template.swing if step % 4 == 2 else 0.0
            timing_jitter = 0.0
            if template.timing_humanize_ms:
                timing_jitter = rng.uniform(
                    -template.timing_humanize_ms / 1000.0,
                    template.timing_humanize_ms / 1000.0,
                )
            start = max(bar_start, bar_start + step * step_dur + swing_delay + timing_jitter)
            end = min(bar_start + bar_dur, start + max(0.02, step_dur * hit.duration_steps))

            velocity = hit.velocity
            if template.velocity_humanize:
                velocity += rng.randint(-template.velocity_humanize, template.velocity_humanize)
            if hit.ghost:
                velocity = min(velocity, 46)
            velocity = max(1, min(127, velocity))

            track.notes.append(NoteData(
                pitch=hit.pitch,
                start=round(start, 6),
                end=round(end, 6),
                velocity=velocity,
                channel=9,
            ))

    track.notes.sort(key=lambda note: (note.start, note.pitch, note.velocity))
    return track


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
                provenance={
                    "module": "midi_studio",
                    "operation": "generate_midi",
                    "model_id": self._model_id or "midi-llm-1b",
                    "prompt": params.prompt,
                    "parameters": asdict(params),
                    "output_kind": "model",
                    "extra": {
                        "token_count": len(new_tokens),
                        "raw_tokens_preview": raw_text[:500],
                    },
                },
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
        provenance = result.provenance or {
            "module": "midi_studio",
            "operation": "save_generation",
            "model_id": self._model_id or "midi-llm-1b",
            "parameters": {
                "generation_time": result.generation_time,
                "token_count": result.token_count,
                "tracks": result.midi_data.track_count,
                "total_notes": result.midi_data.total_notes,
                "duration": result.midi_data.duration,
                "tempo": result.midi_data.tempo,
            },
            "output_kind": "export",
        }
        save_midi(result.midi_data, midi_path, provenance=provenance)
        result.provenance_path = str(sidecar_path_for(midi_path))

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

    rng = random.Random(params.seed) if params.seed is not None else random

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
    _progression_label, prog = select_chord_progression(params, is_minor=is_minor, rng=rng)

    def scale_note(degree: int, octave_offset: int = 0) -> int:
        oct = degree // 7
        deg = degree % 7
        return root + scale[deg] + (oct + octave_offset) * 12

    def chord_tones(degree: int, octave_offset: int = -1) -> list[int]:
        return [
            scale_note(degree, octave_offset),
            scale_note(degree + 2, octave_offset),
            scale_note(degree + 4, octave_offset),
        ]

    # Piano chords track
    piano = TrackData(name="Piano", program=0, channel=0)
    for bar in range(params.duration_bars):
        chord_deg = prog[bar % len(prog)]
        chord_notes = chord_tones(chord_deg)
        t = bar * bar_dur
        for p in chord_notes:
            piano.notes.append(NoteData(
                pitch=max(0, min(127, p)),
                start=t, end=t + bar_dur * 0.95,
                velocity=rng.randint(70, 90),
            ))
    midi_data.tracks.append(piano)

    # Melody track
    melody = TrackData(name="Melody", program=0, channel=1)
    t = 0.0
    prev_deg = 0
    while t < total_dur:
        bar = min(params.duration_bars - 1, int(t // bar_dur))
        chord_deg = prog[bar % len(prog)]
        chord_bias = [chord_deg, chord_deg + 2, chord_deg + 4]
        if rng.random() < 0.65:
            deg = rng.choice(chord_bias)
        else:
            deg = prev_deg + rng.choice([-2, -1, 0, 1, 1, 2])
        deg = max(-3, min(10, deg))
        pitch = scale_note(deg, 1)

        dur_choices = [beat_dur * 0.5, beat_dur, beat_dur * 1.5, beat_dur * 2]
        dur = rng.choice(dur_choices)

        if rng.random() > 0.15:  # 85% note density
            melody.notes.append(NoteData(
                pitch=max(0, min(127, pitch)),
                start=t, end=min(t + dur * 0.9, total_dur),
                velocity=rng.randint(80, 110),
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
                    velocity=rng.randint(80, 100),
                ))
    midi_data.tracks.append(bass)

    drum_track = generate_drum_track(params, rng=rng)
    if drum_track is not None:
        midi_data.tracks.append(drum_track)

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
            provenance={
                "module": "midi_studio",
                "operation": "generate_midi",
                "model_id": "midi-llm-1b",
                "prompt": params.prompt,
                "parameters": asdict(params),
                "output_kind": "demo",
                "extra": {"demo_synthesis": True},
            },
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
