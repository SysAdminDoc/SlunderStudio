"""
Slunder Studio v0.1.21 - Humming-to-melody extraction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from core.midi_utils import MidiData, NoteData, TrackData, save_midi
from core.settings import get_default_output_dir


@dataclass
class LyricMelodyParams:
    input_path: str
    lyrics: str = ""
    tempo: float = 120.0
    output_midi_path: str = ""
    render_diffsinger: bool = True
    min_note_duration: float = 0.12
    max_merge_gap: float = 0.08
    fmin_note: str = "C3"
    fmax_note: str = "C6"
    frame_length: int = 2048
    hop_length: int = 256


@dataclass
class LyricMelodyResult:
    midi_path: str
    vocal_path: str = ""
    notes_count: int = 0
    lyric_units: int = 0
    duration: float = 0.0
    tempo: float = 120.0
    diffsinger_error: str = ""


def generate_lyric_melody(
    params: LyricMelodyParams,
    progress_cb: Callable[[int], None] | None = None,
    step_cb: Callable[[str], None] | None = None,
    log_cb: Callable[[str], None] | None = None,
    cancel_event=None,
) -> LyricMelodyResult:
    """Extract a monophonic humming melody, save MIDI, and optionally render DiffSinger."""
    import librosa

    input_path = Path(params.input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Humming input not found: {input_path}")

    if step_cb:
        step_cb("Loading humming audio...")
    audio, sr = librosa.load(str(input_path), sr=None, mono=True)
    if audio.size == 0:
        raise ValueError("Humming input is empty")
    duration = len(audio) / float(sr)

    if cancel_event and cancel_event.is_set():
        return LyricMelodyResult(midi_path="", duration=duration, tempo=params.tempo)

    if progress_cb:
        progress_cb(15)
    if step_cb:
        step_cb("Extracting humming pitch...")

    f0, voiced_flag, _voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz(params.fmin_note),
        fmax=librosa.note_to_hz(params.fmax_note),
        sr=sr,
        frame_length=params.frame_length,
        hop_length=params.hop_length,
    )
    notes = notes_from_pitch_frames(
        f0,
        voiced_flag,
        sr=sr,
        hop_length=params.hop_length,
        min_duration=params.min_note_duration,
        max_merge_gap=params.max_merge_gap,
    )
    if not notes:
        raise ValueError("No stable humming melody was detected")

    lyric_units = lyric_units_from_text(params.lyrics)
    sing_notes = align_lyrics_to_notes(notes, lyric_units)
    midi_data = MidiData(
        tracks=[TrackData(name="Hummed Vocal Melody", program=53, notes=notes)],
        tempo=float(params.tempo),
        duration=max(note.end for note in notes),
    )

    output_midi = Path(params.output_midi_path) if params.output_midi_path else _default_midi_path(input_path)
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(50)
    if step_cb:
        step_cb("Writing melody MIDI...")
    save_midi(
        midi_data,
        str(output_midi),
        provenance={
            "module": "vocal_suite",
            "operation": "humming_to_lyric_melody",
            "model_id": "librosa-pyin-melody",
            "lyrics": params.lyrics,
            "parameters": asdict(params),
            "source_paths": [str(input_path)],
            "output_kind": "analysis",
            "extra": {
                "lyric_units": lyric_units,
                "sing_notes": sing_notes,
                "source_duration": duration,
            },
        },
    )

    result = LyricMelodyResult(
        midi_path=str(output_midi),
        notes_count=len(notes),
        lyric_units=len(lyric_units),
        duration=midi_data.duration,
        tempo=float(params.tempo),
    )

    if cancel_event and cancel_event.is_set():
        return result

    if params.render_diffsinger:
        if progress_cb:
            progress_cb(70)
        if step_cb:
            step_cb("Rendering DiffSinger vocal...")
        result.vocal_path, result.diffsinger_error = render_diffsinger_from_melody(
            params=params,
            sing_notes=sing_notes,
            source_paths=[str(input_path), str(output_midi)],
        )

    if progress_cb:
        progress_cb(100)
    if log_cb:
        message = f"Lyric melody wrote {output_midi}"
        if result.vocal_path:
            message += f" and {result.vocal_path}"
        log_cb(message)
    return result


def notes_from_pitch_frames(
    f0: np.ndarray,
    voiced_flag: np.ndarray | None,
    *,
    sr: int,
    hop_length: int,
    min_duration: float,
    max_merge_gap: float,
) -> list[NoteData]:
    """Convert pYIN pitch frames into merged MIDI notes."""
    import librosa

    if f0 is None or len(f0) == 0:
        return []
    midi = librosa.hz_to_midi(f0)
    valid = np.isfinite(midi)
    if voiced_flag is not None:
        valid &= voiced_flag.astype(bool)

    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    raw_notes: list[NoteData] = []
    current_pitch = None
    current_start = 0.0
    last_time = 0.0

    for index, is_valid in enumerate(valid):
        time_value = float(frame_times[index])
        pitch = int(np.clip(round(midi[index]), 0, 127)) if is_valid else None
        if pitch == current_pitch:
            last_time = time_value
            continue
        if current_pitch is not None:
            _append_note(raw_notes, current_pitch, current_start, last_time + hop_length / sr, min_duration)
        current_pitch = pitch
        current_start = time_value
        last_time = time_value

    if current_pitch is not None:
        _append_note(raw_notes, current_pitch, current_start, last_time + hop_length / sr, min_duration)

    return merge_adjacent_notes(raw_notes, max_gap=max_merge_gap)


def merge_adjacent_notes(notes: list[NoteData], *, max_gap: float) -> list[NoteData]:
    if not notes:
        return []
    merged = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        if note.pitch == prev.pitch and note.start - prev.end <= max_gap:
            prev.end = max(prev.end, note.end)
            prev.velocity = max(prev.velocity, note.velocity)
        else:
            merged.append(note)
    return merged


def lyric_units_from_text(lyrics: str) -> list[str]:
    units: list[str] = []
    for raw in lyrics.replace("\r", "\n").replace("/", " ").split():
        cleaned = "".join(ch for ch in raw.strip() if ch.isalnum() or ch in "'-")
        if cleaned:
            units.append(cleaned)
    return units


def align_lyrics_to_notes(notes: list[NoteData], lyric_units: list[str]) -> list[dict]:
    aligned = []
    for index, note in enumerate(notes):
        text = lyric_units[index] if index < len(lyric_units) else ""
        aligned.append({
            "pitch": note.pitch,
            "start": note.start,
            "end": note.end,
            "text": text,
        })
    return aligned


def render_diffsinger_from_melody(
    *,
    params: LyricMelodyParams,
    sing_notes: list[dict],
    source_paths: list[str],
) -> tuple[str, str]:
    from engines.diffsinger_engine import SingParams, get_diffsinger, synthesize_vocals

    engine = get_diffsinger()
    if not engine.is_loaded:
        return "", "DiffSinger model not loaded. Melody MIDI was created."

    sing_params = SingParams(
        lyrics=params.lyrics,
        notes=sing_notes,
        tempo=params.tempo,
    )
    result = synthesize_vocals(sing_params)
    if result.error:
        return "", result.error
    provenance = result.provenance or {}
    provenance["operation"] = "lyric_melody_diffsinger_render"
    provenance["source_paths"] = source_paths
    provenance.setdefault("extra", {})["melody_source"] = "humming_to_midi"
    result.provenance = provenance
    vocal_path = engine.save_output(result, name=f"{Path(params.input_path).stem}_lyric_melody")
    return vocal_path or "", ""


def _append_note(notes: list[NoteData], pitch: int, start: float, end: float, min_duration: float):
    if end - start < min_duration:
        return
    notes.append(NoteData(pitch=pitch, start=max(0.0, start), end=max(end, start + min_duration), velocity=96))


def _default_midi_path(input_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_default_output_dir() / "vocals" / "melodies" / f"{input_path.stem}_melody_{timestamp}.mid"
