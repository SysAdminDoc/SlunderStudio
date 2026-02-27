"""
Slunder Studio v0.0.2 — MIDI Utilities
Helpers for MIDI parsing, quantization, import/export, and track manipulation.
Wraps pretty_midi for consistent API across the app.
"""
import os
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field


def _ensure_pretty_midi():
    from core.deps import ensure
    ensure("pretty_midi")


@dataclass
class NoteData:
    """Single MIDI note representation."""
    pitch: int = 60  # 0-127
    start: float = 0.0  # seconds
    end: float = 0.5
    velocity: int = 100  # 0-127
    channel: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def name(self) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return f"{names[self.pitch % 12]}{self.pitch // 12 - 1}"


@dataclass
class TrackData:
    """Single MIDI track."""
    name: str = "Track"
    program: int = 0  # General MIDI program number
    channel: int = 0
    notes: list[NoteData] = field(default_factory=list)
    is_drum: bool = False

    @property
    def note_count(self) -> int:
        return len(self.notes)

    @property
    def duration(self) -> float:
        if not self.notes:
            return 0.0
        return max(n.end for n in self.notes)


@dataclass
class MidiData:
    """Complete MIDI file representation."""
    tracks: list[TrackData] = field(default_factory=list)
    tempo: float = 120.0
    time_signature: tuple = (4, 4)
    duration: float = 0.0

    @property
    def track_count(self) -> int:
        return len(self.tracks)

    @property
    def total_notes(self) -> int:
        return sum(t.note_count for t in self.tracks)


# ── GM Program Names ───────────────────────────────────────────────────────────

GM_PROGRAMS = {
    0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano", 2: "Electric Grand Piano",
    3: "Honky-tonk Piano", 4: "Electric Piano 1", 5: "Electric Piano 2",
    6: "Harpsichord", 7: "Clavinet", 8: "Celesta", 9: "Glockenspiel",
    10: "Music Box", 11: "Vibraphone", 12: "Marimba", 13: "Xylophone",
    14: "Tubular Bells", 15: "Dulcimer", 16: "Drawbar Organ", 17: "Percussive Organ",
    18: "Rock Organ", 19: "Church Organ", 20: "Reed Organ", 21: "Accordion",
    22: "Harmonica", 23: "Tango Accordion", 24: "Nylon Guitar", 25: "Steel Guitar",
    26: "Jazz Guitar", 27: "Clean Guitar", 28: "Muted Guitar", 29: "Overdriven Guitar",
    30: "Distortion Guitar", 31: "Guitar Harmonics", 32: "Acoustic Bass",
    33: "Electric Bass (finger)", 34: "Electric Bass (pick)", 35: "Fretless Bass",
    36: "Slap Bass 1", 37: "Slap Bass 2", 38: "Synth Bass 1", 39: "Synth Bass 2",
    40: "Violin", 41: "Viola", 42: "Cello", 43: "Contrabass",
    44: "Tremolo Strings", 45: "Pizzicato Strings", 46: "Orchestral Harp", 47: "Timpani",
    48: "String Ensemble 1", 49: "String Ensemble 2", 50: "Synth Strings 1",
    56: "Trumpet", 57: "Trombone", 58: "Tuba", 59: "Muted Trumpet",
    60: "French Horn", 61: "Brass Section", 65: "Soprano Sax", 66: "Alto Sax",
    67: "Tenor Sax", 68: "Baritone Sax", 73: "Flute", 74: "Recorder",
    128: "Drums",
}


def get_program_name(program: int, is_drum: bool = False) -> str:
    if is_drum:
        return "Drums"
    return GM_PROGRAMS.get(program, f"Program {program}")


# ── Import / Export ────────────────────────────────────────────────────────────

def load_midi(file_path: str) -> MidiData:
    """Load a MIDI file into MidiData structure."""
    _ensure_pretty_midi()
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(file_path)
    midi_data = MidiData()

    # Tempo
    tempos = pm.get_tempo_changes()
    if len(tempos[1]) > 0:
        midi_data.tempo = float(tempos[1][0])

    # Time signature
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        midi_data.time_signature = (ts.numerator, ts.denominator)

    midi_data.duration = pm.get_end_time()

    # Tracks
    for inst in pm.instruments:
        track = TrackData(
            name=inst.name or get_program_name(inst.program, inst.is_drum),
            program=inst.program,
            channel=9 if inst.is_drum else 0,
            is_drum=inst.is_drum,
        )
        for note in inst.notes:
            track.notes.append(NoteData(
                pitch=note.pitch,
                start=note.start,
                end=note.end,
                velocity=note.velocity,
                channel=track.channel,
            ))
        midi_data.tracks.append(track)

    return midi_data


def save_midi(midi_data: MidiData, file_path: str):
    """Save MidiData to a MIDI file."""
    _ensure_pretty_midi()
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=midi_data.tempo)

    # Time signature
    ts_num, ts_den = midi_data.time_signature
    pm.time_signature_changes.append(
        pretty_midi.TimeSignature(ts_num, ts_den, 0)
    )

    for track in midi_data.tracks:
        inst = pretty_midi.Instrument(
            program=track.program,
            is_drum=track.is_drum,
            name=track.name,
        )
        for note in track.notes:
            inst.notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end,
            ))
        pm.instruments.append(inst)

    pm.write(file_path)


def export_tracks_separately(midi_data: MidiData, output_dir: str) -> list[str]:
    """Export each track as a separate MIDI file."""
    _ensure_pretty_midi()
    import pretty_midi
    paths = []
    os.makedirs(output_dir, exist_ok=True)

    for i, track in enumerate(midi_data.tracks):
        pm = pretty_midi.PrettyMIDI(initial_tempo=midi_data.tempo)
        inst = pretty_midi.Instrument(
            program=track.program, is_drum=track.is_drum, name=track.name,
        )
        for note in track.notes:
            inst.notes.append(pretty_midi.Note(
                velocity=note.velocity, pitch=note.pitch,
                start=note.start, end=note.end,
            ))
        pm.instruments.append(inst)
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in track.name)
        path = os.path.join(output_dir, f"{i:02d}_{safe_name}.mid")
        pm.write(path)
        paths.append(path)

    return paths


# ── Quantization ───────────────────────────────────────────────────────────────

def quantize_notes(notes: list[NoteData], grid: float = 0.25, tempo: float = 120.0) -> list[NoteData]:
    """
    Quantize note start times and durations to nearest grid division.
    grid: fraction of a beat (0.25 = 16th note, 0.5 = 8th, 1.0 = quarter).
    """
    beat_dur = 60.0 / tempo  # seconds per beat
    grid_sec = grid * beat_dur

    quantized = []
    for note in notes:
        q_start = round(note.start / grid_sec) * grid_sec
        q_end = round(note.end / grid_sec) * grid_sec
        if q_end <= q_start:
            q_end = q_start + grid_sec
        quantized.append(NoteData(
            pitch=note.pitch,
            start=round(q_start, 6),
            end=round(q_end, 6),
            velocity=note.velocity,
            channel=note.channel,
        ))
    return quantized


def transpose_notes(notes: list[NoteData], semitones: int) -> list[NoteData]:
    """Transpose all notes by given semitones."""
    return [
        NoteData(
            pitch=max(0, min(127, n.pitch + semitones)),
            start=n.start, end=n.end,
            velocity=n.velocity, channel=n.channel,
        )
        for n in notes
    ]


def scale_velocity(notes: list[NoteData], factor: float) -> list[NoteData]:
    """Scale all velocities by factor (clamped 0-127)."""
    return [
        NoteData(
            pitch=n.pitch, start=n.start, end=n.end,
            velocity=max(0, min(127, int(n.velocity * factor))),
            channel=n.channel,
        )
        for n in notes
    ]


def get_pitch_range(notes: list[NoteData]) -> tuple[int, int]:
    """Get min/max pitch in note list."""
    if not notes:
        return (60, 72)
    pitches = [n.pitch for n in notes]
    return (min(pitches), max(pitches))


def get_time_range(notes: list[NoteData]) -> tuple[float, float]:
    """Get start/end time span."""
    if not notes:
        return (0.0, 4.0)
    return (min(n.start for n in notes), max(n.end for n in notes))
