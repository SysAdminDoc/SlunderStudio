"""
Slunder Studio v0.1.28 - MIDI chord chart export.
Infers bar-level chords from MIDI note content and writes ChordPro or CRD sheets.
"""
from dataclasses import dataclass
from pathlib import Path
import re

from core.midi_utils import MidiData


PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_QUALITIES = [
    ("", (0, 4, 7)),
    ("m", (0, 3, 7)),
    ("dim", (0, 3, 6)),
    ("aug", (0, 4, 8)),
    ("sus2", (0, 2, 7)),
    ("sus4", (0, 5, 7)),
    ("7", (0, 4, 7, 10)),
    ("maj7", (0, 4, 7, 11)),
    ("m7", (0, 3, 7, 10)),
    ("m7b5", (0, 3, 6, 10)),
]

SECTION_RE = re.compile(r"^\[(?P<label>[A-Za-z][A-Za-z0-9 /_-]*)\]\s*$")


@dataclass(frozen=True)
class ChordSegment:
    bar_index: int
    start: float
    end: float
    name: str


def detect_chord_segments(midi_data: MidiData) -> list[ChordSegment]:
    """Infer one chord per bar from non-drum MIDI notes."""
    bar_dur = _bar_duration(midi_data)
    bars = max(1, int((midi_data.duration + bar_dur - 0.000001) // bar_dur))
    segments: list[ChordSegment] = []

    for bar in range(bars):
        start = bar * bar_dur
        end = start + bar_dur
        weights = _pitch_class_weights(midi_data, start, end)
        segments.append(ChordSegment(bar, start, end, _name_chord(weights)))

    return segments


def format_chordpro(
    midi_data: MidiData,
    *,
    title: str = "Untitled",
    artist: str = "",
    lyrics: str = "",
) -> str:
    chords = [segment.name for segment in detect_chord_segments(midi_data)]
    lines = [
        f"{{title: {_clean_directive(title)}}}",
        f"{{tempo: {midi_data.tempo:.0f}}}",
        f"{{time: {midi_data.time_signature[0]}/{midi_data.time_signature[1]}}}",
    ]
    if artist:
        lines.insert(1, f"{{artist: {_clean_directive(artist)}}}")
    lines.append("")
    lines.extend(_chordpro_body(chords, lyrics))
    return "\n".join(lines).rstrip() + "\n"


def format_crd(
    midi_data: MidiData,
    *,
    title: str = "Untitled",
    artist: str = "",
    lyrics: str = "",
) -> str:
    chords = [segment.name for segment in detect_chord_segments(midi_data)]
    lines = [
        f"Title: {_clean_directive(title)}",
        f"Tempo: {midi_data.tempo:.0f} BPM",
        f"Time: {midi_data.time_signature[0]}/{midi_data.time_signature[1]}",
    ]
    if artist:
        lines.insert(1, f"Artist: {_clean_directive(artist)}")
    lines.append("")
    lines.extend(_crd_body(chords, lyrics))
    return "\n".join(lines).rstrip() + "\n"


def save_chord_chart(
    midi_data: MidiData,
    file_path: str,
    *,
    lyrics: str = "",
    title: str = "Untitled",
    artist: str = "",
) -> str:
    """Save a MIDI chord chart as .chordpro or .crd."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".crd":
        text = format_crd(midi_data, title=title, artist=artist, lyrics=lyrics)
    else:
        text = format_chordpro(midi_data, title=title, artist=artist, lyrics=lyrics)
    path.write_text(text, encoding="utf-8", newline="\n")
    return str(path)


def _bar_duration(midi_data: MidiData) -> float:
    beats = max(1, int(midi_data.time_signature[0]))
    tempo = midi_data.tempo if midi_data.tempo > 0 else 120.0
    return (60.0 / tempo) * beats


def _pitch_class_weights(midi_data: MidiData, start: float, end: float) -> dict[int, float]:
    weights = {pc: 0.0 for pc in range(12)}
    for track in midi_data.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            overlap = min(end, note.end) - max(start, note.start)
            if overlap > 0:
                weights[note.pitch % 12] += overlap * max(1, note.velocity)
    return weights


def _name_chord(weights: dict[int, float]) -> str:
    total = sum(weights.values())
    if total <= 0:
        return "N.C."

    best_name = "N.C."
    best_score = float("-inf")
    for root in range(12):
        for suffix, intervals in CHORD_QUALITIES:
            tones = {(root + interval) % 12 for interval in intervals}
            tone_weight = sum(weights[pc] for pc in tones)
            extra_weight = total - tone_weight
            coverage = sum(1 for pc in tones if weights[pc] > 0) / len(tones)
            root_bonus = weights[root] * 0.25
            score = tone_weight + root_bonus + coverage * 24.0 - extra_weight * 0.35
            if score > best_score:
                best_score = score
                best_name = f"{PITCH_NAMES[root]}{suffix}"
    return best_name


def _chordpro_body(chords: list[str], lyrics: str) -> list[str]:
    if not lyrics.strip():
        return [" ".join(f"[{chord}]" for chord in row) for row in _chunks(chords, 4)]

    body: list[str] = []
    chord_index = 0
    for raw_line in lyrics.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            body.append("")
            continue
        section = SECTION_RE.match(line)
        if section:
            body.append(f"{{comment: {section.group('label')}}}")
            continue
        chord = chords[chord_index] if chord_index < len(chords) else "N.C."
        body.append(f"[{chord}]{raw_line}")
        chord_index += 1

    if chord_index < len(chords):
        body.append("")
        body.extend(" ".join(f"[{chord}]" for chord in row) for row in _chunks(chords[chord_index:], 4))
    return body


def _crd_body(chords: list[str], lyrics: str) -> list[str]:
    if not lyrics.strip():
        return [" | ".join(row) for row in _chunks(chords, 4)]

    body: list[str] = []
    chord_index = 0
    for raw_line in lyrics.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            body.append("")
            continue
        section = SECTION_RE.match(line)
        if section:
            body.append(f"[{section.group('label')}]")
            continue
        chord = chords[chord_index] if chord_index < len(chords) else "N.C."
        body.append(chord)
        body.append(raw_line)
        chord_index += 1

    if chord_index < len(chords):
        body.append("")
        body.extend(" | ".join(row) for row in _chunks(chords[chord_index:], 4))
    return body


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _clean_directive(value: str) -> str:
    return (value or "Untitled").replace("\n", " ").strip() or "Untitled"
