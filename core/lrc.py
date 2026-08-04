"""Enhanced LRC formatting from verified lyric-to-note alignment data."""
from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_SECTION_PATTERN = re.compile(r"^\[[^\]]+\]\s*$")
_PARENTHETICAL_PATTERN = re.compile(r"\([^)]*\)")


class LRCValidationError(ValueError):
    """Raised when an Enhanced LRC cannot be derived from verified alignment."""


@dataclass(frozen=True)
class LRCWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class LRCLine:
    words: tuple[LRCWord, ...]
    section_tag: str = ""


def _words(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text)


def _normalized_word(text: str) -> str:
    return "".join(character for character in text.casefold() if character.isalnum())


def _timestamp(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise LRCValidationError("LRC timestamps must be finite and non-negative")
    centiseconds = int(math.floor(seconds * 100.0 + 0.5))
    minutes, remainder = divmod(centiseconds, 6_000)
    return f"{minutes:02d}:{remainder / 100.0:05.2f}"


def _lyric_lines(lyrics: str) -> list[tuple[str, list[str]]]:
    result: list[tuple[str, list[str]]] = []
    for raw_line in str(lyrics or "").replace("\r", "").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _SECTION_PATTERN.fullmatch(line):
            result.append((line, []))
            continue
        content = _PARENTHETICAL_PATTERN.sub(" ", line)
        words = _words(content)
        if words:
            result.append(("", words))
    return result


def _validated_words(aligned_notes: Iterable[dict[str, Any]]) -> list[LRCWord]:
    words: list[LRCWord] = []
    for index, entry in enumerate(aligned_notes):
        if not isinstance(entry, dict):
            raise LRCValidationError(f"Alignment entry {index + 1} is not an object")
        text = str(entry.get("text", "") or "").strip()
        if not text:
            continue
        try:
            start = float(entry["start"])
            end = float(entry["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LRCValidationError(
                f"Alignment entry {index + 1} has no valid start/end time"
            ) from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise LRCValidationError(
                f"Alignment entry {index + 1} has an invalid time range"
            )
        words.append(LRCWord(text=text, start=start, end=end))
    return words


def build_lrc_lines(lyrics: str, aligned_notes: Iterable[dict[str, Any]]) -> tuple[LRCLine, ...]:
    """Group verified aligned words back into the original lyric lines."""
    lyric_lines = _lyric_lines(lyrics)
    expected = [word for _tag, words in lyric_lines for word in words]
    aligned = _validated_words(aligned_notes)
    if not expected:
        raise LRCValidationError("Lyrics contain no timestampable words")
    if len(aligned) != len(expected):
        raise LRCValidationError(
            f"Alignment covers {len(aligned)} of {len(expected)} lyric words; "
            "generate a complete, exact alignment before exporting LRC"
        )
    if any(
        _normalized_word(actual.text) != _normalized_word(expected[index])
        for index, actual in enumerate(aligned[:len(expected)])
    ):
        raise LRCValidationError(
            "Alignment words do not match the lyric draft; export was refused"
        )

    lines: list[LRCLine] = []
    cursor = 0
    for section_tag, line_words in lyric_lines:
        if not line_words:
            lines.append(LRCLine(words=(), section_tag=section_tag))
            continue
        count = len(line_words)
        lines.append(LRCLine(words=tuple(aligned[cursor:cursor + count])))
        cursor += count
    return tuple(lines)


def format_enhanced_lrc(lyrics: str, aligned_notes: Iterable[dict[str, Any]]) -> str:
    """Return Enhanced LRC with line and word-level centisecond timestamps."""
    lines = build_lrc_lines(lyrics, aligned_notes)
    rendered: list[str] = []
    for line in lines:
        if line.section_tag:
            rendered.append(line.section_tag)
            continue
        line_start = _timestamp(line.words[0].start)
        words = " ".join(
            f"<{_timestamp(word.start)}>{word.text}" for word in line.words
        )
        rendered.append(f"[{line_start}]{words}")
    return "\n".join(rendered) + "\n"


def write_enhanced_lrc(
    destination: str | os.PathLike[str],
    lyrics: str,
    aligned_notes: Iterable[dict[str, Any]],
) -> str:
    """Atomically write a validated Enhanced LRC file and return its path."""
    path = Path(destination).expanduser()
    if not path.name:
        raise LRCValidationError("LRC destination must be a file path")
    content = format_enhanced_lrc(lyrics, aligned_notes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return str(path)


__all__ = [
    "LRCLine",
    "LRCValidationError",
    "LRCWord",
    "build_lrc_lines",
    "format_enhanced_lrc",
    "write_enhanced_lrc",
]
