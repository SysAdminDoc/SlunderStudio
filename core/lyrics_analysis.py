"""Deterministic, advisory rhyme and cadence feedback for lyric drafts."""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass


_WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_SECTION_PATTERN = re.compile(r"^\[[^\]]+\]\s*$")
_PARENTHETICAL_PATTERN = re.compile(r"\([^)]*\)")
_VOWEL_PATTERN = re.compile(r"[aeiouy]+", re.IGNORECASE)


@dataclass(frozen=True)
class LyricLineAnalysis:
    """Measured facts for one non-empty lyric line."""

    line_number: int
    text: str
    word_count: int
    syllables: int
    ending_word: str
    rhyme_key: str


@dataclass(frozen=True)
class LyricAnalysis:
    """Advisory metrics; scores are signals, not judgments of lyric quality."""

    lines: tuple[LyricLineAnalysis, ...] = ()
    word_count: int = 0
    syllable_count: int = 0
    rhyme_coverage: float = 0.0
    rhyme_covered_lines: int = 0
    rhyme_eligible_lines: int = 0
    cadence_consistency: float = 0.0
    average_syllables: float = 0.0
    min_syllables: int = 0
    max_syllables: int = 0
    cadence_range: int = 0
    rhyme_families: tuple[tuple[str, int], ...] = ()

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def advisory_notes(self) -> tuple[str, ...]:
        """Return plain-language prompts without suggesting an automatic rewrite."""
        if not self.lines:
            return ("Add two or more lyric lines to see advisory feedback.",)
        notes: list[str] = []
        if self.rhyme_eligible_lines < 2:
            notes.append("Add another content line to compare end sounds.")
        elif self.rhyme_coverage < 50:
            notes.append(
                "Few repeated end sounds were detected; that may be intentional in free verse."
            )
        if self.cadence_range > 4:
            notes.append(
                "Line lengths vary widely; try reading the draft over the intended groove."
            )
        if not notes:
            notes.append("Use these signals as prompts while keeping your own phrasing and flow.")
        return tuple(notes)


def _words(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text)


def estimate_syllables(word: str) -> int:
    """Estimate spoken syllables with a small offline heuristic.

    This deliberately avoids claiming phonetic accuracy: it is only intended to
    surface unusually uneven line lengths without a language model or network
    lookup. Unicode words fall back to vowel-group counting as well.
    """
    normalized = "".join(character for character in word.casefold() if character.isalpha())
    if not normalized:
        return 0
    if len(normalized) <= 3:
        return 1
    groups = _VOWEL_PATTERN.findall(normalized)
    count = len(groups) or 1
    if normalized.endswith("e") and not normalized.endswith(("le", "ye")) and count > 1:
        count -= 1
    if normalized.endswith("ed") and count > 1 and len(normalized) > 3:
        count -= 1
    return max(1, count)


def _rhyme_key(word: str) -> str:
    normalized = "".join(character for character in word.casefold() if character.isalpha())
    if not normalized:
        return ""
    vowels = [match.start() for match in _VOWEL_PATTERN.finditer(normalized)]
    if vowels:
        start = vowels[-1]
        # A one-letter key is too permissive to be useful as an end sound.
        if len(normalized) - start >= 2:
            return normalized[start:]
    return normalized[-3:]


def _content_lines(text: str) -> list[tuple[int, str]]:
    content: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(str(text or "").replace("\r", "").split("\n"), 1):
        line = raw_line.strip()
        if not line or _SECTION_PATTERN.fullmatch(line):
            continue
        line = _PARENTHETICAL_PATTERN.sub(" ", line).strip()
        if line:
            content.append((line_number, line))
    return content


def analyze_lyrics(text: str) -> LyricAnalysis:
    """Analyze end sounds and syllable cadence without mutating ``text``."""
    measured: list[LyricLineAnalysis] = []
    for line_number, line in _content_lines(text):
        words = _words(line)
        if not words:
            continue
        ending = words[-1]
        measured.append(
            LyricLineAnalysis(
                line_number=line_number,
                text=line,
                word_count=len(words),
                syllables=sum(estimate_syllables(word) for word in words),
                ending_word=ending,
                rhyme_key=_rhyme_key(ending),
            )
        )

    if not measured:
        return LyricAnalysis()

    family_counts = Counter(
        line.rhyme_key for line in measured if line.rhyme_key
    )
    eligible = sum(1 for line in measured if line.rhyme_key)
    covered = sum(
        count for key, count in family_counts.items() if key and count >= 2
    )
    rhyme_coverage = (covered / eligible * 100.0) if eligible > 1 else 0.0
    syllables = [line.syllables for line in measured]
    median = float(statistics.median(syllables))
    deviation = statistics.fmean(abs(value - median) for value in syllables)
    cadence_consistency = max(
        0.0,
        min(100.0, 100.0 * (1.0 - deviation / max(median, 1.0))),
    )
    families = tuple(
        sorted(
            ((key, count) for key, count in family_counts.items() if count >= 2),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return LyricAnalysis(
        lines=tuple(measured),
        word_count=sum(line.word_count for line in measured),
        syllable_count=sum(syllables),
        rhyme_coverage=round(rhyme_coverage, 1),
        rhyme_covered_lines=covered,
        rhyme_eligible_lines=eligible,
        cadence_consistency=round(cadence_consistency, 1),
        average_syllables=round(statistics.fmean(syllables), 1),
        min_syllables=min(syllables),
        max_syllables=max(syllables),
        cadence_range=max(syllables) - min(syllables),
        rhyme_families=families,
    )


__all__ = [
    "LyricAnalysis",
    "LyricLineAnalysis",
    "analyze_lyrics",
    "estimate_syllables",
]
