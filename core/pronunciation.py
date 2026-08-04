"""Pronunciation correction data and bit-preserving audio splicing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


PRONUNCIATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PronunciationOverride:
    """One user-supplied phoneme replacement for a rendered time region."""

    unit: str
    start: float
    end: float
    phonemes: tuple[str, ...]

    @classmethod
    def from_values(
        cls,
        unit: str,
        start: float,
        end: float,
        phonemes: Iterable[str],
    ) -> "PronunciationOverride":
        tokens = tuple(str(token).strip() for token in phonemes if str(token).strip())
        if not tokens:
            raise ValueError("Enter at least one phoneme token.")
        start_value = float(start)
        end_value = float(end)
        if not np.isfinite(start_value) or not np.isfinite(end_value):
            raise ValueError("The pronunciation region must use finite times.")
        if start_value < 0 or end_value <= start_value:
            raise ValueError("The pronunciation region must have a positive duration.")
        label = str(unit or "selected region").strip() or "selected region"
        return cls(label, start_value, end_value, tokens)

    @classmethod
    def from_dict(cls, payload: dict) -> "PronunciationOverride":
        """Restore an override from sidecar/project JSON."""
        return cls.from_values(
            payload.get("unit", "selected region"),
            payload.get("start", 0.0),
            payload.get("end", 0.0),
            payload.get("phonemes", ()),
        )

    def to_dict(self) -> dict:
        """Return a JSON-safe, stable representation."""
        return {
            "schema_version": PRONUNCIATION_SCHEMA_VERSION,
            "unit": self.unit,
            "start": self.start,
            "end": self.end,
            "phonemes": list(self.phonemes),
        }


def parse_phoneme_text(value: str) -> tuple[str, ...]:
    """Parse space/comma-separated phoneme tokens entered in the UI."""
    tokens = tuple(token for token in value.replace(",", " ").split() if token)
    if not tokens:
        raise ValueError("Enter at least one phoneme token.")
    return tokens


def _as_frames_channels(audio: object) -> tuple[np.ndarray, bool]:
    data = np.asarray(audio)
    if data.ndim == 1:
        return np.ascontiguousarray(data[:, None], dtype=np.float32), True
    if data.ndim == 2 and data.shape[0] > 0 and data.shape[1] > 0:
        return np.ascontiguousarray(data, dtype=np.float32), False
    raise ValueError("Audio must be a non-empty mono or frames-by-channels array.")


def _match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.shape[1] == channels:
        return audio
    if audio.shape[1] == 1:
        return np.repeat(audio, channels, axis=1)
    if channels == 1:
        return audio.mean(axis=1, keepdims=True, dtype=np.float32)
    raise ValueError("Base and replacement audio have incompatible channel layouts.")


def _resample_frames(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly time-scale a region without adding an audio dependency."""
    if target_length <= 0:
        raise ValueError("The replacement region must contain at least one sample.")
    if len(audio) == target_length:
        return audio.copy()
    if len(audio) <= 0:
        raise ValueError("The replacement audio is empty.")
    source_axis = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    target_axis = np.linspace(0.0, 1.0, target_length, endpoint=False)
    columns = [np.interp(target_axis, source_axis, audio[:, channel]) for channel in range(audio.shape[1])]
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float32)


def apply_pronunciation_override(
    base_audio: object,
    replacement_audio: object,
    sample_rate: int,
    start: float,
    end: float,
    *,
    crossfade_ms: float = 20.0,
) -> np.ndarray:
    """Replace one region while leaving every outside sample bit-identical.

    The replacement is time-scaled to the selected region, then blended only
    inside that region.  The returned array keeps the base layout and length;
    samples before ``start`` and at/after ``end`` are copied without touching
    their values.
    """
    if isinstance(sample_rate, bool) or int(sample_rate) <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    if crossfade_ms < 0:
        raise ValueError("crossfade_ms must be non-negative.")

    base, base_mono = _as_frames_channels(base_audio)
    replacement, _replacement_mono = _as_frames_channels(replacement_audio)
    replacement = _match_channels(replacement, base.shape[1])

    start_sample = int(round(float(start) * int(sample_rate)))
    end_sample = int(round(float(end) * int(sample_rate)))
    start_sample = max(0, min(len(base), start_sample))
    end_sample = max(0, min(len(base), end_sample))
    if end_sample <= start_sample:
        raise ValueError("The pronunciation region is outside the base audio.")

    target_length = end_sample - start_sample
    replacement = _resample_frames(replacement, target_length)
    output = base.copy()
    fade_samples = min(
        int(round(float(crossfade_ms) * int(sample_rate) / 1000.0)),
        target_length // 2,
    )
    if fade_samples:
        fade_in = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)[:, None]
        fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=False, dtype=np.float32)[:, None]
        output[start_sample:start_sample + fade_samples] = (
            base[start_sample:start_sample + fade_samples] * (1.0 - fade_in)
            + replacement[:fade_samples] * fade_in
        )
        output[end_sample - fade_samples:end_sample] = (
            replacement[-fade_samples:] * (1.0 - fade_out)
            + base[end_sample - fade_samples:end_sample] * fade_out
        )
        middle_start = start_sample + fade_samples
        middle_end = end_sample - fade_samples
        if middle_end > middle_start:
            output[middle_start:middle_end] = replacement[fade_samples:-fade_samples]
    else:
        output[start_sample:end_sample] = replacement

    if base_mono:
        return np.ascontiguousarray(output[:, 0], dtype=np.float32)
    return np.ascontiguousarray(output, dtype=np.float32)
