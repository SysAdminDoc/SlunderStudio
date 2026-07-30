"""Validated audio-buffer decoding, channel normalization, and resampling."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np


MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 384_000


class AudioBufferError(ValueError):
    """Raised when an audio buffer cannot safely enter a processing graph."""


def validate_sample_rate(sample_rate: int) -> int:
    """Return a supported integer sample rate or raise a user-facing error."""
    if isinstance(sample_rate, bool):
        raise AudioBufferError("Sample rate must be an integer")
    try:
        value = int(sample_rate)
    except (TypeError, ValueError) as exc:
        raise AudioBufferError("Sample rate must be an integer") from exc
    if value != sample_rate or not MIN_SAMPLE_RATE <= value <= MAX_SAMPLE_RATE:
        raise AudioBufferError(
            f"Sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz"
        )
    return value


def validate_audio_buffer(audio: np.ndarray) -> np.ndarray:
    """Validate a mono or frames-by-channels numeric buffer as finite float32."""
    source = np.asarray(audio)
    if source.ndim not in (1, 2):
        raise AudioBufferError(
            f"Audio must be mono or frames-by-channels, got shape {source.shape}"
        )
    if source.shape[0] == 0 or (source.ndim == 2 and source.shape[1] == 0):
        raise AudioBufferError("Audio contains no frames")
    if not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source):
        raise AudioBufferError("Audio samples must be real numbers")
    frames = source.astype(np.float32, copy=False)
    if not np.all(np.isfinite(frames)):
        raise AudioBufferError("Audio contains NaN or infinite samples")
    return np.ascontiguousarray(frames)


def normalize_channel_layout(
    audio: np.ndarray,
    *,
    target_channels: int = 2,
) -> np.ndarray:
    """Normalize mono/stereo/surround input to mono or stereo deterministically."""
    frames = validate_audio_buffer(audio)
    if target_channels not in (1, 2):
        raise AudioBufferError("Only mono and stereo output layouts are supported")

    if frames.ndim == 1:
        mono = frames
        if target_channels == 1:
            return mono.copy()
        return np.ascontiguousarray(np.column_stack((mono, mono)))

    channels = frames.shape[1]
    if target_channels == 1:
        return np.ascontiguousarray(frames.mean(axis=1, dtype=np.float32))
    if channels == 1:
        return np.ascontiguousarray(np.repeat(frames, 2, axis=1))
    if channels == 2:
        return frames.copy()

    # Conventional WAV/CAF surround order: L, R, C, LFE, surrounds...
    # Quad is conventionally L, R, Ls, Rs and therefore has no center/LFE.
    left = frames[:, 0].copy()
    right = frames[:, 1].copy()
    if channels == 3:
        left += frames[:, 2] * np.float32(0.70710678)
        right += frames[:, 2] * np.float32(0.70710678)
    elif channels == 4:
        left += frames[:, 2] * np.float32(0.70710678)
        right += frames[:, 3] * np.float32(0.70710678)
    else:
        left += frames[:, 2] * np.float32(0.70710678)
        right += frames[:, 2] * np.float32(0.70710678)
        if channels >= 6:
            left += frames[:, 3] * np.float32(0.5)
            right += frames[:, 3] * np.float32(0.5)
            left += frames[:, 4] * np.float32(0.70710678)
            right += frames[:, 5] * np.float32(0.70710678)
            extras = frames[:, 6:]
        else:
            left += frames[:, 3] * np.float32(0.70710678)
            right += frames[:, 4] * np.float32(0.70710678)
            extras = frames[:, 5:]
        for index in range(extras.shape[1]):
            if index % 2:
                right += extras[:, index] * np.float32(0.5)
            else:
                left += extras[:, index] * np.float32(0.5)
    return np.ascontiguousarray(np.column_stack((left, right)), dtype=np.float32)


def resample_audio(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """Polyphase-resample while preserving duration and frames-first layout."""
    frames = validate_audio_buffer(audio)
    source_rate = validate_sample_rate(source_sample_rate)
    target_rate = validate_sample_rate(target_sample_rate)
    if source_rate == target_rate:
        return frames.copy()

    from scipy.signal import resample_poly

    divisor = math.gcd(source_rate, target_rate)
    resampled = resample_poly(
        frames,
        target_rate // divisor,
        source_rate // divisor,
        axis=0,
        padtype="constant",
    )
    expected_frames = max(1, int(round(len(frames) * target_rate / source_rate)))
    if len(resampled) > expected_frames:
        resampled = resampled[:expected_frames]
    elif len(resampled) < expected_frames:
        padding_shape = (expected_frames - len(resampled), *resampled.shape[1:])
        resampled = np.concatenate(
            (resampled, np.zeros(padding_shape, dtype=resampled.dtype)),
            axis=0,
        )
    result = resampled.astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise AudioBufferError("Resampling produced invalid samples")
    return np.ascontiguousarray(result)


def prepare_audio_buffer(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
    *,
    target_channels: int = 2,
) -> np.ndarray:
    """Validate, normalize channels, and resample for a processing graph."""
    normalized = normalize_channel_layout(
        audio,
        target_channels=target_channels,
    )
    source_rate = validate_sample_rate(source_sample_rate)
    target_rate = validate_sample_rate(target_sample_rate)
    if source_rate == target_rate:
        return normalized
    return resample_audio(normalized, source_rate, target_rate)


def decode_audio_file(
    file_path: str | Path,
    *,
    target_sample_rate: Optional[int] = None,
    target_channels: int = 2,
) -> tuple[np.ndarray, int]:
    """Decode a local audio file and optionally prepare it for a project rate."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    import soundfile as sf

    try:
        audio, source_rate = sf.read(
            str(path),
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise AudioBufferError(f"Could not decode {path.name}: {exc}") from exc

    source_rate = validate_sample_rate(source_rate)
    if target_sample_rate is None:
        return (
            normalize_channel_layout(audio, target_channels=target_channels),
            source_rate,
        )
    target_rate = validate_sample_rate(target_sample_rate)
    return (
        prepare_audio_buffer(
            audio,
            source_rate,
            target_rate,
            target_channels=target_channels,
        ),
        target_rate,
    )
