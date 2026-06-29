"""
Slunder Studio v0.1.24 - Vocal pitch correction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from core.provenance import write_provenance_sidecar
from core.settings import get_default_output_dir


@dataclass
class AutoTuneParams:
    input_path: str
    output_path: str = ""
    strength: float = 0.75
    fmin_note: str = "C2"
    fmax_note: str = "C7"
    frame_length: int = 2048
    hop_length: int = 512


@dataclass
class AutoTuneResult:
    output_path: str
    sample_rate: int
    duration: float
    frames_analyzed: int
    voiced_frames: int
    mean_abs_correction: float
    max_abs_correction: float
    strength: float


def autotune_file(
    params: AutoTuneParams,
    progress_cb: Callable[[int], None] | None = None,
    step_cb: Callable[[str], None] | None = None,
    log_cb: Callable[[str], None] | None = None,
    cancel_event=None,
) -> AutoTuneResult:
    """Pitch-correct a vocal file toward the nearest semitone and write a WAV."""
    import librosa
    import soundfile as sf

    input_path = Path(params.input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Auto-tune input not found: {input_path}")

    strength = float(np.clip(params.strength, 0.0, 1.0))
    output_path = Path(params.output_path) if params.output_path else _default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if step_cb:
        step_cb("Loading vocal audio...")
    audio, sr = sf.read(str(input_path), dtype="float32", always_2d=True)
    if audio.size == 0:
        raise ValueError("Auto-tune input is empty")
    audio = _sanitize_audio(audio)
    mono = audio.mean(axis=1)
    duration = len(audio) / float(sr)

    if cancel_event and cancel_event.is_set():
        return _cancelled_result(output_path, sr, duration, strength)

    if progress_cb:
        progress_cb(15)
    if step_cb:
        step_cb("Estimating vocal pitch...")

    f0, voiced_flag, _voiced_prob = librosa.pyin(
        mono,
        fmin=librosa.note_to_hz(params.fmin_note),
        fmax=librosa.note_to_hz(params.fmax_note),
        sr=sr,
        frame_length=params.frame_length,
        hop_length=params.hop_length,
    )
    corrections = compute_frame_corrections(f0, voiced_flag, strength)
    voiced_frames = int(np.count_nonzero(np.isfinite(f0) & voiced_flag))
    mean_abs = float(np.mean(np.abs(corrections[np.nonzero(corrections)]))) if np.any(corrections) else 0.0
    max_abs = float(np.max(np.abs(corrections))) if corrections.size else 0.0

    if cancel_event and cancel_event.is_set():
        return _cancelled_result(output_path, sr, duration, strength)

    if progress_cb:
        progress_cb(35)
    if step_cb:
        step_cb("Correcting pitch...")

    if strength <= 0.0 or voiced_frames == 0 or max_abs < 0.01:
        tuned = audio.copy()
    else:
        tuned = apply_segmented_pitch_shift(
            audio,
            sr,
            corrections,
            hop_length=params.hop_length,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

    if cancel_event and cancel_event.is_set():
        return _cancelled_result(output_path, sr, duration, strength)

    tuned = _match_peak(tuned, audio)
    sf.write(str(output_path), tuned, sr, subtype="PCM_16")
    if progress_cb:
        progress_cb(95)
    write_provenance_sidecar(
        output_path,
        module="vocal_suite",
        operation="vocal_autotune",
        model_id="librosa-pyin-pitch-shift",
        model_name="Librosa pYIN pitch correction",
        parameters=asdict(params) | {"strength": strength},
        source_paths=[str(input_path)],
        export_format="wav",
        output_kind="processed",
        extra={
            "frames_analyzed": int(len(corrections)),
            "voiced_frames": voiced_frames,
            "mean_abs_correction_semitones": mean_abs,
            "max_abs_correction_semitones": max_abs,
        },
    )
    if progress_cb:
        progress_cb(100)
    if log_cb:
        log_cb(f"Auto-tune wrote {output_path}")

    return AutoTuneResult(
        output_path=str(output_path),
        sample_rate=sr,
        duration=duration,
        frames_analyzed=int(len(corrections)),
        voiced_frames=voiced_frames,
        mean_abs_correction=mean_abs,
        max_abs_correction=max_abs,
        strength=strength,
    )


def compute_frame_corrections(
    f0: np.ndarray,
    voiced_flag: np.ndarray | None,
    strength: float,
) -> np.ndarray:
    """Return per-frame semitone shifts toward nearest semitone."""
    if f0 is None:
        return np.zeros(0, dtype=np.float32)
    import librosa

    strength = float(np.clip(strength, 0.0, 1.0))
    midi = librosa.hz_to_midi(f0)
    valid = np.isfinite(midi)
    if voiced_flag is not None:
        valid &= voiced_flag.astype(bool)

    corrections = np.zeros_like(midi, dtype=np.float32)
    corrections[valid] = (np.rint(midi[valid]) - midi[valid]) * strength
    corrections[np.abs(corrections) < 0.01] = 0.0
    return _smooth_and_quantize(corrections)


def apply_segmented_pitch_shift(
    audio: np.ndarray,
    sr: int,
    corrections: np.ndarray,
    *,
    hop_length: int,
    progress_cb: Callable[[int], None] | None = None,
    cancel_event=None,
) -> np.ndarray:
    """Apply chunked pitch shifts with overlap-add crossfades."""
    n_samples, channels = audio.shape
    output = np.zeros_like(audio, dtype=np.float32)
    weights = np.zeros((n_samples, 1), dtype=np.float32)
    segments = _segments_for_corrections(corrections, n_samples, hop_length)
    fade = min(1024, max(128, hop_length))

    for index, (start, end, shift) in enumerate(segments):
        if cancel_event and cancel_event.is_set():
            break
        if end <= start:
            continue

        ext_start = max(0, start - fade)
        ext_end = min(n_samples, end + fade)
        chunk = audio[ext_start:ext_end, :]
        shifted = _pitch_shift_channels(chunk, sr, shift)
        shifted = _fit_length(shifted, len(chunk), channels)

        weight = np.ones(len(chunk), dtype=np.float32)
        if ext_start < start:
            ramp = start - ext_start
            weight[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=False, dtype=np.float32)
        if ext_end > end:
            ramp = ext_end - end
            weight[-ramp:] = np.linspace(1.0, 0.0, ramp, endpoint=False, dtype=np.float32)

        output[ext_start:ext_end, :] += shifted * weight[:, None]
        weights[ext_start:ext_end, :] += weight[:, None]

        if progress_cb and segments:
            progress_cb(35 + int(55 * (index + 1) / len(segments)))

    missing = weights[:, 0] <= 1e-6
    if np.any(missing):
        output[missing, :] = audio[missing, :]
        weights[missing, :] = 1.0
    return output / np.maximum(weights, 1e-6)


def _default_output_path(input_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_default_output_dir() / "vocals" / "autotune" / f"{input_path.stem}_autotune_{timestamp}.wav"


def _sanitize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def _smooth_and_quantize(corrections: np.ndarray) -> np.ndarray:
    if len(corrections) < 3:
        return np.round(corrections / 0.05) * 0.05
    smoothed = corrections.copy()
    for index in range(len(corrections)):
        start = max(0, index - 2)
        end = min(len(corrections), index + 3)
        smoothed[index] = float(np.median(corrections[start:end]))
    smoothed = np.round(smoothed / 0.05) * 0.05
    smoothed[np.abs(smoothed) < 0.025] = 0.0
    return smoothed.astype(np.float32)


def _segments_for_corrections(
    corrections: np.ndarray,
    n_samples: int,
    hop_length: int,
) -> list[tuple[int, int, float]]:
    if len(corrections) == 0:
        return [(0, n_samples, 0.0)]
    segments: list[tuple[int, int, float]] = []
    current_shift = float(corrections[0])
    current_start = 0
    for frame_index in range(1, len(corrections)):
        shift = float(corrections[frame_index])
        boundary = int(min(n_samples, frame_index * hop_length))
        if shift != current_shift:
            segments.append((current_start, boundary, current_shift))
            current_start = boundary
            current_shift = shift
    segments.append((current_start, n_samples, current_shift))
    return segments


def _pitch_shift_channels(chunk: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    if abs(n_steps) < 0.01 or len(chunk) < 2048:
        return chunk.copy()
    import librosa

    shifted = []
    for channel in range(chunk.shape[1]):
        shifted.append(
            librosa.effects.pitch_shift(
                y=chunk[:, channel].astype(np.float32),
                sr=sr,
                n_steps=float(n_steps),
            )
        )
    return np.stack(shifted, axis=1).astype(np.float32)


def _fit_length(audio: np.ndarray, target_samples: int, channels: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] != channels:
        audio = np.repeat(audio[:, :1], channels, axis=1)
    if len(audio) > target_samples:
        return audio[:target_samples, :]
    if len(audio) < target_samples:
        pad = np.zeros((target_samples - len(audio), channels), dtype=np.float32)
        return np.vstack([audio, pad])
    return audio


def _match_peak(processed: np.ndarray, original: np.ndarray) -> np.ndarray:
    original_peak = float(np.max(np.abs(original))) if original.size else 0.0
    processed_peak = float(np.max(np.abs(processed))) if processed.size else 0.0
    if processed_peak <= 1e-8:
        return processed.astype(np.float32)
    target_peak = min(0.98, max(original_peak, 0.1))
    return (processed / processed_peak * target_peak).astype(np.float32)


def _cancelled_result(output_path: Path, sr: int, duration: float, strength: float) -> AutoTuneResult:
    return AutoTuneResult(
        output_path=str(output_path),
        sample_rate=sr,
        duration=duration,
        frames_analyzed=0,
        voiced_frames=0,
        mean_abs_correction=0.0,
        max_abs_correction=0.0,
        strength=strength,
    )
