"""
Slunder Studio v0.0.2 — Smart Mastering
Automated mastering chain: EQ, compression, stereo enhancement,
limiting, and loudness normalization (LUFS targeting).
"""
import os
import time
from typing import Optional, Callable
from dataclasses import dataclass, field

import numpy as np

from core.settings import get_config_dir


@dataclass
class MasteringPreset:
    """Mastering preset configuration."""
    name: str = "Balanced"
    # EQ
    low_shelf_gain: float = 1.5     # dB boost at low end
    low_shelf_freq: float = 80.0    # Hz
    high_shelf_gain: float = 1.0    # dB boost at high end
    high_shelf_freq: float = 12000.0
    mid_gain: float = 0.0           # dB mid scoop/boost
    mid_freq: float = 2500.0
    mid_q: float = 1.0
    # Compression
    comp_threshold: float = -12.0   # dB
    comp_ratio: float = 3.0         # :1
    comp_attack: float = 10.0       # ms
    comp_release: float = 100.0     # ms
    comp_makeup: float = 3.0        # dB
    # Limiting
    limiter_ceiling: float = -0.3   # dB
    limiter_release: float = 50.0   # ms
    # Stereo
    stereo_width: float = 1.0       # 0.0 (mono) to 2.0 (extra wide)
    # Loudness
    target_lufs: float = -14.0      # integrated loudness target


PRESETS = {
    "Balanced": MasteringPreset(name="Balanced"),
    "Loud / Radio": MasteringPreset(
        name="Loud / Radio", comp_threshold=-16, comp_ratio=4.0,
        comp_makeup=5.0, target_lufs=-11.0, limiter_ceiling=-0.1,
        high_shelf_gain=2.0,
    ),
    "Warm / Analog": MasteringPreset(
        name="Warm / Analog", low_shelf_gain=2.5, high_shelf_gain=-0.5,
        comp_threshold=-10, comp_ratio=2.5, comp_attack=20,
        stereo_width=0.9, target_lufs=-14.0,
    ),
    "Bright / Crisp": MasteringPreset(
        name="Bright / Crisp", low_shelf_gain=0.5, high_shelf_gain=3.0,
        mid_gain=1.0, mid_freq=3000, comp_ratio=2.0,
        stereo_width=1.2, target_lufs=-14.0,
    ),
    "Hip-Hop / Trap": MasteringPreset(
        name="Hip-Hop / Trap", low_shelf_gain=4.0, low_shelf_freq=60,
        high_shelf_gain=1.5, comp_threshold=-14, comp_ratio=4.0,
        comp_makeup=4.0, target_lufs=-12.0, stereo_width=1.1,
    ),
    "Cinematic": MasteringPreset(
        name="Cinematic", low_shelf_gain=2.0, high_shelf_gain=1.5,
        comp_threshold=-8, comp_ratio=2.0, comp_attack=30,
        comp_release=200, stereo_width=1.4, target_lufs=-16.0,
    ),
    "Lo-Fi": MasteringPreset(
        name="Lo-Fi", low_shelf_gain=1.0, high_shelf_gain=-3.0,
        high_shelf_freq=8000, comp_threshold=-18, comp_ratio=5.0,
        stereo_width=0.8, target_lufs=-16.0,
    ),
    "Streaming (Spotify)": MasteringPreset(
        name="Streaming (Spotify)", target_lufs=-14.0,
        comp_threshold=-12, comp_ratio=3.0, limiter_ceiling=-1.0,
    ),
}


@dataclass
class MasteringResult:
    """Result from mastering processing."""
    audio: Optional[np.ndarray] = None
    sample_rate: int = 44100
    duration: float = 0.0
    processing_time: float = 0.0
    input_lufs: float = 0.0
    output_lufs: float = 0.0
    peak_db: float = 0.0
    preset_name: str = ""
    error: Optional[str] = None


# ── DSP Functions ──────────────────────────────────────────────────────────────

def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    return 20.0 * np.log10(max(linear, 1e-10))


def measure_lufs(audio: np.ndarray, sr: int) -> float:
    """Simplified LUFS measurement (ITU-R BS.1770 approximation)."""
    if audio.ndim == 1:
        audio = np.column_stack([audio, audio])

    # K-weighting filter (simplified high-shelf + high-pass)
    # Apply simple RMS-based approximation
    block_size = int(0.4 * sr)  # 400ms blocks
    hop = int(0.1 * sr)  # 100ms hop

    powers = []
    for ch in range(min(audio.shape[1], 2)):
        channel = audio[:, ch]
        for start in range(0, len(channel) - block_size, hop):
            block = channel[start:start + block_size]
            power = np.mean(block ** 2)
            if power > 0:
                powers.append(power)

    if not powers:
        return -70.0

    # Gated loudness (relative threshold)
    powers = np.array(powers)
    abs_threshold = 10 ** (-70 / 10)
    above_abs = powers[powers > abs_threshold]

    if len(above_abs) == 0:
        return -70.0

    relative_threshold = np.mean(above_abs) * (10 ** (-10 / 10))
    gated = above_abs[above_abs > relative_threshold]

    if len(gated) == 0:
        return -70.0

    lufs = -0.691 + 10 * np.log10(np.mean(gated))
    return float(lufs)


def apply_eq_shelf(audio: np.ndarray, sr: int, freq: float,
                   gain_db: float, shelf_type: str = "low") -> np.ndarray:
    """Apply a simple shelf EQ using biquad filter."""
    if abs(gain_db) < 0.1:
        return audio

    A = db_to_linear(gain_db / 2.0)
    w0 = 2 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt(2.0)

    if shelf_type == "low":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    else:  # high shelf
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    return _biquad_filter(audio, b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def _biquad_filter(audio: np.ndarray, b0, b1, b2, a1, a2) -> np.ndarray:
    """Apply biquad IIR filter to audio."""
    output = np.zeros_like(audio)
    channels = audio.shape[1] if audio.ndim == 2 else 1

    for ch in range(channels):
        x = audio[:, ch] if audio.ndim == 2 else audio
        y = np.zeros_like(x)
        x1 = x2 = y1 = y2 = 0.0

        for i in range(len(x)):
            y[i] = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, x[i]
            y2, y1 = y1, y[i]

        if audio.ndim == 2:
            output[:, ch] = y
        else:
            output = y

    return output


def apply_compression(audio: np.ndarray, sr: int,
                      threshold_db: float, ratio: float,
                      attack_ms: float, release_ms: float,
                      makeup_db: float) -> np.ndarray:
    """Simple compressor with attack/release envelope."""
    threshold = db_to_linear(threshold_db)
    makeup = db_to_linear(makeup_db)
    attack_coeff = np.exp(-1.0 / (attack_ms * 0.001 * sr))
    release_coeff = np.exp(-1.0 / (release_ms * 0.001 * sr))

    output = np.copy(audio)
    envelope = 0.0

    for i in range(len(audio)):
        sample = np.max(np.abs(audio[i])) if audio.ndim == 2 else abs(audio[i])

        if sample > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * sample
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * sample

        if envelope > threshold:
            gain_reduction = threshold + (envelope - threshold) / ratio
            gain = gain_reduction / max(envelope, 1e-10)
        else:
            gain = 1.0

        output[i] = audio[i] * gain * makeup

    return output


def apply_limiter(audio: np.ndarray, ceiling_db: float,
                  release_ms: float, sr: int) -> np.ndarray:
    """Brick-wall limiter."""
    ceiling = db_to_linear(ceiling_db)
    release_coeff = np.exp(-1.0 / (release_ms * 0.001 * sr))
    output = np.copy(audio)
    gain = 1.0

    for i in range(len(audio)):
        peak = np.max(np.abs(audio[i])) if audio.ndim == 2 else abs(audio[i])

        if peak * gain > ceiling:
            gain = ceiling / max(peak, 1e-10)
        else:
            gain = release_coeff * gain + (1 - release_coeff) * 1.0
            gain = min(gain, 1.0)

        output[i] = audio[i] * gain

    return output


def apply_stereo_width(audio: np.ndarray, width: float) -> np.ndarray:
    """Adjust stereo width. 0=mono, 1=unchanged, 2=extra wide."""
    if audio.ndim != 2 or audio.shape[1] != 2:
        return audio

    mid = (audio[:, 0] + audio[:, 1]) * 0.5
    side = (audio[:, 0] - audio[:, 1]) * 0.5

    side *= width

    output = np.column_stack([mid + side, mid - side])
    return output


def normalize_lufs(audio: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Normalize audio to target LUFS."""
    current = measure_lufs(audio, sr)
    if current < -60:
        return audio

    diff = target_lufs - current
    gain = db_to_linear(diff)
    return audio * gain


# ── Master Chain ───────────────────────────────────────────────────────────────

def master_audio(audio: np.ndarray, sr: int,
                 preset: Optional[MasteringPreset] = None,
                 progress_callback: Optional[Callable] = None) -> MasteringResult:
    """Run the full mastering chain on audio."""
    if preset is None:
        preset = PRESETS["Balanced"]

    t0 = time.time()

    try:
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        input_lufs = measure_lufs(audio, sr)

        if progress_callback:
            progress_callback(0.1, "Applying EQ...")

        # EQ
        processed = apply_eq_shelf(audio, sr, preset.low_shelf_freq,
                                   preset.low_shelf_gain, "low")
        processed = apply_eq_shelf(processed, sr, preset.high_shelf_freq,
                                   preset.high_shelf_gain, "high")

        if progress_callback:
            progress_callback(0.3, "Applying compression...")

        # Compression
        processed = apply_compression(
            processed, sr,
            preset.comp_threshold, preset.comp_ratio,
            preset.comp_attack, preset.comp_release,
            preset.comp_makeup,
        )

        if progress_callback:
            progress_callback(0.5, "Adjusting stereo width...")

        # Stereo width
        processed = apply_stereo_width(processed, preset.stereo_width)

        if progress_callback:
            progress_callback(0.7, "Applying limiter...")

        # Limiter
        processed = apply_limiter(processed, preset.limiter_ceiling,
                                  preset.limiter_release, sr)

        if progress_callback:
            progress_callback(0.85, "Normalizing loudness...")

        # LUFS normalization
        processed = normalize_lufs(processed, sr, preset.target_lufs)

        # Final clip
        processed = np.clip(processed, -1.0, 1.0)

        output_lufs = measure_lufs(processed, sr)
        peak = linear_to_db(np.max(np.abs(processed)))
        proc_time = time.time() - t0

        if progress_callback:
            progress_callback(1.0, "Mastering complete")

        return MasteringResult(
            audio=processed, sample_rate=sr,
            duration=len(processed) / sr,
            processing_time=proc_time,
            input_lufs=input_lufs, output_lufs=output_lufs,
            peak_db=peak, preset_name=preset.name,
        )

    except Exception as e:
        return MasteringResult(error=str(e), processing_time=time.time() - t0)
