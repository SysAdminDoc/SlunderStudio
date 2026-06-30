"""
Slunder Studio v0.1.28 — Smart Mastering
Automated mastering chain: EQ, compression, stereo enhancement,
limiting, and loudness normalization (LUFS targeting).
"""
import time
from typing import Optional, Callable
from dataclasses import dataclass, field

import numpy as np


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


@dataclass(frozen=True)
class LoudnessTarget:
    """Named delivery loudness target."""
    key: str
    label: str
    lufs: float
    category: str
    description: str


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


LUFS_TARGETS = {
    "streaming": LoudnessTarget(
        key="streaming",
        label="Streaming (-14 LUFS)",
        lufs=-14.0,
        category="streaming",
        description="General music streaming normalization target.",
    ),
    "youtube": LoudnessTarget(
        key="youtube",
        label="YouTube (-13 LUFS)",
        lufs=-13.0,
        category="streaming",
        description="Online video music delivery target.",
    ),
    "apple": LoudnessTarget(
        key="apple",
        label="Apple Music (-16 LUFS)",
        lufs=-16.0,
        category="streaming",
        description="Apple Sound Check-style music target.",
    ),
    "podcast": LoudnessTarget(
        key="podcast",
        label="Podcast stereo (-16 LUFS)",
        lufs=-16.0,
        category="spoken-word",
        description="Common stereo podcast delivery target.",
    ),
    "broadcast": LoudnessTarget(
        key="broadcast",
        label="Broadcast (-24 LUFS)",
        lufs=-24.0,
        category="broadcast",
        description="ATSC A/85-style broadcast loudness target.",
    ),
    "cinema": LoudnessTarget(
        key="cinema",
        label="Cinema dialog (-27 LUFS)",
        lufs=-27.0,
        category="cinema",
        description="Cinema/dialogue-oriented loudness target.",
    ),
    "cd": LoudnessTarget(
        key="cd",
        label="CD / loud master (-9 LUFS)",
        lufs=-9.0,
        category="physical",
        description="Loud physical or club master target.",
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


@dataclass(frozen=True)
class DynamicEQBand:
    """Single suggested EQ move for a stem."""
    frequency_hz: float
    gain_db: float
    q: float
    reason: str


@dataclass(frozen=True)
class DynamicEQSuggestion:
    """Stem-aware dynamic EQ recommendation."""
    stem_name: str
    stem_role: str
    bands: tuple[DynamicEQBand, ...] = field(default_factory=tuple)
    rms_db: float = -70.0
    spectral_centroid_hz: float = 0.0
    low_ratio: float = 0.0
    low_mid_ratio: float = 0.0
    mid_ratio: float = 0.0
    presence_ratio: float = 0.0
    high_ratio: float = 0.0


@dataclass(frozen=True)
class ShortTermLoudnessPoint:
    """Short-term loudness snapshot for a time window."""
    time_sec: float
    lufs: float


@dataclass(frozen=True)
class LoudnessMatchResult:
    """Result from matching audio loudness to a reference profile."""
    audio: np.ndarray
    sample_rate: int
    source_lufs: float
    reference_lufs: float
    output_lufs: float
    gain_db: float
    source_short_term: tuple[ShortTermLoudnessPoint, ...] = field(default_factory=tuple)
    reference_short_term: tuple[ShortTermLoudnessPoint, ...] = field(default_factory=tuple)
    output_short_term: tuple[ShortTermLoudnessPoint, ...] = field(default_factory=tuple)
    average_short_term_delta_db: float = 0.0
    max_short_term_delta_db: float = 0.0
    peak_db: float = 0.0


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


def _window_lufs(block: np.ndarray) -> float:
    arr = np.asarray(block, dtype=np.float32)
    if arr.ndim == 2:
        power = np.mean(np.square(arr), axis=0)
        mean_power = float(np.mean(power))
    else:
        mean_power = float(np.mean(np.square(arr)))
    if mean_power <= 1e-12:
        return -70.0
    return float(-0.691 + 10.0 * np.log10(mean_power))


def measure_short_term_lufs(audio: np.ndarray, sr: int,
                            window_sec: float = 3.0,
                            hop_sec: float = 1.0) -> tuple[ShortTermLoudnessPoint, ...]:
    """Measure short-term LUFS windows for reference comparison."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0 or sr <= 0:
        return tuple()

    if arr.ndim == 1:
        arr = np.column_stack([arr, arr])

    window = max(1, int(window_sec * sr))
    hop = max(1, int(hop_sec * sr))
    total = len(arr)
    if total == 0:
        return tuple()

    if total <= window:
        return (ShortTermLoudnessPoint(time_sec=0.0, lufs=_window_lufs(arr)),)

    starts = list(range(0, total - window + 1, hop))
    final_start = total - window
    if starts[-1] != final_start:
        starts.append(final_start)

    return tuple(
        ShortTermLoudnessPoint(
            time_sec=start / sr,
            lufs=_window_lufs(arr[start:start + window]),
        )
        for start in starts
    )


def _profile_delta(output: tuple[ShortTermLoudnessPoint, ...],
                   reference: tuple[ShortTermLoudnessPoint, ...]) -> tuple[float, float]:
    pairs = [
        abs(left.lufs - right.lufs)
        for left, right in zip(output, reference)
        if left.lufs > -60.0 and right.lufs > -60.0
    ]
    if not pairs:
        return 0.0, 0.0
    return float(np.mean(pairs)), float(np.max(pairs))


def match_loudness_to_reference(audio: np.ndarray, sr: int,
                                reference_audio: np.ndarray,
                                reference_sr: Optional[int] = None,
                                ceiling_db: float = -0.3) -> LoudnessMatchResult:
    """Normalize audio to a reference track and retain short-term LUFS profiles."""
    reference_sr = reference_sr or sr
    source = np.asarray(audio, dtype=np.float32)
    reference = np.asarray(reference_audio, dtype=np.float32)

    source_lufs = measure_lufs(source, sr)
    reference_lufs = measure_lufs(reference, reference_sr)
    gain_db = 0.0 if source_lufs < -60.0 or reference_lufs < -60.0 else reference_lufs - source_lufs
    matched = source * db_to_linear(gain_db)

    ceiling = db_to_linear(ceiling_db)
    peak = float(np.max(np.abs(matched))) if matched.size else 0.0
    if peak > ceiling:
        matched = matched * (ceiling / max(peak, 1e-10))

    matched = np.clip(matched, -1.0, 1.0).astype(np.float32)
    source_profile = measure_short_term_lufs(source, sr)
    reference_profile = measure_short_term_lufs(reference, reference_sr)
    output_profile = measure_short_term_lufs(matched, sr)
    avg_delta, max_delta = _profile_delta(output_profile, reference_profile)
    output_lufs = measure_lufs(matched, sr)
    output_peak = linear_to_db(float(np.max(np.abs(matched))) if matched.size else 0.0)

    return LoudnessMatchResult(
        audio=matched,
        sample_rate=sr,
        source_lufs=source_lufs,
        reference_lufs=reference_lufs,
        output_lufs=output_lufs,
        gain_db=gain_db,
        source_short_term=source_profile,
        reference_short_term=reference_profile,
        output_short_term=output_profile,
        average_short_term_delta_db=avg_delta,
        max_short_term_delta_db=max_delta,
        peak_db=output_peak,
    )


def _mono_float(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _stem_role(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("vocal", "voice", "vox", "lead")):
        return "vocal"
    if any(token in lower for token in ("bass", "808", "sub")):
        return "bass"
    if any(token in lower for token in ("drum", "kick", "snare", "perc")):
        return "drums"
    if any(token in lower for token in ("guitar", "piano", "keys", "synth", "pad")):
        return "instrument"
    return "stem"


def _band_ratio(freqs: np.ndarray, power: np.ndarray, low: float, high: float,
                total: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask) or total <= 0.0:
        return 0.0
    return float(np.sum(power[mask]) / total)


def _add_band(bands: list[DynamicEQBand], frequency_hz: float, gain_db: float,
              q: float, reason: str):
    if abs(gain_db) < 0.1:
        return
    bands.append(
        DynamicEQBand(
            frequency_hz=float(frequency_hz),
            gain_db=float(max(-6.0, min(6.0, gain_db))),
            q=float(max(0.2, min(8.0, q))),
            reason=reason,
        )
    )


def _merge_nearby_bands(bands: list[DynamicEQBand]) -> tuple[DynamicEQBand, ...]:
    merged: list[DynamicEQBand] = []
    for band in sorted(bands, key=lambda item: item.frequency_hz):
        if merged and abs(np.log2(max(band.frequency_hz, 1.0) / merged[-1].frequency_hz)) < 0.12:
            previous = merged.pop()
            gain = max(-6.0, min(6.0, previous.gain_db + band.gain_db))
            reason = previous.reason if previous.reason == band.reason else f"{previous.reason}; {band.reason}"
            merged.append(
                DynamicEQBand(
                    frequency_hz=(previous.frequency_hz + band.frequency_hz) / 2.0,
                    gain_db=gain,
                    q=max(previous.q, band.q),
                    reason=reason,
                )
            )
        else:
            merged.append(band)
    return tuple(merged[:6])


def suggest_dynamic_eq_curve(audio: np.ndarray, sr: int,
                             stem_name: str = "Stem") -> DynamicEQSuggestion:
    """Build a deterministic stem-aware EQ recommendation from spectral balance."""
    mono = _mono_float(audio)
    if mono.size == 0 or sr <= 0:
        return DynamicEQSuggestion(stem_name=stem_name, stem_role=_stem_role(stem_name))

    max_samples = min(mono.size, int(sr * 60))
    mono = mono[:max_samples]
    rms = float(np.sqrt(np.mean(np.square(mono))) + 1e-12)
    rms_db = linear_to_db(rms)

    if mono.size < 16 or rms < 1e-8:
        return DynamicEQSuggestion(
            stem_name=stem_name,
            stem_role=_stem_role(stem_name),
            rms_db=rms_db,
        )

    window = np.hanning(mono.size).astype(np.float32)
    spectrum = np.fft.rfft(mono * window)
    power = np.square(np.abs(spectrum))
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sr)
    audible = (freqs >= 20.0) & (freqs <= min(20000.0, sr / 2.0))
    total = float(np.sum(power[audible]) + 1e-12)

    low_ratio = _band_ratio(freqs, power, 20.0, 120.0, total)
    low_mid_ratio = _band_ratio(freqs, power, 120.0, 500.0, total)
    mid_ratio = _band_ratio(freqs, power, 500.0, 4000.0, total)
    presence_ratio = _band_ratio(freqs, power, 4000.0, 9000.0, total)
    high_ratio = _band_ratio(freqs, power, 9000.0, min(20000.0, sr / 2.0), total)
    centroid = float(np.sum(freqs[audible] * power[audible]) / total)

    role = _stem_role(stem_name)
    bands: list[DynamicEQBand] = []

    if low_ratio > 0.52 and role not in {"bass", "drums"}:
        _add_band(bands, 95.0, -2.0, 0.7, "Tames low-end buildup")
    if low_mid_ratio > 0.34:
        _add_band(bands, 280.0, -1.4, 1.0, "Clears boxy low-mids")
    if mid_ratio < 0.12 and role not in {"bass"}:
        _add_band(bands, 2200.0, 1.1, 1.2, "Adds midrange definition")
    if presence_ratio < 0.035 and role in {"vocal", "drums", "instrument"}:
        _add_band(bands, 5200.0, 1.2, 1.3, "Restores presence")
    if high_ratio > 0.28:
        _add_band(bands, 9500.0, -1.2, 1.0, "Softens brittle highs")

    if role == "vocal":
        if low_ratio > 0.12:
            _add_band(bands, 120.0, -1.8, 0.8, "Vocal cleanup below the melody range")
        if low_mid_ratio > 0.18:
            _add_band(bands, 350.0, -1.2, 1.2, "Reduces vocal mud")
        _add_band(bands, 3200.0, 1.5, 1.4, "Improves vocal intelligibility")
        if high_ratio < 0.08:
            _add_band(bands, 12000.0, 0.8, 0.9, "Adds gentle vocal air")
    elif role == "bass":
        if low_ratio < 0.55:
            _add_band(bands, 75.0, 1.4, 0.8, "Reinforces bass fundamental")
        if low_mid_ratio > 0.28:
            _add_band(bands, 260.0, -1.4, 1.1, "Controls bass mud")
        _add_band(bands, 7000.0, -1.0, 0.9, "Leaves high-end space for vocals")
    elif role == "drums":
        _add_band(bands, 65.0, 1.0, 0.8, "Adds kick weight")
        if low_mid_ratio > 0.24:
            _add_band(bands, 240.0, -1.1, 1.1, "Reduces drum boxiness")
        _add_band(bands, 5000.0, 1.2, 1.2, "Highlights drum attack")
    elif role == "instrument":
        if low_mid_ratio > 0.22:
            _add_band(bands, 260.0, -1.0, 1.0, "Keeps instruments out of vocal low-mids")
        _add_band(bands, 2400.0, 0.8, 1.0, "Adds instrument articulation")

    return DynamicEQSuggestion(
        stem_name=stem_name,
        stem_role=role,
        bands=_merge_nearby_bands(bands),
        rms_db=rms_db,
        spectral_centroid_hz=centroid,
        low_ratio=low_ratio,
        low_mid_ratio=low_mid_ratio,
        mid_ratio=mid_ratio,
        presence_ratio=presence_ratio,
        high_ratio=high_ratio,
    )


def apply_dynamic_eq(audio: np.ndarray, sr: int,
                     bands: tuple[DynamicEQBand, ...] | list[DynamicEQBand],
                     strength: float = 1.0) -> np.ndarray:
    """Apply suggested EQ bands using a smooth FFT gain curve."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.size == 0 or sr <= 0 or not bands:
        return np.array(arr, copy=True)

    strength = float(max(0.0, min(1.0, strength)))
    if strength <= 0.0:
        return np.array(arr, copy=True)

    was_mono = arr.ndim == 1
    work = arr[:, None] if was_mono else arr
    output = np.zeros_like(work, dtype=np.float32)
    freqs = np.fft.rfftfreq(work.shape[0], 1.0 / sr)
    safe_freqs = np.maximum(freqs, 20.0)
    gain_db_curve = np.zeros_like(freqs, dtype=np.float64)

    for band in bands:
        frequency = max(20.0, min(float(band.frequency_hz), sr / 2.0))
        bandwidth_octaves = max(0.12, min(2.5, 1.0 / max(float(band.q), 0.2)))
        distance = np.log2(safe_freqs / frequency)
        influence = np.exp(-0.5 * np.square(distance / bandwidth_octaves))
        gain_db_curve += float(band.gain_db) * strength * influence

    gain_db_curve = np.clip(gain_db_curve, -9.0, 6.0)
    linear_curve = db_to_linear(gain_db_curve)

    for channel in range(work.shape[1]):
        spectrum = np.fft.rfft(work[:, channel])
        processed = np.fft.irfft(spectrum * linear_curve, n=work.shape[0])
        output[:, channel] = processed.astype(np.float32)

    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 1.0:
        output /= peak

    return output[:, 0] if was_mono else output


def suggest_and_apply_dynamic_eq(audio: np.ndarray, sr: int,
                                 stem_name: str = "Stem",
                                 strength: float = 0.75) -> tuple[np.ndarray, DynamicEQSuggestion]:
    """Suggest and apply a stem-aware EQ curve in one pass."""
    suggestion = suggest_dynamic_eq_curve(audio, sr, stem_name)
    return apply_dynamic_eq(audio, sr, suggestion.bands, strength), suggestion


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
