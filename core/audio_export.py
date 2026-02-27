"""
Slunder Studio v0.0.2 — Audio Export
Multi-format audio export: WAV, FLAC, MP3, OGG.
Uses soundfile for lossless, ffmpeg subprocess for lossy.
"""
import os
import shutil
import subprocess
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass
class ExportSettings:
    """Export configuration."""
    format: str = "wav"  # wav, flac, mp3, ogg
    sample_rate: int = 48000
    bit_depth: int = 16  # 16, 24, 32 (wav only)
    mp3_bitrate: int = 320  # 128, 192, 256, 320
    ogg_quality: int = 8  # 0-10
    normalize: bool = False
    normalize_target_db: float = -1.0  # peak normalization target
    fade_in_ms: int = 0
    fade_out_ms: int = 0
    # Metadata
    title: str = ""
    artist: str = "Slunder"
    album: str = ""
    year: str = ""
    genre: str = ""


def _find_ffmpeg() -> Optional[str]:
    """Find ffmpeg in PATH or common locations."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    # Check common Windows locations
    for p in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.path.expanduser("~\\ffmpeg\\bin\\ffmpeg.exe"),
    ]:
        if os.path.exists(p):
            return p
    return None


def normalize_audio(audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """Peak normalize audio to target dB."""
    peak = np.abs(audio).max()
    if peak < 1e-8:
        return audio
    target_linear = 10 ** (target_db / 20.0)
    return audio * (target_linear / peak)


def apply_fade(audio: np.ndarray, sr: int, fade_in_ms: int = 0, fade_out_ms: int = 0) -> np.ndarray:
    """Apply fade in/out to audio."""
    result = audio.copy()
    if fade_in_ms > 0:
        n_samples = int(sr * fade_in_ms / 1000)
        n_samples = min(n_samples, len(result))
        fade = np.linspace(0, 1, n_samples)
        if result.ndim == 2:
            result[:n_samples] *= fade[:, np.newaxis]
        else:
            result[:n_samples] *= fade

    if fade_out_ms > 0:
        n_samples = int(sr * fade_out_ms / 1000)
        n_samples = min(n_samples, len(result))
        fade = np.linspace(1, 0, n_samples)
        if result.ndim == 2:
            result[-n_samples:] *= fade[:, np.newaxis]
        else:
            result[-n_samples:] *= fade

    return result


def export_audio(
    source_path: str,
    output_path: str,
    settings: ExportSettings = None,
) -> str:
    """
    Export audio file to target format with optional processing.
    Returns final output path.
    """
    import soundfile as sf

    if settings is None:
        settings = ExportSettings()

    output_path = str(output_path)
    source_path = str(source_path)

    # Ensure correct extension
    ext = f".{settings.format}"
    if not output_path.lower().endswith(ext):
        output_path = os.path.splitext(output_path)[0] + ext

    # Load source
    audio, sr = sf.read(source_path, dtype="float32")

    # Resample if needed
    if sr != settings.sample_rate:
        try:
            import librosa
            if audio.ndim == 2:
                channels = []
                for ch in range(audio.shape[1]):
                    channels.append(librosa.resample(audio[:, ch], orig_sr=sr, target_sr=settings.sample_rate))
                audio = np.column_stack(channels)
            else:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=settings.sample_rate)
            sr = settings.sample_rate
        except ImportError:
            pass  # Keep original sample rate

    # Apply processing
    if settings.fade_in_ms > 0 or settings.fade_out_ms > 0:
        audio = apply_fade(audio, sr, settings.fade_in_ms, settings.fade_out_ms)

    if settings.normalize:
        audio = normalize_audio(audio, settings.normalize_target_db)

    # Export based on format
    if settings.format in ("wav", "flac"):
        subtype_map = {
            (16, "wav"): "PCM_16",
            (24, "wav"): "PCM_24",
            (32, "wav"): "FLOAT",
            (16, "flac"): "PCM_16",
            (24, "flac"): "PCM_24",
        }
        subtype = subtype_map.get((settings.bit_depth, settings.format), "PCM_16")
        sf.write(output_path, audio, sr, subtype=subtype)

    elif settings.format in ("mp3", "ogg"):
        # Write temp WAV, then convert via ffmpeg
        ffmpeg = _find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg for MP3/OGG export.\n"
                "Download from: https://ffmpeg.org/download.html"
            )

        temp_wav = output_path + ".tmp.wav"
        sf.write(temp_wav, audio, sr, subtype="PCM_16")

        try:
            cmd = [ffmpeg, "-y", "-i", temp_wav]

            # Add metadata
            if settings.title:
                cmd += ["-metadata", f"title={settings.title}"]
            if settings.artist:
                cmd += ["-metadata", f"artist={settings.artist}"]
            if settings.album:
                cmd += ["-metadata", f"album={settings.album}"]
            if settings.year:
                cmd += ["-metadata", f"date={settings.year}"]
            if settings.genre:
                cmd += ["-metadata", f"genre={settings.genre}"]

            if settings.format == "mp3":
                cmd += ["-codec:a", "libmp3lame", "-b:a", f"{settings.mp3_bitrate}k"]
            else:  # ogg
                cmd += ["-codec:a", "libvorbis", "-q:a", str(settings.ogg_quality)]

            cmd.append(output_path)

            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
    else:
        raise ValueError(f"Unsupported format: {settings.format}")

    return output_path


def export_from_numpy(
    audio: np.ndarray,
    sr: int,
    output_path: str,
    settings: ExportSettings = None,
) -> str:
    """Export a numpy audio array directly to file."""
    import soundfile as sf

    if settings is None:
        settings = ExportSettings()

    # Write temp WAV then use main export
    temp_path = output_path + ".tmp_src.wav"
    sf.write(temp_path, audio, sr, subtype="FLOAT")

    try:
        return export_audio(temp_path, output_path, settings)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def trim_audio(
    source_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
    fade_in_ms: int = 0,
    fade_out_ms: int = 0,
) -> str:
    """Trim audio to selection with optional fades."""
    import soundfile as sf

    audio, sr = sf.read(source_path, dtype="float32")
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    trimmed = audio[start_sample:end_sample]

    if fade_in_ms > 0 or fade_out_ms > 0:
        trimmed = apply_fade(trimmed, sr, fade_in_ms, fade_out_ms)

    sf.write(output_path, trimmed, sr, subtype="PCM_16")
    return output_path
