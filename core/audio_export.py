"""
Slunder Studio — Audio Export
Multi-format audio export: WAV, FLAC, MP3, OGG.
Uses soundfile for lossless, ffmpeg subprocess for lossy.
"""
import hashlib
import os
import re
import shutil
import subprocess
from typing import Optional
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np

from core.audio_buffers import resample_audio
from core.provenance import read_provenance_sidecar, write_provenance_sidecar


# Every delivery format the app can produce, and what writes it.
LOSSLESS_FORMATS = ("wav", "flac")
LOSSY_FORMATS = ("mp3", "ogg", "opus")
DELIVERY_FORMATS = LOSSLESS_FORMATS + LOSSY_FORMATS

# ffmpeg encoder required for each lossy format.
FORMAT_ENCODERS = {
    "mp3": "libmp3lame",
    "ogg": "libvorbis",
    "opus": "libopus",
}

# Field -> tag name per container standard. ffmpeg maps its generic -metadata
# keys onto ID3v2.4 for MP3 and Vorbis comments for Ogg/Opus/FLAC, so one
# canonical key set covers all of them; the mapping is recorded in provenance
# so an export can be audited against the standard it claims.
METADATA_STANDARDS = {
    "mp3": "ID3v2.4",
    "ogg": "Vorbis comment (RFC 7845 style)",
    "opus": "Vorbis comment (RFC 7845)",
    "flac": "Vorbis comment",
    "wav": "RIFF INFO / BWF",
}


@dataclass
class ExportSettings:
    """Export configuration."""
    format: str = "wav"  # wav, flac, mp3, ogg, opus
    sample_rate: int = 48000
    bit_depth: int = 16  # 16, 24, 32 (wav only)
    mp3_bitrate: int = 320  # 128, 192, 256, 320
    ogg_quality: int = 8  # 0-10
    opus_bitrate: int = 192  # kbps
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
    # Standards-mapped production metadata.
    bpm: float = 0.0
    musical_key: str = ""
    language: str = ""
    lyrics: str = ""
    rights: str = ""
    revision: str = ""
    track_number: str = ""
    isrc: str = ""
    comment: str = ""

    def metadata_tags(self) -> dict:
        """Canonical tag keys for this delivery, empty values dropped."""
        tags = {
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "date": self.year,
            "genre": self.genre,
            "language": self.language,
            "lyrics": self.lyrics,
            "copyright": self.rights,
            "comment": self.comment,
            "track": self.track_number,
            "TSRC": self.isrc,
            "version": self.revision,
        }
        if self.bpm and self.bpm > 0:
            tags["TBPM"] = f"{self.bpm:g}"
        if self.musical_key:
            tags["TKEY"] = self.musical_key
        return {k: str(v) for k, v in tags.items() if str(v).strip()}


@dataclass(frozen=True)
class CodecAvailability:
    """Whether a delivery format can actually be written right now."""
    format: str
    available: bool
    writer: str
    detail: str = ""


def probe_codecs() -> dict[str, CodecAvailability]:
    """Report which delivery formats this installation can write, and why not."""
    results: dict[str, CodecAvailability] = {}
    try:
        import soundfile as sf

        writable = {fmt.lower() for fmt in sf.available_formats()}
    except Exception as exc:
        writable = set()
        sf_error = f"soundfile is unavailable: {exc}"
    else:
        sf_error = ""

    for fmt in LOSSLESS_FORMATS:
        ok = fmt in writable
        results[fmt] = CodecAvailability(
            format=fmt,
            available=ok,
            writer="soundfile",
            detail="" if ok else (sf_error or f"soundfile cannot write {fmt.upper()}"),
        )

    ffmpeg = _find_ffmpeg()
    encoders = _ffmpeg_encoders(ffmpeg) if ffmpeg else set()
    for fmt in LOSSY_FORMATS:
        encoder = FORMAT_ENCODERS[fmt]
        if not ffmpeg:
            detail = "ffmpeg was not found on PATH."
        elif encoder not in encoders:
            detail = f"ffmpeg has no {encoder} encoder."
        else:
            detail = ""
        results[fmt] = CodecAvailability(
            format=fmt,
            available=not detail,
            writer="ffmpeg",
            detail=detail,
        )
    return results


def _ffmpeg_encoders(ffmpeg: Optional[str]) -> set:
    if not ffmpeg:
        return set()
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return set()
    if result.returncode != 0:
        return set()
    names = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(("A", "V", "S")):
            names.add(parts[1])
    return names


def require_codec(fmt: str) -> CodecAvailability:
    """Raise a clear error when the requested format cannot be written."""
    fmt = fmt.lower()
    if fmt not in DELIVERY_FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")
    availability = probe_codecs()[fmt]
    if not availability.available:
        raise RuntimeError(
            f"{fmt.upper()} export is unavailable. {availability.detail}"
        )
    return availability


def deterministic_filename(base: str, *, fmt: str, revision: str = "",
                           variant: str = "") -> str:
    """Build a stable, filesystem-safe delivery filename.

    The same inputs always produce the same name, so re-running an export
    overwrites its own artifact rather than accumulating copies.
    """
    def _slug(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
        return re.sub(r"-{2,}", "-", cleaned).strip("-._")

    parts = [p for p in (_slug(base) or "export", _slug(variant), _slug(revision)) if p]
    return f"{'-'.join(parts)}.{fmt.lower()}"


def _verify_written_file(path: str) -> dict:
    """Reopen and hash a written delivery. Raises if it cannot be read back."""
    import soundfile as sf

    if not os.path.isfile(path):
        raise RuntimeError(f"Export did not produce a file: {path}")
    size = os.path.getsize(path)
    if size <= 0:
        raise RuntimeError(f"Export produced an empty file: {path}")
    try:
        info = sf.info(path)
        frames, channels, samplerate = info.frames, info.channels, info.samplerate
        readable = True
        read_error = ""
    except Exception as exc:
        # Opus and some Ogg variants are not always readable by libsndfile;
        # say so rather than claiming a verification that did not happen.
        frames = channels = samplerate = 0
        readable = False
        read_error = f"{type(exc).__name__}: {exc}"

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return {
        "path": path,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "reopened": readable,
        "reopen_error": read_error,
        "frames": frames,
        "channels": channels,
        "sample_rate": samplerate,
        "duration_sec": round(frames / samplerate, 6) if samplerate else 0.0,
    }


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


def _source_model_license_metadata(source_path: str) -> dict:
    provenance = read_provenance_sidecar(source_path)
    model = provenance.get("model") or {}
    if not model:
        return {}
    keys = (
        "id",
        "name",
        "license",
        "license_url",
        "commercial_use",
        "commercial_use_label",
        "commercial_use_note",
        "license_warning",
        "requires_export_warning",
        "metadata_status",
        "metadata_error",
        "gated",
        "access",
    )
    return {key: model.get(key) for key in keys if key in model}


def get_export_license_warnings(source_path: str) -> list[str]:
    metadata = _source_model_license_metadata(source_path)
    if not metadata:
        return []
    warning = metadata.get("license_warning") or ""
    indeterminate = metadata.get("metadata_status") == "indeterminate"
    unknown_commercial_use = metadata.get("commercial_use") == "unknown"
    if (
        not metadata.get("requires_export_warning")
        and not warning
        and not indeterminate
        and not unknown_commercial_use
    ):
        return []
    model_name = metadata.get("name") or metadata.get("id") or "Source model"
    if warning:
        return [f"{model_name}: {warning}"]
    return [f"{model_name}: Review model license before release."]


def export_audio(
    source_path: str,
    output_path: str,
    settings: ExportSettings = None,
    *,
    module: str = "export",
    operation: str = "export_audio",
    source_asset_ids: Optional[list[str]] = None,
    source_paths: Optional[list[str]] = None,
    provenance_extra: Optional[dict] = None,
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
        audio = resample_audio(audio, sr, settings.sample_rate)
        sr = settings.sample_rate

    # Apply processing
    if settings.fade_in_ms > 0 or settings.fade_out_ms > 0:
        audio = apply_fade(audio, sr, settings.fade_in_ms, settings.fade_out_ms)

    if settings.normalize:
        audio = normalize_audio(audio, settings.normalize_target_db)

    availability = require_codec(settings.format)
    tags = settings.metadata_tags()

    # Export based on format
    if settings.format in LOSSLESS_FORMATS:
        subtype_map = {
            (16, "wav"): "PCM_16",
            (24, "wav"): "PCM_24",
            (32, "wav"): "FLOAT",
            (16, "flac"): "PCM_16",
            (24, "flac"): "PCM_24",
        }
        subtype = subtype_map.get((settings.bit_depth, settings.format), "PCM_16")
        sf.write(output_path, audio, sr, subtype=subtype)
        _write_lossless_tags(output_path, settings.format, tags)

    else:
        # Write temp WAV, then convert via ffmpeg
        ffmpeg = _find_ffmpeg()
        temp_wav = output_path + ".tmp.wav"
        sf.write(temp_wav, audio, sr, subtype="PCM_16")

        try:
            cmd = [ffmpeg, "-y", "-i", temp_wav]
            for key, value in tags.items():
                cmd += ["-metadata", f"{key}={_sanitize_meta(value)}"]

            if settings.format == "mp3":
                cmd += ["-codec:a", "libmp3lame", "-b:a", f"{settings.mp3_bitrate}k"]
            elif settings.format == "ogg":
                cmd += ["-codec:a", "libvorbis", "-q:a", str(settings.ogg_quality)]
            else:  # opus
                cmd += ["-codec:a", "libopus", "-b:a", f"{settings.opus_bitrate}k"]

            cmd.append(output_path)

            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

    verification = _verify_written_file(output_path)

    extra = dict(provenance_extra or {})
    source_model_license = _source_model_license_metadata(source_path)
    if source_model_license:
        extra["source_model_license"] = source_model_license
    license_warnings = get_export_license_warnings(source_path)
    if license_warnings:
        extra["license_warnings"] = license_warnings
    extra["delivery"] = {
        "writer": availability.writer,
        "metadata_standard": METADATA_STANDARDS.get(settings.format, "unknown"),
        "tags": tags,
        "verification": verification,
    }

    write_provenance_sidecar(
        output_path,
        module=module,
        operation=operation,
        parameters={"settings": asdict(settings)},
        source_asset_ids=source_asset_ids or [],
        source_paths=source_paths if source_paths is not None else [source_path],
        export_format=settings.format,
        output_kind="export",
        extra=extra,
    )
    return output_path


def _sanitize_meta(value: str) -> str:
    return (
        str(value)
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\\", "\\\\")
        .replace(";", ",")
    )


def _write_lossless_tags(path: str, fmt: str, tags: dict):
    """Attach metadata to a FLAC/WAV written by soundfile, where supported."""
    if not tags:
        return
    try:
        import soundfile as sf

        with sf.SoundFile(path, mode="r+") as handle:
            for key, value in tags.items():
                field = {
                    "date": "date", "track": "tracknumber", "copyright": "copyright",
                }.get(key, key)
                try:
                    setattr(handle, field, _sanitize_meta(value))
                except Exception:
                    # libsndfile only exposes a fixed set of string fields; the
                    # rest still live in the provenance sidecar.
                    continue
    except Exception:
        # Never fail an otherwise good export because a tag could not be set.
        pass


def export_from_numpy(
    audio: np.ndarray,
    sr: int,
    output_path: str,
    settings: ExportSettings = None,
    *,
    module: str = "export",
    operation: str = "export_from_numpy",
    source_asset_ids: Optional[list[str]] = None,
    source_paths: Optional[list[str]] = None,
    provenance_extra: Optional[dict] = None,
) -> str:
    """Export a numpy audio array directly to file."""
    import soundfile as sf

    if settings is None:
        settings = ExportSettings()

    # Write temp WAV then use main export
    temp_path = output_path + ".tmp_src.wav"
    sf.write(temp_path, audio, sr, subtype="FLOAT")

    try:
        extra = {"input_sample_rate": sr, "input_shape": list(audio.shape)}
        if provenance_extra:
            extra.update(provenance_extra)
        return export_audio(
            temp_path,
            output_path,
            settings,
            module=module,
            operation=operation,
            source_asset_ids=source_asset_ids or [],
            source_paths=source_paths or [],
            provenance_extra=extra,
        )
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
    *,
    source_asset_ids: Optional[list[str]] = None,
) -> str:
    """Trim audio to selection with optional fades."""
    import soundfile as sf

    if end_sec <= start_sec:
        raise ValueError(f"Invalid trim region: start {start_sec}s must be before end {end_sec}s")

    audio, sr = sf.read(source_path, dtype="float32")
    duration = len(audio) / sr
    start_sec = max(0.0, min(start_sec, duration))
    end_sec = max(start_sec, min(end_sec, duration))
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    trimmed = audio[start_sample:end_sample]

    if fade_in_ms > 0 or fade_out_ms > 0:
        trimmed = apply_fade(trimmed, sr, fade_in_ms, fade_out_ms)

    sf.write(output_path, trimmed, sr, subtype="PCM_16")
    write_provenance_sidecar(
        output_path,
        module="export",
        operation="trim_audio",
        parameters={
            "start_sec": start_sec,
            "end_sec": end_sec,
            "fade_in_ms": fade_in_ms,
            "fade_out_ms": fade_out_ms,
            "sample_rate": sr,
        },
        source_asset_ids=source_asset_ids or [],
        source_paths=[source_path],
        export_format=Path(output_path).suffix.lstrip(".").lower() or "wav",
        output_kind="export",
    )
    return output_path
