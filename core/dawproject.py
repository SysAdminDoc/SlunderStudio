"""
Slunder Studio — DAWproject Export
Generates cross-DAW .dawproject archives (ZIP containing project.xml,
metadata.xml, and referenced media files).
"""
import os
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from core.settings import APP_VERSION


DAW_NS = "http://bitwig.com/dawproject"
META_NS = "http://bitwig.com/dawproject"

REQUIRED_ARCHIVE_ENTRIES = {"project.xml", "metadata.xml"}
DAWPROJECT_AUDIO_ASSET_TYPES = frozenset({"audio", "stems", "sfx", "export"})
DEFAULT_MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024


class DAWProjectSecurityError(ValueError):
    """Raised when a DAWproject archive cannot be handled safely."""


@dataclass
class DAWTrack:
    """A single track in the DAWproject."""
    name: str = ""
    media_file: str = ""
    volume: float = 1.0
    pan: float = 0.0
    muted: bool = False
    soloed: bool = False
    color: str = "#89b4fa"
    role: str = "music"


@dataclass
class DAWProjectSpec:
    """Specification for a .dawproject export."""
    title: str = "Untitled"
    artist: str = "Slunder"
    tempo: float = 120.0
    time_signature: str = "4/4"
    sample_rate: int = 48000
    tracks: list[DAWTrack] = field(default_factory=list)


@dataclass
class DAWProjectValidation:
    """Result of validating a .dawproject archive."""
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    entries: list[str] = field(default_factory=list)
    media_refs: list[str] = field(default_factory=list)


def _media_archive_names(spec: DAWProjectSpec) -> list[str]:
    """Return unique, portable archive names in track order."""
    names: list[str] = []
    used: set[str] = set()
    for index, track in enumerate(spec.tracks, 1):
        if not track.media_file:
            names.append("")
            continue
        raw_name = Path(track.media_file).name.strip(" .")
        if not raw_name or raw_name in {".", ".."}:
            raw_name = f"track-{index}.wav"
        stem = Path(raw_name).stem or f"track-{index}"
        suffix = Path(raw_name).suffix
        candidate = raw_name
        serial = 2
        while candidate.casefold() in used:
            candidate = f"{stem}-{serial}{suffix}"
            serial += 1
        used.add(candidate.casefold())
        names.append(candidate)
    return names


def _build_project_xml(spec: DAWProjectSpec) -> str:
    media_names = _media_archive_names(spec)
    root = ET.Element("Project", xmlns=DAW_NS, version="1.0")
    root.set("creator", f"SlunderStudio/{APP_VERSION}")

    transport = ET.SubElement(root, "Transport")
    tempo_el = ET.SubElement(transport, "Tempo")
    tempo_el.set("value", str(spec.tempo))
    ts_el = ET.SubElement(transport, "TimeSignature")
    parts = spec.time_signature.split("/")
    ts_el.set("numerator", parts[0] if len(parts) == 2 else "4")
    ts_el.set("denominator", parts[1] if len(parts) == 2 else "4")

    structure = ET.SubElement(root, "Structure")
    for i, track in enumerate(spec.tracks):
        track_el = ET.SubElement(structure, "Track")
        track_el.set("id", f"track-{i}")
        track_el.set("name", track.name or f"Track {i + 1}")
        track_el.set("color", track.color)

        channel = ET.SubElement(track_el, "Channel")
        vol_el = ET.SubElement(channel, "Volume")
        vol_el.set("value", str(round(track.volume, 4)))
        pan_el = ET.SubElement(channel, "Pan")
        pan_el.set("value", str(round(track.pan, 4)))
        if track.muted:
            mute_el = ET.SubElement(channel, "Mute")
            mute_el.set("value", "true")

    arrangement = ET.SubElement(root, "Arrangement")
    for i, (track, media_name) in enumerate(zip(spec.tracks, media_names)):
        if not track.media_file:
            continue
        lane = ET.SubElement(arrangement, "Lane")
        lane.set("trackRef", f"track-{i}")
        clip = ET.SubElement(lane, "Clip")
        clip.set("time", "0.0")
        audio_ref = ET.SubElement(clip, "Audio")
        audio_ref.set("file", f"media/{media_name}")

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        root, encoding="unicode"
    )


def _build_metadata_xml(spec: DAWProjectSpec) -> str:
    root = ET.Element("MetaData", xmlns=META_NS, version="1.0")

    title_el = ET.SubElement(root, "Title")
    title_el.text = spec.title
    artist_el = ET.SubElement(root, "Artist")
    artist_el.text = spec.artist

    app_el = ET.SubElement(root, "Application")
    app_el.set("name", "SlunderStudio")
    app_el.set("version", APP_VERSION)

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        root, encoding="unicode"
    )


def export_dawproject(
    spec: DAWProjectSpec,
    output_path: str,
) -> str:
    """
    Build a .dawproject archive from a project spec.
    Returns the path to the written archive.
    """
    output_path = str(output_path)
    if not output_path.lower().endswith(".dawproject"):
        output_path += ".dawproject"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.xml", _build_project_xml(spec))
        zf.writestr("metadata.xml", _build_metadata_xml(spec))

        media_names = _media_archive_names(spec)
        for track, media_name in zip(spec.tracks, media_names):
            if not track.media_file or not os.path.isfile(track.media_file):
                continue
            zf.write(track.media_file, f"media/{media_name}")

    return output_path


def _archive_member_error(name: str) -> str:
    """Return a reason when a ZIP member is not a safe relative path."""
    if not isinstance(name, str) or not name:
        return "empty archive member name"
    if "\\" in name:
        return "backslash is not allowed in archive member names"
    if name.startswith("/") or (len(name) >= 2 and name[1] == ":"):
        return "absolute archive member path"

    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts[:-1]):
        return "archive member contains an unsafe path component"
    if parts[-1] in {".", ".."}:
        return "archive member contains an unsafe path component"
    return ""


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    """Detect Unix symlink entries without extracting them first."""
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_IFMT(mode) == stat.S_IFLNK


def _validate_archive_members(
    infos: list[zipfile.ZipInfo],
    result: DAWProjectValidation,
) -> None:
    seen: set[str] = set()
    for info in infos:
        name = info.filename
        reason = _archive_member_error(name)
        if reason:
            result.valid = False
            result.errors.append(f"Unsafe archive entry {name!r}: {reason}")
        if name in seen:
            result.valid = False
            result.errors.append(f"Duplicate archive entry: {name}")
        seen.add(name)
        if _is_zip_symlink(info):
            result.valid = False
            result.errors.append(f"Symbolic-link archive entry is not allowed: {name}")


def extract_dawproject(
    archive_path: str,
    destination_dir: str | Path,
    *,
    max_total_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
) -> Path:
    """Safely extract a validated DAWproject into ``destination_dir``.

    ZIP extraction is intentionally explicit because ``ZipFile.extractall``
    historically makes it easy to overlook ``../``, absolute-path, symlink,
    and decompression-bomb inputs. Existing files are never overwritten.
    """
    if max_total_bytes < 0:
        raise ValueError("max_total_bytes must be non-negative")

    validation = validate_dawproject(archive_path)
    if not validation.valid:
        detail = "; ".join(validation.errors[:4])
        if len(validation.errors) > 4:
            detail += "; ..."
        raise DAWProjectSecurityError(f"DAWproject archive rejected: {detail}")

    destination = Path(destination_dir).resolve(strict=False)
    destination.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    total_bytes = 0
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            for info in infos:
                reason = _archive_member_error(info.filename)
                if reason or _is_zip_symlink(info):
                    raise DAWProjectSecurityError(
                        f"Unsafe archive entry {info.filename!r}"
                    )
                declared_bytes = max(0, int(info.file_size))
                if total_bytes + declared_bytes > max_total_bytes:
                    raise DAWProjectSecurityError(
                        "DAWproject extracted size exceeds the configured limit"
                    )

                target = destination / info.filename
                resolved_target = target.resolve(strict=False)
                if not resolved_target.is_relative_to(destination):
                    raise DAWProjectSecurityError(
                        f"Archive entry resolves outside destination: {info.filename}"
                    )

                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.parent.resolve(strict=False).is_relative_to(destination):
                    raise DAWProjectSecurityError(
                        f"Archive entry parent resolves outside destination: {info.filename}"
                    )
                with archive.open(info, "r") as source, target.open("xb") as output:
                    created.append(target)
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        if total_bytes > max_total_bytes:
                            raise DAWProjectSecurityError(
                                "DAWproject extracted size exceeds the configured limit"
                            )
                        output.write(chunk)
    except Exception:
        for path in reversed(created):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return destination


def spec_from_project(project) -> DAWProjectSpec:
    """Build a DAWproject spec from the existing project asset contract.

    Non-audio assets remain in the Slunder project and are intentionally not
    represented as audio clips. Missing files are omitted so an export can
    still be validated and the caller can report the omission to the user.
    """
    time_signature = getattr(project, "time_signature", (4, 4))
    if not isinstance(time_signature, (tuple, list)) or len(time_signature) != 2:
        time_signature = (4, 4)
    tracks = []
    for asset in getattr(project, "assets", ()) or ():
        asset_type = str(getattr(asset, "asset_type", "") or "audio")
        media_file = str(getattr(asset, "file_path", "") or "")
        if (
            asset_type not in DAWPROJECT_AUDIO_ASSET_TYPES
            or not media_file
            or not os.path.isfile(media_file)
        ):
            continue
        tracks.append(
            DAWTrack(
                name=str(getattr(asset, "name", "") or Path(media_file).stem),
                media_file=media_file,
                role=asset_type,
            )
        )
    return DAWProjectSpec(
        title=str(getattr(project, "name", "Untitled") or "Untitled"),
        artist="Slunder",
        tempo=float(getattr(project, "tempo", 120.0) or 120.0),
        time_signature=f"{time_signature[0]}/{time_signature[1]}",
        tracks=tracks,
    )


def validate_dawproject(archive_path: str) -> DAWProjectValidation:
    """
    Validate a .dawproject archive for structural correctness.
    Checks: ZIP integrity, required files, XML well-formedness,
    required elements, and media reference resolution.
    """
    result = DAWProjectValidation()

    if not os.path.isfile(archive_path):
        result.valid = False
        result.errors.append(f"Archive not found: {archive_path}")
        return result

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                result.valid = False
                result.errors.append(f"Corrupt ZIP entry: {bad}")
                return result

            infos = zf.infolist()
            result.entries = [info.filename for info in infos]
            _validate_archive_members(infos, result)
            if not result.valid:
                return result

            for required in REQUIRED_ARCHIVE_ENTRIES:
                if required not in result.entries:
                    result.valid = False
                    result.errors.append(f"Missing required entry: {required}")

            if not result.valid:
                return result

            project_xml = zf.read("project.xml").decode("utf-8")
            metadata_xml = zf.read("metadata.xml").decode("utf-8")

            _validate_project_xml(project_xml, result)
            _validate_metadata_xml(metadata_xml, result)

            media_files = {e for e in result.entries if e.startswith("media/")}
            for ref in result.media_refs:
                if _archive_member_error(ref) or not ref.startswith("media/"):
                    result.valid = False
                    result.errors.append(f"Unsafe media reference in project.xml: {ref}")
                    continue
                if ref not in media_files:
                    result.valid = False
                    result.errors.append(f"Media reference not found in archive: {ref}")

    except zipfile.BadZipFile as exc:
        result.valid = False
        result.errors.append(f"Invalid ZIP file: {exc}")
    except Exception as exc:
        result.valid = False
        result.errors.append(f"Validation error: {exc}")

    return result


def _validate_project_xml(xml_str: str, result: DAWProjectValidation) -> None:
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as exc:
        result.valid = False
        result.errors.append(f"project.xml is not well-formed XML: {exc}")
        return

    tag = _strip_ns(root.tag)
    if tag != "Project":
        result.valid = False
        result.errors.append(f"project.xml root element is '{tag}', expected 'Project'")

    if not root.get("version"):
        result.warnings.append("project.xml missing version attribute")

    transport = root.find(f".//{{{DAW_NS}}}Transport")
    if transport is None:
        transport = root.find(".//Transport")
    if transport is None:
        result.warnings.append("project.xml missing Transport element")

    structure = root.find(f".//{{{DAW_NS}}}Structure")
    if structure is None:
        structure = root.find(".//Structure")
    if structure is None:
        result.warnings.append("project.xml missing Structure element")

    for audio_el in root.iter():
        if _strip_ns(audio_el.tag) == "Audio":
            file_ref = audio_el.get("file")
            if file_ref:
                result.media_refs.append(file_ref)


def _validate_metadata_xml(xml_str: str, result: DAWProjectValidation) -> None:
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as exc:
        result.valid = False
        result.errors.append(f"metadata.xml is not well-formed XML: {exc}")
        return

    tag = _strip_ns(root.tag)
    if tag != "MetaData":
        result.valid = False
        result.errors.append(f"metadata.xml root element is '{tag}', expected 'MetaData'")

    title = root.find(f".//{{{META_NS}}}Title")
    if title is None:
        title = root.find(".//Title")
    if title is None or not (title.text or "").strip():
        result.warnings.append("metadata.xml missing or empty Title")

    app = root.find(f".//{{{META_NS}}}Application")
    if app is None:
        app = root.find(".//Application")
    if app is None:
        result.warnings.append("metadata.xml missing Application element")


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag
