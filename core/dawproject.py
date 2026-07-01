"""
Slunder Studio v0.1.30 — DAWproject Export
Generates cross-DAW .dawproject archives (ZIP containing project.xml,
metadata.xml, and referenced media files).
"""
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from core.settings import APP_VERSION


DAW_NS = "http://bitwig.com/dawproject"
META_NS = "http://bitwig.com/dawproject"

REQUIRED_ARCHIVE_ENTRIES = {"project.xml", "metadata.xml"}


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


def _build_project_xml(spec: DAWProjectSpec) -> str:
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
    for i, track in enumerate(spec.tracks):
        if not track.media_file:
            continue
        lane = ET.SubElement(arrangement, "Lane")
        lane.set("trackRef", f"track-{i}")
        clip = ET.SubElement(lane, "Clip")
        clip.set("time", "0.0")
        audio_ref = ET.SubElement(clip, "Audio")
        media_name = Path(track.media_file).name
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

        for track in spec.tracks:
            if not track.media_file or not os.path.isfile(track.media_file):
                continue
            media_name = Path(track.media_file).name
            zf.write(track.media_file, f"media/{media_name}")

    return output_path


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

            result.entries = zf.namelist()

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
