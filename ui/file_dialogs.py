"""Shared file-dialog policy for Slunder Studio.

The views should describe the operation they are performing and let this
module own format filters, remembered directories, and filename defaults.
That keeps the picker experience consistent without putting Qt concerns in
the core audio or routing modules.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Sequence

from PySide6.QtWidgets import QFileDialog

from core.audio_export import DELIVERY_FORMATS, probe_codecs
from core.routing import AUDIO_EXTENSIONS, MIDI_EXTENSIONS
from core.settings import Settings


_ALL_FILES_FILTER = "All Files (*)"


def _extension_globs(extensions: Iterable[str]) -> str:
    return " ".join(f"*{extension}" for extension in extensions)


def _filter(label: str, extensions: Iterable[str], *, include_all: bool = True) -> str:
    result = f"{label} ({_extension_globs(extensions)})"
    return f"{result};;{_ALL_FILES_FILTER}" if include_all else result


def audio_import_filter(*, include_all: bool = True) -> str:
    """Return the canonical audio input filter."""
    return _filter("Audio Files", AUDIO_EXTENSIONS, include_all=include_all)


def midi_import_filter(*, include_all: bool = True) -> str:
    """Return the canonical MIDI input filter."""
    return _filter("MIDI Files", MIDI_EXTENSIONS, include_all=include_all)


def project_asset_filter() -> str:
    """Return the combined filter used when importing project assets."""
    return ";;".join(
        (
            _filter("Audio Files", AUDIO_EXTENSIONS, include_all=False),
            _filter("MIDI Files", MIDI_EXTENSIONS, include_all=False),
            _ALL_FILES_FILTER,
        )
    )


def delivery_formats(*, available_only: bool = True) -> tuple[str, ...]:
    """Return delivery formats that the current installation can write."""
    if not available_only:
        return tuple(DELIVERY_FORMATS)
    try:
        availability = probe_codecs()
    except Exception:
        availability = {}
    formats = tuple(
        fmt for fmt in DELIVERY_FORMATS
        if availability.get(fmt) is not None and availability[fmt].available
    )
    # A picker must remain useful when an optional codec probe fails.  WAV is
    # the minimum delivery path and is also the default format in Settings.
    return formats or ("wav",)


def delivery_filter(
    *,
    formats: Sequence[str] | None = None,
    available_only: bool = True,
    include_all: bool = False,
) -> str:
    """Build a save filter from the canonical delivery table."""
    selected = tuple(formats) if formats is not None else delivery_formats(
        available_only=available_only
    )
    selected = tuple(fmt.lower().lstrip(".") for fmt in selected if fmt)
    selected = tuple(fmt for fmt in selected if fmt in DELIVERY_FORMATS)
    if not selected:
        selected = ("wav",)
    parts = [f"{fmt.upper()} (*.{fmt})" for fmt in selected]
    if include_all:
        parts.append(_ALL_FILES_FILTER)
    return ";;".join(parts)


def _configured_dirs() -> dict[str, str]:
    value = Settings().get("general.file_dialog_dirs", {})
    return dict(value) if isinstance(value, dict) else {}


def _valid_directory(value: str | os.PathLike[str] | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    return candidate if candidate.is_dir() else None


def last_directory(
    operation_kind: str,
    *,
    fallback_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve the persisted directory for an operation kind."""
    dirs = _configured_dirs()
    for candidate in (
        dirs.get(operation_kind),
        fallback_dir,
        Settings().get("general.output_dir", ""),
        str(Path.home()),
    ):
        resolved = _valid_directory(candidate)
        if resolved is not None:
            return str(resolved)
    return str(Path.cwd())


def _remember_directory(operation_kind: str, path: str, *, is_directory: bool = False):
    if not path:
        return
    candidate = Path(path).expanduser()
    directory = candidate if is_directory else candidate.parent
    if not directory.is_dir():
        return
    dirs = _configured_dirs()
    if dirs.get(operation_kind) == str(directory):
        return
    dirs[operation_kind] = str(directory)
    Settings().set("general.file_dialog_dirs", dirs)


def _initial_path(
    operation_kind: str,
    default_name: str | os.PathLike[str] | None,
    *,
    fallback_dir: str | os.PathLike[str] | None = None,
) -> str:
    directory = Path(last_directory(operation_kind, fallback_dir=fallback_dir))
    if not default_name:
        return str(directory)
    candidate = Path(default_name).expanduser()
    return str(candidate if candidate.is_absolute() else directory / candidate)


def open_audio_file(
    parent,
    title: str,
    operation_kind: str = "audio_import",
    *,
    dialog=QFileDialog,
    fallback_dir: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Open one audio file and remember its directory."""
    path, selected_filter = dialog.getOpenFileName(
        parent,
        title,
        last_directory(operation_kind, fallback_dir=fallback_dir),
        audio_import_filter(),
    )
    if path:
        _remember_directory(operation_kind, path)
    return path, selected_filter


def open_audio_files(
    parent,
    title: str,
    operation_kind: str = "audio_import",
    *,
    dialog=QFileDialog,
    fallback_dir: str | os.PathLike[str] | None = None,
) -> tuple[list[str], str]:
    """Open multiple audio files and remember the selected directory."""
    paths, selected_filter = dialog.getOpenFileNames(
        parent,
        title,
        last_directory(operation_kind, fallback_dir=fallback_dir),
        audio_import_filter(),
    )
    paths = [str(path) for path in paths if path]
    if paths:
        _remember_directory(operation_kind, paths[0])
    return paths, selected_filter


def open_file(
    parent,
    title: str,
    file_filter: str,
    operation_kind: str = "file_import",
    *,
    dialog=QFileDialog,
    fallback_dir: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Open one arbitrary file with the shared remembered-directory policy."""
    path, selected_filter = dialog.getOpenFileName(
        parent,
        title,
        last_directory(operation_kind, fallback_dir=fallback_dir),
        file_filter,
    )
    if path:
        _remember_directory(operation_kind, path)
    return path, selected_filter


def open_midi_file(
    parent,
    title: str,
    operation_kind: str = "midi_import",
    *,
    dialog=QFileDialog,
    fallback_dir: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Open one MIDI file and remember its directory."""
    path, selected_filter = dialog.getOpenFileName(
        parent,
        title,
        last_directory(operation_kind, fallback_dir=fallback_dir),
        midi_import_filter(),
    )
    if path:
        _remember_directory(operation_kind, path)
    return path, selected_filter


def open_project_files(
    parent,
    title: str,
    operation_kind: str = "project_asset_import",
    *,
    dialog=QFileDialog,
    fallback_dir: str | os.PathLike[str] | None = None,
) -> tuple[list[str], str]:
    """Open one or more project assets using the shared import table."""
    paths, selected_filter = dialog.getOpenFileNames(
        parent,
        title,
        last_directory(operation_kind, fallback_dir=fallback_dir),
        project_asset_filter(),
    )
    paths = [str(path) for path in paths if path]
    if paths:
        _remember_directory(operation_kind, paths[0])
    return paths, selected_filter


def save_file(
    parent,
    title: str,
    default_name: str | os.PathLike[str] | None,
    file_filter: str,
    operation_kind: str,
    *,
    dialog=QFileDialog,
    fallback_dir: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Save a file with a remembered directory."""
    path, selected_filter = dialog.getSaveFileName(
        parent,
        title,
        _initial_path(operation_kind, default_name, fallback_dir=fallback_dir),
        file_filter,
    )
    if path:
        _remember_directory(operation_kind, path)
    return path, selected_filter


def save_audio_file(
    parent,
    title: str,
    default_name: str | os.PathLike[str] | None,
    operation_kind: str = "audio_export",
    *,
    dialog=QFileDialog,
    formats: Sequence[str] | None = None,
    available_only: bool = True,
    fallback_dir: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Save audio using the current delivery codecs and remember its directory."""
    return save_file(
        parent,
        title,
        default_name,
        delivery_filter(formats=formats, available_only=available_only),
        operation_kind,
        dialog=dialog,
        fallback_dir=fallback_dir,
    )


def save_midi_file(
    parent,
    title: str,
    default_name: str | os.PathLike[str] | None,
    operation_kind: str = "midi_export",
    *,
    dialog=QFileDialog,
    fallback_dir: str | os.PathLike[str] | None = None,
) -> tuple[str, str]:
    """Save a MIDI file with a remembered directory."""
    return save_file(
        parent,
        title,
        default_name,
        "MIDI Files (*.mid *.midi)",
        operation_kind,
        dialog=dialog,
        fallback_dir=fallback_dir,
    )


def choose_directory(
    parent,
    title: str,
    operation_kind: str = "directory",
    *,
    dialog=QFileDialog,
    fallback_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Choose and remember a directory."""
    path = dialog.getExistingDirectory(
        parent,
        title,
        last_directory(operation_kind, fallback_dir=fallback_dir),
    )
    if path:
        _remember_directory(operation_kind, path, is_directory=True)
    return path


def extension_for_filter(selected_filter: str, *, default: str = "wav") -> str:
    """Extract a safe extension from a selected save filter."""
    match = re.search(r"\*\.([a-z0-9]+)", str(selected_filter or "").lower())
    return match.group(1) if match else default.lstrip(".").lower()


def ensure_extension(path: str, selected_filter: str, *, default: str = "wav") -> str:
    """Add the selected format extension when the user omitted one."""
    if not path or Path(path).suffix:
        return path
    return f"{path}.{extension_for_filter(selected_filter, default=default)}"
