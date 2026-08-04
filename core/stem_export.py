"""Stem delivery naming templates shared by the UI and export workers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class StemExportTemplate:
    """A deterministic filename convention for a target DAW."""

    id: str
    label: str
    pattern: str
    description: str


STEM_EXPORT_TEMPLATES: tuple[StemExportTemplate, ...] = (
    StemExportTemplate(
        "generic",
        "Generic",
        "{project}_{stem}",
        "Project name followed by the stem role.",
    ),
    StemExportTemplate(
        "ableton_live",
        "Ableton Live",
        "{project}_{index:02d}_{stem}",
        "Two-digit track order keeps stems sorted in the File Browser.",
    ),
    StemExportTemplate(
        "bitwig",
        "Bitwig Studio",
        "{project}_{stem}",
        "Compact names that map directly to Bitwig track names.",
    ),
    StemExportTemplate(
        "cubase",
        "Cubase",
        "{project}_{index:02d}_{stem}",
        "Two-digit track order keeps imported stems aligned with the session.",
    ),
    StemExportTemplate(
        "logic_pro",
        "Logic Pro",
        "{project}_{stem}",
        "Project and role names stay readable in Logic's track list.",
    ),
    StemExportTemplate(
        "pro_tools",
        "Pro Tools",
        "{project}_{stem}_Stem",
        "Adds an explicit Stem suffix for Pro Tools delivery folders.",
    ),
    StemExportTemplate(
        "studio_one",
        "Studio One",
        "{project}_{index:02d}_{stem}",
        "Two-digit track order keeps imported stems aligned with the session.",
    ),
    StemExportTemplate(
        "fl_studio",
        "FL Studio",
        "{project}_{stem}_{index:02d}",
        "Keeps the stem role prominent while retaining stable track order.",
    ),
)

_TEMPLATES_BY_ID = {template.id: template for template in STEM_EXPORT_TEMPLATES}
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str, fallback: str) -> str:
    """Return a portable filename token without path separators or reserved syntax."""
    cleaned = _SAFE_TOKEN.sub("-", str(value or "").strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip(" .-_")
    return cleaned or fallback


def get_stem_export_template(template_id: str | None) -> StemExportTemplate:
    """Return a registered template, falling back safely to Generic."""
    return _TEMPLATES_BY_ID.get(str(template_id or "").strip(), _TEMPLATES_BY_ID["generic"])


def stem_export_filename(
    template_id: str | None,
    project_name: str,
    stem_name: str,
    index: int,
    extension: str = "wav",
) -> str:
    """Render one safe filename from a target template."""
    template = get_stem_export_template(template_id)
    try:
        position = max(1, int(index))
    except (TypeError, ValueError):
        position = 1
    suffix = re.sub(r"[^A-Za-z0-9]+", "", str(extension or "wav").lstrip(".")) or "wav"
    values = {
        "project": _slug(project_name, "slunder-project"),
        "stem": _slug(stem_name, f"stem-{position}"),
        "index": position,
    }
    try:
        rendered = template.pattern.format(**values)
    except (KeyError, ValueError):
        rendered = f"{values['project']}_{values['stem']}"
    rendered = _slug(Path(rendered).stem, f"stem-{position}")
    return f"{rendered}.{suffix.lower()}"


def stem_export_filenames(
    template_id: str | None,
    project_name: str,
    stem_names: Iterable[str],
    extension: str = "wav",
) -> list[str]:
    """Render unique, case-insensitive filenames in stem order."""
    names: list[str] = []
    used: set[str] = set()
    for index, stem_name in enumerate(stem_names, 1):
        candidate = stem_export_filename(
            template_id,
            project_name,
            stem_name,
            index,
            extension,
        )
        path = Path(candidate)
        stem = path.stem
        suffix = path.suffix
        serial = 2
        while candidate.casefold() in used:
            candidate = f"{stem}-{serial}{suffix}"
            serial += 1
        used.add(candidate.casefold())
        names.append(candidate)
    return names

