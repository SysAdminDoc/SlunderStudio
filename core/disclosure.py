"""
AI disclosure and human-authorship reports for project registration.

The report deliberately separates what Slunder Studio observed from what a
person declared.  A project file can prove that text, MIDI, edits, and model
artifacts were stored; it cannot prove who authored the underlying material.
Unknown values therefore remain explicit in both JSON and the copy-pasteable
TSV sheet.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core.provenance import read_provenance_sidecar
from core.settings import APP_VERSION

DISCLOSURE_SCHEMA_VERSION = 1
REPORT_TYPE = "ai-disclosure-human-authorship"

CLASS_GENERATED = "generated"
CLASS_PROCESSED = "processed"
CLASS_HUMAN_AUTHORED = "human-authored"
CLASS_UNKNOWN = "unknown"

_CONTRIBUTION_CATEGORIES = {
    "lyrics": "lyrics",
    "lyric": "lyrics",
    "midi": "midi",
    "notes": "midi",
    "edit": "edit",
    "edits": "edit",
    "take": "take-selection",
    "takes": "take-selection",
    "take-selection": "take-selection",
    "selection": "take-selection",
    "other": "other",
}

_PROCESS_OPERATION_TOKENS = (
    "autotune",
    "convert",
    "export",
    "master",
    "mix",
    "recover",
    "render",
    "separate",
    "stitch",
    "trim",
)
_GENERATION_OPERATION_TOKENS = (
    "generate",
    "synth",
    "clone",
    "diffusion",
)


def parse_human_contributions(text: str) -> list[dict[str, str]]:
    """Parse one user-declared contribution per line.

    A line may start with ``lyrics:``, ``midi:``, ``edit:``, ``takes:`` or
    ``other:``.  Unprefixed lines are retained as ``other`` so a declaration
    is never silently discarded.
    """
    contributions: list[dict[str, str]] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        category = "other"
        description = line
        if ":" in line:
            prefix, remainder = line.split(":", 1)
            mapped = _CONTRIBUTION_CATEGORIES.get(prefix.strip().lower())
            if mapped and remainder.strip():
                category = mapped
                description = remainder.strip()
        contributions.append({
            "category": category,
            "description": description,
            "basis": "user-declared",
        })
    return contributions


def format_human_contributions(contributions: Iterable[Any]) -> str:
    """Format stored contribution declarations for the project editor."""
    lines: list[str] = []
    for entry in contributions or []:
        if isinstance(entry, dict):
            category = _normalise_category(entry.get("category", "other"))
            description = _text(entry.get("description", ""))
        else:
            category = "other"
            description = _text(entry)
        if description:
            lines.append(f"{category}: {description}")
    return "\n".join(lines)


def build_disclosure_report(project: Any, *, generated_at: float | None = None) -> dict[str, Any]:
    """Build a deterministic project-level disclosure report.

    ``project`` is intentionally duck-typed so this module can report loaded
    project snapshots as well as the live ``core.project.Project`` object.
    """
    assets = list(getattr(project, "assets", []) or [])
    asset_elements = [
        _asset_element(asset, index)
        for index, asset in enumerate(sorted(
            assets,
            key=lambda item: (
                _safe_float(getattr(item, "created_at", 0.0)),
                _text(getattr(item, "id", "")),
            ),
        ),
        start=1)
    ]
    declared_contributions = _declared_contributions(project)
    human_elements = [
        _human_element(contribution, index)
        for index, contribution in enumerate(
            declared_contributions,
            start=len(asset_elements) + 1,
        )
    ]
    elements = asset_elements + human_elements
    human_evidence = _human_authorship_evidence(project, assets)

    generated_count = sum(
        element["classification"] == CLASS_GENERATED for element in elements
    )
    processed_count = sum(
        element["classification"] == CLASS_PROCESSED for element in elements
    )
    human_count = sum(
        element["classification"] == CLASS_HUMAN_AUTHORED for element in elements
    )
    unknown_count = sum(
        element["classification"] == CLASS_UNKNOWN for element in elements
    )

    ai_values = [element["ddex"]["IsAIGenerated"] for element in elements]
    if generated_count:
        release_ai_generated: bool | None = True
    elif any(value is None for value in ai_values):
        release_ai_generated = None
    else:
        release_ai_generated = False

    component_types = sorted({
        element["ddex"]["AIComponentType"]
        for element in elements
        if element["ddex"].get("AIComponentType")
    })
    training_values = {
        element["ddex"].get("AITrainingDisclosure")
        for element in elements
        if element["ddex"].get("AITrainingDisclosure") is not None
    }
    training_disclosure: Any = (
        next(iter(training_values)) if len(training_values) == 1 else None
    )

    timestamp = float(generated_at if generated_at is not None else time.time())
    report = {
        "schema_version": DISCLOSURE_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "generated_at": timestamp,
        "generated_at_iso": datetime.fromtimestamp(
            timestamp, timezone.utc
        ).isoformat(),
        "app_version": _text(getattr(project, "app_version", "")) or APP_VERSION,
        "project": {
            "id": _text(getattr(project, "id", "")),
            "name": _text(getattr(project, "name", "Untitled Project")),
            "description": _text(getattr(project, "description", "")),
            "created_at": getattr(project, "created_at", 0.0),
            "updated_at": getattr(project, "updated_at", 0.0),
            "asset_count": len(assets),
            "version_count": len(getattr(project, "versions", []) or []),
        },
        "ddex": {
            "IsAIGenerated": release_ai_generated,
            "AIComponentType": component_types,
            "AITrainingDisclosure": training_disclosure,
        },
        "summary": {
            "generated_elements": generated_count,
            "processed_elements": processed_count,
            "human_authored_elements": human_count,
            "unknown_elements": unknown_count,
            "human_declarations": sum(
                evidence.get("status") == "user-declared"
                for evidence in human_evidence
            ),
        },
        "elements": elements,
        "human_authorship_evidence": human_evidence,
        "limitations": [
            "A stored lyrics field does not prove who wrote the lyrics or whether they were AI-generated.",
            "A MIDI, audio, or project-version record does not prove who performed, drew, edited, or selected it.",
            "Model identity, license, and AI training disclosure are reported only when present in provenance; unknown values are not inferred.",
            "This record documents project evidence and declarations; it is not a legal determination or a distributor acceptance decision.",
        ],
    }
    return _json_safe(report)


def render_disclosure_sheet(report: dict[str, Any]) -> str:
    """Render a copy-pasteable tab-separated disclosure sheet."""
    project = report.get("project") or {}
    ddex = report.get("ddex") or {}
    lines = [
        "Slunder Studio AI disclosure and human-authorship record",
        f"Project\t{_cell(project.get('name', ''))}",
        f"Project ID\t{_cell(project.get('id', ''))}",
        f"Generated at\t{_cell(report.get('generated_at_iso', ''))}",
        "",
        "DDEX release mapping",
        "Field\tValue",
        f"IsAIGenerated\t{_ddex_cell(ddex.get('IsAIGenerated'))}",
        f"AIComponentType\t{_cell(', '.join(ddex.get('AIComponentType') or []) or 'Unknown')}",
        f"AITrainingDisclosure\t{_ddex_cell(ddex.get('AITrainingDisclosure'))}",
        "",
        "Contributing elements",
        "Classification\tElement\tAsset type\tModule\tIsAIGenerated\tAIComponentType\tModel\tRevision\tHash\tLicense\tEvidence",
    ]
    for element in report.get("elements") or []:
        provenance = element.get("provenance") or {}
        model = provenance.get("model") or {}
        ddex_values = element.get("ddex") or {}
        lines.append("\t".join([
            _cell(element.get("classification", CLASS_UNKNOWN)),
            _cell(element.get("name", "")),
            _cell(element.get("asset_type", "")),
            _cell(element.get("module", "")),
            _ddex_cell(ddex_values.get("IsAIGenerated")),
            _cell(ddex_values.get("AIComponentType") or "Unknown"),
            _cell(model.get("id") or model.get("name") or "Unknown"),
            _cell(model.get("revision", "Unknown")),
            _cell(model.get("hash", "Unknown")),
            _cell(model.get("license", "Unknown")),
            _cell(element.get("evidence_status", "unknown")),
        ]))
    lines.extend([
        "",
        "Human-authorship registration evidence",
        "Category\tDescription\tStatus\tBasis",
    ])
    for evidence in report.get("human_authorship_evidence") or []:
        lines.append("\t".join([
            _cell(evidence.get("category", "other")),
            _cell(evidence.get("description", "")),
            _cell(evidence.get("status", "unknown")),
            _cell(evidence.get("basis", "")),
        ]))
    lines.extend(["", "Limitations"])
    lines.extend(f"-\t{_cell(value)}" for value in report.get("limitations") or [])
    return "\n".join(lines) + "\n"


def write_disclosure_report(
    project: Any,
    output_dir: str | Path,
    *,
    stem: str | None = None,
    generated_at: float | None = None,
) -> tuple[Path, Path]:
    """Write JSON and TSV disclosure reports and return ``(json, tsv)`` paths."""
    report = build_disclosure_report(project, generated_at=generated_at)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report_stem = _safe_stem(stem or getattr(project, "name", "project"))
    json_path = directory / f"{report_stem}-ai-disclosure.json"
    tsv_path = directory / f"{report_stem}-ai-disclosure.tsv"
    _atomic_write(json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    _atomic_write(tsv_path, render_disclosure_sheet(report))
    return json_path, tsv_path


def _asset_element(asset: Any, index: int) -> dict[str, Any]:
    provenance, sidecar_path = _asset_provenance(asset)
    model = provenance.get("model") or {}
    if not isinstance(model, dict):
        model = {}
    operation = _text(provenance.get("operation", ""))
    output_kind = _text(provenance.get("output_kind", ""))
    asset_type = _text(getattr(asset, "asset_type", ""))
    module = _text(getattr(asset, "module", ""))
    classification = _classify(operation, output_kind, model.get("id", ""))
    component_type = _component_type(
        asset_type,
        module,
        operation,
        provenance,
    )
    if classification == CLASS_GENERATED:
        is_ai_generated: bool | None = True
    elif classification == CLASS_HUMAN_AUTHORED:
        is_ai_generated = False
    else:
        is_ai_generated = None

    extra = provenance.get("extra") or {}
    if not isinstance(extra, dict):
        extra = {}
    training_disclosure = _first_value(
        extra,
        "AITrainingDisclosure",
        "ai_training_disclosure",
        "training_disclosure",
    )
    if training_disclosure is None:
        training_disclosure = _first_value(
            model,
            "AITrainingDisclosure",
            "ai_training_disclosure",
            "training_disclosure",
        )

    if sidecar_path:
        evidence_status = "provenance-sidecar"
    elif provenance:
        evidence_status = "project-metadata-projection"
    else:
        evidence_status = "no-provenance-record"

    ddex_component_type = (
        component_type
        if classification == CLASS_GENERATED
        or (classification == CLASS_PROCESSED and model.get("id"))
        else None
    )
    element: dict[str, Any] = {
        "index": index,
        "id": _text(getattr(asset, "id", "")),
        "name": _text(getattr(asset, "name", "")),
        "asset_type": asset_type,
        "module": module,
        "file_path": _text(getattr(asset, "file_path", "")),
        "classification": classification,
        "evidence_status": evidence_status,
        "provenance": {
            "available": bool(provenance),
            "sidecar_path": sidecar_path,
            "module": _text(provenance.get("module", "")),
            "operation": operation,
            "output_kind": output_kind,
            "seed": provenance.get("seed"),
            "prompt": _text(provenance.get("prompt", "")),
            "lyrics": _text(provenance.get("lyrics", "")),
            "source_asset_ids": provenance.get("source_asset_ids") or [],
            "source_paths": provenance.get("source_paths") or [],
            "artifact_sha256": (provenance.get("artifact") or {}).get("sha256", ""),
            "model": {
                "id": _text(model.get("id", "")),
                "name": _text(model.get("name", "")),
                "revision": _text(model.get("resolved_revision") or model.get("revision", "")),
                "hash": _text(model.get("hash", "")),
                "license": _text(model.get("license", "")),
                "license_url": _text(model.get("license_url", "")),
                "metadata_status": _text(model.get("metadata_status", "")),
            },
        },
        "ddex": {
            "IsAIGenerated": is_ai_generated,
            "AIComponentType": ddex_component_type,
            "AITrainingDisclosure": training_disclosure,
        },
        "limitations": [],
    }
    if not provenance:
        element["limitations"].append(
            "No provenance sidecar or project provenance projection was available; authorship and AI use are unknown."
        )
    if classification == CLASS_GENERATED and not model.get("id"):
        element["limitations"].append(
            "The artifact is marked generated, but its model identity and license are not recorded."
        )
    if classification == CLASS_PROCESSED:
        element["limitations"].append(
            "Processed classification records a transformation; it does not decide whether the source was AI-generated."
        )
    return element


def _human_element(contribution: dict[str, str], index: int) -> dict[str, Any]:
    """Represent an explicit declaration as a report element."""
    return {
        "index": index,
        "id": f"human-contribution-{index}",
        "name": contribution["description"],
        "asset_type": contribution["category"],
        "module": "project_manager",
        "file_path": "",
        "classification": CLASS_HUMAN_AUTHORED,
        "evidence_status": "user-declared",
        "provenance": {
            "available": False,
            "sidecar_path": "",
            "module": "project_manager",
            "operation": "human_authorship_declaration",
            "output_kind": "human-authored",
            "seed": None,
            "prompt": "",
            "lyrics": "",
            "source_asset_ids": [],
            "source_paths": [],
            "artifact_sha256": "",
            "model": {
                "id": "",
                "name": "",
                "revision": "",
                "hash": "",
                "license": "",
                "license_url": "",
                "metadata_status": "not_applicable",
            },
        },
        "ddex": {
            "IsAIGenerated": False,
            "AIComponentType": None,
            "AITrainingDisclosure": None,
        },
        "limitations": [
            "Human-authored classification is a user declaration; Slunder Studio cannot independently verify it."
        ],
        "human_declaration": {
            "category": contribution["category"],
            "description": contribution["description"],
            "basis": contribution.get("basis") or "user-declared",
        },
    }


def _asset_provenance(asset: Any) -> tuple[dict[str, Any], str]:
    sidecar_path = _text(getattr(asset, "provenance_path", ""))
    if sidecar_path and Path(sidecar_path).is_file():
        record = read_provenance_sidecar(sidecar_path)
        if record:
            return record, sidecar_path

    metadata = getattr(asset, "metadata", {}) or {}
    projection = metadata.get("provenance") if isinstance(metadata, dict) else None
    if isinstance(projection, dict) and projection:
        model = {
            "id": projection.get("model_id", ""),
            "name": projection.get("model_name", ""),
            "revision": projection.get("model_revision", ""),
            "hash": projection.get("model_hash", ""),
            "license": projection.get("model_license", ""),
            "license_url": projection.get("model_license_url", ""),
            "metadata_status": "project-metadata-projection",
        }
        return {
            "app_version": projection.get("app_version", ""),
            "module": projection.get("module", ""),
            "operation": projection.get("operation", ""),
            "output_kind": projection.get("output_kind", ""),
            "seed": projection.get("seed"),
            "prompt": projection.get("prompt", ""),
            "lyrics": projection.get("lyrics", ""),
            "parameters": projection.get("parameters", {}),
            "source_asset_ids": projection.get("source_asset_ids", []),
            "source_paths": projection.get("source_paths", []),
            "export_format": projection.get("export_format", ""),
            "artifact": {"sha256": projection.get("artifact_sha256", "")},
            "model": model,
            "extra": {},
        }, ""
    return {}, ""


def _human_authorship_evidence(project: Any, assets: list[Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, contribution in enumerate(_declared_contributions(project), start=1):
        evidence.append({
            "id": f"human-declaration-{index}",
            "category": contribution["category"],
            "description": contribution["description"],
            "status": "user-declared",
            "basis": contribution.get("basis") or "user-declared",
            "classification": CLASS_HUMAN_AUTHORED,
        })

    lyrics = _text(getattr(project, "lyrics_text", ""))
    if lyrics:
        evidence.append({
            "id": "observed-lyrics-field",
            "category": "lyrics",
            "description": "Lyrics text is stored in the project; authorship is not established by storage alone.",
            "status": "observed-not-proof",
            "basis": "project.lyrics_text",
            "classification": CLASS_UNKNOWN,
        })
    midi_assets = [
        asset for asset in assets
        if _text(getattr(asset, "asset_type", "")).lower() == "midi"
    ]
    if midi_assets:
        evidence.append({
            "id": "observed-midi-assets",
            "category": "midi",
            "description": f"{len(midi_assets)} MIDI asset(s) are stored in the project; who drew or edited the notes is not established.",
            "status": "observed-not-proof",
            "basis": "project.assets",
            "classification": CLASS_UNKNOWN,
        })
    versions = getattr(project, "versions", []) or []
    if versions:
        evidence.append({
            "id": "observed-version-history",
            "category": "edit",
            "description": f"{len(versions)} project version(s) record that edits or saves occurred; the editor is not established.",
            "status": "observed-not-proof",
            "basis": "project.versions",
            "classification": CLASS_UNKNOWN,
        })
    if getattr(project, "mixer_state", {}) or {}:
        evidence.append({
            "id": "observed-mixer-state",
            "category": "edit",
            "description": "Mixer settings are stored in the project; they show a recorded edit state, not who made it.",
            "status": "observed-not-proof",
            "basis": "project.mixer_state",
            "classification": CLASS_UNKNOWN,
        })
    return evidence


def _declared_contributions(project: Any) -> list[dict[str, str]]:
    contributions: list[dict[str, str]] = []
    for raw in getattr(project, "human_contributions", []) or []:
        contribution = _normalise_contribution(raw)
        if contribution["description"]:
            contributions.append(contribution)
    return contributions


def _normalise_contribution(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {
            "category": _normalise_category(value.get("category", "other")),
            "description": _text(value.get("description", "")).strip(),
            "basis": _text(value.get("basis", "user-declared")) or "user-declared",
        }
    return {
        "category": "other",
        "description": _text(value).strip(),
        "basis": "user-declared",
    }


def _normalise_category(value: Any) -> str:
    normalized = _text(value).strip().lower()
    return _CONTRIBUTION_CATEGORIES.get(normalized, "other")


def _classify(operation: str, output_kind: str, model_id: Any) -> str:
    operation = _text(operation).lower()
    output_kind = _text(output_kind).lower()
    model_id = _text(model_id)
    if any(token in operation for token in _PROCESS_OPERATION_TOKENS):
        return CLASS_PROCESSED
    if output_kind in {"processed", "export"}:
        return CLASS_PROCESSED
    if any(token in operation for token in _GENERATION_OPERATION_TOKENS):
        return CLASS_GENERATED
    if output_kind in {"model", "demo"} and model_id:
        return CLASS_GENERATED
    return CLASS_UNKNOWN


def _component_type(
    asset_type: str,
    module: str,
    operation: str,
    provenance: dict[str, Any],
) -> str | None:
    extra = provenance.get("extra") or {}
    if isinstance(extra, dict):
        explicit = _first_value(extra, "AIComponentType", "ai_component_type", "component_type")
        if explicit:
            return _text(explicit)
    haystack = " ".join((asset_type, module, operation, _text(provenance.get("prompt", "")))).lower()
    if any(token in haystack for token in ("vocal", "voice", "singer", "rvc", "diffsinger")):
        return "vocal"
    if any(token in haystack for token in ("lyric", "lyrics")):
        return "lyric"
    if any(token in haystack for token in ("midi", "melody", "note")):
        return "melody"
    if any(token in haystack for token in ("sfx", "instrument", "drum", "sound-effect")):
        return "instrument"
    if any(token in haystack for token in ("song", "composition", "ai_producer", "song_forge")):
        return "full-composition"
    return None


def _first_value(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _safe_stem(value: Any) -> str:
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", _text(value)).strip(" .")
    return (stem or "project")[:120]


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _atomic_write(path: Path, contents: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _cell(value: Any) -> str:
    return _text(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def _ddex_cell(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    return _cell(value)


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return _text(value)
