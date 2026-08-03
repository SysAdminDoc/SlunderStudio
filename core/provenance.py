"""
Slunder Studio - Generation provenance sidecars.
Writes reproducibility metadata next to generated and exported artifacts.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from core.settings import APP_VERSION

PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_SUFFIX = ".provenance.json"
UNKNOWN_LICENSE_WARNING = (
    "Model license metadata is indeterminate; review the model license before release."
)
logger = logging.getLogger(__name__)


def sidecar_path_for(artifact_path: str | Path) -> Path:
    """Return the adjacent provenance sidecar path for an artifact."""
    return Path(str(artifact_path) + PROVENANCE_SUFFIX)


def find_provenance_sidecar(artifact_path: str | Path) -> Optional[Path]:
    """Return an existing sidecar for an artifact, if present."""
    path = Path(artifact_path)
    if path.name.endswith(PROVENANCE_SUFFIX) and path.is_file():
        return path
    sidecar = sidecar_path_for(path)
    return sidecar if sidecar.is_file() else None


def file_sha256(path: str | Path) -> str:
    """Hash an artifact for later tamper/reproduction checks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _hash_file_map(file_hashes: dict[str, str]) -> str:
    if not file_hashes:
        return ""
    h = hashlib.sha256()
    for rel_path, digest in sorted(file_hashes.items()):
        h.update(rel_path.encode("utf-8", errors="replace"))
        h.update(b"\0")
        h.update(str(digest).encode("ascii", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def collect_model_metadata(
    model_id: str = "",
    *,
    model_name: str = "",
    model_source: str = "",
    model_revision: str = "",
    model_hash: str = "",
    model_license: str = "",
) -> dict[str, Any]:
    """Collect model registry and download-manifest metadata when available."""
    metadata: dict[str, Any] = {
        "id": model_id or "",
        "name": model_name or "",
        "source": model_source or "",
        "revision": model_revision or "",
        "resolved_revision": "",
        "hash": model_hash or "",
        "license": model_license or "",
        "license_url": "",
        "commercial_use": "",
        "commercial_use_label": "",
        "commercial_use_note": "",
        "license_warning": "",
        "requires_export_warning": False,
        "metadata_status": "not_applicable" if not model_id else "indeterminate",
        "metadata_error": "" if not model_id else "metadata_not_collected",
        "file_hash_count": 0,
        "trusted_source": None,
        "requires_remote_code": False,
        "allows_unsafe_weights": False,
        "execution_consent_required": False,
        "serialization": "",
        "gated": None,
        "access": "",
    }
    if not model_id:
        return metadata

    try:
        from core.model_manager import ModelManager

        mgr = ModelManager()
        info = mgr.get_model_info(model_id)
        if info is not None:
            license_meta = info.license_metadata()
            metadata.update({
                "name": metadata["name"] or info.name,
                "source": metadata["source"] or info.source,
                "revision": metadata["revision"] or info.revision,
                "license": metadata["license"] or info.license,
                "license_url": license_meta.get("license_url", ""),
                "commercial_use": license_meta.get("commercial_use", ""),
                "commercial_use_label": license_meta.get("commercial_use_label", ""),
                "commercial_use_note": license_meta.get("commercial_use_note", ""),
                "license_warning": license_meta.get("license_warning", ""),
                "requires_export_warning": license_meta.get("requires_export_warning", False),
                "trusted_source": info.trusted_source,
                "requires_remote_code": info.requires_remote_code,
                "allows_unsafe_weights": info.allows_unsafe_weights,
                "execution_consent_required": (
                    info.requires_remote_code or info.allows_unsafe_weights
                ),
                "gated": info.gated,
                "access": license_meta.get("access", ""),
            })

        manifest = mgr.get_download_manifest(model_id)
        if manifest:
            file_hashes = manifest.get("file_hashes") or {}
            metadata.update({
                "source": metadata["source"] or manifest.get("source", ""),
                "revision": metadata["revision"] or manifest.get("revision", ""),
                "resolved_revision": manifest.get("resolved_revision", ""),
                "license": metadata["license"] or manifest.get("license", ""),
                "license_url": metadata["license_url"] or manifest.get("license_url", ""),
                "commercial_use": metadata["commercial_use"] or manifest.get("commercial_use", ""),
                "commercial_use_label": metadata["commercial_use_label"] or manifest.get("commercial_use_label", ""),
                "commercial_use_note": metadata["commercial_use_note"] or manifest.get("commercial_use_note", ""),
                "license_warning": metadata["license_warning"] or manifest.get("license_warning", ""),
                "requires_export_warning": (
                    metadata["requires_export_warning"]
                    or manifest.get("requires_export_warning", False)
                ),
                "hash": metadata["hash"] or _hash_file_map(file_hashes),
                "file_hash_count": len(file_hashes),
                "total_bytes": manifest.get("total_bytes", 0),
                "access": metadata["access"] or manifest.get("access", ""),
                "serialization": manifest.get("serialization", ""),
            })
        if info is None and not manifest:
            _mark_model_metadata_indeterminate(metadata, "model_not_registered")
        elif info is None and not any(
            manifest.get(key)
            for key in ("license", "commercial_use", "license_warning")
        ):
            _mark_model_metadata_indeterminate(metadata, "manifest_incomplete")
        else:
            metadata["metadata_status"] = "known"
            metadata["metadata_error"] = ""
    except (ImportError, OSError, TypeError, ValueError, KeyError, AttributeError) as exc:
        _mark_model_metadata_indeterminate(metadata, type(exc).__name__)
        logger.warning(
            "Model license metadata unavailable for %s: %s",
            model_id,
            type(exc).__name__,
        )
    except Exception:  # noqa: BLE001 - preserve export with a fail-closed warning
        _mark_model_metadata_indeterminate(metadata, "unexpected_error")
        logger.exception("Unexpected model metadata failure for %s", model_id)

    return metadata


def _mark_model_metadata_indeterminate(
    metadata: dict[str, Any],
    reason: str,
) -> None:
    """Make metadata failures explicit and warn-worthy without blocking export."""
    metadata.update({
        "license": "unknown",
        "license_url": "",
        "commercial_use": "unknown",
        "commercial_use_label": "Unknown",
        "commercial_use_note": "",
        "license_warning": UNKNOWN_LICENSE_WARNING,
        "requires_export_warning": True,
        "metadata_status": "indeterminate",
        "metadata_error": reason,
        "trusted_source": None,
    })


def write_provenance_sidecar(
    artifact_path: str | Path,
    *,
    module: str,
    operation: str,
    model_id: str = "",
    model_name: str = "",
    model_source: str = "",
    model_revision: str = "",
    model_hash: str = "",
    model_license: str = "",
    seed: Optional[int] = None,
    prompt: str = "",
    lyrics: str = "",
    parameters: Optional[dict[str, Any]] = None,
    source_asset_ids: Optional[list[str]] = None,
    source_paths: Optional[list[str]] = None,
    export_format: str = "",
    output_kind: str = "model",
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a JSON provenance sidecar next to an artifact and return its path."""
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Cannot write provenance for missing artifact: {artifact}")

    model = collect_model_metadata(
        model_id,
        model_name=model_name,
        model_source=model_source,
        model_revision=model_revision,
        model_hash=model_hash,
        model_license=model_license,
    )
    sidecar = sidecar_path_for(artifact)
    payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "created_at": time.time(),
        "created_at_iso": datetime.now(timezone.utc).isoformat(),
        "module": module,
        "operation": operation,
        "output_kind": output_kind,
        "artifact": {
            "path": str(artifact),
            "name": artifact.name,
            "format": export_format or artifact.suffix.lstrip(".").lower(),
            "size_bytes": artifact.stat().st_size,
            "sha256": file_sha256(artifact),
        },
        "model": model,
        "seed": seed,
        "prompt": prompt or "",
        "lyrics": lyrics or "",
        "parameters": _json_safe(parameters or {}),
        "source_asset_ids": _json_safe(source_asset_ids or []),
        "source_paths": _json_safe(source_paths or []),
        "export_format": export_format or artifact.suffix.lstrip(".").lower(),
        "extra": _json_safe(extra or {}),
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(sidecar)
    return sidecar


def read_provenance_sidecar(path: str | Path) -> dict[str, Any]:
    """Read a provenance sidecar or the sidecar adjacent to an artifact."""
    sidecar = find_provenance_sidecar(path)
    if sidecar is None:
        candidate = Path(path)
        if candidate.is_file():
            sidecar = candidate
        else:
            return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}


def project_metadata_from_provenance(
    provenance: dict[str, Any],
    sidecar_path: str | Path = "",
) -> dict[str, Any]:
    """Return a compact project-asset metadata projection of a provenance record."""
    model = provenance.get("model") or {}
    artifact = provenance.get("artifact") or {}
    return {
        "provenance": {
            "sidecar_path": str(sidecar_path) if sidecar_path else "",
            "app_version": provenance.get("app_version", ""),
            "module": provenance.get("module", ""),
            "operation": provenance.get("operation", ""),
            "output_kind": provenance.get("output_kind", ""),
            "model_id": model.get("id", ""),
            "model_name": model.get("name", ""),
            "model_license": model.get("license", ""),
            "model_license_url": model.get("license_url", ""),
            "model_commercial_use": model.get("commercial_use", ""),
            "model_commercial_use_label": model.get("commercial_use_label", ""),
            "model_license_warning": model.get("license_warning", ""),
            "model_revision": model.get("resolved_revision") or model.get("revision", ""),
            "model_hash": model.get("hash", ""),
            "seed": provenance.get("seed"),
            "prompt": provenance.get("prompt", ""),
            "lyrics": provenance.get("lyrics", ""),
            "parameters": provenance.get("parameters", {}),
            "source_asset_ids": provenance.get("source_asset_ids", []),
            "source_paths": provenance.get("source_paths", []),
            "export_format": provenance.get("export_format") or artifact.get("format", ""),
            "artifact_sha256": artifact.get("sha256", ""),
        }
    }
