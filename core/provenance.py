"""
Slunder Studio - Generation provenance sidecars.
Writes reproducibility metadata next to generated and exported artifacts.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import platform
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from core.settings import APP_VERSION

PROVENANCE_SCHEMA_VERSION = 2
PROVENANCE_SUFFIX = ".provenance.json"
UNKNOWN_LICENSE_WARNING = (
    "Model license metadata is indeterminate; review the model license before release."
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProvenanceDiff:
    """One recorded-vs-current value that prevents a guaranteed re-render."""

    field: str
    recorded: Any
    current: Any
    reason: str

    def format(self) -> str:
        return (
            f"{self.field}: recorded={self.recorded!r}, "
            f"current={self.current!r} ({self.reason})"
        )


@dataclass(frozen=True)
class ProvenanceCompatibility:
    """Fail-closed compatibility result for a provenance-backed render."""

    compatible: bool
    diffs: tuple[ProvenanceDiff, ...] = ()


class ProvenanceCompatibilityError(RuntimeError):
    """Raised when a render cannot be guaranteed to match its source artifact."""

    def __init__(self, message: str, diffs: tuple[ProvenanceDiff, ...] = ()):
        self.diffs = tuple(diffs)
        detail = "; ".join(diff.format() for diff in self.diffs)
        super().__init__(f"{message}{': ' + detail if detail else ''}")


@dataclass(frozen=True)
class RerenderResult:
    """Result of a provenance-backed render and byte comparison."""

    original_path: str
    rerendered_path: str
    identical: bool
    differences: tuple[ProvenanceDiff, ...] = ()


_RERENDERERS: dict[str, Any] = {}


def register_rerenderer(operation_key: str, renderer) -> None:
    """Register a lazy renderer for ``<module>:<operation>`` provenance keys."""
    key = str(operation_key or "").strip()
    if not key or ":" not in key:
        raise ValueError("Rerenderer keys must be '<module>:<operation>'")
    if not callable(renderer):
        raise TypeError("Rerenderer must be callable")
    _RERENDERERS[key] = renderer


def runtime_fingerprint() -> dict[str, Any]:
    """Return deterministic runtime inputs that can affect rendered bytes."""
    package_names = (
        "numpy",
        "scipy",
        "soundfile",
        "librosa",
        "torch",
        "diffusers",
        "transformers",
        "onnxruntime",
    )
    packages = {}
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = ""
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
    }


def _source_file_hashes(source_paths: list[str | Path]) -> dict[str, str]:
    hashes = {}
    for raw_path in source_paths:
        path = Path(raw_path)
        try:
            hashes[str(path)] = file_sha256(path) if path.is_file() else ""
        except OSError:
            hashes[str(path)] = ""
    return hashes


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
        "source_hashes": _source_file_hashes(source_paths or []),
        "export_format": export_format or artifact.suffix.lstrip(".").lower(),
        "runtime": runtime_fingerprint(),
        "rerender_key": f"{module}:{operation}",
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


def _current_model_snapshot(model_id: str) -> dict[str, Any]:
    """Read the installed registry/manifest identity without loading weights."""
    if not model_id:
        return {"id": "", "revision": "", "hash": "", "available": True}
    try:
        from core.model_manager import ModelManager

        manager = ModelManager()
        info = manager.get_model_info(model_id)
        manifest = manager.get_download_manifest(model_id) or {}
        revision = (
            manifest.get("resolved_revision")
            or manifest.get("revision")
            or (info.revision if info else "")
        )
        file_hashes = manifest.get("file_hashes") or {}
        return {
            "id": model_id,
            "revision": revision,
            "hash": _hash_file_map(file_hashes),
            "available": bool(manifest) or bool(info and info.pip_managed),
        }
    except Exception as exc:  # fail closed in the caller
        return {
            "id": model_id,
            "revision": "",
            "hash": "",
            "available": False,
            "error": type(exc).__name__,
        }


def check_provenance_compatibility(
    provenance: dict[str, Any],
    *,
    current_app_version: Optional[str] = None,
    current_runtime: Optional[dict[str, Any]] = None,
    current_model: Optional[dict[str, Any]] = None,
) -> ProvenanceCompatibility:
    """Compare recorded render inputs with the current local environment.

    Missing identity fields are differences, not assumptions.  This means old
    sidecars can still be opened and inspected, but cannot make a bit-identical
    re-render promise until they are regenerated with the versioned contract.
    """
    diffs: list[ProvenanceDiff] = []
    if not isinstance(provenance, dict):
        return ProvenanceCompatibility(
            False,
            (ProvenanceDiff("provenance", provenance, {}, "record is not an object"),),
        )

    schema = provenance.get("schema_version")
    if schema != PROVENANCE_SCHEMA_VERSION:
        diffs.append(ProvenanceDiff(
            "schema_version",
            schema,
            PROVENANCE_SCHEMA_VERSION,
            "provenance contract version differs",
        ))

    expected_app = current_app_version or APP_VERSION
    recorded_app = provenance.get("app_version", "")
    if not recorded_app:
        diffs.append(ProvenanceDiff(
            "app_version", recorded_app, expected_app, "recorded application version is missing"
        ))
    elif recorded_app != expected_app:
        diffs.append(ProvenanceDiff(
            "app_version", recorded_app, expected_app, "application version changed"
        ))

    expected_runtime = current_runtime or runtime_fingerprint()
    recorded_runtime = provenance.get("runtime")
    if not isinstance(recorded_runtime, dict):
        diffs.append(ProvenanceDiff(
            "runtime", recorded_runtime, expected_runtime, "runtime fingerprint is missing"
        ))
    else:
        for field in ("python", "system", "machine"):
            recorded = recorded_runtime.get(field, "")
            current = expected_runtime.get(field, "")
            if recorded != current:
                diffs.append(ProvenanceDiff(
                    f"runtime.{field}", recorded, current, "render runtime changed"
                ))
        recorded_packages = recorded_runtime.get("packages")
        current_packages = expected_runtime.get("packages")
        if not isinstance(recorded_packages, dict):
            diffs.append(ProvenanceDiff(
                "runtime.packages", recorded_packages, current_packages,
                "package fingerprint is missing",
            ))
        else:
            for package, current in (current_packages or {}).items():
                recorded = recorded_packages.get(package, "")
                if recorded != current:
                    diffs.append(ProvenanceDiff(
                        f"runtime.packages.{package}", recorded, current,
                        "dependency version changed",
                    ))

    model = provenance.get("model") or {}
    output_kind = str(provenance.get("output_kind", "model"))
    requires_model = bool(model.get("id")) and output_kind == "model"
    if requires_model:
        current = current_model or _current_model_snapshot(str(model.get("id", "")))
        recorded_revision = model.get("resolved_revision") or model.get("revision", "")
        current_revision = current.get("revision", "")
        if not current.get("available"):
            diffs.append(ProvenanceDiff(
                "model.available", True, False,
                "recorded model is not available in the local registry/cache",
            ))
        if not recorded_revision:
            diffs.append(ProvenanceDiff(
                "model.revision", recorded_revision, current_revision,
                "immutable model revision is missing",
            ))
        elif recorded_revision != current_revision:
            diffs.append(ProvenanceDiff(
                "model.revision", recorded_revision, current_revision,
                "model revision changed",
            ))
        recorded_hash = model.get("hash", "")
        current_hash = current.get("hash", "")
        if not recorded_hash:
            diffs.append(ProvenanceDiff(
                "model.hash", recorded_hash, current_hash,
                "model file hash is missing",
            ))
        elif recorded_hash != current_hash:
            diffs.append(ProvenanceDiff(
                "model.hash", recorded_hash, current_hash,
                "model files changed",
            ))

    source_paths = provenance.get("source_paths") or []
    source_hashes = provenance.get("source_hashes")
    if source_paths and not isinstance(source_hashes, dict):
        diffs.append(ProvenanceDiff(
            "source_hashes", source_hashes, {}, "source file hashes are missing"
        ))
    elif source_paths:
        for raw_path in source_paths:
            path = str(raw_path)
            recorded = source_hashes.get(path, "")
            current = ""
            try:
                current = file_sha256(path) if Path(path).is_file() else ""
            except OSError:
                current = ""
            if not recorded:
                diffs.append(ProvenanceDiff(
                    f"source_hashes.{path}", recorded, current,
                    "source file hash is missing",
                ))
            elif recorded != current:
                diffs.append(ProvenanceDiff(
                    f"source_hashes.{path}", recorded, current,
                    "source file changed or is unavailable",
                ))

    return ProvenanceCompatibility(not diffs, tuple(diffs))


def _rerender_progress(progress_cb, value, *_message):
    if progress_cb:
        progress_cb(int(float(value) * 100 if float(value) <= 1 else float(value)))


def _rerender_sfx(provenance, output_dir, progress_cb=None, cancel_event=None):
    from engines.sfx_engine import SFXEngine, SFXParams

    fields = {field.name for field in dataclasses.fields(SFXParams)}
    payload = {
        key: value for key, value in (provenance.get("parameters") or {}).items()
        if key in fields
    }
    if provenance.get("seed") is not None:
        payload["seed"] = int(provenance["seed"])
    payload["batch_size"] = 1
    if provenance.get("output_kind") == "demo":
        payload["allow_demo_output"] = True
    engine = SFXEngine()
    engine._output_dir = str(output_dir)
    if provenance.get("output_kind") == "model":
        from core.model_manager import ModelManager

        engine = ModelManager().load_model(str((provenance.get("model") or {}).get("id", "")))
        engine._output_dir = str(output_dir)
    result = engine.generate(
        SFXParams(**payload),
        progress_callback=lambda value, *message: _rerender_progress(
            progress_cb, value, *message
        ),
    )
    if result.error or not result.file_path:
        raise RuntimeError(result.error or "SFX rerender produced no artifact")
    return result.file_path


def _rerender_ace_step(provenance, output_dir, progress_cb=None, cancel_event=None):
    from engines.ace_step_engine import (
        GenerationParams,
        _load_managed_engine,
    )
    from core.model_manager import ModelManager

    fields = {field.name for field in dataclasses.fields(GenerationParams)}
    payload = {
        key: value for key, value in (provenance.get("parameters") or {}).items()
        if key in fields
    }
    if provenance.get("seed") is not None:
        payload["seed"] = int(provenance["seed"])
    source_paths = provenance.get("source_paths") or []
    if source_paths and not payload.get("source_audio_path"):
        payload["source_audio_path"] = str(source_paths[0])
    operation = str(provenance.get("operation", ""))
    payload["long_form"] = operation == "generate_long_form" or bool(payload.get("long_form"))
    params = GenerationParams(**payload)
    engine = _load_managed_engine(ModelManager())
    previous_output_dir = engine._output_dir
    engine._output_dir = Path(output_dir)
    try:
        if payload["long_form"]:
            result = engine.generate_long_form(
                params,
                progress_cb=lambda value: _rerender_progress(progress_cb, value),
                cancel_event=cancel_event,
            )
        else:
            result = engine.generate(
                params,
                progress_cb=lambda value: _rerender_progress(progress_cb, value),
                cancel_event=cancel_event,
            )
    finally:
        engine._output_dir = previous_output_dir
    if not result.audio_path:
        raise RuntimeError("ACE-Step rerender produced no artifact")
    return result.audio_path


def _rerender_autotune(provenance, output_dir, progress_cb=None, cancel_event=None):
    from engines.vocal_tuning import AutoTuneParams, autotune_file

    fields = {field.name for field in dataclasses.fields(AutoTuneParams)}
    payload = {
        key: value for key, value in (provenance.get("parameters") or {}).items()
        if key in fields
    }
    source_paths = provenance.get("source_paths") or []
    if source_paths and not payload.get("input_path"):
        payload["input_path"] = str(source_paths[0])
    if not payload.get("input_path"):
        raise ValueError("Auto-tune provenance does not name an input file")
    payload["output_path"] = str(
        Path(output_dir) / f"{Path(payload['input_path']).stem}_rerendered.wav"
    )
    result = autotune_file(
        AutoTuneParams(**payload),
        progress_cb=lambda value: _rerender_progress(progress_cb, value),
        cancel_event=cancel_event,
    )
    return result.output_path


def rerender_from_provenance(
    artifact_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    progress_cb=None,
    cancel_event=None,
) -> RerenderResult:
    """Re-render an artifact only when its provenance is locally compatible."""
    requested = Path(artifact_path)
    if requested.name.endswith(PROVENANCE_SUFFIX):
        source_artifact = Path(str(requested)[:-len(PROVENANCE_SUFFIX)])
    else:
        source_artifact = requested
    provenance = read_provenance_sidecar(requested)
    if not provenance:
        raise ProvenanceCompatibilityError(
            "No readable provenance sidecar was found",
            (ProvenanceDiff("provenance", "", "", "sidecar is missing or invalid"),),
        )
    compatibility = check_provenance_compatibility(provenance)
    if not compatibility.compatible:
        raise ProvenanceCompatibilityError(
            "Artifact cannot be guaranteed reproducible", compatibility.diffs
        )

    key = str(
        provenance.get("rerender_key")
        or f"{provenance.get('module', '')}:{provenance.get('operation', '')}"
    )
    renderer = _RERENDERERS.get(key)
    if renderer is None:
        raise ProvenanceCompatibilityError(
            "Artifact renderer is not registered",
            (ProvenanceDiff("rerender_key", key, "", "no renderer is available"),),
        )

    target_dir = Path(output_dir) if output_dir else source_artifact.parent / "rerenders"
    target_dir.mkdir(parents=True, exist_ok=True)
    rerendered = renderer(
        provenance,
        target_dir,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )
    rerendered_path = Path(rerendered)
    if not rerendered_path.is_file():
        raise RuntimeError(f"Rerenderer did not produce a file: {rerendered_path}")

    expected_hash = str((provenance.get("artifact") or {}).get("sha256", ""))
    actual_hash = file_sha256(rerendered_path)
    differences = () if expected_hash and expected_hash == actual_hash else (
        ProvenanceDiff(
            "artifact.sha256", expected_hash, actual_hash,
            "rerendered bytes differ from the recorded artifact",
        ),
    )
    return RerenderResult(
        original_path=str(source_artifact),
        rerendered_path=str(rerendered_path),
        identical=not differences,
        differences=differences,
    )


register_rerenderer("sfx:generate", _rerender_sfx)
register_rerenderer("song_forge:generate", _rerender_ace_step)
register_rerenderer("song_forge:generate_long_form", _rerender_ace_step)
register_rerenderer("vocal_suite:vocal_autotune", _rerender_autotune)


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
            "source_hashes": provenance.get("source_hashes", {}),
            "export_format": provenance.get("export_format") or artifact.get("format", ""),
            "artifact_sha256": artifact.get("sha256", ""),
            "rerender_key": provenance.get("rerender_key", ""),
        }
    }
