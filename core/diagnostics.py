"""
Slunder Studio v0.1.20 - Redacted diagnostics and health report export.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from core.audio_export import _find_ffmpeg
from core.job_state import JobStatus, JobStore
from core.model_manager import ModelManager
from core.project import PROJECT_SCHEMA_VERSION
from core.provenance import PROVENANCE_SCHEMA_VERSION
from core.settings import (
    APP_NAME,
    APP_VERSION,
    SETTINGS_SCHEMA_VERSION,
    Settings,
    get_config_dir,
    get_trash_dir,
)


REPORT_SCHEMA_VERSION = 1
HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9][A-Za-z0-9_-]{7,}")
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(token|hf_token|hugging_face_hub_token|authorization|api_key|secret|password)"
    r"\s*[:=]\s*[\"']?[^\"'\s;,]+"
)
SENSITIVE_KEYS = {
    "token",
    "hf_token",
    "hugging_face_hub_token",
    "authorization",
    "api_key",
    "secret",
    "password",
    "access_token",
}
DEPENDENCY_PROBES = (
    ("PySide6", "PySide6"),
    ("numpy", "numpy"),
    ("soundfile", "soundfile"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("huggingface_hub", "huggingface_hub"),
    ("librosa", "librosa"),
    ("pyinstaller", "PyInstaller"),
)


def redact_text(value: Any, replacements: Optional[Iterable[tuple[str, str]]] = None) -> str:
    """Redact tokens and known local paths from a string value."""
    text = "" if value is None else str(value)
    for raw, alias in sorted(replacements or [], key=lambda item: len(item[0]), reverse=True):
        if not raw:
            continue
        flags = re.IGNORECASE if os.name == "nt" else 0
        text = re.sub(re.escape(raw), alias, text, flags=flags)
    text = HF_TOKEN_RE.sub("[REDACTED_HF_TOKEN]", text)
    text = SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    return text


def redact_path(path: Any, replacements: Optional[Iterable[tuple[str, str]]] = None) -> str:
    """Return a redacted path string suitable for support reports."""
    return redact_text(path, replacements)


def collect_health_report(
    *,
    include_private: bool = False,
    settings: Optional[Settings] = None,
    model_manager: Optional[ModelManager] = None,
    job_store: Optional[JobStore] = None,
) -> dict[str, Any]:
    """Collect app health state without exposing tokens or creative inputs by default."""
    settings = settings or Settings()
    replacements = _redaction_replacements(settings)
    model_manager = model_manager or ModelManager()
    job_store = job_store or JobStore()

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": _iso(time.time()),
        "privacy": {
            "private_job_inputs_included": bool(include_private),
            "hf_tokens_redacted": True,
            "local_paths_redacted": True,
        },
        "app": _app_info(replacements),
        "schemas": {
            "settings": SETTINGS_SCHEMA_VERSION,
            "project": PROJECT_SCHEMA_VERSION,
            "provenance": PROVENANCE_SCHEMA_VERSION,
        },
        "paths": _path_info(settings, replacements),
        "settings_repair": _sanitize(settings.repair_status, replacements, include_private=True),
        "dependencies": _dependency_info(),
        "gpu": model_manager.get_gpu_status(),
        "ffmpeg": _ffmpeg_info(replacements),
        "models": _model_info(model_manager, replacements),
        "recent_job_failures": _job_failures(job_store, replacements, include_private),
        "crash_log": _crash_log_info(replacements, include_private),
    }
    return _sanitize(report, replacements, include_private=True)


def format_health_report_text(report: dict[str, Any]) -> str:
    """Create a compact support-readable text report from collected JSON."""
    lines = [
        f"Slunder Studio Health Report v{report.get('schema_version', REPORT_SCHEMA_VERSION)}",
        f"Generated: {report.get('generated_at', '')}",
        "",
        "App",
        f"- Version: {report.get('app', {}).get('version', '')}",
        f"- Platform: {report.get('app', {}).get('platform', '')}",
        f"- Python: {report.get('app', {}).get('python', '')}",
        f"- Frozen: {report.get('app', {}).get('frozen', False)}",
        "",
        "Runtime",
        f"- GPU: {report.get('gpu', {}).get('name', 'unknown')} "
        f"({report.get('gpu', {}).get('free_gb', 0)} GB free)",
        f"- ffmpeg: {'available' if report.get('ffmpeg', {}).get('available') else 'missing'}",
        f"- Config: {report.get('paths', {}).get('config_dir', '')}",
        f"- Output: {report.get('paths', {}).get('output_dir', '')}",
        f"- Model cache: {report.get('paths', {}).get('model_cache_dir', '')}",
        "",
        "Models",
    ]
    for model in report.get("models", []):
        lines.append(
            f"- {model.get('id')}: {model.get('status')} | "
            f"{model.get('license')} | commercial={model.get('commercial_use_label')} | "
            f"cache={model.get('cache_state')}"
        )

    lines.extend(["", "Recent Job Failures"])
    failures = report.get("recent_job_failures", [])
    if not failures:
        lines.append("- None")
    for job in failures:
        error = job.get("error") or ("error present" if job.get("error_present") else "no error text")
        lines.append(
            f"- {job.get('updated_at', '')} {job.get('kind')} {job.get('status')}: "
            f"{job.get('label', '')} ({error})"
        )

    lines.extend(["", "Dependency Versions"])
    for name, details in report.get("dependencies", {}).items():
        version = details.get("version") or "not installed"
        lines.append(f"- {name}: {version}")

    return "\n".join(lines) + "\n"


def export_health_report(
    target_path: str | Path,
    *,
    include_private: bool = False,
    settings: Optional[Settings] = None,
    model_manager: Optional[ModelManager] = None,
    job_store: Optional[JobStore] = None,
) -> Path:
    """Write a ZIP bundle containing redacted JSON and text health reports."""
    path = Path(target_path).expanduser()
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)

    report = collect_health_report(
        include_private=include_private,
        settings=settings,
        model_manager=model_manager,
        job_store=job_store,
    )
    replacements = _redaction_replacements(settings or Settings())
    json_blob = redact_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
        replacements,
    )
    text_blob = redact_text(format_health_report_text(json.loads(json_blob)), replacements)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("health-report.json", json_blob)
        bundle.writestr("health-report.txt", text_blob)
    return path


def _app_info(replacements: Iterable[tuple[str, str]]) -> dict[str, Any]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "python": platform.python_version(),
        "python_executable": redact_path(sys.executable, replacements),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "frozen": bool(getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS")),
    }


def _path_info(settings: Settings, replacements: Iterable[tuple[str, str]]) -> dict[str, Any]:
    return {
        "config_dir": redact_path(get_config_dir(), replacements),
        "output_dir": redact_path(settings.get("general.output_dir", ""), replacements),
        "model_cache_dir": redact_path(settings.get("model_hub.cache_dir", ""), replacements),
        "trash_dir": redact_path(get_trash_dir(), replacements),
    }


def _dependency_info() -> dict[str, dict[str, Any]]:
    dependencies: dict[str, dict[str, Any]] = {}
    for import_name, distribution_name in DEPENDENCY_PROBES:
        installed = importlib.util.find_spec(import_name) is not None
        try:
            version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            version = ""
        dependencies[import_name] = {
            "installed": installed,
            "version": version,
        }
    return dependencies


def _ffmpeg_info(replacements: Iterable[tuple[str, str]]) -> dict[str, Any]:
    path = _find_ffmpeg()
    info: dict[str, Any] = {
        "available": bool(path),
        "path": redact_path(path or "", replacements),
        "version": "",
    }
    if not path:
        return info
    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        first_line = (result.stdout or result.stderr or "").splitlines()[0:1]
        info["version"] = redact_text(first_line[0] if first_line else "", replacements)
    except (OSError, subprocess.SubprocessError):
        info["version"] = "unavailable"
    return info


def _model_info(manager: ModelManager, replacements: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model_id, info in sorted(manager.registry.items()):
        status = manager._status.get(model_id)  # diagnostic snapshot of manager state
        manifest = manager.get_download_manifest(model_id)
        partial = manager.has_partial_download(model_id)
        cache_state = "downloaded" if manifest else ("partial" if partial else "not_downloaded")
        models.append({
            "id": model_id,
            "name": info.name,
            "category": info.category.value,
            "status": getattr(status, "value", str(status or "unknown")),
            "cache_state": cache_state,
            "cache_dir": redact_path(manager.get_cache_dir(model_id), replacements),
            "expected_disk_gb": info.disk_gb,
            "expected_vram_gb": info.vram_gb,
            "source": info.source,
            "revision": info.revision,
            "license": info.license,
            "license_url": info.license_url,
            "commercial_use": info.commercial_use,
            "commercial_use_label": info.commercial_use_label,
            "license_warning": info.license_warning,
            "gated": info.gated,
            "access": info.access_label,
            "manifest": _manifest_summary(manifest),
        })
    return models


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    if not manifest:
        return {"present": False}
    return {
        "present": True,
        "file_count": manifest.get("file_count", 0),
        "total_bytes": manifest.get("total_bytes", 0),
        "revision": manifest.get("revision", ""),
        "resolved_revision": manifest.get("resolved_revision", ""),
        "trusted_source": bool(manifest.get("trusted_source", False)),
        "file_hash_count": len(manifest.get("file_hashes", {}) or {}),
    }


def _job_failures(
    store: JobStore,
    replacements: Iterable[tuple[str, str]],
    include_private: bool,
) -> list[dict[str, Any]]:
    statuses = {
        JobStatus.FAILED,
        JobStatus.RECOVERABLE,
        JobStatus.CANCELLED,
    }
    records = store.list_records(status=statuses)[:10]
    summaries: list[dict[str, Any]] = []
    for record in records:
        summary: dict[str, Any] = {
            "id": record.id,
            "kind": record.kind,
            "label": redact_text(record.label, replacements),
            "status": record.status,
            "created_at": _iso(record.created_at),
            "updated_at": _iso(record.updated_at),
            "finished_at": _iso(record.finished_at),
            "progress": record.progress,
            "recoverable": record.recoverable,
            "input_keys": sorted(record.inputs.keys()),
            "output_keys": sorted(record.outputs.keys()),
            "message_present": bool(record.message),
            "error_present": bool(record.error),
        }
        if include_private:
            summary["message"] = _truncate(redact_text(record.message, replacements))
            summary["error"] = _truncate(redact_text(record.error, replacements))
            summary["inputs"] = _sanitize(record.inputs, replacements, include_private=True)
            summary["outputs"] = _sanitize(record.outputs, replacements, include_private=True)
            summary["metadata"] = _sanitize(record.metadata, replacements, include_private=True)
        summaries.append(summary)
    return summaries


def _crash_log_info(
    replacements: Iterable[tuple[str, str]],
    include_private: bool,
) -> dict[str, Any]:
    path = get_config_dir() / "crash.log"
    info: dict[str, Any] = {
        "path": redact_path(path, replacements),
        "exists": path.exists(),
        "size_bytes": 0,
        "modified_at": "",
        "tail": "",
    }
    if not path.exists():
        return info
    try:
        stat = path.stat()
        info["size_bytes"] = stat.st_size
        info["modified_at"] = _iso(stat.st_mtime)
        if include_private:
            raw_tail = path.read_text(encoding="utf-8", errors="replace")[-4000:]
            info["tail"] = redact_text(raw_tail, replacements)
    except OSError:
        info["read_error"] = True
    return info


def _redaction_replacements(settings: Settings) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    candidates = [
        ("<SLUNDER_CONFIG>", get_config_dir()),
        ("<SLUNDER_OUTPUT>", settings.get("general.output_dir", "")),
        ("<SLUNDER_MODEL_CACHE>", settings.get("model_hub.cache_dir", "")),
        ("<SLUNDER_TRASH>", get_trash_dir()),
        ("<USER_HOME>", Path.home()),
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(("<APPDATA>", appdata))

    for alias, raw_path in candidates:
        if not raw_path:
            continue
        try:
            path = Path(raw_path).expanduser()
        except TypeError:
            continue
        variants = {
            str(path),
            str(path).replace("\\", "/"),
        }
        try:
            resolved = path.resolve()
            variants.add(str(resolved))
            variants.add(str(resolved).replace("\\", "/"))
        except OSError:
            pass
        for variant in variants:
            if len(variant) > 2:
                replacements.append((variant, alias))
    return replacements


def _sanitize(
    value: Any,
    replacements: Iterable[tuple[str, str]],
    *,
    include_private: bool,
) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower()
            if _is_sensitive_key(normalized_key):
                result[str(key)] = "[REDACTED]"
                continue
            result[str(key)] = _sanitize(item, replacements, include_private=include_private)
        return result
    if isinstance(value, list):
        return [_sanitize(item, replacements, include_private=include_private) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item, replacements, include_private=include_private) for item in value]
    if isinstance(value, Path):
        return redact_path(value, replacements)
    if isinstance(value, str):
        return redact_text(value, replacements)
    return value


def _is_sensitive_key(normalized_key: str) -> bool:
    if normalized_key in SENSITIVE_KEYS:
        return True
    parts = set(re.split(r"[^a-z0-9]+", normalized_key))
    if {"api", "key"}.issubset(parts):
        return True
    return bool(parts & {"password", "secret", "authorization", "token"})


def _truncate(value: str, limit: int = 1000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


def _iso(timestamp: Any) -> str:
    if not timestamp:
        return ""
    try:
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except (TypeError, ValueError, OSError):
        return ""
