"""Security gates for local model snapshots used by Transformers loaders."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class ModelSecurityError(RuntimeError):
    """Raised when model provenance or executable-content policy is not satisfied."""


def _private_key_paths(value: Any, path: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            if str(key).startswith("_"):
                yield key_path
            else:
                yield from _private_key_paths(child, key_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _private_key_paths(child, f"{path}[{index}]")


def _config_files(snapshot_root: Path) -> Iterator[Path]:
    try:
        candidates = sorted(snapshot_root.rglob("config.json"))
    except OSError as exc:
        raise ModelSecurityError(
            f"Unable to enumerate Transformers config files in {snapshot_root}"
        ) from exc

    for candidate in candidates:
        if candidate.is_symlink():
            raise ModelSecurityError(
                f"Transformers snapshot contains a symlinked config.json: {candidate}"
            )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ModelSecurityError(
                f"Transformers config disappeared before it could be checked: {candidate}"
            ) from exc
        if not resolved.is_relative_to(snapshot_root):
            raise ModelSecurityError(
                f"Transformers config escapes its snapshot: {candidate}"
            )
        yield candidate


def assert_safe_transformers_snapshot(snapshot_path: str | Path) -> tuple[Path, ...]:
    """Reject unsafe configuration before any Transformers ``from_pretrained`` call.

    Transformers versions retained for ACE-Step can import a module named by a
    private configuration key. Every local ``config.json`` is therefore parsed
    recursively, and all underscore-prefixed keys are rejected. Malformed JSON
    is also rejected rather than delegated to a model loader with version-
    dependent behavior.
    """
    raw_path = Path(snapshot_path).expanduser()
    if not raw_path.is_absolute():
        raise ModelSecurityError(
            "Transformers loading requires an absolute local snapshot path."
        )
    try:
        snapshot_root = raw_path.resolve(strict=True)
    except OSError as exc:
        raise ModelSecurityError(
            f"Transformers snapshot does not exist: {raw_path}"
        ) from exc
    if not snapshot_root.is_dir():
        raise ModelSecurityError(
            f"Transformers snapshot is not a directory: {snapshot_root}"
        )

    checked: list[Path] = []
    for config_path in _config_files(snapshot_root):
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelSecurityError(
                f"Transformers config is not valid UTF-8 JSON: {config_path}"
            ) from exc
        if not isinstance(payload, dict):
            raise ModelSecurityError(
                f"Transformers config must contain a JSON object: {config_path}"
            )
        private_keys = tuple(_private_key_paths(payload))
        if private_keys:
            joined = ", ".join(private_keys[:4])
            if len(private_keys) > 4:
                joined += ", ..."
            raise ModelSecurityError(
                f"Transformers config contains underscore-prefixed key(s) in "
                f"{config_path}: {joined}"
            )
        checked.append(config_path)
    return tuple(checked)
