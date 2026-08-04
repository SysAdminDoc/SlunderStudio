"""Resolve registry-declared song-generation adapter functions."""
from __future__ import annotations

import importlib
from typing import Callable

from core.song_generator_registry import (
    SONG_GENERATOR_REGISTRY,
    SongGeneratorConfig,
    SongGeneratorRegistry,
)


class SongGeneratorAdapterError(RuntimeError):
    """Raised when a configured song generator cannot be used safely."""


def resolve_song_generator(
    model_id: str | None = None,
    *,
    operation: str = "generate_song",
    registry: SongGeneratorRegistry = SONG_GENERATOR_REGISTRY,
) -> tuple[SongGeneratorConfig, Callable]:
    """Return the callable declared by one enabled generator configuration."""
    selected_id = model_id
    if not selected_id:
        active_ids = registry.active_model_ids()
        selected_id = active_ids[0] if active_ids else None
    config = registry.get(selected_id or "")
    if config is None:
        raise SongGeneratorAdapterError(
            f"No song generator is registered for {selected_id or 'the default model'}."
        )
    if not config.enabled:
        reason = config.availability_reason or "the generator is disabled"
        raise SongGeneratorAdapterError(f"{config.display_name} is unavailable: {reason}.")
    if operation not in config.operations:
        raise SongGeneratorAdapterError(
            f"{config.display_name} does not provide the {operation} operation."
        )

    try:
        module = importlib.import_module(config.adapter_module)
        function_name = (
            config.generation_fn if operation == "generate_song" else operation
        )
        function = getattr(module, function_name)
    except (ImportError, AttributeError) as exc:
        raise SongGeneratorAdapterError(
            f"Could not load {config.display_name}'s {operation} adapter: {exc}"
        ) from exc
    if not callable(function):
        raise SongGeneratorAdapterError(
            f"{config.display_name}'s {operation} adapter is not callable."
        )
    return config, function
