"""Maintained python-audio-separator adapter.

The dependency is optional and never installed implicitly. The adapter keeps
the existing ``SeparationResult`` contract while recording checkpoint-level
license and resource metadata for every output.
"""
from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from core.deps import ensure
from core.audio_export import write_audio_file
from core.provenance import write_provenance_sidecar
from core.separator_registry import (
    SeparatorCheckpoint,
    get_separator_checkpoint,
    separator_artifact_policy,
)
from core.settings import get_config_dir
from engines.demucs_engine import SeparationResult, StemResult, restore_native_audio


def _stem_name_from_path(path: Path) -> str:
    """Normalize common Audio Separator output names to app stem names."""
    name = re.sub(r"[_-]+", " ", path.stem).lower()
    for token in ("vocals", "vocal", "voice"):
        if token in name:
            return "vocals"
    if "drum" in name:
        return "drums"
    if "bass" in name:
        return "bass"
    if "guitar" in name:
        return "guitar"
    if "piano" in name:
        return "piano"
    if "other" in name:
        return "other"
    if "instrument" in name or "music" in name:
        return "instrumental"
    return path.stem


class AudioSeparatorEngine:
    """Adapter around ``audio_separator.separator.Separator``."""

    backend_id = "audio-separator"

    def __init__(self):
        self._separator = None
        self._checkpoint: SeparatorCheckpoint | None = None
        self._device = "cpu"
        self._model_dir = get_config_dir() / "models" / "audio-separator"
        self._output_dir = get_config_dir() / "generations" / "stems"
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_loaded(self) -> bool:
        return self._separator is not None and self._checkpoint is not None

    @property
    def model_name(self) -> str:
        return self._checkpoint.name if self._checkpoint else ""

    @property
    def checkpoint_id(self) -> str:
        return self._checkpoint.id if self._checkpoint else ""

    def load_model(
        self,
        checkpoint_id: str = "audio-separator-bs-roformer",
        device: str = "auto",
        progress_callback: Optional[Callable] = None,
    ) -> None:
        checkpoint = get_separator_checkpoint(checkpoint_id)
        if checkpoint.backend_id != self.backend_id:
            raise ValueError(f"Checkpoint {checkpoint_id} is not an Audio Separator checkpoint")

        from core.model_manager import ModelManager, OfflineModeError

        model_path = self._model_dir / checkpoint.model_filename
        if ModelManager().is_offline and not model_path.is_file():
            raise OfflineModeError(
                f"Offline Mode: separator checkpoint {checkpoint.model_filename} is not cached"
            )

        ensure("audio_separator", pip_name="audio-separator")
        try:
            from audio_separator.separator import Separator

            if progress_callback:
                progress_callback(0.05, f"Loading {checkpoint.name}...")
            self._separator = Separator(
                output_dir=str(self._output_dir),
                model_file_dir=str(self._model_dir),
                output_format="WAV",
                use_soundfile=True,
            )
            self._separator.load_model(model_filename=checkpoint.model_filename)
            self._checkpoint = checkpoint
            self._device = "cpu" if device == "auto" else device
            if progress_callback:
                progress_callback(1.0, f"{checkpoint.name} loaded")
        except ImportError as exc:
            self._separator = None
            self._checkpoint = None
            raise RuntimeError(f"Failed to import audio-separator: {exc}") from exc
        except Exception:
            self._separator = None
            self._checkpoint = None
            raise

    def unload_model(self) -> None:
        self._separator = None
        self._checkpoint = None

    def separate(
        self,
        input_path: str,
        progress_callback: Optional[Callable] = None,
    ) -> SeparationResult:
        if not self.is_loaded:
            return SeparationResult(error="Audio Separator model not loaded")

        started = time.time()
        checkpoint = self._checkpoint
        assert checkpoint is not None
        run_dir = self._output_dir / f"{Path(input_path).stem}_{uuid.uuid4().hex[:12]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        previous_output_dir = getattr(self._separator, "output_dir", None)

        try:
            source_info = sf.info(input_path)
            source_sample_rate = int(source_info.samplerate)
            source_frames = int(source_info.frames)
            source_duration = source_frames / source_sample_rate
            artifact_policy = separator_artifact_policy(
                checkpoint,
                source_duration,
            )
            checkpoint_metadata = checkpoint.metadata()
            checkpoint_metadata["artifact_policy"] = artifact_policy
            checkpoint_metadata["source_sample_rate"] = source_sample_rate
            checkpoint_metadata["source_duration"] = source_duration
            if progress_callback:
                progress_callback(0.1, "Preparing audio separator...")
            if previous_output_dir is not None:
                self._separator.output_dir = str(run_dir)
            output_files = self._separator.separate(input_path)
            if isinstance(output_files, (str, os.PathLike)):
                output_files = [output_files]

            stems: list[StemResult] = []
            for output in output_files or []:
                output_path = Path(output)
                if not output_path.is_absolute():
                    output_path = run_dir / output_path
                if not output_path.is_file():
                    continue
                output_info = sf.info(output_path)
                audio, sample_rate = sf.read(
                    output_path,
                    dtype="float32",
                    always_2d=True,
                )
                native_audio = restore_native_audio(
                    audio,
                    source_sample_rate,
                    source_frames,
                    int(sample_rate),
                )
                if (
                    int(sample_rate) != source_sample_rate
                    or len(audio) != source_frames
                ):
                    bit_depth = {
                        "PCM_16": 16,
                        "PCM_24": 24,
                        "PCM_32": 32,
                        "FLOAT": 32,
                    }.get(output_info.subtype, 16)
                    write_audio_file(
                        output_path,
                        native_audio,
                        source_sample_rate,
                        file_format="wav",
                        bit_depth=bit_depth,
                    )
                stem_name = _stem_name_from_path(output_path)
                provenance = write_provenance_sidecar(
                    output_path,
                    module="stem_separation",
                    operation="separate_stem",
                    model_id=checkpoint.id,
                    model_name=checkpoint.name,
                    model_license=checkpoint.checkpoint_license,
                    prompt="",
                    parameters={
                        "input_path": str(input_path),
                        "backend_id": self.backend_id,
                        "checkpoint": checkpoint.metadata(),
                        "device": self._device,
                        "source_sample_rate": source_sample_rate,
                        "source_duration": source_duration,
                        "artifact_policy": artifact_policy,
                    },
                    source_paths=[str(input_path)],
                    export_format="wav",
                    output_kind="model",
                    extra={
                        "checkpoint_metadata": checkpoint.metadata(),
                        "artifact_policy": artifact_policy,
                    },
                )
                stems.append(
                    StemResult(
                        name=stem_name,
                        audio=native_audio,
                        sample_rate=source_sample_rate,
                        file_path=str(output_path),
                        provenance_path=str(provenance),
                    )
                )

            if progress_callback:
                progress_callback(1.0, f"Separation complete ({time.time() - started:.1f}s)")
            if not stems:
                return SeparationResult(
                    error="Audio Separator produced no output files",
                    separation_time=time.time() - started,
                    model_name=checkpoint.name,
                    backend_id=self.backend_id,
                    checkpoint_id=checkpoint.id,
                    checkpoint_metadata=checkpoint_metadata,
                    source_sample_rate=source_sample_rate,
                    source_duration=source_duration,
                    artifact_policy=artifact_policy,
                    long_file_warning=artifact_policy.get("warning", ""),
                )
            return SeparationResult(
                stems=stems,
                sample_rate=source_sample_rate,
                duration=source_duration,
                separation_time=time.time() - started,
                model_name=checkpoint.name,
                backend_id=self.backend_id,
                checkpoint_id=checkpoint.id,
                checkpoint_metadata=checkpoint_metadata,
                source_sample_rate=source_sample_rate,
                source_duration=source_duration,
                artifact_policy=artifact_policy,
                long_file_warning=artifact_policy.get("warning", ""),
            )
        except Exception as exc:
            return SeparationResult(
                error=str(exc),
                separation_time=time.time() - started,
                model_name=checkpoint.name,
                backend_id=self.backend_id,
                checkpoint_id=checkpoint.id,
                checkpoint_metadata=checkpoint.metadata(),
            )
        finally:
            if previous_output_dir is not None:
                self._separator.output_dir = previous_output_dir


_engine: AudioSeparatorEngine | None = None


def get_audio_separator() -> AudioSeparatorEngine:
    global _engine
    if _engine is None:
        _engine = AudioSeparatorEngine()
    return _engine


def separate_stems(
    input_path: str,
    checkpoint_id: str = "audio-separator-bs-roformer",
    progress_callback: Optional[Callable] = None,
) -> SeparationResult:
    engine = get_audio_separator()
    if not engine.is_loaded or engine.checkpoint_id != checkpoint_id:
        engine.load_model(checkpoint_id, progress_callback=progress_callback)
    return engine.separate(input_path, progress_callback=progress_callback)


def load_model(cache_dir: str | None = None, **kwargs) -> AudioSeparatorEngine:
    """ModelManager loader entry point."""
    engine = get_audio_separator()
    engine.load_model(
        kwargs.get("checkpoint_id", "audio-separator-bs-roformer"),
        device=kwargs.get("device", "auto"),
    )
    return engine
