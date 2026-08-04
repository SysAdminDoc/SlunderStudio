"""Queue-facing helpers for durable jobs and selected-output export."""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from core.job_state import JobRecord, JobStore, extract_output_paths


@dataclass(frozen=True)
class JobResourceEstimate:
    """A transparent estimate shown beside a durable job."""

    duration_minutes: Optional[float] = None
    output_gb: Optional[float] = None
    ram_gb: Optional[float] = None
    vram_gb: Optional[float] = None
    basis: str = "Declared by the job"

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None

    @classmethod
    def from_record(cls, record: JobRecord) -> "JobResourceEstimate":
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        inputs = record.inputs if isinstance(record.inputs, dict) else {}
        declared = metadata.get("resource_estimate", {})
        if not isinstance(declared, dict):
            declared = {}
        duration = cls._number(declared.get("duration_minutes"))
        output_gb = cls._number(declared.get("output_gb"))
        ram_gb = cls._number(declared.get("ram_gb"))
        vram_gb = cls._number(declared.get("vram_gb"))
        basis = str(declared.get("basis", "Declared by the job") or "Declared by the job")

        if duration is None:
            seconds = cls._number(inputs.get("duration"))
            count = cls._number(
                inputs.get("batch_count", inputs.get("count", 1))
            ) or 1.0
            if seconds is not None:
                duration = max(0.1, seconds / 60.0 * count)
                basis = "Derived from requested duration and batch count"
        if output_gb is None:
            seconds = cls._number(inputs.get("duration"))
            count = cls._number(
                inputs.get("batch_count", inputs.get("count", 1))
            ) or 1.0
            if seconds is not None:
                # Conservative stereo float-WAV envelope, rounded for display.
                output_gb = max(0.001, seconds * count * 4 * 2 / 1_000_000_000)
        return cls(duration, output_gb, ram_gb, vram_gb, basis)

    def summary(self) -> str:
        parts = []
        if self.duration_minutes is not None:
            parts.append(f"work ≈ {self.duration_minutes:.1f} min")
        if self.output_gb is not None:
            parts.append(f"output ≈ {self.output_gb:.2f} GB")
        if self.ram_gb is not None:
            parts.append(f"RAM ≈ {self.ram_gb:.1f} GB")
        if self.vram_gb is not None:
            parts.append(f"VRAM ≈ {self.vram_gb:.1f} GB")
        if not parts:
            parts.append("resource estimate not declared")
        return " · ".join(parts)


def estimate_job_resources(record: JobRecord) -> JobResourceEstimate:
    """Return a bounded estimate without importing an engine or model."""
    return JobResourceEstimate.from_record(record)


def _unique_destination(destination: Path, name: str) -> Path:
    candidate = destination / name
    if not candidate.exists():
        return candidate
    stem, suffix = Path(name).stem, Path(name).suffix
    for index in range(2, 10_001):
        candidate = destination / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique export name for {name}")


def export_selected_outputs(
    records: Iterable[JobRecord],
    selected_paths: Iterable[str | Path],
    destination: str | Path,
) -> list[str]:
    """Copy selected completed outputs with collision-safe names.

    Only paths declared by the supplied job records are eligible.  Sources
    must be regular files and symlinks are rejected before copying.
    """
    record_list = list(records)
    allowed = {
        str(Path(path).resolve())
        for record in record_list
        for path in extract_output_paths(record)
        if isinstance(path, (str, Path))
    }
    selected = []
    for raw in selected_paths:
        source = Path(raw)
        if not source.is_absolute():
            raise ValueError("Job output export requires absolute source paths")
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Selected output is unavailable: {source}") from exc
        if str(resolved) not in allowed:
            raise ValueError("Selected path is not declared by a queued job")
        if source.is_symlink() or not resolved.is_file():
            raise ValueError(f"Selected output is not a regular file: {source}")
        selected.append(resolved)

    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir():
        raise ValueError("Job export destination is not a directory")

    written = []
    for source in dict.fromkeys(selected):
        output = _unique_destination(target, source.name)
        shutil.copy2(source, output)
        written.append(str(output))
    return written


__all__ = [
    "JobResourceEstimate",
    "estimate_job_resources",
    "export_selected_outputs",
]
