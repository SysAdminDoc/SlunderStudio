"""Reproducible, runner-neutral engine evaluation contracts and measurements."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import psutil

from core.mastering import measure_lufs, measure_true_peak_db


EVALUATION_SCHEMA_VERSION = 1
LISTENER_RUBRIC_VERSION = 1


@dataclass(frozen=True)
class EvaluationCase:
    """Fixed input contract for one reproducible engine measurement."""

    case_id: str
    engine_id: str
    prompt: str
    seed: int
    duration_seconds: float
    language: str
    expected_structure: tuple[str, ...] = ()
    lyric_timing_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_structure"] = list(self.expected_structure)
        return payload


DEFAULT_EVALUATION_CASES: tuple[EvaluationCase, ...] = (
    EvaluationCase(
        case_id="lyrics-en-001",
        engine_id="lyrics",
        prompt="Write a concise verse and chorus about a night train leaving the city.",
        seed=18001,
        duration_seconds=0.0,
        language="en",
        expected_structure=("verse", "chorus"),
        lyric_timing_required=False,
    ),
    EvaluationCase(
        case_id="lyrics-es-001",
        engine_id="lyrics",
        prompt="Escribe una estrofa y un estribillo sobre volver a casa bajo la lluvia.",
        seed=18002,
        duration_seconds=0.0,
        language="es",
        expected_structure=("verse", "chorus"),
        lyric_timing_required=False,
    ),
    EvaluationCase(
        case_id="song-structure-001",
        engine_id="song_forge",
        prompt="A restrained electronic song about rebuilding after a blackout.",
        seed=18101,
        duration_seconds=30.0,
        language="en",
        expected_structure=("intro", "verse", "chorus", "outro"),
        lyric_timing_required=True,
    ),
    EvaluationCase(
        case_id="midi-groove-001",
        engine_id="midi",
        prompt="Four-bar minor-key groove with kick, snare, bass, and a sparse lead.",
        seed=18201,
        duration_seconds=8.0,
        language="en",
        expected_structure=("four_bars", "kick", "snare", "bass", "lead"),
    ),
    EvaluationCase(
        case_id="separation-vocal-001",
        engine_id="separation",
        prompt="Fixed stereo source-separation fixture.",
        seed=18301,
        duration_seconds=10.0,
        language="und",
        expected_structure=("vocals", "instrumental"),
    ),
)


@dataclass
class EvaluationOutput:
    """Optional runner-returned evidence for one evaluation case."""

    artifacts: list[str | Path] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)
    adherence: dict[str, Any] | None = None
    lyric_timing: dict[str, Any] | None = None
    structure: dict[str, Any] | None = None
    status: str = "completed"
    failure: str = ""


@dataclass
class EvaluationMeasurement:
    case: dict[str, Any]
    status: str
    latency_ms: float
    peak_ram_mb: float | None = None
    peak_vram_mb: float | None = None
    failure: str = ""
    adherence: dict[str, Any] | None = None
    lyric_timing: dict[str, Any] | None = None
    structure: dict[str, Any] | None = None
    loudness_lufs: list[float] = field(default_factory=list)
    true_peak_dbtp: list[float] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def runtime_fingerprint() -> dict[str, Any]:
    """Capture runtime identity without importing optional engine packages."""
    fingerprint: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_revision": _git_revision(),
        "packages": _package_versions(
            ("numpy", "scipy", "soundfile", "PySide6", "torch", "transformers")
        ),
    }
    try:
        fingerprint["cpu_count"] = psutil.cpu_count(logical=True)
        fingerprint["ram_total_mb"] = round(psutil.virtual_memory().total / 1e6, 2)
    except Exception:
        fingerprint["cpu_count"] = None
        fingerprint["ram_total_mb"] = None

    try:
        import torch

        fingerprint["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        }
    except (ImportError, OSError, RuntimeError):
        fingerprint["cuda"] = {"available": False, "device_count": 0, "devices": []}
    return fingerprint


class _ResourceSampler:
    """Small background sampler for process RSS and CUDA peak allocation."""

    def __init__(self):
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_ram_bytes = 0
        self.peak_vram_bytes: int | None = None
        self._torch = None

    def __enter__(self):
        try:
            import torch

            if torch.cuda.is_available():
                self._torch = torch
                torch.cuda.reset_peak_memory_stats()
                self.peak_vram_bytes = 0
        except (ImportError, OSError, RuntimeError):
            self._torch = None
        self._sample()
        self._thread = threading.Thread(target=self._run, name="evaluation-resource-sampler", daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        try:
            self.peak_ram_bytes = max(self.peak_ram_bytes, self._process.memory_info().rss)
        except (psutil.Error, OSError):
            pass
        if self._torch is not None:
            try:
                self.peak_vram_bytes = max(
                    self.peak_vram_bytes or 0,
                    int(self._torch.cuda.max_memory_allocated()),
                )
            except (RuntimeError, AttributeError):
                pass

    def _run(self):
        while not self._stop.wait(0.05):
            self._sample()

    def __exit__(self, *_exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    try:
        relative = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        relative = str(path)
    record: dict[str, Any] = {
        "path": relative,
        "exists": path.is_file(),
        "kind": path.suffix.lower().lstrip("."),
    }
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = _sha256(path)
    return record


def _audio_metrics(path: Path) -> tuple[float, float] | None:
    if path.suffix.lower() not in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
        return None
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return (
            float(measure_lufs(np.asarray(audio), int(sample_rate))),
            float(measure_true_peak_db(np.asarray(audio), int(sample_rate))),
        )
    except (OSError, RuntimeError, ValueError):
        return None


def _coerce_output(value: Any) -> EvaluationOutput:
    if isinstance(value, EvaluationOutput):
        return value
    if value is None:
        return EvaluationOutput(status="skipped", failure="Runner returned no output")
    if isinstance(value, dict):
        return EvaluationOutput(
            artifacts=list(value.get("artifacts", ())),
            model=dict(value.get("model", {})),
            adherence=value.get("adherence"),
            lyric_timing=value.get("lyric_timing"),
            structure=value.get("structure"),
            status=str(value.get("status", "completed")),
            failure=str(value.get("failure", "")),
        )
    raise TypeError("Evaluation runner must return EvaluationOutput, dict, or None")


def blinded_listener_rubric(measurements: Iterable[EvaluationMeasurement]) -> dict[str, Any]:
    """Create a model-agnostic listening sheet keyed only by blind sample IDs."""
    samples = []
    for measurement in measurements:
        artifact_hash = next(
            (item.get("sha256", "") for item in measurement.artifacts if item.get("sha256")),
            "no-artifact",
        )
        blind_id = hashlib.sha256(
            f"{measurement.case['case_id']}:{artifact_hash}".encode("utf-8")
        ).hexdigest()[:12]
        samples.append(
            {
                "blind_id": blind_id,
                "case_id": measurement.case["case_id"],
                "scales": {
                    "overall_quality": {"min": 1, "max": 5},
                    "prompt_adherence": {"min": 1, "max": 5},
                    "artifact_free": {"min": 1, "max": 5},
                    "musical_coherence": {"min": 1, "max": 5},
                },
                "free_text": True,
            }
        )
    return {
        "schema_version": LISTENER_RUBRIC_VERSION,
        "blinded": True,
        "instructions": "Rate samples by blind_id only; do not reveal model, revision, or runner identity to listeners.",
        "samples": samples,
    }


def run_evaluation(
    runner: Callable[[EvaluationCase, Path], Any],
    *,
    cases: Iterable[EvaluationCase] = DEFAULT_EVALUATION_CASES,
    artifact_dir: str | Path,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run fixed cases and return a JSON-serializable report."""
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    measurements: list[EvaluationMeasurement] = []

    for case in cases:
        case_dir = root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        output = EvaluationOutput(status="failed", failure="Runner did not start")
        with _ResourceSampler() as resources:
            try:
                output = _coerce_output(runner(case, case_dir))
            except Exception as exc:  # runner failures belong in the report
                output = EvaluationOutput(status="failed", failure=f"{type(exc).__name__}: {exc}")
        artifacts = []
        loudness = []
        true_peak = []
        for raw_path in output.artifacts:
            path = Path(raw_path)
            if not path.is_absolute():
                path = case_dir / path
            artifacts.append(_artifact_record(path, root))
            audio_metrics = _audio_metrics(path)
            if audio_metrics is not None:
                loudness.append(audio_metrics[0])
                true_peak.append(audio_metrics[1])
        measurements.append(
            EvaluationMeasurement(
                case=case.to_dict(),
                status=output.status,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                peak_ram_mb=round(resources.peak_ram_bytes / 1e6, 3),
                peak_vram_mb=(
                    round(resources.peak_vram_bytes / 1e6, 3)
                    if resources.peak_vram_bytes is not None else None
                ),
                failure=output.failure,
                adherence=output.adherence,
                lyric_timing=output.lyric_timing,
                structure=output.structure,
                loudness_lufs=loudness,
                true_peak_dbtp=true_peak,
                artifacts=artifacts,
                model=output.model,
            )
        )

    serialized = [item.to_dict() for item in measurements]
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "generated_at": time.time(),
        "runtime": runtime or runtime_fingerprint(),
        "cases": serialized,
        "listener_rubric": blinded_listener_rubric(measurements),
        "release_gate": {
            "gated": False,
            "fad_used": False,
            "reason": "FAD is not a release gate; combine fixed measurements with the blinded listener rubric and review failures manually.",
        },
    }


def write_evaluation_report(report: dict[str, Any], path: str | Path) -> Path:
    """Atomically write an evaluation report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)
    return destination
