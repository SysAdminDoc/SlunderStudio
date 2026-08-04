"""Measured tradeoffs for quantized model variants.

The model registry owns the immutable source and coarse hardware estimates.  This
module owns the repeatable local benchmark contract used to replace estimates
with measurements once a variant is installed.  It deliberately accepts a
runner callback so the core stays independent of optional llama.cpp and
Transformers packages.
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from core.evaluation import _ResourceSampler


VARIANT_BENCHMARK_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VariantBenchmarkCase:
    """A fixed prompt used to compare two quantizations of one base model."""

    case_id: str
    prompt: str
    expected_sections: tuple[str, ...] = ()
    max_tokens: int = 128

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expected_sections"] = list(self.expected_sections)
        return payload


DEFAULT_VARIANT_BENCHMARK_CASES: tuple[VariantBenchmarkCase, ...] = (
    VariantBenchmarkCase(
        case_id="lyrics-variant-en-001",
        prompt="Write a short verse and chorus about a night train leaving the city.",
        expected_sections=("verse", "chorus"),
    ),
    VariantBenchmarkCase(
        case_id="lyrics-variant-en-002",
        prompt="Write an intimate verse and chorus about repairing a friendship after a storm.",
        expected_sections=("verse", "chorus"),
    ),
    VariantBenchmarkCase(
        case_id="lyrics-variant-es-001",
        prompt="Escribe una estrofa y un estribillo sobre volver a casa bajo la lluvia.",
        expected_sections=("estrofa", "estribillo"),
    ),
)


@dataclass(frozen=True)
class VariantMeasurement:
    """One aggregate measurement for a quantized variant.

    ``quality_score`` is supplied by the caller.  The built-in scorer is a
    structural proxy, while a listener rubric or task-specific evaluator may
    provide a stronger score.  Keeping the scorer outside this record prevents
    a latency benchmark from being presented as a universal quality claim.
    """

    model_id: str
    quantization: str
    quality_metric: str
    quality_score: Optional[float]
    latency_tokens_per_second: Optional[float]
    disk_bytes: int
    peak_ram_mb: Optional[float]
    peak_vram_mb: Optional[float]
    sample_count: int
    hardware: str
    measured_at: float
    status: str = "completed"
    failure: str = ""

    @property
    def disk_gb(self) -> float:
        return self.disk_bytes / (1024**3)

    @property
    def complete(self) -> bool:
        return (
            self.status == "completed"
            and self.quality_score is not None
            and self.latency_tokens_per_second is not None
            and self.disk_bytes > 0
            and self.sample_count > 0
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = VARIANT_BENCHMARK_SCHEMA_VERSION
        payload["disk_gb"] = round(self.disk_gb, 4)
        payload["complete"] = self.complete
        return payload


def _coerce_runner_output(value: Any) -> tuple[str, int, Optional[float], Optional[float]]:
    """Normalize runner output to text, token count, quality, and VRAM."""
    if isinstance(value, str):
        return value, len(value.split()), None, None
    if not isinstance(value, dict):
        raise TypeError("variant runner must return text or a mapping")
    text = str(value.get("text", value.get("output", "")) or "")
    raw_tokens = value.get("token_count", value.get("tokens", len(text.split())))
    try:
        token_count = max(0, int(raw_tokens))
    except (TypeError, ValueError):
        token_count = len(text.split())
    raw_quality = value.get("quality_score")
    quality = None if raw_quality is None else float(raw_quality)
    raw_vram = value.get("peak_vram_mb")
    vram = None if raw_vram is None else float(raw_vram)
    return text, token_count, quality, vram


def structural_quality_score(
    case: VariantBenchmarkCase,
    output: str,
) -> float:
    """Return a deterministic, intentionally modest structure-quality proxy.

    This is useful for smoke comparisons and CI fixtures; it is not a listener
    score and must be labelled as structural adherence in reports.
    """
    words = len(output.split())
    if words == 0:
        return 0.0
    length_score = min(1.0, words / 32.0)
    lowered = output.casefold()
    expected = tuple(section.casefold() for section in case.expected_sections)
    structure_score = (
        sum(1 for section in expected if section in lowered) / len(expected)
        if expected else 1.0
    )
    return round((length_score * 0.35) + (structure_score * 0.65), 4)


def _hardware_label() -> str:
    """Return a stable human-readable benchmark host label."""
    return f"{platform.system()} {platform.machine()} / {platform.processor() or 'unknown CPU'}"


def measure_variant(
    model_id: str,
    quantization: str,
    model_path: str | Path,
    runner: Callable[[VariantBenchmarkCase], Any],
    *,
    quality_scorer: Callable[[VariantBenchmarkCase, str], float] = structural_quality_score,
    quality_metric: str = "structural_adherence",
    cases: Iterable[VariantBenchmarkCase] = DEFAULT_VARIANT_BENCHMARK_CASES,
    hardware: str | None = None,
) -> VariantMeasurement:
    """Measure a local model variant without importing an optional engine.

    The runner is called once per fixed case and may return ``{"text": ...,
    "token_count": ..., "quality_score": ..., "peak_vram_mb": ...}``.  A
    supplied quality score takes precedence over ``quality_scorer`` for that
    case, which lets a task-specific evaluator or blind-review import its own
    result.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model variant file not found: {path}")
    benchmark_cases = tuple(cases)
    if not benchmark_cases:
        raise ValueError("At least one benchmark case is required")

    quality_scores: list[float] = []
    throughput: list[float] = []
    failures: list[str] = []
    runner_vram: list[float] = []

    with _ResourceSampler() as resources:
        for case in benchmark_cases:
            started = time.perf_counter()
            try:
                text, token_count, supplied_quality, supplied_vram = _coerce_runner_output(runner(case))
                elapsed = max(time.perf_counter() - started, 1e-9)
                if token_count > 0:
                    throughput.append(token_count / elapsed)
                score = supplied_quality
                if score is None:
                    score = float(quality_scorer(case, text))
                if not 0.0 <= score <= 1.0:
                    raise ValueError("quality score must be between 0 and 1")
                quality_scores.append(score)
                if supplied_vram is not None:
                    runner_vram.append(supplied_vram)
            except Exception as exc:  # keep the other fixed cases measurable
                failures.append(f"{case.case_id}: {type(exc).__name__}: {exc}")

    peak_ram_mb = round(resources.peak_ram_bytes / 1e6, 3) if resources.peak_ram_bytes else None
    sampled_vram_mb = (
        round(resources.peak_vram_bytes / 1e6, 3)
        if resources.peak_vram_bytes is not None else None
    )
    peak_vram_mb = max([value for value in (sampled_vram_mb, *runner_vram) if value is not None], default=None)
    status = "completed" if len(quality_scores) == len(benchmark_cases) else (
        "partial" if quality_scores else "failed"
    )
    return VariantMeasurement(
        model_id=model_id,
        quantization=quantization,
        quality_metric=quality_metric,
        quality_score=(round(statistics.fmean(quality_scores), 4) if quality_scores else None),
        latency_tokens_per_second=(
            round(statistics.fmean(throughput), 3) if throughput else None
        ),
        disk_bytes=path.stat().st_size,
        peak_ram_mb=peak_ram_mb,
        peak_vram_mb=peak_vram_mb,
        sample_count=len(quality_scores),
        hardware=hardware or _hardware_label(),
        measured_at=time.time(),
        status=status,
        failure="; ".join(failures),
    )


def compare_variants(measurements: Iterable[VariantMeasurement]) -> list[dict[str, Any]]:
    """Return a stable, UI/CLI-friendly tradeoff table."""
    rows = [measurement.to_dict() for measurement in measurements]
    rows.sort(
        key=lambda row: (
            -(row["quality_score"] if row["quality_score"] is not None else -1.0),
            -(row["latency_tokens_per_second"] or 0.0),
            row["disk_bytes"],
            row["quantization"],
        )
    )
    return rows


def write_variant_measurement(measurement: VariantMeasurement, path: str | Path) -> Path:
    """Atomically persist a benchmark result for later comparison."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(measurement.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(destination)
    return destination
