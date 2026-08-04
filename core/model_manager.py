"""
Slunder Studio — Model Manager
Central singleton managing model lifecycle: download, load, unload, and GPU memory.
Enforces one-large-model-at-a-time GPU residency for 16GB VRAM budget.
"""
import gc
import os
import json
import time
import threading
import hashlib
import importlib.util
from enum import Enum
from typing import Any, Callable, Optional
from pathlib import Path
from dataclasses import dataclass, field, replace

from PySide6.QtCore import QObject, Signal

from core.settings import Settings, get_config_dir
from core.admission import (
    AdmissionBusyError,
    AdmissionCancelledError,
    global_admission_controller,
)
from core.device import configured_cuda_index
from core.trash import TrashEntry, TrashError, TrashManager
from core.model_security import ModelSecurityError
from core.model_signatures import (
    SIGNATURE_MISSING,
    SIGNATURE_UNSIGNED,
    SIGNATURE_VERIFIED,
    SignatureVerification,
    signature_metadata_label,
    verify_oms_signature,
)
from core.engine_contract import (
    ActivationOutcome,
    CapabilityReadiness,
    EngineActivationResult,
    ModelReadiness,
    RunMode,
    get_capability,
)
from core.song_generator_registry import (
    SONG_GENERATOR_REGISTRY,
    default_local_mirror,
)
from core.ace_step_contract import (
    ACE_STEP_CAPABILITIES,
    ACE_STEP_DISPLAY_NAME,
    ACE_STEP_IGNORE_PATTERNS,
    ACE_STEP_LICENSE,
    ACE_STEP_LICENSE_URL,
    ACE_STEP_MODEL_ID,
    ACE_STEP_REVISION,
    ACE_STEP_SOURCE,
)


class OfflineModeError(RuntimeError):
    """Raised when a network operation is attempted while offline mode is enabled."""
    pass


class StaleModelRequestError(RuntimeError):
    """Raised when a load finished after a newer load/unload request superseded it."""
    pass


class ModelReleaseError(RuntimeError):
    """Raised when an active model cannot release its runtime resources."""
    pass


class DownloadInFlightError(RuntimeError):
    """Raised when a model download is already running for the same model."""
    pass


class ModelUpdateError(RuntimeError):
    """Raised when a model update cannot be installed or rolled back safely."""
    pass


EXECUTABLE_MODEL_WARNING = (
    "This model revision contains executable repository code or pickle-backed "
    "weights. Loading it can execute code with your user permissions. Review "
    "the pinned source and revision before granting consent."
)
EXECUTABLE_EXTENSIONS = {".py", ".pyc", ".pyd", ".so", ".dll", ".dylib"}
PICKLE_WEIGHT_EXTENSIONS = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
SAFER_WEIGHT_EXTENSIONS = {".safetensors", ".gguf", ".onnx"}


def is_commit_sha(value: str) -> bool:
    return len(value or "") == 40 and all(char in "0123456789abcdefABCDEF" for char in value)


# ── Model Registry ─────────────────────────────────────────────────────────────

class ModelCategory(str, Enum):
    SONG_FORGE = "song_forge"
    LYRICS = "lyrics"
    MIDI = "midi"
    VOCAL = "vocal"
    SEPARATION = "separation"
    SFX = "sfx"
    ALIGNMENT = "alignment"
    EXTRAS = "extras"


class ModelStatus(str, Enum):
    NOT_DOWNLOADED = "not_downloaded"
    PARTIAL = "partial"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


@dataclass(frozen=True)
class ModelUpdate:
    """An immutable upstream model target discovered by an update check."""

    model_id: str
    source: str
    current_revision: str
    target_revision: str
    status: str = "error"
    release_notes: tuple[str, ...] = ()
    target_title: str = ""
    published_at: str = ""
    checked_at: float = 0.0
    source_url: str = ""
    error: str = ""

    @property
    def available(self) -> bool:
        return (
            self.status == "available"
            and is_commit_sha(self.target_revision)
            and self.target_revision != self.current_revision
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source": self.source,
            "current_revision": self.current_revision,
            "target_revision": self.target_revision,
            "status": self.status,
            "release_notes": list(self.release_notes),
            "target_title": self.target_title,
            "published_at": self.published_at,
            "checked_at": self.checked_at,
            "source_url": self.source_url,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["ModelUpdate"]:
        if not isinstance(data, dict):
            return None
        notes = data.get("release_notes", ())
        if isinstance(notes, str):
            notes = (notes,)
        elif isinstance(notes, (list, tuple)):
            notes = tuple(str(note) for note in notes if str(note).strip())
        else:
            notes = ()
        try:
            checked_at = float(data.get("checked_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            checked_at = 0.0
        return cls(
            model_id=str(data.get("model_id", "")),
            source=str(data.get("source", "")),
            current_revision=str(data.get("current_revision", "")),
            target_revision=str(data.get("target_revision", "")),
            status=str(data.get("status", "error")),
            release_notes=notes,
            target_title=str(data.get("target_title", "")),
            published_at=str(data.get("published_at", "")),
            checked_at=checked_at,
            source_url=str(data.get("source_url", "")),
            error=str(data.get("error", "")),
        )


@dataclass(frozen=True)
class ModelHealthReport:
    """Integrity and optional runtime health result for one model revision."""

    model_id: str
    revision: str
    healthy: bool
    checks: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    checked_at: float = 0.0

    @property
    def is_success(self) -> bool:
        return self.healthy

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "healthy": self.healthy,
            "checks": list(self.checks),
            "errors": list(self.errors),
            "checked_at": self.checked_at,
        }


COMMERCIAL_USE_ALLOWED = "allowed"
COMMERCIAL_USE_TERMS = "check_terms"
COMMERCIAL_USE_LIMITED = "limited"
COMMERCIAL_USE_NON_COMMERCIAL = "non_commercial"
COMMERCIAL_USE_UNKNOWN = "unknown"

COMMERCIAL_USE_LABELS = {
    COMMERCIAL_USE_ALLOWED: "Allowed",
    COMMERCIAL_USE_TERMS: "Check terms",
    COMMERCIAL_USE_LIMITED: "Limited",
    COMMERCIAL_USE_NON_COMMERCIAL: "Non-commercial only",
    COMMERCIAL_USE_UNKNOWN: "Unknown",
}


RECOMMENDATION_TASKS = (
    "best vocal isolation",
    "fastest",
    "lowest vram",
    "best song generation",
    "best lyrics",
    "midi composition",
    "singing voice synthesis",
    "voice conversion",
    "voice cloning",
    "multi-stem separation",
    "sfx generation",
    "alignment",
)


def normalize_task_label(value: str) -> str:
    """Normalize task labels for registry and UI comparisons."""
    return " ".join(str(value or "").strip().lower().split())


def vram_tier_for_gb(vram_gb: float, *, available: bool = True) -> str:
    """Return the published hardware tier used by the recommendation UI."""
    if not available or float(vram_gb or 0) <= 0:
        return "CPU"
    value = float(vram_gb)
    if value <= 4:
        return "≤4 GB"
    if value <= 6:
        return "4–6 GB"
    if value <= 8:
        return "6–8 GB"
    if value <= 12:
        return "8–12 GB"
    if value <= 16:
        return "12–16 GB"
    if value <= 20:
        return "16–20 GB"
    if value <= 24:
        return "20–24 GB"
    return "≥24 GB"


@dataclass
class ModelInfo:
    """Metadata for a registered model."""
    model_id: str
    name: str
    description: str
    category: ModelCategory
    vram_gb: float
    disk_gb: float
    license: str
    source: str  # HuggingFace repo ID
    loader_module: str  # e.g., "engines.ace_step_engine"
    loader_fn: str  # e.g., "load_model"
    is_core: bool = False  # Core models shown prominently in onboarding
    requires: list[str] = field(default_factory=list)  # dependency model IDs
    tags: list[str] = field(default_factory=list)
    allow_patterns: list[str] = field(default_factory=list)  # HF download filter
    ignore_patterns: list[str] = field(default_factory=list)
    revision: str = ""
    pip_managed: bool = False  # True = model managed by pip package, not HF download
    gated: bool = False  # True = requires HF login + license acceptance
    trusted_source: bool = True
    trust_note: str = "Built-in registry source"
    requires_remote_code: bool = False
    allows_unsafe_weights: bool = False
    commercial_use: str = COMMERCIAL_USE_UNKNOWN
    commercial_use_note: str = ""
    license_url: str = ""
    task_labels: list[str] = field(default_factory=list)
    task_scores: dict[str, float] = field(default_factory=dict)
    measurement_basis: str = ""
    measurement_source: str = ""
    measurement_date: str = ""
    vram_tier: str = ""
    cpu_supported: bool = True
    mps_supported: bool = True
    signature_path: str = ""
    signature_identity: str = ""
    signature_oidc_issuer: str = ""
    signature_public_key: str = ""
    signature_certificate_chain: list[str] = field(default_factory=list)
    generator_id: str = ""
    local_mirror: str = ""
    variant_family: str = ""
    quantization: str = ""
    quantization_bits: int = 0
    quality_label: str = ""
    benchmark_quality_score: Optional[float] = None
    benchmark_latency_tokens_per_second: Optional[float] = None
    benchmark_hardware: str = ""
    benchmark_sample_count: int = 0
    benchmark_method: str = ""

    def __post_init__(self) -> None:
        """Give every registry entry a portable, revision-pinned mirror key."""
        if not self.local_mirror:
            revision = self.revision or "package-managed"
            self.local_mirror = default_local_mirror(self.model_id, revision)

    @property
    def commercial_use_label(self) -> str:
        return COMMERCIAL_USE_LABELS.get(self.commercial_use, COMMERCIAL_USE_LABELS[COMMERCIAL_USE_UNKNOWN])

    @property
    def access_label(self) -> str:
        return "Gated / token required" if self.gated else "Open download"

    @property
    def requires_export_warning(self) -> bool:
        return self.commercial_use in {
            COMMERCIAL_USE_TERMS,
            COMMERCIAL_USE_LIMITED,
            COMMERCIAL_USE_NON_COMMERCIAL,
            COMMERCIAL_USE_UNKNOWN,
        }

    @property
    def license_warning(self) -> str:
        if self.commercial_use == COMMERCIAL_USE_NON_COMMERCIAL:
            return "Outputs may not be cleared for commercial use with this model."
        if self.commercial_use == COMMERCIAL_USE_LIMITED:
            return "Commercial use is limited by the model license; verify eligibility before release."
        if self.commercial_use == COMMERCIAL_USE_TERMS:
            return "Commercial use is governed by model-specific terms; review the license before release."
        if self.commercial_use == COMMERCIAL_USE_UNKNOWN:
            return "Commercial-use rights are unknown; review the model license before release."
        return ""

    def license_metadata(self) -> dict[str, Any]:
        return {
            "license": self.license,
            "license_url": self.license_url,
            "commercial_use": self.commercial_use,
            "commercial_use_label": self.commercial_use_label,
            "commercial_use_note": self.commercial_use_note,
            "license_warning": self.license_warning,
            "requires_export_warning": self.requires_export_warning,
            "gated": self.gated,
            "access": self.access_label,
            "generator_id": self.generator_id,
            "local_mirror": self.local_mirror,
        }

    @property
    def advertised_vram_tier(self) -> str:
        """Return the explicit registry tier, falling back to the VRAM estimate."""
        return self.vram_tier or vram_tier_for_gb(self.vram_gb)

    @property
    def variant_label(self) -> str:
        """Return a compact quantization label for cards and diagnostics."""
        if self.quantization:
            return self.quantization
        return "Full precision / package managed"

    @property
    def has_local_benchmark(self) -> bool:
        """Whether this entry carries a complete numeric local benchmark."""
        return (
            self.benchmark_quality_score is not None
            and self.benchmark_latency_tokens_per_second is not None
            and self.benchmark_sample_count > 0
        )


@dataclass(frozen=True)
class HardwareFit:
    """Whether a model is a GPU fit, CPU fallback, or unavailable."""

    status: str
    fits: bool
    tier: str
    reason: str


def model_hardware_fit(
    info: ModelInfo,
    hardware: Optional[dict[str, Any]] = None,
) -> HardwareFit:
    """Classify a model against the detected hardware without loading it."""
    profile = hardware if hardware is not None else get_gpu_info()
    backend = str(profile.get("backend", "cuda" if profile.get("available") else "cpu"))
    available = bool(profile.get("available"))
    total_gb = float(profile.get("total_gb", 0) or 0)

    if backend == "mps":
        if not info.mps_supported:
            return HardwareFit(
                "unsupported",
                False,
                "MPS",
                f"{info.name} does not declare Apple Silicon support.",
            )
        if total_gb > 0 and total_gb < info.vram_gb:
            return HardwareFit(
                "cpu-fallback" if info.cpu_supported else "unsupported",
                False,
                "MPS",
                f"Detected unified memory {total_gb:.1f} GB is below the {info.vram_gb:.1f} GB estimate.",
            )
        return HardwareFit("mps", True, "MPS", "Declared Apple Silicon path fits the detected hardware.")

    if not available:
        if info.cpu_supported:
            return HardwareFit("cpu", True, "CPU", "Runs on CPU; GPU acceleration is unavailable.")
        return HardwareFit("unsupported", False, "CPU", "This model does not declare CPU support.")

    tier = vram_tier_for_gb(total_gb)
    if total_gb >= info.vram_gb:
        return HardwareFit(
            "cuda",
            True,
            tier,
            f"{total_gb:.1f} GB detected; the registry estimate is {info.vram_gb:.1f} GB.",
        )
    if info.cpu_supported:
        return HardwareFit(
            "cpu-fallback",
            False,
            tier,
            f"{total_gb:.1f} GB detected; needs about {info.vram_gb:.1f} GB for the declared GPU path.",
        )
    return HardwareFit(
        "unsupported",
        False,
        tier,
        f"{total_gb:.1f} GB detected; needs about {info.vram_gb:.1f} GB and has no CPU fallback.",
    )


def model_supports_task(info: Optional[ModelInfo], task: str) -> bool:
    """Return whether a registry model carries the requested task label."""
    if info is None:
        return False
    needle = normalize_task_label(task)
    return bool(needle) and any(
        normalize_task_label(label) == needle for label in info.task_labels
    )


def model_tasks(registry: Optional[dict[str, ModelInfo]] = None) -> tuple[str, ...]:
    """Return stable task filter values from the active model registry."""
    values = registry.values() if registry is not None else BUILTIN_MODELS.values()
    labels = {normalize_task_label(label) for info in values for label in info.task_labels}
    ordered = [task for task in RECOMMENDATION_TASKS if task in labels]
    ordered.extend(sorted(labels - set(ordered)))
    return tuple(ordered)


def model_variants(
    model_id_or_family: str,
    registry: Optional[dict[str, ModelInfo]] = None,
) -> list[ModelInfo]:
    """Return all quantization variants for one model or variant family."""
    active_registry = registry if registry is not None else BUILTIN_MODELS
    requested = active_registry.get(model_id_or_family)
    family = requested.variant_family if requested is not None else model_id_or_family
    if not family:
        return [requested] if requested is not None else []
    return sorted(
        [info for info in active_registry.values() if info.variant_family == family],
        key=lambda info: (info.quantization_bits or 999, info.model_id),
    )


def _task_score(info: ModelInfo, task: str) -> float:
    needle = normalize_task_label(task)
    for label, score in info.task_scores.items():
        if normalize_task_label(label) == needle:
            try:
                return float(score)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def recommend_models_for_task(
    task: str,
    registry: Optional[dict[str, ModelInfo]] = None,
    hardware: Optional[dict[str, Any]] = None,
) -> list[ModelInfo]:
    """Rank models for a task, preferring models that fit this hardware."""
    active_registry = registry if registry is not None else BUILTIN_MODELS
    candidates = [info for info in active_registry.values() if model_supports_task(info, task)]
    if not candidates:
        return []

    fits = [info for info in candidates if model_hardware_fit(info, hardware).fits]
    ranked = fits or candidates
    normalized = normalize_task_label(task)
    if normalized == "lowest vram":
        return sorted(
            ranked,
            key=lambda info: (info.vram_gb, -_task_score(info, normalized), info.name.lower()),
        )
    return sorted(
        ranked,
        key=lambda info: (
            -_task_score(info, normalized),
            info.vram_gb,
            info.name.lower(),
        ),
    )


def recommend_model_for_task(
    task: str,
    registry: Optional[dict[str, ModelInfo]] = None,
    hardware: Optional[dict[str, Any]] = None,
) -> Optional[ModelInfo]:
    """Return the top task recommendation, or ``None`` when no label exists."""
    ranked = recommend_models_for_task(task, registry=registry, hardware=hardware)
    return ranked[0] if ranked else None


def hash_file_sha256(path: Path) -> str:
    """Hash a model file for tamper detection."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _serialization_summary(file_hashes: dict[str, str]) -> str:
    suffixes = {Path(path).suffix.lower() for path in file_hashes}
    safer = suffixes & SAFER_WEIGHT_EXTENSIONS
    unsafe = suffixes & PICKLE_WEIGHT_EXTENSIONS
    if safer and unsafe:
        return "mixed"
    if ".safetensors" in safer:
        return "safetensors"
    if ".gguf" in safer:
        return "gguf"
    if ".onnx" in safer:
        return "onnx"
    if unsafe:
        return "pickle"
    return "unknown"


# ── Built-in Model Registry ───────────────────────────────────────────────────

BUILTIN_MODELS: dict[str, ModelInfo] = {
    ACE_STEP_MODEL_ID: ModelInfo(
        model_id=ACE_STEP_MODEL_ID,
        name=ACE_STEP_DISPLAY_NAME,
        description=(
            "Official ACE-Step 1.5 XL Turbo Diffusers synthesizer for 48 kHz "
            "stereo songs, source repainting, reference covers, and extensions."
        ),
        category=ModelCategory.SONG_FORGE,
        vram_gb=16.0,
        disk_gb=10.4,
        license=ACE_STEP_LICENSE,
        source=ACE_STEP_SOURCE,
        loader_module="engines.ace_step_engine",
        loader_fn="load_model",
        is_core=True,
        tags=list(ACE_STEP_CAPABILITIES),
        task_labels=["best song generation", "long-form generation", "source-conditioned editing"],
        task_scores={"best song generation": 1.0},
        measurement_basis=(
            "Official GPU guide: XL is supported with CPU offload at 16–20 GB, "
            "uses about 9 GB for weights, and is fully supported at 20 GB+."
        ),
        measurement_source="https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/GPU_COMPATIBILITY.md",
        measurement_date="2026-08-03",
        vram_tier="16–20 GB",
        ignore_patterns=list(ACE_STEP_IGNORE_PATTERNS),
        revision=ACE_STEP_REVISION,
        trust_note=(
            "Pinned official Diffusers conversion; repository code and the "
            "unused pickle-backed converter artifact are excluded."
        ),
        commercial_use=COMMERCIAL_USE_ALLOWED,
        commercial_use_note=(
            "MIT model weights; generated output rights still depend on prompts "
            "and source material."
        ),
        license_url=ACE_STEP_LICENSE_URL,
        generator_id="ace-step",
    ),
    "llama-3.1-8b-q4": ModelInfo(
        model_id="llama-3.1-8b-q4",
        name="LLaMA 3.1 8B Instruct (Q4_K_M)",
        description="High-quality lyrics generation and AI Producer planning. Best quality/VRAM ratio.",
        category=ModelCategory.LYRICS,
        vram_gb=5.0,
        disk_gb=4.9,
        license="Llama 3.1 Community",
        source="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        loader_module="engines.lyrics_engine",
        loader_fn="load_model",
        is_core=True,
        tags=["lyrics", "text", "creative writing", "LLM"],
        task_labels=["best lyrics"],
        task_scores={"best lyrics": 1.0},
        measurement_basis="Published Q4 model-card footprint and the registry's 5.0 GB runtime estimate; not an independent Slunder benchmark.",
        measurement_source="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        measurement_date="2026-08-03",
        vram_tier="4–6 GB",
        allow_patterns=["Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"],
        revision="bf5b95e96dac0462e2a09145ec66cae9a3f12067",
        commercial_use=COMMERCIAL_USE_TERMS,
        commercial_use_note="Meta Llama Community License allows commercial use subject to model terms and policy.",
        license_url="https://www.llama.com/llama3_1/license/",
    ),
    "llama-3.2-3b-q4": ModelInfo(
        model_id="llama-3.2-3b-q4",
        name="LLaMA 3.2 3B Instruct (Q4_K_M)",
        description="Fast lyrics generation for quick iteration. Lower VRAM, slightly lower quality.",
        category=ModelCategory.LYRICS,
        vram_gb=2.5,
        disk_gb=2.0,
        license="Llama 3.2 Community",
        source="bartowski/Llama-3.2-3B-Instruct-GGUF",
        loader_module="engines.lyrics_engine",
        loader_fn="load_model",
        tags=["lyrics", "text", "fast", "LLM"],
        task_labels=["best lyrics", "fastest", "lowest vram"],
        task_scores={"best lyrics": 0.8, "fastest": 1.0},
        measurement_basis="Published Q4 model-card footprint and the registry's 2.5 GB runtime estimate; lower-resource option, not an independent speed benchmark.",
        measurement_source="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
        measurement_date="2026-08-03",
        vram_tier="≤4 GB",
        allow_patterns=["Llama-3.2-3B-Instruct-Q4_K_M.gguf"],
        revision="5ab33fa94d1d04e903623ae72c95d1696f09f9e8",
        commercial_use=COMMERCIAL_USE_TERMS,
        commercial_use_note="Meta Llama Community License allows commercial use subject to model terms and policy.",
        license_url="https://www.llama.com/llama3_2/license/",
    ),
    "qwen-2.5-14b-q4": ModelInfo(
        model_id="qwen-2.5-14b-q4",
        name="Qwen 2.5 14B Instruct (Q4_K_M)",
        description="Premium lyrics quality with 29+ language support. Requires 10GB VRAM.",
        category=ModelCategory.LYRICS,
        vram_gb=10.0,
        disk_gb=8.5,
        license="Apache 2.0",
        source="bartowski/Qwen2.5-14B-Instruct-GGUF",
        loader_module="engines.lyrics_engine",
        loader_fn="load_model",
        tags=["lyrics", "multilingual", "premium", "LLM"],
        task_labels=["best lyrics"],
        task_scores={"best lyrics": 1.1},
        measurement_basis="Published Q4 model-card footprint and the registry's 10.0 GB runtime estimate; quality-oriented option, not an independent Slunder benchmark.",
        measurement_source="https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF",
        measurement_date="2026-08-03",
        vram_tier="8–12 GB",
        allow_patterns=["Qwen2.5-14B-Instruct-Q4_K_M.gguf"],
        revision="05244aa5d871c661c80082a15d3bce44714d068d",
        commercial_use=COMMERCIAL_USE_ALLOWED,
        commercial_use_note="Apache 2.0 model license.",
        license_url="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct",
    ),
    "midi-llm-1b": ModelInfo(
        model_id="midi-llm-1b",
        name="MIDI-LLM (Llama 3.2 1B)",
        description="Text-to-MIDI multitrack composition. Generates jazz, classical, pop, and more.",
        category=ModelCategory.MIDI,
        vram_gb=3.0,
        disk_gb=2.8,
        license="Llama 3.2 Community",
        source="slseanwu/MIDI-LLM_Llama-3.2-1B",
        loader_module="engines.midi_llm_engine",
        loader_fn="load_model",
        is_core=True,
        tags=["MIDI", "composition", "multitrack", "instrumental"],
        task_labels=["midi composition", "fastest"],
        task_scores={"midi composition": 1.0, "fastest": 0.8},
        measurement_basis="Published model snapshot and the registry's 3.0 GB runtime estimate; no independent latency claim is made.",
        measurement_source="https://huggingface.co/slseanwu/MIDI-LLM_Llama-3.2-1B",
        measurement_date="2026-08-03",
        vram_tier="≤4 GB",
        revision="8b82ab9ec144348900e9ea4623b123e0b12f60b3",
        ignore_patterns=["*.bin", "*.pt", "*.pth", "*.ckpt", "*.pkl", "*.pickle"],
        commercial_use=COMMERCIAL_USE_TERMS,
        commercial_use_note="Derived from Llama 3.2; commercial use is governed by Meta Llama terms.",
        license_url="https://www.llama.com/llama3_2/license/",
    ),
    "diffsinger": ModelInfo(
        model_id="diffsinger",
        name="DiffSinger (openvpi)",
        description="Singing voice synthesis from MIDI score + lyrics. Install via pip, voice models downloaded separately.",
        category=ModelCategory.VOCAL,
        vram_gb=5.0,
        disk_gb=1.0,
        license="Apache 2.0",
        source="",
        loader_module="engines.diffsinger_engine",
        loader_fn="load_model",
        pip_managed=True,
        tags=["singing", "voice synthesis", "MIDI"],
        task_labels=["singing voice synthesis", "lowest vram"],
        task_scores={"singing voice synthesis": 1.0},
        measurement_basis="Published engine/runtime requirements plus the registry's 5.0 GB estimate; voice checkpoints vary and are not independently benchmarked here.",
        measurement_source="https://github.com/openvpi/DiffSinger",
        measurement_date="2026-08-03",
        vram_tier="4–6 GB",
        commercial_use=COMMERCIAL_USE_ALLOWED,
        commercial_use_note="Apache 2.0 engine; individual voice models may carry separate terms.",
        license_url="https://github.com/openvpi/DiffSinger",
    ),
    "rvc-v2": ModelInfo(
        model_id="rvc-v2",
        name="RVC v2",
        description="Voice timbre conversion. Transform any vocal to a target voice.",
        category=ModelCategory.VOCAL,
        vram_gb=3.0,
        disk_gb=0.3,
        license="MIT",
        source="lj1995/VoiceConversionWebUI",
        loader_module="engines.rvc_engine",
        loader_fn="load_model",
        tags=["voice conversion", "AI cover", "timbre"],
        task_labels=["voice conversion", "fastest", "lowest vram"],
        task_scores={"voice conversion": 1.0, "fastest": 0.7},
        measurement_basis="Published RVC runtime footprint and the registry's 3.0 GB estimate; profile/checkpoint size varies.",
        measurement_source="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI",
        measurement_date="2026-08-03",
        vram_tier="≤4 GB",
        allow_patterns=["hubert_base.pt", "rmvpe.pt", "pretrained_v2/*"],
        revision="5836e9ea8ad6b7852f906acfa440e65a36e72396",
        allows_unsafe_weights=True,
        trust_note="Pinned upstream revision; pickle-backed weights require per-revision consent.",
        commercial_use=COMMERCIAL_USE_ALLOWED,
        commercial_use_note="MIT engine; individual voice profiles require separate consent metadata.",
        license_url="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI",
    ),
    "gpt-sovits-v2": ModelInfo(
        model_id="gpt-sovits-v2",
        name="GPT-SoVITS v2",
        description="Zero-shot voice cloning from 5-second reference. Supports speech and singing.",
        category=ModelCategory.VOCAL,
        vram_gb=6.0,
        disk_gb=2.5,
        license="MIT",
        source="lj1995/GPT-SoVITS",
        loader_module="engines.rvc_engine",
        loader_fn="load_model",
        tags=["voice cloning", "TTS", "zero-shot"],
        task_labels=["voice cloning"],
        task_scores={"voice cloning": 1.0},
        measurement_basis="Published GPT-SoVITS runtime footprint and the registry's 6.0 GB estimate; reference/model size varies.",
        measurement_source="https://github.com/RVC-Boss/GPT-SoVITS",
        measurement_date="2026-08-03",
        vram_tier="4–6 GB",
        revision="336b2ec4e8d4ac74740798dd40af44e74659ecaf",
        allows_unsafe_weights=True,
        trust_note="Pinned upstream revision; pickle-backed weights require per-revision consent.",
        commercial_use=COMMERCIAL_USE_ALLOWED,
        commercial_use_note="MIT engine; individual voice profiles require separate consent metadata.",
        license_url="https://github.com/RVC-Boss/GPT-SoVITS",
    ),
    "demucs-v4": ModelInfo(
        model_id="demucs-v4",
        name="Demucs v4 (htdemucs)",
        description="Audio source separation into vocals, drums, bass, and other stems. Managed by demucs pip package.",
        category=ModelCategory.SEPARATION,
        vram_gb=4.0,
        disk_gb=0.08,
        license="MIT",
        source="",
        loader_module="engines.demucs_engine",
        loader_fn="load_model",
        pip_managed=True,
        tags=["stem separation", "vocals", "drums", "remixing"],
        task_labels=["multi-stem separation", "fastest", "lowest vram"],
        task_scores={"multi-stem separation": 1.0, "fastest": 0.9},
        measurement_basis="Audio Separator's published Demucs listing reports vocals 10.0, drums 9.4, bass 11.3 SDR for htdemucs; registry run estimate is 4.0 GB VRAM.",
        measurement_source="https://github.com/nomadkaraoke/python-audio-separator",
        measurement_date="2026-08-03",
        vram_tier="≤4 GB",
        commercial_use=COMMERCIAL_USE_ALLOWED,
        commercial_use_note="MIT model package.",
        license_url="https://github.com/facebookresearch/demucs",
    ),
    "audio-separator": ModelInfo(
        model_id="audio-separator",
        name="Audio Separator (MDX/MDXC/Roformer)",
        description="Maintained adapter for selectable UVR-family and Roformer source-separation checkpoints.",
        category=ModelCategory.SEPARATION,
        vram_gb=4.0,
        disk_gb=1.0,
        license="MIT wrapper; checkpoint terms vary",
        source="nomadkaraoke/python-audio-separator",
        loader_module="engines.audio_separator_engine",
        loader_fn="load_model",
        pip_managed=True,
        tags=["stem separation", "MDX", "MDXC", "Roformer", "UVR"],
        task_labels=["best vocal isolation", "vocal separation"],
        task_scores={"best vocal isolation": 12.9755},
        measurement_basis="Audio Separator's published model listing reports BS-Roformer vocals SDR 12.9 and instrumental SDR 17.0; the checkpoint registry estimates 4.0 GB VRAM.",
        measurement_source="https://github.com/nomadkaraoke/python-audio-separator",
        measurement_date="2026-08-03",
        vram_tier="≤4 GB",
        trusted_source=True,
        trust_note="The adapter is MIT; each downloaded checkpoint must be reviewed separately.",
        commercial_use=COMMERCIAL_USE_UNKNOWN,
        commercial_use_note="Checkpoint license and commercial-use rights are not inferred from the MIT wrapper.",
        license_url="https://github.com/nomadkaraoke/python-audio-separator",
    ),
    "stable-audio-open": ModelInfo(
        model_id="stable-audio-open",
        name="Stable Audio Open",
        description="Text-to-SFX generation. Ambient textures, risers, drops, up to 47s. Requires HuggingFace login.",
        category=ModelCategory.SFX,
        vram_gb=8.0,
        disk_gb=2.5,
        license="Stability Community",
        source="stabilityai/stable-audio-open-1.0",
        loader_module="engines.sfx_engine",
        loader_fn="load_model",
        tags=["SFX", "sound effects", "ambient", "foley"],
        task_labels=["sfx generation"],
        task_scores={"sfx generation": 1.0},
        measurement_basis="Published Stable Audio Open model-card footprint and the registry's 8.0 GB runtime estimate; no independent Slunder quality benchmark.",
        measurement_source="https://huggingface.co/stabilityai/stable-audio-open-1.0",
        measurement_date="2026-08-03",
        vram_tier="6–8 GB",
        gated=True,
        revision="f21265c1e2710b3bd2386596943f0007f55f802e",
        ignore_patterns=["*.bin", "*.pt", "*.pth", "*.ckpt", "*.pkl", "*.pickle"],
        commercial_use=COMMERCIAL_USE_LIMITED,
        commercial_use_note="Stability Community License; commercial use has eligibility limits and requires license acceptance.",
        license_url="https://huggingface.co/stabilityai/stable-audio-open-1.0",
    ),
    "whisper-tiny": ModelInfo(
        model_id="whisper-tiny",
        name="Whisper tiny",
        description="Lyrics-to-audio alignment and transcription. Very lightweight.",
        category=ModelCategory.ALIGNMENT,
        vram_gb=1.0,
        disk_gb=0.15,
        license="MIT",
        source="openai/whisper-tiny",
        loader_module="engines.audio_analyzer",
        loader_fn="load_model",
        tags=["alignment", "transcription", "lyrics sync"],
        task_labels=["alignment", "lowest vram"],
        task_scores={"alignment": 1.0},
        measurement_basis="Published Whisper tiny model-card footprint and the registry's 1.0 GB runtime estimate.",
        measurement_source="https://huggingface.co/openai/whisper-tiny",
        measurement_date="2026-08-03",
        vram_tier="≤4 GB",
        revision="169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        # The loader uses the repository's safe Transformers checkpoint;
        # pickle-backed pytorch_model.bin remains excluded.
        ignore_patterns=["*.bin"],
        commercial_use=COMMERCIAL_USE_ALLOWED,
        commercial_use_note="MIT model license.",
    ),
    "musicgen-medium": ModelInfo(
        model_id="musicgen-medium",
        name="MusicGen Medium",
        description="Quick 30-second instrumental sketches from text prompts.",
        category=ModelCategory.EXTRAS,
        vram_gb=5.0,
        disk_gb=3.3,
        license="CC-BY-NC",
        source="facebook/musicgen-medium",
        loader_module="engines.ace_step_engine",
        loader_fn="load_model",
        tags=["instrumental", "short clips", "sketching"],
        task_labels=["fastest", "lowest vram", "instrumental sketches"],
        task_scores={"fastest": 0.6},
        measurement_basis="Published MusicGen model-card footprint and the registry's 5.0 GB runtime estimate; this is a non-commercial sketching path.",
        measurement_source="https://huggingface.co/facebook/musicgen-medium",
        measurement_date="2026-08-03",
        vram_tier="4–6 GB",
        revision="d3bd7b00761b78ad7a8a05145ee31e7832e9916c",
        allows_unsafe_weights=True,
        trust_note="Pinned upstream revision; pickle-backed weights require per-revision consent.",
        commercial_use=COMMERCIAL_USE_NON_COMMERCIAL,
        commercial_use_note="CC-BY-NC model weights are not cleared for commercial use.",
        license_url="https://huggingface.co/facebook/musicgen-medium",
    ),
}


def _register_song_generator_models() -> None:
    """Project active generator configurations into the model registry."""
    for config in SONG_GENERATOR_REGISTRY.values():
        if not config.enabled:
            continue
        existing = BUILTIN_MODELS.get(config.model_id)
        if existing is not None:
            identity = {
                "source": (existing.source, config.source),
                "revision": (existing.revision, config.revision),
                "loader_module": (existing.loader_module, config.adapter_module),
                "loader_fn": (existing.loader_fn, config.loader_fn),
                "local_mirror": (existing.local_mirror, config.local_mirror),
            }
            mismatches = [
                name for name, (actual, expected) in identity.items()
                if actual != expected
            ]
            if mismatches:
                raise RuntimeError(
                    f"Song generator {config.model_id} disagrees with the model "
                    f"registry: {', '.join(mismatches)}."
                )
            existing.generator_id = config.generator_id
            continue

        BUILTIN_MODELS[config.model_id] = ModelInfo(
            model_id=config.model_id,
            name=config.display_name,
            description=config.description or f"{config.display_name} song generator.",
            category=ModelCategory.SONG_FORGE,
            vram_gb=config.vram_gb,
            disk_gb=config.disk_gb,
            license=config.license,
            source=config.source,
            loader_module=config.adapter_module,
            loader_fn=config.loader_fn,
            is_core=config.is_core,
            tags=list(config.capabilities),
            task_labels=["best song generation"],
            task_scores={"best song generation": 1.0},
            revision=config.revision,
            vram_tier=vram_tier_for_gb(config.vram_gb),
            trust_note=(
                "Registry-declared generator with a pinned source revision and "
                "portable local mirror identity."
            ),
            commercial_use=config.commercial_use,
            license_url=config.license_url,
            generator_id=config.generator_id,
            local_mirror=config.local_mirror,
        )


_register_song_generator_models()


def _register_gguf_quantized_variants() -> None:
    """Register paired Q4/Q8 GGUF entries with honest benchmark provenance."""
    specs = (
        {
            "base_id": "llama-3.1-8b-q4",
            "family": "llama-3.1-8b-instruct",
            "quantization": "Q4_K_M",
            "bits": 4,
            "quality": "Good / recommended",
            "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        },
        {
            "base_id": "llama-3.2-3b-q4",
            "family": "llama-3.2-3b-instruct",
            "quantization": "Q4_K_M",
            "bits": 4,
            "quality": "Good / recommended",
            "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        },
        {
            "base_id": "qwen-2.5-14b-q4",
            "family": "qwen-2.5-14b-instruct",
            "quantization": "Q4_K_M",
            "bits": 4,
            "quality": "Good / recommended",
            "file": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        },
    )
    q8_specs = (
        {
            "base_id": "llama-3.1-8b-q4",
            "model_id": "llama-3.1-8b-q8",
            "family": "llama-3.1-8b-instruct",
            "file": "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            "disk_gb": 7.9542,
            "vram_gb": 9.0,
            "quality": "Extremely high",
        },
        {
            "base_id": "llama-3.2-3b-q4",
            "model_id": "llama-3.2-3b-q8",
            "family": "llama-3.2-3b-instruct",
            "file": "Llama-3.2-3B-Instruct-Q8_0.gguf",
            "disk_gb": 3.1869,
            "vram_gb": 4.0,
            "quality": "Extremely high",
        },
        {
            "base_id": "qwen-2.5-14b-q4",
            "model_id": "qwen-2.5-14b-q8",
            "family": "qwen-2.5-14b-instruct",
            "file": "Qwen2.5-14B-Instruct-Q8_0.gguf",
            "disk_gb": 14.6233,
            "vram_gb": 16.0,
            "quality": "Extremely high",
        },
    )

    benchmark_method = (
        "Upstream imatrix model-card quality label and exact published file size; "
        "run core.model_variants.measure_variant for local numeric quality, token "
        "latency, RAM, and peak VRAM measurements."
    )
    for spec in specs:
        info = BUILTIN_MODELS[spec["base_id"]]
        BUILTIN_MODELS[spec["base_id"]] = replace(
            info,
            variant_family=spec["family"],
            quantization=spec["quantization"],
            quantization_bits=spec["bits"],
            quality_label=spec["quality"],
            benchmark_method=benchmark_method,
        )

    for spec in q8_specs:
        base = BUILTIN_MODELS[spec["base_id"]]
        BUILTIN_MODELS[spec["model_id"]] = replace(
            base,
            model_id=spec["model_id"],
            name=f"{base.name.split(' (', 1)[0]} (Q8_0)",
            description=(
                f"Higher-quality 8-bit variant of {base.name.split(' (', 1)[0]}; "
                "uses more disk and VRAM for lower quantization loss."
            ),
            is_core=False,
            vram_gb=spec["vram_gb"],
            disk_gb=spec["disk_gb"],
            allow_patterns=[spec["file"]],
            variant_family=spec["family"],
            quantization="Q8_0",
            quantization_bits=8,
            quality_label=spec["quality"],
            measurement_basis=(
                f"Hugging Face model card labels Q8_0 as {spec['quality'].lower()}; "
                f"the immutable file is {spec['disk_gb']:.4f} GiB. "
                "Local latency, quality, and peak-VRAM figures are not inferred "
                "from this catalog entry."
            ),
            measurement_date="2026-08-03",
            benchmark_method=benchmark_method,
            local_mirror="",
        )


_register_gguf_quantized_variants()


MODEL_RUNTIME_PACKAGES: dict[str, tuple[tuple[str, str], ...]] = {
    ACE_STEP_MODEL_ID: (
        ("torch", "torch"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
    ),
    "llama-3.1-8b-q4": (("llama-cpp-python", "llama_cpp"),),
    "llama-3.1-8b-q8": (("llama-cpp-python", "llama_cpp"),),
    "llama-3.2-3b-q4": (("llama-cpp-python", "llama_cpp"),),
    "llama-3.2-3b-q8": (("llama-cpp-python", "llama_cpp"),),
    "qwen-2.5-14b-q4": (("llama-cpp-python", "llama_cpp"),),
    "qwen-2.5-14b-q8": (("llama-cpp-python", "llama_cpp"),),
    "midi-llm-1b": (
        ("torch", "torch"),
        ("transformers", "transformers"),
    ),
    "diffsinger": (("onnxruntime", "onnxruntime"),),
    "rvc-v2": (("torch", "torch"),),
    "gpt-sovits-v2": (("torch", "torch"),),
    "demucs-v4": (
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("demucs", "demucs"),
    ),
    "audio-separator": (
        ("torch", "torch"),
        ("onnxruntime", "onnxruntime"),
        ("audio_separator", "audio-separator"),
    ),
    "stable-audio-open": (
        ("torch", "torch"),
        ("stable-audio-tools", "stable_audio_tools"),
    ),
    "whisper-tiny": (
        ("torch", "torch"),
        ("transformers", "transformers"),
    ),
    "musicgen-medium": (
        ("torch", "torch"),
        ("transformers", "transformers"),
    ),
}


# ── GPU Utilities ──────────────────────────────────────────────────────────────

_TORCH_MODULE_UNSET = object()
_torch_module_cache: Any = _TORCH_MODULE_UNSET


def _get_torch_module():
    """Import torch at most once, retaining a failed-import sentinel."""
    global _torch_module_cache
    if _torch_module_cache is _TORCH_MODULE_UNSET:
        try:
            import torch
        except (ImportError, OSError):
            _torch_module_cache = None
        else:
            _torch_module_cache = torch
    return _torch_module_cache


def get_gpu_info() -> dict:
    """Get GPU VRAM info. Returns dict with total_gb, used_gb, free_gb, name."""
    try:
        torch = _get_torch_module()
        if torch is not None and torch.cuda.is_available():
            index = configured_cuda_index(torch)
            props = torch.cuda.get_device_properties(index)
            total = props.total_memory / (1024**3)
            reserved = torch.cuda.memory_reserved(index) / (1024**3)
            allocated = torch.cuda.memory_allocated(index) / (1024**3)
            return {
                "available": True,
                "backend": "cuda",
                "name": props.name,
                "index": index,
                "total_gb": round(total, 1),
                "used_gb": round(allocated, 1),
                "reserved_gb": round(reserved, 1),
                "free_gb": round(total - reserved, 1),
            }
        if torch is not None:
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is not None and mps.is_available():
                return {
                    "available": True,
                    "backend": "mps",
                    "name": "Apple Silicon (MPS)",
                    "index": 0,
                    "total_gb": 0,
                    "used_gb": 0,
                    "reserved_gb": 0,
                    "free_gb": 0,
                }
    except (ImportError, RuntimeError, AttributeError):
        pass

    return {
        "available": False,
        "backend": "cpu",
        "name": "No GPU detected",
        "total_gb": 0,
        "used_gb": 0,
        "reserved_gb": 0,
        "free_gb": 0,
    }


def cleanup_gpu():
    """Aggressive GPU memory cleanup."""
    try:
        torch = _get_torch_module()
        if torch is not None and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except (ImportError, RuntimeError):
        pass


# ── Model Manager ──────────────────────────────────────────────────────────────

class ModelManager(QObject):
    """
    Central model lifecycle manager. Singleton.

    Enforces one-large-model-at-a-time GPU residency.
    Provides download, load, unload, and status tracking for all models.

    Signals:
        gpu_status_changed(dict)   - GPU info dict updated
        status_changed(str, str)   - (model_id, new_status)
    """
    gpu_status_changed = Signal(dict)
    status_changed = Signal(str, str)

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._initialized = True

        self._registry: dict[str, ModelInfo] = dict(BUILTIN_MODELS)
        self._status: dict[str, ModelStatus] = {}
        self._current_model_id: Optional[str] = None
        self._current_model: Any = None
        self._readiness_cache: dict[str, ModelReadiness] = {}
        self._readiness_cache_state = None
        self._disk_usage_cache: Optional[float] = None
        self._disk_usage_cache_path: Optional[str] = None
        self._settings = Settings()
        self._trash = TrashManager()
        self._restore_installed_revisions()

        # Guards every read/write of _status, _current_model_id, _current_model,
        # and the lifecycle request ticket. Held only for short critical
        # sections; signals are always emitted outside it.
        self._state_lock = threading.RLock()
        # Serializes the expensive part of a load so only one model is ever
        # being brought onto the GPU at a time.
        self._load_lock = threading.RLock()
        # Monotonic ticket. Every load/unload request claims one; a loader whose
        # ticket is no longer the pending one must discard its result instead of
        # overwriting a newer request.
        self._request_counter = 0
        self._pending_request = 0
        self._downloads_in_flight: set[str] = set()
        self._model_errors: dict[str, str] = {}

        # Initialize status for all registered models
        for model_id in self._registry:
            if self._is_model_cached(model_id):
                self._status[model_id] = ModelStatus.DOWNLOADED
            elif self.has_partial_download(model_id):
                self._status[model_id] = ModelStatus.PARTIAL
            else:
                self._status[model_id] = ModelStatus.NOT_DOWNLOADED

    # ── Registry ───────────────────────────────────────────────────────────────

    @property
    def registry(self) -> dict[str, ModelInfo]:
        return self._registry

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        return self._registry.get(model_id)

    def _restore_installed_revisions(self) -> None:
        """Restore only source-matched immutable revisions from the settings ledger."""
        records = self._settings.get("model_hub.installed_revisions", {}) or {}
        if not isinstance(records, dict):
            return
        for model_id, raw in records.items():
            info = self._registry.get(model_id)
            if info is None:
                continue
            if isinstance(raw, dict):
                revision = str(raw.get("revision", ""))
                source = str(raw.get("source", ""))
            else:
                revision = str(raw or "")
                source = info.source
            if (
                not revision
                or not is_commit_sha(revision)
                or source != info.source
            ):
                continue
            if revision == info.revision:
                continue
            self._registry[model_id] = replace(
                info,
                revision=revision,
                local_mirror=default_local_mirror(model_id, revision),
            )

    def _set_installed_revision(self, model_id: str, revision: str) -> None:
        """Persist and expose one verified revision without changing built-in code."""
        info = self._registry.get(model_id)
        if info is None or not is_commit_sha(revision):
            raise ModelUpdateError(f"Invalid immutable revision for {model_id}")
        base = BUILTIN_MODELS.get(model_id)
        records = self._settings.get("model_hub.installed_revisions", {}) or {}
        if not isinstance(records, dict):
            records = {}
        if base is not None and revision == base.revision:
            records.pop(model_id, None)
        else:
            records[model_id] = {
                "revision": revision,
                "source": info.source,
                "updated_at": time.time(),
            }
        self._settings.set("model_hub.installed_revisions", records)
        self._registry[model_id] = replace(
            info,
            revision=revision,
            local_mirror=default_local_mirror(model_id, revision),
        )
        self._readiness_cache.clear()
        self._readiness_cache_state = None

    def get_model_update(self, model_id: str) -> Optional[ModelUpdate]:
        """Return the last persisted update check for a model."""
        raw = self._settings.get("model_hub.update_checks", {}) or {}
        if not isinstance(raw, dict):
            return None
        update = ModelUpdate.from_dict(raw.get(model_id))
        info = self._registry.get(model_id)
        if update is None or info is None or update.source != info.source:
            return None
        return update

    def get_model_rollback(self, model_id: str) -> Optional[dict[str, Any]]:
        """Return the recoverable last-good revision metadata, if one exists."""
        raw = self._settings.get("model_hub.update_backups", {}) or {}
        if not isinstance(raw, dict) or not isinstance(raw.get(model_id), dict):
            return None
        backup = dict(raw[model_id])
        if not backup.get("trash_entry_id") or not is_commit_sha(str(backup.get("revision", ""))):
            return None
        return backup

    def get_model_license_metadata(self, model_id: str) -> dict[str, Any]:
        info = self.get_model_info(model_id)
        metadata = info.license_metadata() if info else {
            "license": "unknown",
            "license_url": "",
            "commercial_use": COMMERCIAL_USE_UNKNOWN,
            "commercial_use_label": COMMERCIAL_USE_LABELS[COMMERCIAL_USE_UNKNOWN],
            "commercial_use_note": "",
            "license_warning": "Commercial-use rights are unknown; review the model license before release.",
            "requires_export_warning": True,
            "gated": False,
            "access": "Unknown",
            "generator_id": "",
            "local_mirror": "",
        }
        manifest = self.get_download_manifest(model_id)
        if manifest:
            for key in (
                "license",
                "license_url",
                "commercial_use",
                "commercial_use_label",
                "commercial_use_note",
                "license_warning",
                "requires_export_warning",
                "gated",
                "access",
                "generator_id",
                "local_mirror",
            ):
                if key in manifest and manifest[key] not in ("", None):
                    metadata[key] = manifest[key]
        return metadata

    @staticmethod
    def _license_consent_key(info: ModelInfo) -> str:
        revision = info.revision or "package-managed"
        return f"{info.model_id}@{revision}"

    def has_license_acceptance(self, model_id: str) -> bool:
        """Return whether this exact model revision is cleared for use."""
        info = self.get_model_info(model_id)
        if info is None:
            return False
        if info.license.strip() and info.commercial_use == COMMERCIAL_USE_ALLOWED:
            return True
        consents = self._settings.get("model_hub.license_consents", {}) or {}
        consent = consents.get(self._license_consent_key(info), {}) if isinstance(consents, dict) else {}
        return bool(
            isinstance(consent, dict)
            and consent.get("approved") is True
            and consent.get("source") == info.source
            and consent.get("revision") == info.revision
            and consent.get("license") == info.license
            and consent.get("commercial_use") == info.commercial_use
        )

    def record_license_acceptance(self, model_id: str, accepted: bool = True) -> None:
        """Persist acceptance for one exact source/revision and its license."""
        info = self.get_model_info(model_id)
        if info is None:
            raise ModelSecurityError(f"Unknown model: {model_id}")
        consents = self._settings.get("model_hub.license_consents", {}) or {}
        if not isinstance(consents, dict):
            consents = {}
        key = self._license_consent_key(info)
        if accepted:
            consents[key] = {
                "approved": True,
                "source": info.source,
                "revision": info.revision,
                "license": info.license,
                "commercial_use": info.commercial_use,
            }
        else:
            consents.pop(key, None)
        self._settings.set("model_hub.license_consents", consents)

    def require_model_license_acceptance(self, model_id: str) -> None:
        """Fail closed for non-commercial or unlicensed model weights."""
        info = self.get_model_info(model_id)
        if info is None:
            raise ModelSecurityError(f"Unknown model: {model_id}")
        if not self.has_license_acceptance(model_id):
            license_name = info.license.strip() or "unlicensed weights"
            raise ModelSecurityError(
                f"{info.name} requires explicit acceptance of {license_name} "
                f"for revision {info.revision or 'package-managed'}."
            )

    def get_model_signature_metadata(self, model_id: str) -> dict[str, Any]:
        """Return the persisted OMS state without treating an unsigned model as verified."""
        info = self.get_model_info(model_id)
        manifest = self.get_download_manifest(model_id)
        if manifest:
            metadata = {
                "signature_status": manifest.get("signature_status", SIGNATURE_UNSIGNED),
                "signature_reason": manifest.get("signature_reason", ""),
                "signature_path": manifest.get("signature_path", ""),
                "signature_identity": manifest.get("signature_identity", ""),
                "signature_oidc_issuer": manifest.get("signature_oidc_issuer", ""),
                "signature_verifier": manifest.get("signature_verifier", ""),
            }
        else:
            expected = bool(info and info.signature_path)
            metadata = {
                "signature_status": SIGNATURE_MISSING if expected else SIGNATURE_UNSIGNED,
                "signature_reason": (
                    "An OMS signature is expected when this model is downloaded."
                    if expected
                    else "No OMS signature was published with this model revision."
                ),
                "signature_path": info.signature_path if info else "",
                "signature_identity": info.signature_identity if info else "",
                "signature_oidc_issuer": info.signature_oidc_issuer if info else "",
                "signature_verifier": (
                    "OMS public key"
                    if info and info.signature_public_key
                    else "OMS certificate chain"
                    if info and info.signature_certificate_chain
                    else "Sigstore identity"
                    if info and info.signature_identity and info.signature_oidc_issuer
                    else ""
                ),
            }
        metadata["label"] = signature_metadata_label(metadata)
        return metadata

    def get_status(self, model_id: str) -> ModelStatus:
        with self._state_lock:
            return self._status.get(model_id, ModelStatus.NOT_DOWNLOADED)

    def get_model_error(self, model_id: str) -> str:
        """Return the latest actionable lifecycle error for a model, if any."""
        with self._state_lock:
            return self._model_errors.get(model_id, "")

    def get_missing_runtime_packages(self, model_id: str) -> tuple[str, ...]:
        """Return declared runtime packages that cannot currently be imported."""
        return tuple(
            display_name
            for display_name, module_name in MODEL_RUNTIME_PACKAGES.get(
                model_id,
                (),
            )
            if importlib.util.find_spec(module_name) is None
        )

    def get_model_readiness(self, model_id: str) -> ModelReadiness:
        """Report installation, verification, loadability, and activation separately."""
        cache_state = self._readiness_state_token()
        if cache_state != self._readiness_cache_state:
            self._readiness_cache.clear()
            self._readiness_cache_state = cache_state
        cached = self._readiness_cache.get(model_id)
        if cached is not None:
            return cached

        info = self._registry.get(model_id)
        if info is None:
            readiness = ModelReadiness(
                model_id=model_id,
                installed=False,
                verified=False,
                loadable=False,
                active=False,
                status="unknown",
                remedy=f"Unknown model: {model_id}",
            )
            self._readiness_cache[model_id] = readiness
            return readiness

        with self._state_lock:
            active = bool(
                self._current_model_id == model_id
                and self._current_model is not None
                and self._status.get(model_id) == ModelStatus.LOADED
            )
        missing = self.get_missing_runtime_packages(model_id)
        if info.pip_managed:
            installed = not missing
            verified = installed
        else:
            installed = self._is_model_cached(model_id)
            verified = installed and self.verify_download(
                model_id,
                full_hash=False,
            )[0]

        consent_ready = not (
            info.requires_remote_code or info.allows_unsafe_weights
        ) or self.has_executable_model_consent(model_id)
        loadable = bool(
            active
            or (
                installed
                and verified
                and not missing
                and consent_ready
            )
        )

        remedy = ""
        if active:
            remedy = ""
        elif self.get_status(model_id) == ModelStatus.LOADING:
            remedy = f"Wait for {info.name} activation to finish."
        elif missing:
            remedy = (
                f"Install {', '.join(missing)} and activate {info.name} "
                "in Model Hub."
            )
        elif not installed:
            remedy = (
                f"Install and activate {info.name} in Model Hub."
                if info.pip_managed
                else f"Download {info.name} in Model Hub."
            )
        elif not verified:
            remedy = f"Re-download {info.name}; its local cache is not verified."
        elif not consent_ready:
            remedy = (
                f"Review and approve the pinned {info.name} revision in Model Hub."
            )
        elif self.get_status(model_id) == ModelStatus.ERROR:
            detail = self.get_model_error(model_id)
            remedy = (
                f"{detail} Retry {info.name} activation in Model Hub."
                if detail
                else f"Retry {info.name} activation in Model Hub."
            )
        else:
            remedy = f"Activate {info.name} in Model Hub."

        readiness = ModelReadiness(
            model_id=model_id,
            installed=installed,
            verified=verified,
            loadable=loadable,
            active=active,
            status=self.get_status(model_id).value,
            missing_packages=missing,
            remedy=remedy,
        )
        self._readiness_cache[model_id] = readiness
        return readiness

    def get_capability_readiness(
        self,
        capability_id: str,
        *,
        allow_demo: bool = False,
        profile_ready: bool = False,
    ) -> CapabilityReadiness:
        """Resolve one action to model, explicit demo, or a concrete remedy."""
        capability = get_capability(capability_id)
        snapshots = [
            self.get_model_readiness(model_id)
            for model_id in capability.model_ids
        ]
        active = next((item for item in snapshots if item.active), None)
        candidate = active or next(
            (item for item in snapshots if item.loadable),
            snapshots[0] if snapshots else None,
        )

        if capability.profile_requirement and not profile_ready:
            return CapabilityReadiness(
                capability=capability,
                mode=RunMode.UNAVAILABLE,
                can_run=False,
                model_id=(active or candidate).model_id if (active or candidate) else "",
                active_model_id=active.model_id if active else "",
                profile_ready=False,
                remedy=f"Select {capability.profile_requirement}.",
            )

        if not capability.model_output_available:
            return CapabilityReadiness(
                capability=capability,
                mode=RunMode.UNAVAILABLE,
                can_run=False,
                model_id=(active or candidate).model_id if (active or candidate) else "",
                active_model_id=active.model_id if active else "",
                profile_ready=profile_ready,
                remedy=(
                    capability.unavailable_reason
                    or f"{capability.label} is unavailable until a verified local engine is bundled."
                ),
            )

        if (
            capability.requires_activation
            and not capability.auto_activates
            and active is None
        ):
            if (
                allow_demo
                and capability.supports_demo
                and not capability.demo_requires_activation
            ):
                return CapabilityReadiness(
                    capability=capability,
                    mode=RunMode.DEMO,
                    can_run=True,
                    model_id=candidate.model_id if candidate else "",
                    profile_ready=profile_ready,
                )
            remedy = (
                candidate.remedy
                if candidate
                else f"No model is registered for {capability.label}."
            )
            return CapabilityReadiness(
                capability=capability,
                mode=RunMode.UNAVAILABLE,
                can_run=False,
                model_id=candidate.model_id if candidate else "",
                profile_ready=profile_ready,
                remedy=remedy,
            )

        if active is not None:
            return CapabilityReadiness(
                capability=capability,
                mode=RunMode.MODEL,
                can_run=True,
                model_id=active.model_id,
                active_model_id=active.model_id,
                profile_ready=profile_ready,
            )

        if capability.auto_activates and candidate and candidate.loadable:
            return CapabilityReadiness(
                capability=capability,
                mode=RunMode.MODEL,
                can_run=True,
                model_id=candidate.model_id,
                profile_ready=profile_ready,
            )

        if (
            allow_demo
            and capability.supports_demo
            and not capability.demo_requires_activation
        ):
            return CapabilityReadiness(
                capability=capability,
                mode=RunMode.DEMO,
                can_run=True,
                model_id=candidate.model_id if candidate else "",
                profile_ready=profile_ready,
            )

        return CapabilityReadiness(
            capability=capability,
            mode=RunMode.UNAVAILABLE,
            can_run=False,
            model_id=candidate.model_id if candidate else "",
            profile_ready=profile_ready,
            remedy=(
                candidate.remedy
                if candidate
                else f"No model is registered for {capability.label}."
            ),
        )

    def get_models_by_category(self, category: ModelCategory) -> list[ModelInfo]:
        return [m for m in self._registry.values() if m.category == category]

    def get_core_models(self) -> list[ModelInfo]:
        return [m for m in self._registry.values() if m.is_core]

    @property
    def is_offline(self) -> bool:
        return bool(self._settings.get("model_hub.offline_mode", False))

    @property
    def current_model_id(self) -> Optional[str]:
        with self._state_lock:
            return self._current_model_id

    @property
    def current_model(self) -> Any:
        with self._state_lock:
            return self._current_model

    def lifecycle_snapshot(self) -> dict[str, Any]:
        """Return one consistent view of the lifecycle state for observers."""
        with self._state_lock:
            return {
                "current_model_id": self._current_model_id,
                "has_model": self._current_model is not None,
                "pending_request": self._pending_request,
                "status": {mid: status.value for mid, status in self._status.items()},
            }

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _claim_request(self) -> int:
        """Claim the newest lifecycle ticket, superseding any in-flight loader."""
        with self._state_lock:
            self._request_counter += 1
            self._pending_request = self._request_counter
            return self._pending_request

    def _release_model_object(self, model: Any):
        """Release a model object or raise while retaining ownership on failure."""
        if model is None:
            return
        hook = None
        hook_name = ""
        for candidate_name in ("unload_model", "cleanup", "to"):
            candidate = getattr(model, candidate_name, None)
            if callable(candidate):
                hook = candidate
                hook_name = candidate_name
                break
        if hook is None:
            return
        try:
            hook("cpu") if hook_name == "to" else hook()
        except Exception as exc:
            raise ModelReleaseError(
                f"Failed to release {type(model).__name__}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def load_model(self, model_id: str, loader_fn: Optional[Callable] = None) -> Any:
        """
        Load a model onto GPU. Unloads current model first.
        If loader_fn is provided, uses it. Otherwise looks up registry loader.
        Returns the loaded model object.

        Loads are serialized; a load superseded by a newer load or unload
        request raises StaleModelRequestError instead of installing itself.
        """
        info = self._registry.get(model_id)
        if info is None:
            raise ValueError(f"Unknown model: {model_id}")
        self.require_verified_model(model_id)

        ticket = self._claim_request()

        with self._state_lock:
            already = (
                self._current_model
                if self._current_model_id == model_id and self._current_model is not None
                else None
            )
        if already is not None:
            if self.get_status(model_id) == ModelStatus.ERROR:
                detail = self.get_model_error(model_id)
                raise ModelReleaseError(
                    detail or f"{model_id} remains active after a release failure."
                )
            self._set_status(model_id, ModelStatus.LOADED)
            return already

        with self._load_lock:
            with self._state_lock:
                if self._pending_request != ticket:
                    raise StaleModelRequestError(
                        f"Loading {model_id} was superseded by a newer request."
                    )
                already = (
                    self._current_model
                    if self._current_model_id == model_id
                    and self._current_model is not None
                    else None
                )
            if already is not None:
                if self.get_status(model_id) == ModelStatus.ERROR:
                    detail = self.get_model_error(model_id)
                    raise ModelReleaseError(
                        detail or f"{model_id} remains active after a release failure."
                    )
                self._set_status(model_id, ModelStatus.LOADED)
                return already

            # Unload current model without disturbing our own ticket.
            self._unload_locked()

            self._set_status(model_id, ModelStatus.LOADING)

            try:
                if loader_fn is not None:
                    model = loader_fn()
                else:
                    # Dynamic import from registry
                    model = self._dynamic_load(info)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                self._record_model_error(model_id, error_msg)
                self._set_status(model_id, ModelStatus.ERROR)
                cleanup_gpu()
                raise

            with self._state_lock:
                superseded = self._pending_request != ticket
                if not superseded:
                    self._current_model = model
                    self._current_model_id = model_id

            if superseded:
                try:
                    self._release_model_object(model)
                except ModelReleaseError as exc:
                    self._record_model_error(model_id, str(exc))
                    self._set_status(model_id, ModelStatus.ERROR)
                    cleanup_gpu()
                    raise
                del model
                cleanup_gpu()
                raise StaleModelRequestError(
                    f"Loading {model_id} was superseded by a newer request."
                )

            self._set_status(model_id, ModelStatus.LOADED)
            self._emit_gpu_status()
            return model

    def activate_model(
        self,
        model_id: str,
        *,
        progress_cb=None,
        step_cb=None,
        log_cb=None,
        cancel_event=None,
    ) -> EngineActivationResult:
        """Verify and activate one local model through the shared worker contract."""
        info = self._registry.get(model_id)
        if info is None:
            return EngineActivationResult(
                model_id=model_id,
                outcome=ActivationOutcome.FAILED,
                error=f"Unknown model: {model_id}",
            )
        if cancel_event and cancel_event.is_set():
            return EngineActivationResult(
                model_id=model_id,
                outcome=ActivationOutcome.CANCELLED,
                message="Activation cancelled before loading.",
            )
        if progress_cb:
            progress_cb(5)
        if step_cb:
            step_cb(f"Verifying {info.name}...")
        try:
            engine = self.load_model(model_id)
        except StaleModelRequestError as exc:
            # A newer activation or an unload replaced this request while it ran.
            return EngineActivationResult(
                model_id=model_id,
                outcome=ActivationOutcome.CANCELLED,
                message=str(exc),
            )
        except Exception as exc:
            return EngineActivationResult(
                model_id=model_id,
                outcome=ActivationOutcome.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )

        if cancel_event and cancel_event.is_set():
            deactivation = self.deactivate_model(model_id)
            if not deactivation.is_success:
                return EngineActivationResult(
                    model_id=model_id,
                    outcome=ActivationOutcome.FAILED,
                    error=deactivation.error,
                )
            return EngineActivationResult(
                model_id=model_id,
                outcome=ActivationOutcome.CANCELLED,
                message="Activation cancelled; loaded resources were released.",
            )
        if progress_cb:
            progress_cb(100)
        if step_cb:
            step_cb(f"{info.name} active.")
        if log_cb:
            log_cb(f"Activated model {model_id}.")
        return EngineActivationResult(
            model_id=model_id,
            outcome=ActivationOutcome.ACTIVE,
            message=f"{info.name} is active.",
            engine=engine,
        )

    def deactivate_model(self, model_id: Optional[str] = None) -> EngineActivationResult:
        """Deactivate the current model, optionally requiring an exact identity."""
        with self._state_lock:
            current_id = self._current_model_id
        target = model_id or current_id or ""
        if model_id and current_id not in {None, model_id}:
            return EngineActivationResult(
                model_id=model_id,
                outcome=ActivationOutcome.FAILED,
                error=(
                    f"{model_id} is not active; current model is "
                    f"{current_id}."
                ),
            )
        try:
            self.unload()
        except ModelReleaseError as exc:
            return EngineActivationResult(
                model_id=target,
                outcome=ActivationOutcome.FAILED,
                error=str(exc),
            )
        return EngineActivationResult(
            model_id=target,
            outcome=ActivationOutcome.INACTIVE,
            message=f"{target or 'Model'} is inactive.",
        )

    def unload(self):
        """Unload the current model and free GPU memory.

        Claims a new lifecycle ticket first, so a load still in flight is
        superseded and discards its result rather than resurrecting a model the
        user just unloaded.
        """
        self._claim_request()
        with self._load_lock:
            self._unload_locked()

    def unload_if_current(self, model_id: str) -> bool:
        """Unload only if model_id is the active model. Returns True if unloaded.

        Waits for any in-flight load to settle first, so deleting or
        quarantining a cache cannot race a load that is about to install it.
        """
        with self._load_lock:
            with self._state_lock:
                if self._current_model_id != model_id:
                    return False
                self._claim_request()
            try:
                self._unload_locked()
            except ModelReleaseError:
                return False
            return True

    def _unload_locked(self):
        """Unload the current model. Caller must hold _load_lock."""
        with self._state_lock:
            model = self._current_model
            model_id = self._current_model_id
        if model is None:
            return True

        try:
            self._release_model_object(model)
        except ModelReleaseError as exc:
            if model_id:
                self._record_model_error(model_id, str(exc))
                self._set_status(model_id, ModelStatus.ERROR)
            raise

        with self._state_lock:
            if self._current_model is model:
                self._current_model = None
                self._current_model_id = None
        del model
        cleanup_gpu()

        if model_id:
            if self._is_model_cached(model_id):
                self._set_status(model_id, ModelStatus.DOWNLOADED)
            else:
                self._set_status(model_id, ModelStatus.NOT_DOWNLOADED)

        self._emit_gpu_status()
        return True

    def _dynamic_load(self, info: ModelInfo) -> Any:
        """Load a verified local model while denying loader-initiated network access."""
        import importlib
        module = importlib.import_module(info.loader_module)
        loader = getattr(module, info.loader_fn)
        model_path = self.get_cache_dir(info.model_id)
        previous_offline = {
            key: os.environ.get(key)
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        }
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            return loader(
                cache_dir=str(model_path),
                model_path=str(model_path),
                model_id=info.model_id,
                source=info.source,
                revision=info.revision,
                local_files_only=True,
                trust_remote_code=info.requires_remote_code,
                prefer_safetensors=not info.allows_unsafe_weights,
                execution_consent=self.has_executable_model_consent(info.model_id),
            )
        finally:
            for key, value in previous_offline.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def require_verified_model(self, model_id: str) -> dict[str, Any]:
        """Return a verified manifest or fail before model code can execute."""
        info = self._registry.get(model_id)
        if info is None:
            raise ModelSecurityError(f"Unknown model: {model_id}")
        if info.generator_id:
            self.require_model_license_acceptance(model_id)
        if info.pip_managed:
            return {
                "model_id": model_id,
                "pip_managed": True,
                "revision": "package-managed",
            }
        self._validate_registry_revision(info)
        ok, reason = self.verify_download(model_id, full_hash=True)
        if not ok:
            raise ModelSecurityError(
                f"{info.name} cannot load because its local cache is unverified: {reason}"
            )
        if info.requires_remote_code or info.allows_unsafe_weights:
            if not self.has_executable_model_consent(model_id):
                raise ModelSecurityError(
                    f"{info.name} requires explicit consent for revision "
                    f"{info.revision}. {EXECUTABLE_MODEL_WARNING}"
                )
        return self.get_download_manifest(model_id)

    def approve_executable_model(
        self,
        model_id: str,
        revision: str,
        *,
        acknowledged: bool,
    ) -> None:
        """Persist explicit consent for one exact executable model revision."""
        info = self._registry.get(model_id)
        if info is None:
            raise ModelSecurityError(f"Unknown model: {model_id}")
        if revision != info.revision:
            raise ModelSecurityError(
                f"Consent revision mismatch for {model_id}: expected {info.revision}"
            )
        if not (info.requires_remote_code or info.allows_unsafe_weights):
            raise ModelSecurityError(f"{info.name} does not require executable-model consent")
        if not acknowledged:
            raise ModelSecurityError(EXECUTABLE_MODEL_WARNING)
        consents = dict(self._settings.get("model_hub.execution_consents", {}) or {})
        consents[self._execution_consent_key(info)] = {
            "approved": True,
            "source": info.source,
            "revision": info.revision,
            "approved_at": time.time(),
            "warning": EXECUTABLE_MODEL_WARNING,
        }
        self._settings.set("model_hub.execution_consents", consents)
        self._readiness_cache.clear()
        self._readiness_cache_state = None

    def has_executable_model_consent(self, model_id: str) -> bool:
        """Return whether the current exact source revision has recorded consent."""
        info = self._registry.get(model_id)
        if info is None or not (info.requires_remote_code or info.allows_unsafe_weights):
            return False
        consents = self._settings.get("model_hub.execution_consents", {}) or {}
        consent = (
            consents.get(self._execution_consent_key(info), {})
            if isinstance(consents, dict)
            else {}
        )
        return bool(
            consent.get("approved") is True
            and consent.get("source") == info.source
            and consent.get("revision") == info.revision
        )

    @staticmethod
    def _execution_consent_key(info: ModelInfo) -> str:
        return f"{info.model_id}@{info.revision}"

    @staticmethod
    def _validate_registry_revision(info: ModelInfo) -> None:
        if info.source and not info.pip_managed and not is_commit_sha(info.revision):
            raise ModelSecurityError(
                f"{info.name} must use an immutable 40-character commit revision"
            )

    @staticmethod
    def _remote_value(remote: Any, *names: str, default: Any = None) -> Any:
        """Read HuggingFace model metadata across supported client versions."""
        for name in names:
            if isinstance(remote, dict) and name in remote:
                return remote[name]
            value = getattr(remote, name, None)
            if value is not None:
                return value
        return default

    @classmethod
    def _release_notes_from_remote(
        cls,
        remote: Any,
        commits: Any = (),
    ) -> tuple[tuple[str, ...], str]:
        """Extract bounded, explicit notes instead of silently treating a SHA as notes."""
        notes: list[str] = []
        card_data = cls._remote_value(remote, "card_data", "cardData", default={})
        if isinstance(card_data, dict):
            for key in ("release_notes", "release-notes", "changelog", "changes"):
                value = card_data.get(key)
                if isinstance(value, str):
                    notes.extend(line.strip() for line in value.splitlines() if line.strip())
                elif isinstance(value, (list, tuple)):
                    notes.extend(str(item).strip() for item in value if str(item).strip())
                if notes:
                    break

        commit_title = ""
        try:
            commit = next(iter(commits), None)
        except TypeError:
            commit = None
        if commit is not None:
            commit_title = str(
                cls._remote_value(commit, "title", "commit_title", default="") or ""
            ).strip()
            message = str(
                cls._remote_value(commit, "message", "commit_message", default="") or ""
            ).strip()
            if message and message != commit_title:
                notes.extend(line.strip() for line in message.splitlines() if line.strip())
            elif commit_title:
                notes.append(commit_title)

        clean_notes: list[str] = []
        for note in notes:
            note = " ".join(str(note).split())[:1000]
            if note and note not in clean_notes:
                clean_notes.append(note)
            if len(clean_notes) >= 8:
                break
        if not clean_notes:
            clean_notes.append(
                "No upstream release notes were published; review the immutable commit before installing."
            )
        return tuple(clean_notes), commit_title

    def _fetch_model_update(self, info: ModelInfo, api: Any) -> ModelUpdate:
        """Fetch one candidate and reject mutable or unverifiable targets."""
        current_revision = info.revision
        remote = api.model_info(info.source)
        target_revision = str(
            self._remote_value(remote, "sha", "commit_id", "revision", default="") or ""
        )
        if not is_commit_sha(target_revision):
            raise ModelUpdateError(
                f"{info.name} update did not resolve to an immutable commit SHA"
            )

        commits: Any = ()
        list_commits = getattr(api, "list_repo_commits", None)
        if callable(list_commits):
            try:
                commits = list_commits(
                    info.source,
                    revision=target_revision,
                    limit=1,
                )
            except TypeError:
                commits = list_commits(info.source, revision=target_revision)
            except Exception:
                commits = ()
        release_notes, commit_title = self._release_notes_from_remote(remote, commits)
        published = self._remote_value(
            remote,
            "last_modified",
            "lastModified",
            "created_at",
            default="",
        )
        published_at = str(published or "")
        return ModelUpdate(
            model_id=info.model_id,
            source=info.source,
            current_revision=current_revision,
            target_revision=target_revision,
            status="available" if target_revision != current_revision else "up_to_date",
            release_notes=release_notes,
            target_title=commit_title,
            published_at=published_at,
            checked_at=time.time(),
            source_url=f"https://huggingface.co/{info.source}/commit/{target_revision}",
        )

    def check_for_updates(
        self,
        model_ids: Optional[list[str] | tuple[str, ...]] = None,
        *,
        api: Any = None,
        progress_cb=None,
        step_cb=None,
        log_cb=None,
        cancel_event=None,
    ) -> dict[str, ModelUpdate]:
        """Check registered HuggingFace models and persist immutable candidates."""
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        selected = list(model_ids) if model_ids is not None else list(self._registry)
        stored = self._settings.get("model_hub.update_checks", {}) or {}
        if not isinstance(stored, dict):
            stored = {}
        results: dict[str, ModelUpdate] = {}
        total = max(len(selected), 1)
        for index, model_id in enumerate(selected):
            if cancel_event is not None and cancel_event.is_set():
                from core.workers import CancelledJobError

                raise CancelledJobError("Model update check cancelled")
            info = self._registry.get(model_id)
            if info is None:
                continue
            if step_cb:
                step_cb(f"Checking {info.name} for updates...")
            if info.pip_managed or not info.source:
                result = ModelUpdate(
                    model_id=model_id,
                    source=info.source,
                    current_revision=info.revision or "package-managed",
                    target_revision=info.revision or "package-managed",
                    status="not_applicable",
                    release_notes=("Package-managed engine; update through its runtime package.",),
                    checked_at=time.time(),
                )
            elif self.is_offline:
                result = ModelUpdate(
                    model_id=model_id,
                    source=info.source,
                    current_revision=info.revision,
                    target_revision="",
                    status="offline",
                    release_notes=("Offline Mode is enabled; no remote update check was attempted.",),
                    checked_at=time.time(),
                    error="Model update checks are disabled in Offline Mode.",
                )
            else:
                try:
                    self._validate_registry_revision(info)
                    result = self._fetch_model_update(info, api)
                except Exception as exc:
                    result = ModelUpdate(
                        model_id=model_id,
                        source=info.source,
                        current_revision=info.revision,
                        target_revision="",
                        status="error",
                        release_notes=("No update was installed because the check failed.",),
                        checked_at=time.time(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
            results[model_id] = result
            stored[model_id] = result.to_dict()
            if progress_cb:
                progress_cb(int((index + 1) * 100 / total))
            if log_cb and result.available:
                log_cb(f"Update available for {info.name}: {result.target_revision[:12]}")
        self._settings.set("model_hub.update_checks", stored)
        return results

    def validate_model_health(
        self,
        model_id: str,
        *,
        revision: Optional[str] = None,
        runtime_check: Optional[Callable[[str, str], Any]] = None,
    ) -> ModelHealthReport:
        """Validate the exact cached revision before it becomes the active target."""
        info = self._registry.get(model_id)
        if info is None:
            return ModelHealthReport(
                model_id=model_id,
                revision=revision or "",
                healthy=False,
                errors=(f"Unknown model: {model_id}",),
                checked_at=time.time(),
            )
        target = revision or info.revision
        checks: list[str] = []
        errors: list[str] = []
        if not is_commit_sha(target) and not info.pip_managed:
            errors.append("Health target is not an immutable commit SHA")
        else:
            ok, reason = self.verify_download(
                model_id,
                full_hash=True,
                expected_revision=target,
            )
            if ok:
                checks.append("completion marker and source identity")
                checks.append("all cached file hashes")
                checks.append("signature and executable-file policy")
            else:
                errors.append(reason)

        if not errors and runtime_check is not None:
            try:
                runtime_result = runtime_check(model_id, target)
                healthy = getattr(runtime_result, "healthy", runtime_result)
                if callable(healthy):
                    healthy = healthy()
                if healthy is False:
                    errors.append("Runtime health check reported failure")
                else:
                    checks.append("runtime health check")
            except Exception as exc:
                errors.append(f"Runtime health check failed: {type(exc).__name__}: {exc}")

        return ModelHealthReport(
            model_id=model_id,
            revision=target,
            healthy=not errors,
            checks=tuple(checks),
            errors=tuple(errors),
            checked_at=time.time(),
        )

    def install_model_update(
        self,
        model_id: str,
        *,
        target_revision: Optional[str] = None,
        progress_cb=None,
        speed_cb=None,
        downloaded_cb=None,
        cancel_event=None,
        runtime_check: Optional[Callable[[str, str], Any]] = None,
        step_cb=None,
        log_cb=None,
    ) -> ModelHealthReport:
        """Install only the checked target, retaining the prior cache for rollback."""
        info = self._registry.get(model_id)
        if info is None:
            raise ModelUpdateError(f"Unknown model: {model_id}")
        update = self.get_model_update(model_id)
        target = target_revision or (update.target_revision if update else "")
        if update is None or not update.available:
            raise ModelUpdateError(f"No immutable update is available for {info.name}")
        if target != update.target_revision or not is_commit_sha(target):
            raise ModelUpdateError(
                f"Update target for {info.name} must match the checked immutable revision"
            )
        if target == info.revision:
            raise ModelUpdateError(f"{info.name} is already at revision {target}")

        if step_cb:
            step_cb(f"Preserving the last good {info.name} revision...")

        if self.current_model_id == model_id and not self.unload_if_current(model_id):
            raise ModelUpdateError(
                f"Cannot update {info.name} while its active model resources cannot be released"
            )

        cache_path = self.get_cache_dir(model_id)
        backup: Optional[TrashEntry] = None
        if cache_path.exists():
            current_ok, reason = self.verify_download(model_id, full_hash=True)
            if not current_ok:
                raise ModelUpdateError(
                    f"Refusing to replace an unverified {info.name} cache: {reason}"
                )
            try:
                backup = self._trash.trash_path(
                    cache_path,
                    category="model-update",
                    label=f"{info.name}-last-good",
                    metadata={
                        "model_id": model_id,
                        "source": info.source,
                        "revision": info.revision,
                        "target_revision": target,
                    },
                )
            except TrashError as exc:
                raise ModelUpdateError(
                    f"Could not preserve the last good {info.name} cache: {exc}"
                ) from exc

        try:
            self._download_at_revision(
                model_id,
                target,
                progress_cb=progress_cb,
                speed_cb=speed_cb,
                downloaded_cb=downloaded_cb,
                cancel_event=cancel_event,
            )
            if step_cb:
                step_cb(f"Validating {info.name} update health...")
            health = self.validate_model_health(
                model_id,
                revision=target,
                runtime_check=runtime_check,
            )
            if not health.healthy:
                raise ModelUpdateError(
                    f"Health validation failed for {info.name}: "
                    + "; ".join(health.errors)
                )
            self._set_installed_revision(model_id, target)
            backups = self._settings.get("model_hub.update_backups", {}) or {}
            if not isinstance(backups, dict):
                backups = {}
            if backup is not None:
                backups[model_id] = {
                    "trash_entry_id": backup.id,
                    "revision": info.revision,
                    "source": info.source,
                    "target_revision": target,
                    "health": health.to_dict(),
                    "created_at": time.time(),
                }
            self._settings.set("model_hub.update_backups", backups)
            updates = self._settings.get("model_hub.update_checks", {}) or {}
            if isinstance(updates, dict):
                updates[model_id] = replace(
                    update,
                    status="installed",
                    current_revision=target,
                ).to_dict()
                self._settings.set("model_hub.update_checks", updates)
            self._set_status(model_id, ModelStatus.DOWNLOADED)
            if log_cb:
                log_cb(f"Installed {info.name} revision {target[:12]}.")
            return health
        except Exception as exc:
            rollback_error = self._restore_update_backup(
                model_id,
                backup,
                expected_revision=info.revision,
            )
            if rollback_error:
                self._set_status(model_id, ModelStatus.ERROR)
                raise ModelUpdateError(
                    f"Update failed and automatic rollback failed: {rollback_error}"
                ) from exc
            self._set_status(model_id, ModelStatus.DOWNLOADED)
            from core.workers import CancelledJobError

            if isinstance(exc, CancelledJobError):
                raise
            if isinstance(exc, ModelUpdateError):
                raise
            raise ModelUpdateError(
                f"Update failed for {info.name}: {type(exc).__name__}: {exc}"
            ) from exc

    def _restore_update_backup(
        self,
        model_id: str,
        backup: Optional[TrashEntry],
        *,
        expected_revision: str,
    ) -> str:
        """Move a failed candidate aside and restore the previous cache."""
        cache_path = self.get_cache_dir(model_id)
        if cache_path.exists():
            try:
                self._trash.trash_path(
                    cache_path,
                    category="model-update-failed",
                    label=f"{model_id}-failed-update",
                    metadata={"model_id": model_id},
                )
            except TrashError as exc:
                return str(exc)
        if backup is None:
            return "no last-good cache was available"
        try:
            self._trash.restore(backup.id)
            ok, reason = self.verify_download(
                model_id,
                full_hash=True,
                expected_revision=expected_revision,
            )
            if not ok:
                return f"restored cache failed verification: {reason}"
            return ""
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    def rollback_model_update(
        self,
        model_id: str,
        *,
        progress_cb=None,
        step_cb=None,
        log_cb=None,
        cancel_event=None,
    ) -> ModelHealthReport:
        """Restore the last-good cache and discard the currently installed update."""
        info = self._registry.get(model_id)
        backup = self.get_model_rollback(model_id)
        if info is None:
            raise ModelUpdateError(f"Unknown model: {model_id}")
        if backup is None:
            raise ModelUpdateError(f"No last-good revision is available for {info.name}")
        if cancel_event is not None and cancel_event.is_set():
            from core.workers import CancelledJobError

            raise CancelledJobError("Model rollback cancelled")
        old_revision = str(backup.get("revision", ""))
        if backup.get("source") != info.source or not is_commit_sha(old_revision):
            raise ModelUpdateError(f"Rollback metadata for {info.name} is invalid")
        if self.current_model_id == model_id and not self.unload_if_current(model_id):
            raise ModelUpdateError(
                f"Cannot roll back {info.name} while its active model resources cannot be released"
            )

        current_entry: Optional[TrashEntry] = None
        cache_path = self.get_cache_dir(model_id)
        if step_cb:
            step_cb(f"Restoring the last good {info.name} revision...")
        if cache_path.exists():
            try:
                current_entry = self._trash.trash_path(
                    cache_path,
                    category="model-update-rejected",
                    label=f"{info.name}-rejected-update",
                    metadata={"model_id": model_id, "revision": info.revision},
                )
            except TrashError as exc:
                raise ModelUpdateError(f"Could not preserve the installed update: {exc}") from exc
        try:
            self._trash.restore(str(backup["trash_entry_id"]))
            health = self.validate_model_health(
                model_id,
                revision=old_revision,
            )
            if not health.healthy:
                raise ModelUpdateError(
                    f"Rollback health validation failed for {info.name}: "
                    + "; ".join(health.errors)
                )
            self._set_installed_revision(model_id, old_revision)
            backups = self._settings.get("model_hub.update_backups", {}) or {}
            if isinstance(backups, dict):
                backups.pop(model_id, None)
                self._settings.set("model_hub.update_backups", backups)
            self._set_status(model_id, ModelStatus.DOWNLOADED)
            if progress_cb:
                progress_cb(100)
            if log_cb:
                log_cb(f"Rolled back {info.name} to revision {old_revision[:12]}.")
            return health
        except Exception as exc:
            # Put the installed candidate back if restoring the old cache failed.
            if current_entry is not None and not cache_path.exists():
                try:
                    self._trash.restore(current_entry.id)
                except TrashError:
                    pass
            if isinstance(exc, ModelUpdateError):
                raise
            raise ModelUpdateError(
                f"Rollback failed for {info.name}: {type(exc).__name__}: {exc}"
            ) from exc

    # ── Download Management ────────────────────────────────────────────────────

    COMPLETE_MARKER = ".slunder_complete"

    def is_downloaded(self, model_id: str) -> bool:
        return self._is_model_cached(model_id)

    def get_cache_dir(self, model_id: str) -> Path:
        """Get the cache directory for a specific model."""
        base = Path(self._settings.get("model_hub.cache_dir", str(get_config_dir() / "models")))
        return base / model_id.replace("/", "--")

    def get_cache_limit_gb(self) -> float:
        """Return the configured model-cache admission limit, or zero for unlimited."""
        try:
            return max(0.0, float(self._settings.get("general.max_cache_gb", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def download_model(self, model_id: str, progress_cb=None, speed_cb=None,
                       downloaded_cb=None, cancel_event=None):
        """
        Download a model from HuggingFace Hub with real progress tracking.
        Writes a completion marker on success so partial downloads are detected.
        """
        return self._download_revision(
            model_id,
            progress_cb=progress_cb,
            speed_cb=speed_cb,
            downloaded_cb=downloaded_cb,
            cancel_event=cancel_event,
        )

    def _download_at_revision(
        self,
        model_id: str,
        revision: str,
        *,
        progress_cb=None,
        speed_cb=None,
        downloaded_cb=None,
        cancel_event=None,
    ):
        """Download one previously checked immutable revision."""
        if not is_commit_sha(revision):
            raise ModelUpdateError("Update downloads require a 40-character commit SHA")
        return self._download_revision(
            model_id,
            progress_cb=progress_cb,
            speed_cb=speed_cb,
            downloaded_cb=downloaded_cb,
            cancel_event=cancel_event,
            revision_override=revision,
        )

    def _download_revision(
        self,
        model_id: str,
        *,
        progress_cb=None,
        speed_cb=None,
        downloaded_cb=None,
        cancel_event=None,
        revision_override: Optional[str] = None,
    ):
        """Run one serialized download, optionally targeting a checked SHA."""
        if self.is_offline:
            raise OfflineModeError(
                "Model downloads are disabled while Offline Mode is enabled. "
                "Disable Offline Mode in Settings > GPU and Models to download models."
            )

        info = self._registry.get(model_id)
        if info is None:
            raise ValueError(f"Unknown model: {model_id}")
        self._validate_registry_revision(info)

        try:
            admission_lease = global_admission_controller().acquire(
                "download",
                key=model_id,
                cancel_event=cancel_event,
            )
        except AdmissionBusyError as exc:
            raise DownloadInFlightError(str(exc)) from exc
        except AdmissionCancelledError as exc:
            from core.workers import CancelledJobError

            raise CancelledJobError(
                str(exc), outputs={"model_id": model_id}
            ) from exc
        try:
            # Recheck storage after admission: a preceding download may have
            # completed while this request was waiting for the global slot.
            limit = self.get_cache_limit_gb()
            if limit and not self._is_model_cached(model_id):
                used = self.get_total_disk_usage()
                if used + info.disk_gb > limit:
                    raise RuntimeError(
                        f"Model cache limit is {limit:.1f} GB; {info.name} needs about "
                        f"{info.disk_gb:.1f} GB and {used:.1f} GB is already used. "
                        "Increase the limit in Settings > Advanced > Cache or remove "
                        "an installed model from Model Hub."
                    )
            with self._state_lock:
                if model_id in self._downloads_in_flight:
                    raise DownloadInFlightError(
                        f"A download for {info.name} is already running."
                    )
                self._downloads_in_flight.add(model_id)
            try:
                return self._download_model_locked(
                    info,
                    model_id,
                    progress_cb=progress_cb,
                    speed_cb=speed_cb,
                    downloaded_cb=downloaded_cb,
                    cancel_event=cancel_event,
                    revision_override=revision_override,
                )
            finally:
                with self._state_lock:
                    self._downloads_in_flight.discard(model_id)
        finally:
            admission_lease.release()

    def _download_model_locked(self, info: ModelInfo, model_id: str, progress_cb=None,
                               speed_cb=None, downloaded_cb=None, cancel_event=None,
                               revision_override: Optional[str] = None):
        """Download body. Only one call per model_id runs at a time."""
        # Pip-managed models (Demucs, DiffSinger) handle their own downloads
        if info.pip_managed:
            self._set_status(model_id, ModelStatus.DOWNLOADED)
            if progress_cb:
                progress_cb(100)
            return True

        if not info.source:
            raise ValueError(f"No download source for model: {model_id}")

        download_revision = revision_override or info.revision
        if not is_commit_sha(download_revision):
            raise ModelSecurityError(
                f"{info.name} must use an immutable 40-character commit revision"
            )

        cache_path = self.get_cache_dir(model_id)
        self._quarantine_incompatible_cache(model_id, cache_path)

        self._set_status(model_id, ModelStatus.DOWNLOADING)

        def _mark_cancelled_download():
            self._set_status(
                model_id,
                ModelStatus.PARTIAL if self.has_partial_download(model_id) else ModelStatus.NOT_DOWNLOADED,
            )

        def _raise_if_cancelled():
            if cancel_event and cancel_event.is_set():
                _mark_cancelled_download()
                from core.workers import CancelledJobError
                raise CancelledJobError("Download cancelled", outputs={"model_id": model_id})

        try:
            from huggingface_hub import snapshot_download
            from tqdm.auto import tqdm as _BaseTqdm
            import time as _time

            class _CancelableTqdm(_BaseTqdm):
                """Abort HF transfer progress at the next downloaded chunk."""

                def __init__(self, *args, **kwargs):
                    _raise_if_cancelled()
                    kwargs.setdefault("disable", True)
                    super().__init__(*args, **kwargs)

                def update(self, n=1):
                    _raise_if_cancelled()
                    return super().update(n)

            _raise_if_cancelled()

            cache_dir = str(cache_path.parent)

            kwargs = {
                "repo_id": info.source,
                "cache_dir": cache_dir,
                "local_dir": str(cache_path),
                "revision": download_revision,
                # A single transfer worker makes cancellation deterministic:
                # once the active file raises, no queued file can begin.
                "max_workers": 1,
                "tqdm_class": _CancelableTqdm,
            }

            if info.allow_patterns:
                kwargs["allow_patterns"] = info.allow_patterns
            if info.ignore_patterns:
                kwargs["ignore_patterns"] = info.ignore_patterns

            if info.gated:
                token = self._get_hf_token()
                if not token:
                    raise PermissionError(
                        f"{info.name} is a gated model.\n"
                        f"Paste your HF token in Settings > GPU and Models."
                    )
                kwargs["token"] = token

            # -- Progress tracking via disk polling --
            # Instead of fragile tqdm monkey-patching, we poll actual file
            # sizes on disk. Works with any huggingface_hub version.
            _outer_progress_cb = progress_cb
            _outer_speed_cb = speed_cb
            _outer_downloaded_cb = downloaded_cb
            _expected_bytes = max(int(info.disk_gb * 1024**3), 1)
            _poll_state = {"last_bytes": 0, "last_time": _time.monotonic()}
            _download_done = threading.Event()

            def _poll_progress():
                """Poll download dir every 500ms and report progress."""
                while not _download_done.is_set():
                    try:
                        current = 0
                        if cache_path.exists():
                            for f in cache_path.rglob("*"):
                                try:
                                    if f.is_file() and f.name != self.COMPLETE_MARKER:
                                        current += f.stat().st_size
                                except OSError:
                                    pass

                        if current > 0:
                            pct = min(int(current * 100 / _expected_bytes), 99)
                            if _outer_progress_cb:
                                _outer_progress_cb(pct)

                            if _outer_downloaded_cb:
                                _outer_downloaded_cb(
                                    f"{current / 1024**3:.2f} GB / "
                                    f"{_expected_bytes / 1024**3:.2f} GB"
                                )

                            now = _time.monotonic()
                            elapsed = now - _poll_state["last_time"]
                            if elapsed >= 1.0:
                                delta = current - _poll_state["last_bytes"]
                                speed = delta / elapsed if elapsed > 0 else 0
                                _poll_state["last_time"] = now
                                _poll_state["last_bytes"] = current
                                if _outer_speed_cb and speed > 0:
                                    if speed >= 1024**2:
                                        _outer_speed_cb(f"{speed/1024**2:.1f} MB/s")
                                    elif speed >= 1024:
                                        _outer_speed_cb(f"{speed/1024:.0f} KB/s")
                                    else:
                                        _outer_speed_cb(f"{speed:.0f} B/s")
                    except Exception:
                        pass
                    _download_done.wait(0.5)

            poll_thread = threading.Thread(target=_poll_progress, daemon=True)
            poll_thread.start()

            # Remove stale completion marker before downloading
            marker = cache_path / self.COMPLETE_MARKER
            if marker.exists():
                marker.unlink()

            try:
                _raise_if_cancelled()
                resolved_path = snapshot_download(**kwargs)
            finally:
                _download_done.set()
                poll_thread.join(timeout=2)

            # -- Write completion marker --
            resolved_revision = self._resolve_hf_revision(
                info,
                kwargs.get("token"),
                requested_revision=download_revision,
            )
            self._write_complete_marker(
                model_id,
                cache_path,
                resolved_path=resolved_path,
                resolved_revision=resolved_revision,
                revision=download_revision,
            )

            verified, verification_reason = self.verify_download(
                model_id,
                full_hash=True,
                expected_revision=download_revision,
            )
            if not verified:
                self._set_status(model_id, ModelStatus.ERROR)
                raise ModelSecurityError(
                    f"{info.name} was downloaded but cannot be activated: "
                    f"{verification_reason}"
                )

            self._set_status(model_id, ModelStatus.DOWNLOADED)
            if progress_cb:
                progress_cb(100)
            return True

        except Exception as e:
            if cancel_event and cancel_event.is_set():
                _mark_cancelled_download()
            else:
                self._set_status(model_id, ModelStatus.ERROR)
            raise

    def _quarantine_incompatible_cache(
        self,
        model_id: str,
        cache_path: Optional[Path] = None,
    ) -> Optional[TrashEntry]:
        """Recoverably move a completed cache whose immutable identity changed."""
        info = self._registry.get(model_id)
        if info is None or info.pip_managed:
            return None

        path = cache_path or self.get_cache_dir(model_id)
        marker = path / self.COMPLETE_MARKER
        if not marker.is_file():
            return None

        try:
            manifest = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}

        if (
            manifest.get("source") == info.source
            and manifest.get("revision") == info.revision
            and manifest.get("resolved_revision") == info.revision
        ):
            return None

        if self.current_model_id == model_id and not self.unload_if_current(model_id):
            return None

        return self._trash.trash_path(
            path,
            category="model",
            label=f"{info.name}-incompatible-cache",
            metadata={
                "model_id": model_id,
                "model_name": info.name,
                "migration": "immutable model identity changed",
                "old_source": manifest.get("source", ""),
                "old_revision": manifest.get("revision", ""),
                "new_source": info.source,
                "new_revision": info.revision,
            },
        )

    def _resolve_hf_revision(
        self,
        info: ModelInfo,
        token: Optional[str] = None,
        *,
        requested_revision: Optional[str] = None,
    ) -> str:
        """Resolve a HuggingFace revision to a commit SHA when online."""
        requested = requested_revision or info.revision
        if not info.source:
            return requested
        if is_commit_sha(requested):
            return requested
        if self.is_offline:
            return requested
        try:
            from huggingface_hub import HfApi
            model = HfApi().model_info(info.source, revision=requested, token=token)
            return getattr(model, "sha", "") or requested
        except Exception:
            return requested

    def _write_complete_marker(
        self,
        model_id: str,
        cache_path: Path,
        resolved_path: str = "",
        resolved_revision: str = "",
        revision: str = "",
    ):
        """Write a marker file indicating download is complete with metadata."""
        import time as _time
        info = self._registry.get(model_id)
        file_count = 0
        total_size = 0
        file_hashes: dict[str, str] = {}
        for f in cache_path.rglob("*"):
            if f.is_file() and f.name != self.COMPLETE_MARKER:
                file_count += 1
                total_size += f.stat().st_size
                try:
                    file_hashes[str(f.relative_to(cache_path)).replace("\\", "/")] = hash_file_sha256(f)
                except OSError:
                    pass

        marker = cache_path / self.COMPLETE_MARKER
        license_meta = info.license_metadata() if info else {}
        serialization = _serialization_summary(file_hashes)
        signature = (
            verify_oms_signature(cache_path, info)
            if info is not None
            else SignatureVerification(
                status=SIGNATURE_UNSIGNED,
                reason="No OMS signature was published with this model revision.",
            )
        )
        declared_revision = revision or (info.revision if info else "")
        local_mirror = (
            default_local_mirror(model_id, declared_revision)
            if declared_revision
            else info.local_mirror if info else ""
        )
        marker.write_text(json.dumps({
            "model_id": model_id,
            "generator_id": info.generator_id if info else "",
            "local_mirror": local_mirror,
            "timestamp": _time.time(),
            "file_count": file_count,
            "total_bytes": total_size,
            "source": info.source if info else "",
            "revision": declared_revision,
            "resolved_revision": resolved_revision or declared_revision,
            "license": info.license if info else "unknown",
            "license_url": license_meta.get("license_url", ""),
            "commercial_use": license_meta.get("commercial_use", COMMERCIAL_USE_UNKNOWN),
            "commercial_use_label": license_meta.get("commercial_use_label", COMMERCIAL_USE_LABELS[COMMERCIAL_USE_UNKNOWN]),
            "commercial_use_note": license_meta.get("commercial_use_note", ""),
            "license_warning": license_meta.get("license_warning", ""),
            "requires_export_warning": license_meta.get("requires_export_warning", True),
            "gated": bool(info.gated) if info else False,
            "access": license_meta.get("access", "Unknown"),
            "trusted_source": bool(info.trusted_source) if info else False,
            "trust_note": info.trust_note if info else "",
            "requires_remote_code": bool(info.requires_remote_code) if info else False,
            "allows_unsafe_weights": bool(info.allows_unsafe_weights) if info else False,
            "execution_consent_required": bool(
                info and (info.requires_remote_code or info.allows_unsafe_weights)
            ),
            "serialization": serialization,
            "allow_patterns": info.allow_patterns if info else [],
            "ignore_patterns": info.ignore_patterns if info else [],
            "resolved_path": resolved_path,
            "file_hashes": file_hashes,
            **signature.as_manifest_fields(),
        }, indent=2))

    def get_download_manifest(self, model_id: str) -> dict:
        cache_path = self.get_cache_dir(model_id)
        marker = cache_path / self.COMPLETE_MARKER
        if not marker.exists():
            return {}
        try:
            return json.loads(marker.read_text())
        except Exception:
            return {}

    def verify_download(
        self,
        model_id: str,
        *,
        full_hash: bool = True,
        expected_revision: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Verify a download is complete. Returns (ok, reason).
        Checks for completion marker and basic file count sanity.
        """
        info = self._registry.get(model_id)
        if not info:
            return False, "Unknown model"
        if info.pip_managed:
            return True, "pip managed"
        try:
            self._validate_registry_revision(info)
        except ModelSecurityError as exc:
            return False, str(exc)

        expected = expected_revision or info.revision
        if info.source and not is_commit_sha(expected):
            return False, "Expected revision is not an immutable 40-character commit SHA"

        cache_path = self.get_cache_dir(model_id)
        marker = cache_path / self.COMPLETE_MARKER
        if not marker.exists():
            if cache_path.exists() and any(
                f for f in cache_path.iterdir() if f.name != self.COMPLETE_MARKER
            ):
                return False, "Partial download (no completion marker)"
            return False, "Not downloaded"

        try:
            meta = json.loads(marker.read_text())
            if meta.get("model_id") != model_id:
                return False, "Manifest model ID mismatch"
            identity_fields = {
                "generator_id": info.generator_id,
                "local_mirror": default_local_mirror(model_id, expected),
            }
            identity_changed = False
            for key, expected_identity in identity_fields.items():
                stored = meta.get(key)
                if stored not in (None, "") and stored != expected_identity:
                    return False, f"Manifest {key} mismatch"
                if stored != expected_identity:
                    meta[key] = expected_identity
                    identity_changed = True
            if meta.get("source") != info.source:
                return False, "Manifest source mismatch"
            if meta.get("revision") != expected:
                return False, "Manifest revision mismatch"
            if meta.get("resolved_revision") != expected:
                return False, "Manifest resolved revision mismatch"
            expected_files = meta.get("file_count", 0)
            actual_paths = {
                str(f.relative_to(cache_path)).replace("\\", "/")
                for f in cache_path.rglob("*")
                if f.is_file() and f.name != self.COMPLETE_MARKER
            }
            actual_files = len(actual_paths)
            if actual_files != expected_files:
                return False, f"File count mismatch ({actual_files}/{expected_files})"
            file_hashes = meta.get("file_hashes", {})
            if not isinstance(file_hashes, dict) or set(file_hashes) != actual_paths:
                return False, "Manifest does not hash every cached file"

            executable_files = sorted(
                path for path in actual_paths if Path(path).suffix.lower() in EXECUTABLE_EXTENSIONS
            )
            if executable_files and not info.requires_remote_code:
                return False, f"Undeclared executable model code: {executable_files[0]}"
            unsafe_weights = sorted(
                path for path in actual_paths if Path(path).suffix.lower() in PICKLE_WEIGHT_EXTENSIONS
            )
            if unsafe_weights and not info.allows_unsafe_weights:
                return False, f"Unsafe weight format is not approved: {unsafe_weights[0]}"

            for rel_path, expected_hash in file_hashes.items():
                path = cache_path / rel_path
                if path.is_symlink():
                    return False, f"Symlinked model file is not allowed: {rel_path}"
                resolved = path.resolve(strict=False)
                if not resolved.is_relative_to(cache_path.resolve(strict=False)):
                    return False, f"Model path escapes cache: {rel_path}"
                if not path.is_file():
                    return False, f"Missing hashed file: {rel_path}"
                if full_hash:
                    actual_hash = hash_file_sha256(path)
                    if actual_hash != expected_hash:
                        return False, f"Hash mismatch: {rel_path}"

            signature = verify_oms_signature(cache_path, info)
            signature_fields = signature.as_manifest_fields()
            if identity_changed or any(meta.get(key) != value for key, value in signature_fields.items()):
                meta.update(signature_fields)
                marker.write_text(json.dumps(meta, indent=2))
            if not signature.is_acceptable:
                return False, f"OMS signature {signature.status}: {signature.reason}"
            if signature.status == SIGNATURE_VERIFIED:
                return True, "OK (OMS signature verified)"
            return True, "OK (unsigned)"
        except Exception as e:
            return False, f"Marker corrupted: {e}"

    def _get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from environment, settings, or huggingface-cli login."""
        import os
        # 1. Environment variable
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token
        # 2. Settings
        token = self._settings.get("model_hub.hf_token", "")
        if token:
            return token
        # 3. huggingface-cli login token
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
            if token:
                return token
        except Exception:
            pass
        return None

    def _is_model_cached(self, model_id: str) -> bool:
        """Check if model download completed successfully (has completion marker)."""
        info = self._registry.get(model_id)

        # Pip-managed engines are installed only when their declared runtime imports.
        if info and info.pip_managed:
            return not self.get_missing_runtime_packages(model_id)

        ok, _reason = self.verify_download(model_id, full_hash=False)
        return ok

    def has_partial_download(self, model_id: str) -> bool:
        """Check if there are leftover files from an incomplete download."""
        info = self._registry.get(model_id)
        if info and info.pip_managed:
            return False
        cache_path = self.get_cache_dir(model_id)
        if not cache_path.exists():
            return False
        marker = cache_path / self.COMPLETE_MARKER
        if marker.exists():
            return False
        # Has files but no marker = partial
        return any(f for f in cache_path.iterdir() if f.name != self.COMPLETE_MARKER)

    def delete_model_cache(self, model_id: str) -> Optional[TrashEntry]:
        """Move a model cache directory to trash instead of deleting it."""
        info = self._registry.get(model_id)
        if info is None or info.pip_managed:
            return None

        cache_path = self.get_cache_dir(model_id)
        if not cache_path.exists():
            return None

        if self.current_model_id == model_id and not self.unload_if_current(model_id):
            return None

        try:
            entry = self._trash.trash_path(
                cache_path,
                category="model",
                label=info.name or model_id,
                metadata={
                    "model_id": model_id,
                    "model_name": info.name,
                    "source": info.source,
                    "revision": info.revision,
                    "status": self.get_status(model_id).value,
                },
            )
        except TrashError as e:
            return None

        self._set_status(model_id, ModelStatus.NOT_DOWNLOADED)
        return entry

    def restore_model_cache(self, trash_entry_id: str) -> bool:
        """Restore a trashed model cache directory and refresh status."""
        try:
            entry = self._trash.restore(trash_entry_id)
        except TrashError as e:
            model_id = ""
            err = str(e)
        else:
            model_id = entry.metadata.get("model_id", "")
            err = ""

        if err:
            return False
        if not model_id or model_id not in self._registry:
            return False

        ok, _reason = self.verify_download(model_id)
        self._set_status(
            model_id,
            ModelStatus.DOWNLOADED if ok else ModelStatus.PARTIAL,
        )
        return True

    def get_total_disk_usage(self) -> float:
        """Get total disk usage of all downloaded models in GB."""
        base = Path(self._settings.get("model_hub.cache_dir", str(get_config_dir() / "models")))
        cache_path = str(base.resolve(strict=False))
        if (
            self._disk_usage_cache is not None
            and self._disk_usage_cache_path == cache_path
        ):
            return self._disk_usage_cache

        total = 0.0
        if base.exists():
            for f in base.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        usage = total / (1024**3)
        self._disk_usage_cache = usage
        self._disk_usage_cache_path = cache_path
        return usage

    # ── GPU Status ─────────────────────────────────────────────────────────────

    def get_gpu_status(self) -> dict:
        """Get current GPU status including loaded model info."""
        gpu = get_gpu_info()
        with self._state_lock:
            current_id = self._current_model_id
        gpu["current_model"] = current_id
        gpu["current_model_name"] = (
            self._registry[current_id].name
            if current_id and current_id in self._registry
            else None
        )
        return gpu

    def admission_snapshot(self) -> dict[str, Any]:
        """Return central model-work capacity without exposing queue internals."""
        return global_admission_controller().snapshot().to_dict()

    def _emit_gpu_status(self):
        self.gpu_status_changed.emit(self.get_gpu_status())

    # ── Internal ───────────────────────────────────────────────────────────────

    def _readiness_state_token(self):
        """Return cheap lifecycle state used to invalidate filesystem readiness."""
        with self._state_lock:
            statuses = tuple(
                sorted((model_id, status.value) for model_id, status in self._status.items())
            )
            current_model_id = self._current_model_id
            has_current_model = self._current_model is not None
        registry = tuple(
            sorted(
                (
                    model_id,
                    info.revision,
                    bool(info.pip_managed),
                    bool(info.requires_remote_code),
                    bool(info.allows_unsafe_weights),
                )
                for model_id, info in self._registry.items()
            )
        )
        return registry, statuses, current_model_id, has_current_model

    def _set_status(self, model_id: str, status: ModelStatus):
        with self._state_lock:
            self._status[model_id] = status
            if status != ModelStatus.ERROR:
                self._model_errors.pop(model_id, None)
            self._readiness_cache.clear()
            self._readiness_cache_state = None
            self._disk_usage_cache = None
            self._disk_usage_cache_path = None
        # Emitted outside the lock: receivers may call back into the manager.
        self.status_changed.emit(model_id, status.value)

    def _record_model_error(self, model_id: str, error: str):
        with self._state_lock:
            self._model_errors[model_id] = str(error)
            self._readiness_cache.clear()
            self._readiness_cache_state = None
