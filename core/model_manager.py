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
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

from core.settings import Settings, get_config_dir
from core.trash import TrashEntry, TrashError, TrashManager
from core.engine_contract import (
    ActivationOutcome,
    CapabilityReadiness,
    EngineActivationResult,
    ModelReadiness,
    RunMode,
    get_capability,
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


class ModelSecurityError(RuntimeError):
    """Raised when model provenance or executable-content policy is not satisfied."""
    pass


class StaleModelRequestError(RuntimeError):
    """Raised when a load finished after a newer load/unload request superseded it."""
    pass


class DownloadInFlightError(RuntimeError):
    """Raised when a model download is already running for the same model."""
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
        }


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
        vram_gb=4.0,
        disk_gb=10.4,
        license=ACE_STEP_LICENSE,
        source=ACE_STEP_SOURCE,
        loader_module="engines.ace_step_engine",
        loader_fn="load_model",
        is_core=True,
        tags=list(ACE_STEP_CAPABILITIES),
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
        commercial_use=COMMERCIAL_USE_ALLOWED,
        commercial_use_note="MIT model package.",
        license_url="https://github.com/facebookresearch/demucs",
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
        revision="d3bd7b00761b78ad7a8a05145ee31e7832e9916c",
        allows_unsafe_weights=True,
        trust_note="Pinned upstream revision; pickle-backed weights require per-revision consent.",
        commercial_use=COMMERCIAL_USE_NON_COMMERCIAL,
        commercial_use_note="CC-BY-NC model weights are not cleared for commercial use.",
        license_url="https://huggingface.co/facebook/musicgen-medium",
    ),
}


MODEL_RUNTIME_PACKAGES: dict[str, tuple[tuple[str, str], ...]] = {
    ACE_STEP_MODEL_ID: (
        ("torch", "torch"),
        ("diffusers", "diffusers"),
        ("transformers", "transformers"),
    ),
    "llama-3.1-8b-q4": (("llama-cpp-python", "llama_cpp"),),
    "llama-3.2-3b-q4": (("llama-cpp-python", "llama_cpp"),),
    "qwen-2.5-14b-q4": (("llama-cpp-python", "llama_cpp"),),
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

def get_gpu_info() -> dict:
    """Get GPU VRAM info. Returns dict with total_gb, used_gb, free_gb, name."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            return {
                "available": True,
                "name": props.name,
                "total_gb": round(total, 1),
                "used_gb": round(allocated, 1),
                "reserved_gb": round(reserved, 1),
                "free_gb": round(total - reserved, 1),
            }
    except (ImportError, RuntimeError, AttributeError):
        pass

    return {
        "available": False,
        "name": "No GPU detected",
        "total_gb": 0,
        "used_gb": 0,
        "reserved_gb": 0,
        "free_gb": 0,
    }


def cleanup_gpu():
    """Aggressive GPU memory cleanup."""
    try:
        import torch
        if torch.cuda.is_available():
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
        model_loading(str)         - model_id starting to load
        model_loaded(str)          - model_id successfully loaded
        model_unloaded(str)        - model_id unloaded
        model_error(str, str)      - (model_id, error_message)
        download_started(str)      - model_id download started
        download_progress(str, int) - (model_id, 0-100)
        download_completed(str)    - model_id download finished
        download_error(str, str)   - (model_id, error_message)
        gpu_status_changed(dict)   - GPU info dict updated
        status_changed(str, str)   - (model_id, new_status)
    """
    model_loading = Signal(str)
    model_loaded = Signal(str)
    model_unloaded = Signal(str)
    model_error = Signal(str, str)
    download_started = Signal(str)
    download_progress = Signal(str, int)
    download_completed = Signal(str)
    download_error = Signal(str, str)
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
            ):
                if key in manifest and manifest[key] not in ("", None):
                    metadata[key] = manifest[key]
        return metadata

    def get_status(self, model_id: str) -> ModelStatus:
        with self._state_lock:
            return self._status.get(model_id, ModelStatus.NOT_DOWNLOADED)

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
            remedy = f"Retry {info.name} activation in Model Hub."
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
            if allow_demo and capability.supports_demo:
                return CapabilityReadiness(
                    capability=capability,
                    mode=RunMode.DEMO,
                    can_run=True,
                    model_id=(active or candidate).model_id if (active or candidate) else "",
                    active_model_id=active.model_id if active else "",
                    profile_ready=profile_ready,
                )
            return CapabilityReadiness(
                capability=capability,
                mode=RunMode.UNAVAILABLE,
                can_run=False,
                model_id=(active or candidate).model_id if (active or candidate) else "",
                active_model_id=active.model_id if active else "",
                profile_ready=profile_ready,
                remedy=(
                    f"{capability.label} currently exposes an explicit demo "
                    "pipeline only; enable its Demo option to continue."
                ),
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
        """Best-effort teardown of a model object we are about to drop."""
        if model is None:
            return
        try:
            if hasattr(model, "unload_model"):
                model.unload_model()
            elif hasattr(model, "cleanup"):
                model.cleanup()
            elif hasattr(model, "to"):
                model.to("cpu")
        except Exception:
            pass

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
                self._set_status(model_id, ModelStatus.LOADED)
                return already

            # Unload current model without disturbing our own ticket.
            self._unload_locked()

            self._set_status(model_id, ModelStatus.LOADING)
            self.model_loading.emit(model_id)

            try:
                if loader_fn is not None:
                    model = loader_fn()
                else:
                    # Dynamic import from registry
                    model = self._dynamic_load(info)
            except Exception as e:
                self._set_status(model_id, ModelStatus.ERROR)
                error_msg = f"{type(e).__name__}: {e}"
                self.model_error.emit(model_id, error_msg)
                cleanup_gpu()
                raise

            with self._state_lock:
                superseded = self._pending_request != ticket
                if not superseded:
                    self._current_model = model
                    self._current_model_id = model_id

            if superseded:
                self._release_model_object(model)
                del model
                cleanup_gpu()
                raise StaleModelRequestError(
                    f"Loading {model_id} was superseded by a newer request."
                )

            self._set_status(model_id, ModelStatus.LOADED)
            self.model_loaded.emit(model_id)
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
            self.deactivate_model(model_id)
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
        self.unload()
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
            self._unload_locked()
            return True

    def _unload_locked(self):
        """Unload the current model. Caller must hold _load_lock."""
        with self._state_lock:
            model = self._current_model
            model_id = self._current_model_id
            self._current_model = None
            self._current_model_id = None
        if model is None:
            return

        self._release_model_object(model)
        del model
        cleanup_gpu()

        if model_id:
            if self._is_model_cached(model_id):
                self._set_status(model_id, ModelStatus.DOWNLOADED)
            else:
                self._set_status(model_id, ModelStatus.NOT_DOWNLOADED)
            self.model_unloaded.emit(model_id)

        self._emit_gpu_status()

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
                execution_consent=bool(
                    info.requires_remote_code or info.allows_unsafe_weights
                ),
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

    # ── Download Management ────────────────────────────────────────────────────

    COMPLETE_MARKER = ".slunder_complete"

    def is_downloaded(self, model_id: str) -> bool:
        return self._is_model_cached(model_id)

    def get_cache_dir(self, model_id: str) -> Path:
        """Get the cache directory for a specific model."""
        base = Path(self._settings.get("model_hub.cache_dir", str(get_config_dir() / "models")))
        return base / model_id.replace("/", "--")

    def download_model(self, model_id: str, progress_cb=None, speed_cb=None,
                       downloaded_cb=None, cancel_event=None):
        """
        Download a model from HuggingFace Hub with real progress tracking.
        Writes a completion marker on success so partial downloads are detected.
        """
        if self.is_offline:
            raise OfflineModeError(
                "Model downloads are disabled while Offline Mode is enabled. "
                "Disable Offline Mode in Settings > GPU & Models to download models."
            )

        info = self._registry.get(model_id)
        if info is None:
            raise ValueError(f"Unknown model: {model_id}")
        self._validate_registry_revision(info)

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
            )
        finally:
            with self._state_lock:
                self._downloads_in_flight.discard(model_id)

    def _download_model_locked(self, info: ModelInfo, model_id: str, progress_cb=None,
                               speed_cb=None, downloaded_cb=None, cancel_event=None):
        """Download body. Only one call per model_id runs at a time."""
        # Pip-managed models (Demucs, DiffSinger) handle their own downloads
        if info.pip_managed:
            self._set_status(model_id, ModelStatus.DOWNLOADED)
            self.download_completed.emit(model_id)
            if progress_cb:
                progress_cb(100)
            return True

        if not info.source:
            raise ValueError(f"No download source for model: {model_id}")

        cache_path = self.get_cache_dir(model_id)
        self._quarantine_incompatible_cache(model_id, cache_path)

        self._set_status(model_id, ModelStatus.DOWNLOADING)
        self.download_started.emit(model_id)

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
                "revision": info.revision,
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
                        f"Paste your HF token in Settings > GPU & Models."
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
            resolved_revision = self._resolve_hf_revision(info, kwargs.get("token"))
            self._write_complete_marker(
                model_id,
                cache_path,
                resolved_path=resolved_path,
                resolved_revision=resolved_revision,
            )

            self._set_status(model_id, ModelStatus.DOWNLOADED)
            self.download_completed.emit(model_id)
            if progress_cb:
                progress_cb(100)
            return True

        except Exception as e:
            if cancel_event and cancel_event.is_set():
                _mark_cancelled_download()
            else:
                self._set_status(model_id, ModelStatus.ERROR)
                self.download_error.emit(model_id, str(e))
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

        self.unload_if_current(model_id)

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

    def _resolve_hf_revision(self, info: ModelInfo, token: Optional[str] = None) -> str:
        """Resolve a HuggingFace revision to a commit SHA when online."""
        if not info.source:
            return info.revision
        if is_commit_sha(info.revision):
            return info.revision
        if self.is_offline:
            return info.revision
        try:
            from huggingface_hub import HfApi
            model = HfApi().model_info(info.source, revision=info.revision, token=token)
            return getattr(model, "sha", "") or info.revision
        except Exception:
            return info.revision

    def _write_complete_marker(
        self,
        model_id: str,
        cache_path: Path,
        resolved_path: str = "",
        resolved_revision: str = "",
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
        marker.write_text(json.dumps({
            "model_id": model_id,
            "timestamp": _time.time(),
            "file_count": file_count,
            "total_bytes": total_size,
            "source": info.source if info else "",
            "revision": info.revision if info else "",
            "resolved_revision": resolved_revision or (info.revision if info else ""),
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

    def verify_download(self, model_id: str, *, full_hash: bool = True) -> tuple[bool, str]:
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
            if meta.get("source") != info.source:
                return False, "Manifest source mismatch"
            if meta.get("revision") != info.revision:
                return False, "Manifest revision mismatch"
            if meta.get("resolved_revision") != info.revision:
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
            return True, "OK"
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

        self.unload_if_current(model_id)

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
            self.model_error.emit(model_id, str(e))
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
            if model_id:
                self.model_error.emit(model_id, err)
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
            self._readiness_cache.clear()
            self._readiness_cache_state = None
            self._disk_usage_cache = None
            self._disk_usage_cache_path = None
        # Emitted outside the lock: receivers may call back into the manager.
        self.status_changed.emit(model_id, status.value)
