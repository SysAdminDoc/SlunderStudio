"""
Slunder Studio v0.0.2 — Model Manager
Central singleton managing model lifecycle: download, load, unload, and GPU memory.
Enforces one-large-model-at-a-time GPU residency for 16GB VRAM budget.
"""
import gc
import os
import json
import time
import threading
from enum import Enum
from typing import Any, Callable, Optional
from pathlib import Path
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

from core.settings import Settings, get_config_dir

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
    pip_managed: bool = False  # True = model managed by pip package, not HF download
    gated: bool = False  # True = requires HF login + license acceptance


# ── Built-in Model Registry ───────────────────────────────────────────────────

BUILTIN_MODELS: dict[str, ModelInfo] = {
    "ace-step-v1.5": ModelInfo(
        model_id="ace-step-v1.5",
        name="ACE-Step v1.5",
        description="Full song generation with vocals and instrumentals. Under 10s per song, <8GB VRAM.",
        category=ModelCategory.SONG_FORGE,
        vram_gb=3.5,
        disk_gb=8.3,
        license="Apache 2.0",
        source="ACE-Step/ACE-Step-v1-3.5B",
        loader_module="engines.ace_step_engine",
        loader_fn="load_model",
        is_core=True,
        tags=["song", "vocals", "instrumental", "music generation"],
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
    ),
}


# ── GPU Utilities ──────────────────────────────────────────────────────────────

def get_gpu_info() -> dict:
    """Get GPU VRAM info. Returns dict with total_gb, used_gb, free_gb, name."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_mem / (1024**3)
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
    except (ImportError, RuntimeError):
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
        self._settings = Settings()

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

    def get_status(self, model_id: str) -> ModelStatus:
        return self._status.get(model_id, ModelStatus.NOT_DOWNLOADED)

    def get_models_by_category(self, category: ModelCategory) -> list[ModelInfo]:
        return [m for m in self._registry.values() if m.category == category]

    def get_core_models(self) -> list[ModelInfo]:
        return [m for m in self._registry.values() if m.is_core]

    @property
    def current_model_id(self) -> Optional[str]:
        return self._current_model_id

    @property
    def current_model(self) -> Any:
        return self._current_model

    # ── Model Loading ──────────────────────────────────────────────────────────

    def load_model(self, model_id: str, loader_fn: Optional[Callable] = None) -> Any:
        """
        Load a model onto GPU. Unloads current model first.
        If loader_fn is provided, uses it. Otherwise looks up registry loader.
        Returns the loaded model object.
        """
        if self._current_model_id == model_id and self._current_model is not None:
            return self._current_model

        # Unload current model
        self.unload()

        self._set_status(model_id, ModelStatus.LOADING)
        self.model_loading.emit(model_id)

        try:
            if loader_fn is not None:
                model = loader_fn()
            else:
                # Dynamic import from registry
                info = self._registry.get(model_id)
                if info is None:
                    raise ValueError(f"Unknown model: {model_id}")
                model = self._dynamic_load(info)

            self._current_model = model
            self._current_model_id = model_id
            self._set_status(model_id, ModelStatus.LOADED)
            self.model_loaded.emit(model_id)
            self._emit_gpu_status()
            return model

        except Exception as e:
            self._set_status(model_id, ModelStatus.ERROR)
            error_msg = f"{type(e).__name__}: {e}"
            self.model_error.emit(model_id, error_msg)
            cleanup_gpu()
            raise

    def unload(self):
        """Unload the current model and free GPU memory."""
        if self._current_model is not None:
            model_id = self._current_model_id
            try:
                # Try calling model's own cleanup if available
                if hasattr(self._current_model, "cleanup"):
                    self._current_model.cleanup()
                elif hasattr(self._current_model, "to"):
                    self._current_model.to("cpu")
            except Exception:
                pass

            del self._current_model
            self._current_model = None

            cleanup_gpu()

            if model_id:
                if self._is_model_cached(model_id):
                    self._set_status(model_id, ModelStatus.DOWNLOADED)
                else:
                    self._set_status(model_id, ModelStatus.NOT_DOWNLOADED)
                self.model_unloaded.emit(model_id)
                self._current_model_id = None

            self._emit_gpu_status()

    def _dynamic_load(self, info: ModelInfo) -> Any:
        """Dynamically import and call a model's loader function."""
        import importlib
        module = importlib.import_module(info.loader_module)
        loader = getattr(module, info.loader_fn)
        cache_dir = self._settings.get("model_hub.cache_dir")
        return loader(
            cache_dir=cache_dir,
            model_id=info.model_id,
            source=info.source,
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
        info = self._registry.get(model_id)
        if info is None:
            raise ValueError(f"Unknown model: {model_id}")

        # Pip-managed models (Demucs, DiffSinger) handle their own downloads
        if info.pip_managed:
            self._set_status(model_id, ModelStatus.DOWNLOADED)
            self.download_completed.emit(model_id)
            if progress_cb:
                progress_cb(100)
            return

        if not info.source:
            raise ValueError(f"No download source for model: {model_id}")

        self._set_status(model_id, ModelStatus.DOWNLOADING)
        self.download_started.emit(model_id)

        cache_path = self.get_cache_dir(model_id)

        try:
            from huggingface_hub import snapshot_download
            import time as _time

            cache_dir = str(cache_path.parent)

            kwargs = {
                "repo_id": info.source,
                "cache_dir": cache_dir,
                "local_dir": str(cache_path),
            }

            if info.allow_patterns:
                kwargs["allow_patterns"] = info.allow_patterns

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
                snapshot_download(**kwargs)
            finally:
                _download_done.set()
                poll_thread.join(timeout=2)

            # -- Write completion marker --
            self._write_complete_marker(model_id, cache_path)

            self._set_status(model_id, ModelStatus.DOWNLOADED)
            self.download_completed.emit(model_id)
            if progress_cb:
                progress_cb(100)

        except Exception as e:
            self._set_status(model_id, ModelStatus.ERROR)
            self.download_error.emit(model_id, str(e))
            raise

    def _write_complete_marker(self, model_id: str, cache_path: Path):
        """Write a marker file indicating download is complete with metadata."""
        import time as _time
        file_count = 0
        total_size = 0
        for f in cache_path.rglob("*"):
            if f.is_file() and f.name != self.COMPLETE_MARKER:
                file_count += 1
                total_size += f.stat().st_size

        marker = cache_path / self.COMPLETE_MARKER
        marker.write_text(json.dumps({
            "model_id": model_id,
            "timestamp": _time.time(),
            "file_count": file_count,
            "total_bytes": total_size,
            "source": self._registry[model_id].source if model_id in self._registry else "",
        }, indent=2))

    def verify_download(self, model_id: str) -> tuple[bool, str]:
        """
        Verify a download is complete. Returns (ok, reason).
        Checks for completion marker and basic file count sanity.
        """
        info = self._registry.get(model_id)
        if not info:
            return False, "Unknown model"
        if info.pip_managed:
            return True, "pip managed"

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
            expected_files = meta.get("file_count", 0)
            actual_files = sum(
                1 for f in cache_path.rglob("*")
                if f.is_file() and f.name != self.COMPLETE_MARKER
            )
            if actual_files < expected_files:
                return False, f"Missing files ({actual_files}/{expected_files})"
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

        # Pip-managed models are always considered "downloaded" if the package exists
        if info and info.pip_managed:
            return True

        cache_path = self.get_cache_dir(model_id)

        # Must have completion marker — bare files without it = partial download
        marker = cache_path / self.COMPLETE_MARKER
        if marker.exists():
            return True

        # Also check HuggingFace default cache (for models downloaded outside Slunder)
        if info and info.source:
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            model_dir = hf_cache / f"models--{info.source.replace('/', '--')}"
            # HF cache uses refs/main as its own completion signal
            if model_dir.exists() and (model_dir / "refs" / "main").exists():
                return True
        return False

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

    def get_total_disk_usage(self) -> float:
        """Get total disk usage of all downloaded models in GB."""
        total = 0.0
        base = Path(self._settings.get("model_hub.cache_dir", str(get_config_dir() / "models")))
        if base.exists():
            for f in base.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total / (1024**3)

    # ── GPU Status ─────────────────────────────────────────────────────────────

    def get_gpu_status(self) -> dict:
        """Get current GPU status including loaded model info."""
        gpu = get_gpu_info()
        gpu["current_model"] = self._current_model_id
        gpu["current_model_name"] = (
            self._registry[self._current_model_id].name
            if self._current_model_id and self._current_model_id in self._registry
            else None
        )
        return gpu

    def _emit_gpu_status(self):
        self.gpu_status_changed.emit(self.get_gpu_status())

    # ── Internal ───────────────────────────────────────────────────────────────

    def _set_status(self, model_id: str, status: ModelStatus):
        self._status[model_id] = status
        self.status_changed.emit(model_id, status.value)
