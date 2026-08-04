"""Central, cancellation-aware admission control for model work."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


class AdmissionError(RuntimeError):
    """Base error for a resource admission decision."""


class AdmissionBusyError(AdmissionError):
    """Raised when the same keyed resource is already queued or active."""


class AdmissionCancelledError(AdmissionError):
    """Raised when a queued request is cancelled before acquiring capacity."""


class AdmissionTimeoutError(AdmissionError):
    """Raised when a queued request exceeds its caller-supplied wait limit."""


@dataclass(frozen=True)
class AdmissionSnapshot:
    """Serializable capacity state for diagnostics and support reports."""

    limits: dict[str, int]
    active: dict[str, int]
    queued: dict[str, int]
    active_download_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "limits": dict(self.limits),
            "active": dict(self.active),
            "queued": dict(self.queued),
            "active_download_keys": list(self.active_download_keys),
        }


class AdmissionLease:
    """One acquired slot; release is idempotent and context-manager friendly."""

    def __init__(self, controller: "AdmissionController", kind: str, key: str):
        self._controller = controller
        self.kind = kind
        self.key = key
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._controller._release(self.kind, self.key)

    def __enter__(self) -> "AdmissionLease":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


class AdmissionController:
    """Bound model downloads and inference without busy-waiting.

    Downloads are limited globally and keyed by model ID so two requests for
    the same cache can never both enter. Inference is a separate capacity pool;
    its default of one protects the one-large-model GPU policy while still
    allowing downloads to resume independently.
    """

    DOWNLOAD = "download"
    INFERENCE = "inference"
    _KINDS = (DOWNLOAD, INFERENCE)

    def __init__(self, *, max_downloads: int = 2, max_inference: int = 1):
        self._lock = threading.RLock()
        self._limits = {
            self.DOWNLOAD: self._positive_limit(max_downloads, self.DOWNLOAD),
            self.INFERENCE: self._positive_limit(max_inference, self.INFERENCE),
        }
        self._semaphores = {
            kind: threading.BoundedSemaphore(limit)
            for kind, limit in self._limits.items()
        }
        self._active = {kind: 0 for kind in self._KINDS}
        self._queued = {kind: 0 for kind in self._KINDS}
        self._download_keys: set[str] = set()

    @staticmethod
    def _positive_limit(value: int, kind: str) -> int:
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{kind} admission limit must be an integer") from exc
        if limit < 1:
            raise ValueError(f"{kind} admission limit must be at least one")
        return limit

    @property
    def limits(self) -> dict[str, int]:
        with self._lock:
            return dict(self._limits)

    def snapshot(self) -> AdmissionSnapshot:
        with self._lock:
            return AdmissionSnapshot(
                limits=dict(self._limits),
                active=dict(self._active),
                queued=dict(self._queued),
                active_download_keys=tuple(sorted(self._download_keys)),
            )

    def acquire(
        self,
        kind: str,
        *,
        key: str = "",
        cancel_event: Optional[threading.Event] = None,
        timeout: Optional[float] = None,
        wait_cb: Optional[Callable[[], None]] = None,
    ) -> AdmissionLease:
        """Wait for one slot, checking cancellation at short intervals."""
        if kind not in self._KINDS:
            raise ValueError(f"Unknown admission kind: {kind}")
        normalized_key = str(key or "")
        with self._lock:
            if kind == self.DOWNLOAD and normalized_key:
                if normalized_key in self._download_keys:
                    raise AdmissionBusyError(
                        f"A download for {normalized_key} is already queued or active"
                    )
                self._download_keys.add(normalized_key)
            self._queued[kind] += 1

        started = time.monotonic()
        notified = False
        acquired = False
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise AdmissionCancelledError(
                        f"{kind.title()} admission cancelled while waiting"
                    )
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise AdmissionTimeoutError(
                        f"Timed out waiting for {kind} admission capacity"
                    )
                if not notified and wait_cb is not None:
                    wait_cb()
                    notified = True
                if self._semaphores[kind].acquire(timeout=0.1):
                    acquired = True
                    with self._lock:
                        self._queued[kind] -= 1
                        self._active[kind] += 1
                    return AdmissionLease(self, kind, normalized_key)
        finally:
            if not acquired:
                with self._lock:
                    self._queued[kind] -= 1
                    if kind == self.DOWNLOAD and normalized_key:
                        self._download_keys.discard(normalized_key)

    def _release(self, kind: str, key: str) -> None:
        with self._lock:
            if self._active[kind] <= 0:
                return
            self._active[kind] -= 1
            if kind == self.DOWNLOAD and key:
                self._download_keys.discard(key)
        self._semaphores[kind].release()


_global_lock = threading.Lock()
_global_controller: Optional[AdmissionController] = None


def global_admission_controller() -> AdmissionController:
    """Return the process-wide controller configured at first use."""
    global _global_controller
    if _global_controller is None:
        with _global_lock:
            if _global_controller is None:
                try:
                    from core.settings import Settings

                    settings = Settings()
                    max_downloads = settings.get("model_hub.max_concurrent_downloads", 2)
                    max_inference = settings.get("model_hub.max_concurrent_inference", 1)
                except Exception:  # noqa: BLE001 - safe defaults are the admission boundary
                    max_downloads, max_inference = 2, 1
                try:
                    _global_controller = AdmissionController(
                        max_downloads=max_downloads,
                        max_inference=max_inference,
                    )
                except ValueError:
                    _global_controller = AdmissionController()
    return _global_controller


def reset_global_admission_controller() -> None:
    """Reset the lazy singleton for tests and controlled application restarts."""
    global _global_controller
    with _global_lock:
        if _global_controller is not None:
            snapshot = _global_controller.snapshot()
            if any(snapshot.active.values()) or any(snapshot.queued.values()):
                raise RuntimeError("Cannot reset admission control while work is active")
        _global_controller = None


MODEL_INFERENCE_JOB_KINDS = frozenset(
    {
        "lyrics_generation",
        "song_generation",
        "midi_generation",
        "ai_producer",
        "vocal_synthesis",
        "vocal_pronunciation_correction",
        "lyric_melody",
        "voice_conversion",
        "voice_clone",
        "vocal_autotune",
        "stem_separation",
        "sfx_generation",
        "reference_analysis",
        "model_activation",
        "model_update",
        "model_rollback",
    }
)


def admission_kind_for_job(job_kind: str) -> Optional[str]:
    """Map an explicit durable job kind to a model resource pool."""
    return (
        AdmissionController.INFERENCE
        if str(job_kind or "") in MODEL_INFERENCE_JOB_KINDS
        else None
    )
