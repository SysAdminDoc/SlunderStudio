"""
Slunder Studio — Threading & Worker System
InferenceWorker and DownloadWorker primitives with cancellation support and
durable job progress.
"""
import threading
import time
import traceback
import logging
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal

from core.job_state import JobLog, JobStore, extract_output_paths


logger = logging.getLogger(__name__)


JOB_PROGRESS_PERSIST_INTERVAL = 0.1
WORKER_SHUTDOWN_TIMEOUT_MS = 10_000

_worker_registry_lock = threading.RLock()
_active_workers: set[QThread] = set()


def _register_worker(worker: QThread) -> None:
    with _worker_registry_lock:
        _active_workers.add(worker)


def active_workers() -> tuple[QThread, ...]:
    """Return running workers and discard completed thread wrappers."""
    with _worker_registry_lock:
        completed = {worker for worker in _active_workers if not worker.isRunning()}
        _active_workers.difference_update(completed)
        return tuple(_active_workers)


def shutdown_workers(timeout_ms: int = WORKER_SHUTDOWN_TIMEOUT_MS) -> bool:
    """Cancel and join every running inference/download worker.

    A worker that ignores cancellation is left running and makes the operation
    fail closed, so callers never unload a model while a worker may still use it.
    """
    workers = active_workers()
    cancellation_failed = False
    for worker in workers:
        try:
            worker.cancel()
        except Exception:  # noqa: BLE001 - shutdown must continue to other jobs
            cancellation_failed = True

    deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000
    for worker in workers:
        if not worker.isRunning():
            continue
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
        try:
            worker.wait(remaining_ms)
        except RuntimeError:
            cancellation_failed = True

    still_running = any(worker.isRunning() for worker in workers)
    return not cancellation_failed and not still_running


def _result_is_cancelled(result: Any) -> bool:
    """Accept either the typed cancellation property or its method variant."""
    for name in ("cancelled", "is_cancelled"):
        value = getattr(result, name, False)
        if callable(value):
            try:
                value = value()
            except TypeError:
                # ``EngineRunResult.cancelled`` is a class factory that shares
                # a name with the instance-level cancellation property used by
                # other result contracts. It requires a capability argument and
                # is not itself a cancellation flag.
                continue
        if value:
            return True
    return False


class CancelledJobError(RuntimeError):
    """Raised by long-running tasks after cleaning or reporting partial outputs.

    `outputs` are the paths the task owns. `preserved` lists the subset that
    finished and was verified before cancellation: those are kept, only the
    rest are removed. A task that passes no `preserved` keeps nothing, which
    matches the old all-or-nothing behaviour.
    """

    def __init__(self, message: str = "Job cancelled", outputs: Any = None,
                 preserved: Any = None, result: Any = None):
        super().__init__(message)
        self.outputs = outputs
        self.preserved = preserved
        self.result = result


class InferenceWorker(QThread):
    """
    Base worker thread for AI model inference.
    All model operations MUST run through this to avoid GUI freezing.

    Signals:
        progress(int)     - 0-100 percentage
        step_info(str)    - current step description
        log(str)          - log messages for console
        finished(object)  - result payload on success
        error(str)        - error message on failure
    """
    progress = Signal(int)
    step_info = Signal(str)
    log = Signal(str)
    token = Signal(str)
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()
    thread_stopped = Signal()

    def __init__(
        self,
        task_fn: Callable,
        *args,
        job_kind: str = "",
        job_label: str = "",
        job_inputs: Optional[dict[str, Any]] = None,
        job_metadata: Optional[dict[str, Any]] = None,
        job_store: Optional[JobStore] = None,
        **kwargs,
    ):
        super().__init__()
        self.task_fn = task_fn
        self.args = args
        self.kwargs = kwargs
        self._cancel_event = threading.Event()
        self._result = None
        self._job_store = job_store or JobStore()
        self._job_record = None
        self.job_id = ""
        self._last_job_progress_persisted_at = 0.0
        self._job_log: Optional[JobLog] = None
        if job_kind:
            self._job_record = self._job_store.create(
                job_kind,
                job_label or getattr(task_fn, "__name__", "Inference job"),
                inputs=job_inputs or {},
                metadata=job_metadata or {},
            )
            self.job_id = self._job_record.id
            self._job_log = JobLog(self.job_id)

    def run(self):
        if self.job_id:
            self._job_store.mark_running(self.job_id, "Starting")
        if self._job_log:
            self._job_log.info(f"Job started: {self.job_id}")
        try:
            self._result = self.task_fn(
                *self.args,
                **self.kwargs,
                progress_cb=self._emit_progress,
                step_cb=self._emit_step,
                log_cb=self._emit_log,
                cancel_event=self._cancel_event,
            )
            output_paths = extract_output_paths(self._result)
            outputs = {"paths": output_paths} if output_paths else {}
            result_metadata = _extract_job_metadata(self._result)
            if self._cancel_event.is_set() or _result_is_cancelled(self._result):
                self._job_store.cleanup_outputs(output_paths)
                if self.job_id:
                    self._job_store.mark_cancelled(
                        self.job_id,
                        outputs=outputs,
                        metadata=result_metadata,
                    )
                if self._job_log:
                    self._job_log.warn("Cancelled; partial outputs cleaned.")
                self.log.emit("Worker cancelled; partial outputs cleaned.")
                self.cancelled.emit()
            else:
                semantic_success = getattr(self._result, "is_success", None)
                if callable(semantic_success):
                    semantic_success = semantic_success()
                if semantic_success is False:
                    semantic_error = str(
                        getattr(self._result, "error", "")
                        or "Task returned an unsuccessful result"
                    )
                    if self.job_id:
                        self._job_store.mark_failed(
                            self.job_id,
                            semantic_error,
                            outputs=outputs,
                            metadata=result_metadata,
                        )
                    if self._job_log:
                        self._job_log.error(semantic_error)
                else:
                    if self.job_id:
                        self._job_store.mark_completed(
                            self.job_id,
                            outputs=outputs,
                            metadata=result_metadata,
                        )
                    if self._job_log:
                        self._job_log.info(
                            f"Completed with {len(output_paths)} output(s)."
                        )
                self.finished.emit(self._result)
        except CancelledJobError as e:
            output_paths = extract_output_paths(e.outputs)
            preserved = extract_output_paths(e.preserved)
            preserved_set = set(preserved)
            # Only outputs the task did not vouch for are removed.
            partial_paths = [p for p in output_paths if p not in preserved_set]
            outputs = {}
            if output_paths:
                outputs["paths"] = output_paths
            if preserved:
                outputs["preserved_paths"] = preserved
            self._job_store.cleanup_outputs(partial_paths)
            if self.job_id:
                self._job_store.mark_cancelled(self.job_id, outputs=outputs)
            if self._job_log:
                self._job_log.warn(
                    f"CancelledJobError: {e} "
                    f"(kept {len(preserved)}, removed {len(partial_paths)})"
                )
            self.log.emit(str(e))
            self._result = e.result if e.result is not None else self._result
            self.cancelled.emit()
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Inference worker failed")
            if self._job_log:
                self._job_log.error(f"{type(e).__name__}: {e}")
            self.log.emit(f"Worker error:\n{tb}")
            if self.job_id:
                self._job_store.mark_failed(self.job_id, f"{type(e).__name__}: {e}")
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            try:
                if self._job_log:
                    self._job_log.save()
            finally:
                # This is distinct from the result-bearing ``finished``
                # signal: receivers can release their QThread wrapper only
                # after all task cleanup has completed, without calling wait.
                self.thread_stopped.emit()

    def start(self, *args, **kwargs):
        """Register the thread before it can begin work."""
        _register_worker(self)
        try:
            return super().start(*args, **kwargs)
        except Exception:
            with _worker_registry_lock:
                _active_workers.discard(self)
            raise

    @property
    def result(self) -> Any:
        """Last result, including the partial batch reported on cancellation."""
        return self._result

    def cancel(self):
        """Request cancellation. Task must check cancel_event.is_set() periodically."""
        self._cancel_event.set()
        if self.job_id:
            self._job_store.request_cancel(self.job_id)

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _emit_progress(self, pct: int):
        if self.job_id:
            now = time.monotonic()
            if now - self._last_job_progress_persisted_at >= JOB_PROGRESS_PERSIST_INTERVAL:
                self._job_store.update_progress(self.job_id, pct)
                self._last_job_progress_persisted_at = now
        self.progress.emit(pct)

    def _emit_step(self, message: str):
        if self.job_id:
            self._job_store.update_message(self.job_id, message)
        if self._job_log:
            self._job_log.info(message)
        self.step_info.emit(message)

    def _emit_log(self, message: str):
        if self.job_id:
            self._job_store.update_message(self.job_id, message)
        if self._job_log:
            self._job_log.info(message)
        self.log.emit(message)


def _extract_job_metadata(result: Any) -> dict[str, Any]:
    """Read optional bounded result metadata without coupling workers to engines."""
    metadata = getattr(result, "job_metadata", None)
    if callable(metadata):
        try:
            value = metadata()
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            return {}
    return metadata if isinstance(metadata, dict) else {}


class DownloadWorker(QThread):
    """
    Worker specifically for model downloads with byte-level progress.

    Signals:
        progress(int)         - 0-100 percentage
        speed(str)            - download speed string (e.g., "12.3 MB/s")
        downloaded(str)       - bytes downloaded string (e.g., "234 MB / 1.2 GB")
        finished(str)         - model ID on success
        error(str)            - error message on failure
    """
    progress = Signal(int)
    speed = Signal(str)
    downloaded = Signal(str)
    finished = Signal(str)
    error = Signal(str)
    cancelled = Signal(str)

    def __init__(
        self,
        download_fn: Callable,
        model_id: str,
        model_name: str = "",
        job_store: Optional[JobStore] = None,
    ):
        super().__init__()
        self.download_fn = download_fn
        self.model_id = model_id
        self._cancel_event = threading.Event()
        self._job_store = job_store or JobStore()
        self._last_job_progress_persisted_at = 0.0
        self._job_record = self._job_store.create(
            "model_download",
            model_name or model_id,
            inputs={"model_id": model_id},
            metadata={"model_id": model_id},
        )
        self.job_id = self._job_record.id

    def run(self):
        self._job_store.mark_running(self.job_id, "Starting download")
        try:
            download_result = self.download_fn(
                self.model_id,
                progress_cb=self._emit_progress,
                speed_cb=self.speed.emit,
                downloaded_cb=self.downloaded.emit,
                cancel_event=self._cancel_event,
            )
            # A download may finish and write its marker just as cancellation
            # is requested. An explicit True result means that artifact is
            # complete and must be retained.
            if download_result is True or not self._cancel_event.is_set():
                self._job_store.mark_completed(
                    self.job_id,
                    outputs={"model_id": self.model_id},
                )
                self.finished.emit(self.model_id)
            else:
                self._job_store.mark_cancelled(
                    self.job_id,
                    outputs={"model_id": self.model_id},
                    recoverable=True,
                )
                self.cancelled.emit(self.model_id)
        except CancelledJobError:
            self._job_store.mark_cancelled(
                self.job_id,
                outputs={"model_id": self.model_id},
                recoverable=True,
            )
            self.cancelled.emit(self.model_id)
        except Exception as e:
            logger.exception("Download worker failed for model %s", self.model_id)
            self._job_store.mark_failed(
                self.job_id,
                f"{type(e).__name__}: {e}",
                outputs={"model_id": self.model_id},
            )
            self.error.emit(f"{type(e).__name__}: {e}")

    def start(self, *args, **kwargs):
        """Register the thread before it can begin downloading."""
        _register_worker(self)
        try:
            return super().start(*args, **kwargs)
        except Exception:
            with _worker_registry_lock:
                _active_workers.discard(self)
            raise

    def cancel(self):
        self._cancel_event.set()
        self._job_store.request_cancel(self.job_id)

    def _emit_progress(self, pct: int):
        now = time.monotonic()
        if now - self._last_job_progress_persisted_at >= JOB_PROGRESS_PERSIST_INTERVAL:
            self._job_store.update_progress(self.job_id, pct)
            self._last_job_progress_persisted_at = now
        self.progress.emit(pct)
