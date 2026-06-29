"""
Slunder Studio v0.1.19 — Threading & Worker System
InferenceWorker base class, WorkflowQueue for multi-step pipelines,
cancellation support, and progress aggregation.
"""
import threading
import traceback
from typing import Any, Callable, Optional
from collections import deque

from PySide6.QtCore import QThread, Signal, QObject, QTimer

from core.job_state import JobStore, extract_output_paths


class CancelledJobError(RuntimeError):
    """Raised by long-running tasks after cleaning or reporting partial outputs."""

    def __init__(self, message: str = "Job cancelled", outputs: Any = None):
        super().__init__(message)
        self.outputs = outputs


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
    finished = Signal(object)
    error = Signal(str)
    cancelled = Signal()

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
        if job_kind:
            self._job_record = self._job_store.create(
                job_kind,
                job_label or getattr(task_fn, "__name__", "Inference job"),
                inputs=job_inputs or {},
                metadata=job_metadata or {},
            )
            self.job_id = self._job_record.id

    def run(self):
        if self.job_id:
            self._job_store.mark_running(self.job_id, "Starting")
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
            if self._cancel_event.is_set():
                self._job_store.cleanup_outputs(output_paths)
                if self.job_id:
                    self._job_store.mark_cancelled(self.job_id, outputs=outputs)
                self.log.emit("Worker cancelled; partial outputs cleaned.")
                self.cancelled.emit()
            else:
                if self.job_id:
                    self._job_store.mark_completed(self.job_id, outputs=outputs)
                self.finished.emit(self._result)
        except CancelledJobError as e:
            output_paths = extract_output_paths(e.outputs)
            outputs = {"paths": output_paths} if output_paths else {}
            self._job_store.cleanup_outputs(output_paths)
            if self.job_id:
                self._job_store.mark_cancelled(self.job_id, outputs=outputs)
            self.log.emit(str(e))
            self.cancelled.emit()
        except Exception as e:
            tb = traceback.format_exc()
            self.log.emit(f"Worker error:\n{tb}")
            if self.job_id:
                self._job_store.mark_failed(self.job_id, f"{type(e).__name__}: {e}")
            self.error.emit(f"{type(e).__name__}: {e}")

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
            self._job_store.update_progress(self.job_id, pct)
        self.progress.emit(pct)

    def _emit_step(self, message: str):
        if self.job_id:
            self._job_store.update_message(self.job_id, message)
        self.step_info.emit(message)

    def _emit_log(self, message: str):
        if self.job_id:
            self._job_store.update_message(self.job_id, message)
        self.log.emit(message)


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
            self.download_fn(
                self.model_id,
                progress_cb=self._emit_progress,
                speed_cb=self.speed.emit,
                downloaded_cb=self.downloaded.emit,
                cancel_event=self._cancel_event,
            )
            if not self._cancel_event.is_set():
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
            self._job_store.mark_failed(
                self.job_id,
                f"{type(e).__name__}: {e}",
                outputs={"model_id": self.model_id},
            )
            self.error.emit(f"{type(e).__name__}: {e}")

    def cancel(self):
        self._cancel_event.set()
        self._job_store.request_cancel(self.job_id)

    def _emit_progress(self, pct: int):
        self._job_store.update_progress(self.job_id, pct)
        self.progress.emit(pct)


class WorkflowStep:
    """A single step in a multi-step workflow pipeline."""

    def __init__(self, name: str, task_fn: Callable, *args, **kwargs):
        self.name = name
        self.task_fn = task_fn
        self.args = args
        self.kwargs = kwargs
        self.result: Any = None
        self.completed: bool = False
        self.error: Optional[str] = None


class WorkflowQueue(QObject):
    """
    Sequential task runner for multi-step AI pipelines.
    Example: lyrics -> song generation -> mastering

    Each step's result is passed as first arg to the next step.
    Steps run sequentially, swapping models via ModelManager as needed.

    Signals:
        step_started(str, int, int)  - (step_name, current_index, total_steps)
        step_progress(int)           - 0-100 for current step
        step_completed(str, object)  - (step_name, result)
        step_error(str, str)         - (step_name, error_message)
        all_completed(list)          - list of all results
        workflow_error(str)          - fatal error, workflow aborted
        overall_progress(int)        - 0-100 across all steps
    """
    step_started = Signal(str, int, int)
    step_progress = Signal(int)
    step_completed = Signal(str, object)
    step_error = Signal(str, str)
    all_completed = Signal(list)
    workflow_error = Signal(str)
    overall_progress = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._steps: list[WorkflowStep] = []
        self._current_index = 0
        self._worker: Optional[InferenceWorker] = None
        self._cancel_event = threading.Event()
        self._results: list = []
        self._running = False

    def add_step(self, name: str, task_fn: Callable, *args, **kwargs):
        """Add a step to the workflow pipeline."""
        self._steps.append(WorkflowStep(name, task_fn, *args, **kwargs))

    def clear(self):
        """Clear all steps."""
        self._steps.clear()
        self._results.clear()
        self._current_index = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def total_steps(self) -> int:
        return len(self._steps)

    def start(self):
        """Begin executing the workflow from the first step."""
        if not self._steps:
            return
        self._current_index = 0
        self._results = []
        self._cancel_event.clear()
        self._running = True
        self._run_current_step()

    def cancel(self):
        """Cancel the current workflow."""
        self._cancel_event.set()
        if self._worker:
            self._worker.cancel()
        self._running = False

    def _run_current_step(self):
        """Execute the current step."""
        if self._current_index >= len(self._steps):
            self._running = False
            self.all_completed.emit(self._results)
            return

        if self._cancel_event.is_set():
            self._running = False
            return

        step = self._steps[self._current_index]
        self.step_started.emit(step.name, self._current_index, len(self._steps))

        # Pass previous step's result as first argument if available
        args = list(step.args)
        if self._results:
            args.insert(0, self._results[-1])

        self._worker = InferenceWorker(step.task_fn, *args, **step.kwargs)
        self._worker.progress.connect(self._on_step_progress)
        self._worker.finished.connect(self._on_step_finished)
        self._worker.error.connect(self._on_step_error)
        self._worker.start()

    def _on_step_progress(self, pct: int):
        """Handle progress from current step."""
        self.step_progress.emit(pct)
        # Calculate overall progress
        base = int((self._current_index / len(self._steps)) * 100)
        step_contribution = int((pct / 100) * (100 / len(self._steps)))
        self.overall_progress.emit(min(base + step_contribution, 100))

    def _on_step_finished(self, result):
        """Handle completion of current step."""
        step = self._steps[self._current_index]
        step.result = result
        step.completed = True
        self._results.append(result)

        self.step_completed.emit(step.name, result)
        self._current_index += 1

        # Small delay before next step to allow GPU cleanup
        QTimer.singleShot(100, self._run_current_step)

    def _on_step_error(self, error_msg: str):
        """Handle error in current step."""
        step = self._steps[self._current_index]
        step.error = error_msg
        self.step_error.emit(step.name, error_msg)
        self.workflow_error.emit(f"Workflow failed at step '{step.name}': {error_msg}")
        self._running = False


class DebouncedCallback:
    """
    Debounce a callback so it only fires after a delay with no new calls.
    Useful for reactive settings changes that trigger expensive operations.
    """

    def __init__(self, callback: Callable, delay_ms: int = 300):
        self._callback = callback
        self._delay_ms = delay_ms
        self._timer = None

    def __call__(self, *args, **kwargs):
        if self._timer is not None:
            self._timer.stop()
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(lambda: self._callback(*args, **kwargs))
        self._timer.start(self._delay_ms)
