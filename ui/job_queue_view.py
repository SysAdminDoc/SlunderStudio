"""Durable job queue panel used by Song Forge and other batch workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.job_queue import estimate_job_resources, export_selected_outputs
from core.job_state import JobRecord, JobStatus, JobStore, extract_output_paths
from core.i18n import tr
from ui.accessibility import install_accessibility
from ui.file_dialogs import choose_directory
from ui.theme import Palette


class JobQueueView(QWidget):
    """Inspect, requeue, and export durable jobs without touching active jobs."""

    job_requeued = Signal(object)

    def __init__(self, parent=None, toast_mgr=None, job_store: Optional[JobStore] = None):
        super().__init__(parent)
        self.toast_mgr = toast_mgr
        self._job_store = job_store or JobStore()
        self._records: list[JobRecord] = []
        self._setup_ui()
        # A process restart cannot leave a worker marked active forever.  The
        # ledger converts those records into explicit recoverable queue items.
        self._job_store.recover_stale_jobs()
        self.refresh()

    @property
    def job_store(self) -> JobStore:
        return self._job_store

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(tr("job_queue.title"))
        title.setStyleSheet(
            f"color: {Palette.TEXT}; font-weight: bold; font-size: 9.75pt;"
        )
        header.addWidget(title)
        self._count_label = QLabel(tr("job_queue.count", count=0))
        self._count_label.setStyleSheet(
            f"color: {Palette.OVERLAY0}; font-size: 8.25pt;"
        )
        header.addWidget(self._count_label)
        header.addStretch()
        self._refresh_btn = QPushButton(tr("job_queue.refresh"))
        self._refresh_btn.setMinimumHeight(28)
        self._refresh_btn.clicked.connect(self.refresh)
        header.addWidget(self._refresh_btn)
        layout.addLayout(header)

        self._jobs = QListWidget()
        self._jobs.setMinimumHeight(150)
        self._jobs.currentItemChanged.connect(self._on_job_selected)
        layout.addWidget(self._jobs, 1)

        self._details = QLabel(tr("job_queue.details_empty"))
        self._details.setWordWrap(True)
        self._details.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8.25pt; padding: 4px;"
        )
        layout.addWidget(self._details)

        action_row = QHBoxLayout()
        self._resume_btn = QPushButton(tr("job_queue.resume"))
        self._resume_btn.setEnabled(False)
        self._resume_btn.clicked.connect(lambda: self._requeue_selected(resume=True))
        action_row.addWidget(self._resume_btn)

        self._retry_btn = QPushButton(tr("job_queue.retry"))
        self._retry_btn.setEnabled(False)
        self._retry_btn.clicked.connect(lambda: self._requeue_selected(resume=False))
        action_row.addWidget(self._retry_btn)

        self._export_btn = QPushButton(tr("job_queue.export_selected"))
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_selected)
        action_row.addWidget(self._export_btn)
        action_row.addStretch()
        layout.addLayout(action_row)

        self._outputs_label = QLabel(tr("job_queue.completed_outputs"))
        self._outputs_label.setStyleSheet(
            f"color: {Palette.TEXT}; font-weight: bold; font-size: 8.75pt;"
        )
        layout.addWidget(self._outputs_label)
        self._outputs = QListWidget()
        self._outputs.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        layout.addWidget(self._outputs, 1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8.25pt;"
        )
        layout.addWidget(self._status)

        install_accessibility(
            self,
            tr("job_queue.accessibility.name"),
            named_controls=[
                (self._jobs, tr("job_queue.accessibility.jobs_name"), tr("job_queue.accessibility.jobs_description")),
                (self._refresh_btn, tr("job_queue.accessibility.refresh_name"), tr("job_queue.accessibility.refresh_description")),
                (self._resume_btn, tr("job_queue.accessibility.resume_name"), tr("job_queue.accessibility.resume_description")),
                (self._retry_btn, tr("job_queue.accessibility.retry_name"), tr("job_queue.accessibility.retry_description")),
                (self._export_btn, tr("job_queue.accessibility.export_name"), tr("job_queue.accessibility.export_description")),
                (self._outputs, tr("job_queue.accessibility.outputs_name"), tr("job_queue.accessibility.outputs_description")),
            ],
            tab_order=[
                self._jobs,
                self._refresh_btn,
                self._resume_btn,
                self._retry_btn,
                self._export_btn,
                self._outputs,
            ],
        )

    def refresh(self):
        """Reload jobs and preserve the selected record when possible."""
        selected_id = self.selected_record.id if self.selected_record else ""
        self._records = self._job_store.list_records()[:200]
        self._jobs.blockSignals(True)
        try:
            self._jobs.clear()
            selected_item = None
            for record in self._records:
                item = QListWidgetItem(self._job_label(record))
                item.setData(Qt.ItemDataRole.UserRole, record.id)
                self._jobs.addItem(item)
                if record.id == selected_id:
                    selected_item = item
        finally:
            self._jobs.blockSignals(False)
        self._count_label.setText(tr("job_queue.count", count=len(self._records)))
        if selected_item is not None:
            self._jobs.setCurrentItem(selected_item)
        elif self._jobs.count():
            self._jobs.setCurrentRow(0)
        else:
            self._on_job_selected(None, None)
        self._refresh_outputs()

    @property
    def selected_record(self) -> Optional[JobRecord]:
        item = self._jobs.currentItem()
        if item is None:
            return None
        job_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        return next((record for record in self._records if record.id == job_id), None)

    @staticmethod
    def _job_label(record: JobRecord) -> str:
        estimate = estimate_job_resources(record).summary()
        return tr(
            "job_queue.job_label",
            status=record.status,
            label=record.label,
            progress=record.progress,
            estimate=estimate,
        )

    def _on_job_selected(self, _current, _previous):
        record = self.selected_record
        if record is None:
            self._details.setText(tr("job_queue.details_empty"))
            self._resume_btn.setEnabled(False)
            self._retry_btn.setEnabled(False)
            return
        estimate = estimate_job_resources(record)
        details = tr(
            "job_queue.details",
            label=record.label,
            status=record.status,
            progress=record.progress,
            estimate=estimate.summary(),
            basis=estimate.basis,
        )
        if record.error:
            details += "\n" + tr("job_queue.error", error=record.error[:240])
        self._details.setText(details)
        resumable = record.status == JobStatus.RECOVERABLE or (
            record.status == JobStatus.CANCELLED and record.recoverable
        )
        retryable = record.status in {
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.RECOVERABLE,
        }
        self._resume_btn.setEnabled(resumable)
        self._retry_btn.setEnabled(retryable)

    def _refresh_outputs(self):
        self._outputs.clear()
        seen: set[str] = set()
        for record in self._records:
            if record.status != JobStatus.COMPLETED:
                continue
            for raw_path in extract_output_paths(record):
                try:
                    path = Path(raw_path).resolve(strict=True)
                except (OSError, RuntimeError, ValueError):
                    continue
                if str(path) in seen or not path.is_file():
                    continue
                seen.add(str(path))
                item = QListWidgetItem(
                    tr("job_queue.output_item", label=record.label, name=path.name)
                )
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                item.setFlags(
                    item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(Qt.CheckState.Checked)
                self._outputs.addItem(item)
        self._export_btn.setEnabled(self._outputs.count() > 0)
        if self._outputs.count() == 0:
            self._outputs.addItem(tr("job_queue.no_outputs"))
            self._outputs.item(0).setFlags(Qt.ItemFlag.NoItemFlags)

    def _requeue_selected(self, *, resume: bool):
        record = self.selected_record
        if record is None:
            return
        queued = self._job_store.requeue(record.id, resume=resume)
        if queued is None:
            self._status.setText(tr("job_queue.requeue_unavailable"))
            return
        self.refresh()
        self._status.setText(tr("job_queue.queued", label=queued.label))
        self.job_requeued.emit(queued)

    def _export_selected(self):
        destination = choose_directory(
            self,
            tr("job_queue.export_selected"),
            operation_kind="job_queue_output_export",
        )
        if destination:
            self.export_selected_to(destination)

    def export_selected_to(self, destination: str | Path) -> list[str]:
        """Export checked output rows; exposed for headless verification."""
        selected = [
            str(item.data(Qt.ItemDataRole.UserRole))
            for index in range(self._outputs.count())
            if (item := self._outputs.item(index)) is not None
            and item.checkState() == Qt.CheckState.Checked
            and item.data(Qt.ItemDataRole.UserRole)
        ]
        if not selected:
            self._status.setText(tr("job_queue.check_outputs"))
            return []
        try:
            written = export_selected_outputs(self._records, selected, destination)
        except (OSError, RuntimeError, ValueError) as exc:
            self._status.setText(tr("job_queue.export_failed", error=exc))
            if self.toast_mgr:
                self.toast_mgr.error(tr("job_queue.export_failed", error=exc))
            return []
        self._status.setText(tr("job_queue.exported", count=len(written)))
        if self.toast_mgr:
            self.toast_mgr.success(tr("job_queue.toast_exported", count=len(written)))
        return written


__all__ = ["JobQueueView"]
