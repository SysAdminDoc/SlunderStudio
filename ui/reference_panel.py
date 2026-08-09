"""
Slunder Studio — Reference Panel
Reference track analysis UI: drag-drop audio, view analysis results,
"Match This" one-click generation, and reference library management.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QFileDialog, QListWidget, QListWidgetItem, QScrollArea,
    QGroupBox, QGridLayout, QProgressBar, QCheckBox, QComboBox,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from ui.theme import Palette
from ui.accessibility import install_accessibility
from ui.waveform_widget import WaveformWidget
from ui.file_dialogs import open_audio_file
from core.routing import is_audio_path
from core.i18n import tr


class AnalysisCard(QFrame):
    """Displays a single analysis metric."""
    def __init__(self, label: str, value: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"QFrame {{ background: {Palette.BASE}; border: 1px solid {Palette.SURFACE0}; border-radius: 6px; padding: 8px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self._label = QLabel(label)
        self._label.setStyleSheet(f"color: {Palette.SUBTEXT0}; font-size: 7.5pt; font-weight: bold;")
        layout.addWidget(self._label)

        self._value = QLabel(value)
        self._value.setStyleSheet(f"color: {Palette.TEXT}; font-size: 10.5pt; font-weight: bold;")
        layout.addWidget(self._value)

    def set_value(self, value: str):
        self._value.setText(value)


class ReferencePanel(QWidget):
    """
    Reference track analysis panel.
    Drag-drop or browse for audio -> analyze -> display results -> "Match This"
    """
    match_requested = Signal(dict)  # Emits analysis dict for Song Forge to use
    tags_extracted = Signal(str)  # Emits tag string for quick population
    reference_to_midi = Signal(dict)  # Emits effective constraints for MIDI Studio

    def __init__(self, parent=None):
        super().__init__(parent)
        self._analysis = None
        self._worker = None
        self._analysis_workers = set()
        # Terminal worker signals can arrive while the native QThread is still
        # unwinding.  Keep their payloads here until _release_worker_later has
        # observed a stopped thread, so results never expose live file handles
        # or native worker state to the rest of the UI.
        self._pending_analysis_events = {}
        self._pending_waveform_analysis = None
        # Monotonic token so a result from a superseded file is discarded.
        self._analysis_token = 0
        self._pending_path = ""
        self.setAcceptDrops(True)
        self._setup_ui()
        install_accessibility(
            self,
            tr("reference.accessibility.name"),
            named_controls=[
                (self._browse_btn, tr("reference.accessibility.browse_name"), tr("reference.accessibility.browse_description")),
                (self._match_btn, tr("reference.accessibility.match_name"), tr("reference.accessibility.match_description")),
                (self._use_tags_btn, tr("reference.accessibility.tags_name"), tr("reference.accessibility.tags_description")),
                (self._cancel_btn, tr("reference.accessibility.cancel_name"), tr("reference.accessibility.cancel_description")),
                (self._bpm_override_check, tr("reference.accessibility.bpm_override_name"), tr("reference.accessibility.bpm_override_description")),
                (self._bpm_override_spin, tr("reference.accessibility.bpm_value_name"), tr("reference.accessibility.bpm_value_description")),
                (self._key_override_check, tr("reference.accessibility.key_override_name"), tr("reference.accessibility.key_override_description")),
                (self._key_override_combo, tr("reference.accessibility.key_value_name"), tr("reference.accessibility.key_value_description")),
                (self._sections_override_check, tr("reference.accessibility.sections_override_name"), tr("reference.accessibility.sections_override_description")),
                (self._sections_table, tr("reference.accessibility.sections_table_name"), tr("reference.accessibility.sections_table_description")),
                (self._apply_corrections_btn, tr("reference.accessibility.apply_name"), tr("reference.accessibility.apply_description")),
                (self._reset_corrections_btn, tr("reference.accessibility.reset_name"), tr("reference.accessibility.reset_description")),
                (self._to_midi_btn, tr("reference.accessibility.midi_name"), tr("reference.accessibility.midi_description")),
                (self._asset_list if hasattr(self, "_asset_list") else None, tr("reference.accessibility.library_name"), tr("reference.accessibility.library_description")),
            ],
        )

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title = QLabel(tr("reference.title"))
        title.setStyleSheet(f"color: {Palette.TEXT}; font-weight: bold; font-size: 9.75pt;")
        header.addWidget(title)
        header.addStretch()

        self._browse_btn = QPushButton(tr("reference.browse"))
        self._browse_btn.setMinimumHeight(26)
        self._browse_btn.setProperty("class", "secondary")
        self._browse_btn.clicked.connect(self._browse_file)
        header.addWidget(self._browse_btn)

        layout.addLayout(header)

        # Drop zone / file info
        self._drop_zone = QLabel(tr("reference.drop_zone"))
        self._drop_zone.setAlignment(Qt.AlignCenter)
        self._drop_zone.setMinimumHeight(60)
        self._drop_zone.setStyleSheet(
            f"QLabel {{ background: {Palette.MANTLE}; border: 2px dashed {Palette.SURFACE1}; border-radius: 8px; "
            f"color: {Palette.OVERLAY0}; font-size: 9pt; }}"
        )
        layout.addWidget(self._drop_zone)

        # Mini waveform
        self._waveform = WaveformWidget(show_controls=False)
        self._waveform.audio_load_finished.connect(
            self._on_reference_waveform_finished
        )
        self._waveform.setMinimumHeight(60)
        self._waveform.hide()
        layout.addWidget(self._waveform)

        # Scrollable analysis results
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        results_widget = QWidget()
        self._results_layout = QVBoxLayout(results_widget)
        self._results_layout.setContentsMargins(0, 0, 0, 0)
        self._results_layout.setSpacing(6)

        # Metrics grid
        self._metrics_group = QGroupBox(tr("reference.analysis_group"))
        self._metrics_group.setStyleSheet(
            f"QGroupBox {{ color: {Palette.SUBTEXT0}; border: 1px solid {Palette.SURFACE0}; border-radius: 6px; "
            f"margin-top: 8px; padding-top: 14px; font-size: 8.25pt; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}"
        )
        metrics_grid = QGridLayout(self._metrics_group)
        metrics_grid.setSpacing(6)

        self._bpm_card = AnalysisCard(tr("reference.metrics.bpm"))
        self._key_card = AnalysisCard(tr("reference.metrics.key"))
        self._energy_card = AnalysisCard(tr("reference.metrics.energy"))
        self._brightness_card = AnalysisCard(tr("reference.metrics.brightness"))
        self._density_card = AnalysisCard(tr("reference.metrics.density"))
        self._duration_card = AnalysisCard(tr("reference.metrics.duration"))

        metrics_grid.addWidget(self._bpm_card, 0, 0)
        metrics_grid.addWidget(self._key_card, 0, 1)
        metrics_grid.addWidget(self._energy_card, 1, 0)
        metrics_grid.addWidget(self._brightness_card, 1, 1)
        metrics_grid.addWidget(self._density_card, 2, 0)
        metrics_grid.addWidget(self._duration_card, 2, 1)

        self._metrics_group.hide()
        self._results_layout.addWidget(self._metrics_group)

        # Suggested tags
        self._tags_label = QLabel("")
        self._tags_label.setWordWrap(True)
        self._tags_label.setStyleSheet(f"color: {Palette.TEAL}; font-size: 8.25pt; padding: 4px;")
        self._tags_label.hide()
        self._results_layout.addWidget(self._tags_label)

        self._clap_label = QLabel("")
        self._clap_label.setWordWrap(True)
        self._clap_label.setStyleSheet(f"color: {Palette.BLUE}; font-size: 8.25pt; padding: 4px;")
        self._clap_label.hide()
        self._results_layout.addWidget(self._clap_label)

        # Sections
        self._sections_label = QLabel("")
        self._sections_label.setWordWrap(True)
        self._sections_label.setStyleSheet(f"color: {Palette.SUBTEXT0}; font-size: 8.25pt; padding: 4px;")
        self._sections_label.hide()
        self._results_layout.addWidget(self._sections_label)

        # Editable constraints retain the raw measurements while allowing a
        # user to correct BPM, key, and section boundaries before routing.
        self._corrections_group = QGroupBox(tr("reference.corrections.title"))
        self._corrections_group.setStyleSheet(
            f"QGroupBox {{ color: {Palette.SUBTEXT0}; border: 1px solid {Palette.SURFACE0}; border-radius: 6px; "
            f"margin-top: 8px; padding-top: 14px; font-size: 8.25pt; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}"
        )
        corrections_layout = QVBoxLayout(self._corrections_group)
        corrections_layout.setSpacing(5)

        self._correction_confidence_label = QLabel("")
        self._correction_confidence_label.setWordWrap(True)
        self._correction_confidence_label.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8pt; padding: 2px;"
        )
        corrections_layout.addWidget(self._correction_confidence_label)

        bpm_row = QHBoxLayout()
        self._bpm_override_check = QCheckBox(tr("reference.corrections.override_bpm"))
        self._bpm_override_spin = QDoubleSpinBox()
        self._bpm_override_spin.setRange(20.0, 300.0)
        self._bpm_override_spin.setDecimals(1)
        self._bpm_override_spin.setSingleStep(0.5)
        self._bpm_override_spin.setSuffix(" BPM")
        self._bpm_override_spin.setEnabled(False)
        self._bpm_override_check.toggled.connect(self._bpm_override_spin.setEnabled)
        bpm_row.addWidget(self._bpm_override_check)
        bpm_row.addWidget(self._bpm_override_spin)
        bpm_row.addStretch()
        corrections_layout.addLayout(bpm_row)

        key_row = QHBoxLayout()
        self._key_override_check = QCheckBox(tr("reference.corrections.override_key"))
        self._key_override_combo = QComboBox()
        self._key_override_combo.addItem(
            tr("reference.corrections.use_detected"), ""
        )
        from engines.audio_analyzer import KEY_NAMES

        for key_name in KEY_NAMES:
            self._key_override_combo.addItem(f"{key_name} major", f"{key_name} major")
            self._key_override_combo.addItem(f"{key_name} minor", f"{key_name} minor")
        self._key_override_combo.setEnabled(False)
        self._key_override_check.toggled.connect(self._key_override_combo.setEnabled)
        key_row.addWidget(self._key_override_check)
        key_row.addWidget(self._key_override_combo, 1)
        corrections_layout.addLayout(key_row)

        self._sections_override_check = QCheckBox(
            tr("reference.corrections.override_sections")
        )
        corrections_layout.addWidget(self._sections_override_check)

        self._sections_table = QTableWidget(0, 3)
        self._sections_table.setHorizontalHeaderLabels([
            tr("reference.corrections.section_label"),
            tr("reference.corrections.start"),
            tr("reference.corrections.end"),
        ])
        self._sections_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._sections_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._sections_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self._sections_table.setMinimumHeight(96)
        self._sections_table.setMaximumHeight(180)
        corrections_layout.addWidget(self._sections_table)

        correction_btns = QHBoxLayout()
        self._apply_corrections_btn = QPushButton(
            tr("reference.corrections.apply")
        )
        self._apply_corrections_btn.setProperty("class", "secondary")
        self._apply_corrections_btn.clicked.connect(self._apply_corrections)
        correction_btns.addWidget(self._apply_corrections_btn)

        self._reset_corrections_btn = QPushButton(
            tr("reference.corrections.reset")
        )
        self._reset_corrections_btn.setProperty("class", "secondary")
        self._reset_corrections_btn.clicked.connect(self._reset_corrections)
        correction_btns.addWidget(self._reset_corrections_btn)

        self._to_midi_btn = QPushButton(tr("reference.corrections.use_midi"))
        self._to_midi_btn.setProperty("class", "secondary")
        self._to_midi_btn.clicked.connect(self._on_send_to_midi)
        correction_btns.addWidget(self._to_midi_btn)
        corrections_layout.addLayout(correction_btns)

        self._correction_status = QLabel("")
        self._correction_status.setWordWrap(True)
        self._correction_status.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8pt; padding: 2px;"
        )
        corrections_layout.addWidget(self._correction_status)
        self._corrections_group.hide()
        self._apply_corrections_btn.setEnabled(False)
        self._reset_corrections_btn.setEnabled(False)
        self._to_midi_btn.setEnabled(False)
        self._results_layout.addWidget(self._corrections_group)

        self._results_layout.addStretch()

        scroll.setWidget(results_widget)
        layout.addWidget(scroll, 1)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._match_btn = QPushButton(tr("reference.match"))
        self._match_btn.setMinimumHeight(32)
        self._match_btn.setEnabled(False)
        self._match_btn.clicked.connect(self._on_match)
        btn_row.addWidget(self._match_btn)

        self._use_tags_btn = QPushButton(tr("reference.use_tags"))
        self._use_tags_btn.setMinimumHeight(32)
        self._use_tags_btn.setProperty("class", "secondary")
        self._use_tags_btn.setEnabled(False)
        self._use_tags_btn.clicked.connect(self._on_use_tags)
        btn_row.addWidget(self._use_tags_btn)

        self._cancel_btn = QPushButton(tr("reference.cancel"))
        self._cancel_btn.setMinimumHeight(32)
        self._cancel_btn.setProperty("class", "secondary")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self.cancel_analysis)
        btn_row.addWidget(self._cancel_btn)

        layout.addLayout(btn_row)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8.25pt;"
        )
        layout.addWidget(self._progress_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("referenceAnalysisProgress")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setAccessibleName(tr("reference.accessibility.progress_name"))
        self._progress_bar.setAccessibleDescription(
            tr("reference.accessibility.progress_description")
        )
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

    def _browse_file(self):
        path, _ = open_audio_file(
            self,
            tr("reference.dialog_select"),
            operation_kind="reference_track_import",
            dialog=QFileDialog,
        )
        if path:
            self._analyze_file(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and any(
            is_audio_path(url.toLocalFile()) for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path and is_audio_path(path):
                self._analyze_file(path)

    def load_reference_file(self, file_path: str):
        """Public entry point used by cross-module routes."""
        self._analyze_file(file_path)

    def _analyze_file(self, file_path: str):
        """Analyze off the GUI thread; a newer selection supersedes this one."""
        from pathlib import Path

        from core.workers import InferenceWorker
        from engines.audio_analyzer import analyze_track

        self.cancel_analysis()
        self._pending_waveform_analysis = None

        self._pending_path = str(file_path)
        self._analysis_token += 1
        token = self._analysis_token

        self._drop_zone.setText(
            tr("reference.status.analyzing", name=Path(file_path).name)
        )
        self._drop_zone.setStyleSheet(
            f"QLabel {{ background: {Palette.MANTLE}; border: 2px solid {Palette.BLUE}; border-radius: 8px; "
            f"color: {Palette.BLUE}; font-size: 9pt; }}"
        )
        self._cancel_btn.setVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)

        worker = InferenceWorker(
            analyze_track,
            str(file_path),
            job_kind="reference_analysis",
            job_label=Path(file_path).name,
        )
        worker.step_info.connect(
            lambda text, t=token: self._on_analysis_step(t, text)
        )
        worker.progress.connect(
            lambda pct, t=token: self._on_analysis_progress(t, pct)
        )
        worker.finished.connect(
            lambda result, t=token, p=str(file_path), w=worker:
            self._on_analysis_done(t, p, result, w)
        )
        worker.error.connect(
            lambda message, t=token, w=worker: self._on_analysis_error(t, message, w)
        )
        worker.cancelled.connect(
            lambda t=token, w=worker: self._on_analysis_cancelled(t, w)
        )
        # InferenceWorker's result-bearing ``finished`` signal fires before
        # QThread has fully stopped.  The distinct completion signal lets the
        # wrapper be released after task cleanup without a blocking wait.
        worker.thread_stopped.connect(
            lambda w=worker: self._release_worker_later(w)
        )
        self._analysis_workers.add(worker)
        self._worker = worker
        worker.start()

    def cancel_analysis(self):
        """Request cancellation without blocking or dropping a live worker."""
        current = getattr(self, "_worker", None)
        workers = set(getattr(self, "_analysis_workers", set()))
        if current is not None:
            workers.add(current)
        running_workers = [worker for worker in workers if worker.isRunning()]
        if not running_workers:
            if current is not None:
                self._release_worker_later(current)
            return
        self._analysis_token += 1
        for worker in running_workers:
            worker.cancel()
            self._release_worker_later(worker)
        if current in running_workers:
            self._cancel_btn.setEnabled(False)
            self._progress_label.setText(tr("reference.status.cancelling"))
            return

    def _release_worker_later(self, worker):
        """Retain a QThread wrapper until its native thread has stopped."""
        if worker is None:
            return
        if worker.isRunning():
            QTimer.singleShot(10, lambda: self._release_worker_later(worker))
            return
        event = self._pending_analysis_events.pop(worker, None)
        self._analysis_workers.discard(worker)
        if self._worker is worker:
            self._worker = None
            self._cancel_btn.setEnabled(True)
            self._cancel_btn.setVisible(False)
        if event is not None:
            self._finalize_analysis_event(event)

    def _is_current(self, token: int) -> bool:
        """A result from a superseded selection must never be applied."""
        return token == self._analysis_token

    def _on_analysis_step(self, token: int, text: str):
        if self._is_current(token):
            self._progress_label.setText(text)

    def _on_analysis_progress(self, token: int, percent: int):
        if self._is_current(token):
            self._progress_bar.setValue(max(0, min(100, int(percent))))

    def _on_analysis_done(self, token: int, file_path: str, analysis, worker=None):
        from pathlib import Path

        if worker is None:
            if not self._is_current(token) or analysis is None:
                return
            self._cancel_btn.setVisible(False)
            self._progress_label.setText("")
            self._progress_bar.hide()
            self._display_analysis(analysis, Path(file_path).name)
            return
        self._pending_analysis_events[worker] = (
            "done",
            token,
            str(file_path),
            analysis,
        )
        self._release_worker_later(worker)

    def _on_analysis_error(self, token: int, message: str, worker=None):
        if worker is None:
            self._show_analysis_error(token, message)
            return
        self._pending_analysis_events[worker] = ("error", token, message)
        self._release_worker_later(worker)

    def _show_analysis_error(self, token: int, message: str):
        if not self._is_current(token):
            return
        self._cancel_btn.setVisible(False)
        self._progress_label.setText("")
        self._progress_bar.hide()
        if "librosa" in message.lower() or "import" in message.lower():
            self._drop_zone.setText(tr("reference.status.unavailable"))
        else:
            self._drop_zone.setText(
                tr("reference.status.failed", error=message[:60])
            )

    def _on_analysis_cancelled(self, token: int, worker=None):
        if worker is None:
            self._show_analysis_cancelled(token)
            return
        self._pending_analysis_events[worker] = ("cancelled", token)
        self._release_worker_later(worker)

    def _show_analysis_cancelled(self, token: int):
        if self._is_current(token):
            self._cancel_btn.setVisible(False)
            self._progress_label.setText("")
            self._progress_bar.hide()
            self._drop_zone.setText(tr("reference.status.cancelled"))

    def _finalize_analysis_event(self, event):
        """Apply a terminal event only after the worker's native thread stops."""
        kind, token, *payload = event
        if kind == "done":
            file_path, analysis = payload
            if not self._is_current(token) or analysis is None:
                return
            from pathlib import Path

            self._cancel_btn.setVisible(False)
            self._progress_label.setText("")
            self._progress_bar.hide()
            self._display_analysis(analysis, Path(file_path).name)
        elif kind == "error":
            self._show_analysis_error(token, payload[0])
        elif kind == "cancelled":
            self._show_analysis_cancelled(token)

    def _display_analysis(self, analysis, filename: str):
        """Show analysis results in the panel."""
        if hasattr(analysis, "clone"):
            # Analysis cache entries are shared; corrections belong to this
            # editor session and must not mutate the cached raw result.
            analysis = analysis.clone()
        self._pending_waveform_analysis = (self._analysis_token, analysis)

        # Update drop zone
        self._drop_zone.setText(filename)
        self._drop_zone.setStyleSheet(
            f"QLabel {{ background: {Palette.MANTLE}; border: 2px solid {Palette.GREEN}; border-radius: 8px; "
            f"color: {Palette.GREEN}; font-size: 9pt; font-weight: bold; }}"
        )

        # Show waveform
        try:
            self._waveform.load_audio(analysis.file_path)
            self._waveform.show()
        except Exception:
            self._pending_waveform_analysis = None
            self._commit_analysis(analysis)
            return

        if not self._waveform._audio_load_workers:
            self._pending_waveform_analysis = None
            self._commit_analysis(analysis)

        self._populate_correction_editor(analysis)
        self._refresh_analysis_display(analysis)
        self._corrections_group.show()

    def _refresh_analysis_display(self, analysis):
        """Render raw/effective analysis without losing correction lineage."""
        from engines.audio_analyzer import _bpm_to_tag

        # Update metrics
        self._bpm_card.set_value(f"{analysis.effective_bpm:.0f}")
        self._key_card.set_value(analysis.effective_key)
        self._energy_card.set_value(f"{analysis.energy_mean:.2f}")
        self._brightness_card.set_value(
            tr("reference.metrics.brightness_value", value=analysis.brightness_mean)
        )
        self._density_card.set_value(
            tr("reference.metrics.density_value", value=analysis.onset_density)
        )
        self._duration_card.set_value(
            tr("reference.metrics.duration_value", value=analysis.duration)
        )
        self._metrics_group.show()

        # Heuristic tags are explicitly presented as suggestions, not facts.
        if analysis.suggested_tags or analysis.suggested_tempo_tag:
            tag_list = list(analysis.suggested_tags)
            if analysis.corrected_bpm is not None and analysis.effective_bpm > 0:
                tag_list.append(_bpm_to_tag(analysis.effective_bpm))
            elif analysis.suggested_tempo_tag:
                tag_list.append(analysis.suggested_tempo_tag)
            tag_str = ", ".join(tag_list)
            self._tags_label.setText(tr("reference.suggested_tags", tags=tag_str))
            self._tags_label.show()
        else:
            self._tags_label.hide()

        clap_tags = getattr(analysis, "clap_style_tags", [])
        if clap_tags:
            backend = getattr(analysis, "clap_backend", "audio-clap")
            self._clap_label.setText(
                tr(
                    "reference.clap_tags",
                    backend=backend,
                    tags=", ".join(clap_tags),
                )
            )
            self._clap_label.show()
        else:
            self._clap_label.hide()

        # Sections
        sections = analysis.effective_sections
        if sections:
            parts = [
                tr(
                    "reference.section",
                    label=s["label"],
                    start=s["start"],
                    end=s["end"],
                )
                for s in sections[:6]
            ]
            self._sections_label.setText(
                tr("reference.structure", sections=" | ".join(parts))
            )
            self._sections_label.show()
        else:
            self._sections_label.hide()

        bpm_alternatives = ", ".join(
            f"{float(candidate.get('value', 0.0)):.0f} BPM"
            for candidate in getattr(analysis, "bpm_alternatives", [])
        )
        key_alternatives = ", ".join(
            str(candidate.get("value", ""))
            for candidate in getattr(analysis, "key_alternatives", [])
            if candidate.get("value")
        )
        self._correction_confidence_label.setText(
            tr(
                "reference.corrections.confidence",
                bpm=f"{analysis.bpm_confidence:.0%}",
                musical_key=f"{analysis.key_confidence:.0%}",
                bpm_alternates=bpm_alternatives or tr("reference.corrections.none"),
                key_alternates=key_alternatives or tr("reference.corrections.none"),
            )
        )
        raw_summary = tr(
            "reference.corrections.raw_summary",
            bpm=analysis.bpm,
            musical_key=analysis.key or tr("reference.corrections.unknown"),
        )
        if analysis.has_corrections:
            self._correction_status.setText(
                tr("reference.corrections.active", summary=raw_summary)
            )
        else:
            self._correction_status.setText(raw_summary)

    def _populate_correction_editor(self, analysis):
        """Load raw/effective values into the editable correction controls."""
        bpm = analysis.effective_bpm or 120.0
        self._bpm_override_spin.setValue(
            max(self._bpm_override_spin.minimum(), min(self._bpm_override_spin.maximum(), bpm))
        )
        self._bpm_override_check.setChecked(analysis.corrected_bpm is not None)

        key_value = analysis.corrected_key or ""
        key_index = self._key_override_combo.findData(key_value)
        self._key_override_combo.setCurrentIndex(max(0, key_index))
        self._key_override_check.setChecked(analysis.corrected_key is not None)

        sections = analysis.effective_sections
        self._sections_table.setRowCount(len(sections))
        for row, section in enumerate(sections):
            self._sections_table.setItem(
                row, 0, QTableWidgetItem(str(section.get("label", "Section")))
            )
            self._sections_table.setItem(
                row, 1, QTableWidgetItem(f"{float(section.get('start', 0.0)):.3f}")
            )
            self._sections_table.setItem(
                row, 2, QTableWidgetItem(f"{float(section.get('end', 0.0)):.3f}")
            )
        self._sections_override_check.setChecked(analysis.corrected_sections is not None)

    def _read_section_table(self) -> list[dict]:
        sections = []
        for row in range(self._sections_table.rowCount()):
            label_item = self._sections_table.item(row, 0)
            start_item = self._sections_table.item(row, 1)
            end_item = self._sections_table.item(row, 2)
            sections.append({
                "label": label_item.text().strip() if label_item else "",
                "start": float(start_item.text()) if start_item else 0.0,
                "end": float(end_item.text()) if end_item else 0.0,
            })
        return sections

    def _apply_corrections(self) -> bool:
        """Validate and apply the complete editor state atomically."""
        if self._analysis is None:
            return False
        candidate = self._analysis.clone()
        candidate.clear_corrections()
        try:
            bpm = self._bpm_override_spin.value() if self._bpm_override_check.isChecked() else None
            key = (
                str(self._key_override_combo.currentData() or "")
                if self._key_override_check.isChecked()
                else None
            )
            sections = (
                self._read_section_table()
                if self._sections_override_check.isChecked()
                else None
            )
            candidate.apply_corrections(bpm=bpm, key=key or None, sections=sections)
        except (TypeError, ValueError) as exc:
            self._correction_status.setText(
                tr("reference.corrections.invalid", error=str(exc))
            )
            return False

        self._analysis.corrected_bpm = candidate.corrected_bpm
        self._analysis.corrected_key = candidate.corrected_key
        self._analysis.corrected_sections = candidate.corrected_sections
        self._populate_correction_editor(self._analysis)
        self._refresh_analysis_display(self._analysis)
        self._correction_status.setText(tr("reference.corrections.applied"))
        return True

    def _reset_corrections(self):
        if self._analysis is None:
            return
        self._analysis.clear_corrections()
        self._populate_correction_editor(self._analysis)
        self._refresh_analysis_display(self._analysis)
        self._correction_status.setText(tr("reference.corrections.reset_done"))

    def _on_send_to_midi(self):
        if self._analysis is None:
            return
        if not self._apply_corrections():
            return
        self.reference_to_midi.emit(self._analysis.to_generation_constraints())

    def _on_match(self):
        """Emit match request with full analysis."""
        if self._analysis:
            self.match_requested.emit(self._analysis.to_dict())

    def _on_use_tags(self):
        """Emit just the extracted tags."""
        if self._analysis:
            self.tags_extracted.emit(self._analysis.to_ace_step_tags())

    def get_energy_curve(self) -> list[float]:
        """Get the reference energy curve for overlay on mood editor."""
        if self._analysis and self._analysis.energy_curve:
            return self._analysis.energy_curve
        return []

    def clear(self):
        self._analysis = None
        self._pending_waveform_analysis = None
        self._drop_zone.setText(tr("reference.drop_zone"))
        self._drop_zone.setStyleSheet(
            f"QLabel {{ background: {Palette.MANTLE}; border: 2px dashed {Palette.SURFACE1}; border-radius: 8px; "
            f"color: {Palette.OVERLAY0}; font-size: 9pt; }}"
        )
        self._waveform.hide()
        self._metrics_group.hide()
        self._tags_label.hide()
        self._clap_label.hide()
        self._sections_label.hide()
        self._corrections_group.hide()
        self._correction_status.clear()
        self._sections_table.setRowCount(0)
        self._match_btn.setEnabled(False)
        self._use_tags_btn.setEnabled(False)
        self._apply_corrections_btn.setEnabled(False)
        self._reset_corrections_btn.setEnabled(False)
        self._to_midi_btn.setEnabled(False)

    def _on_reference_waveform_finished(self, _success: bool):
        pending = self._pending_waveform_analysis
        if pending is None:
            return
        token, analysis = pending
        self._pending_waveform_analysis = None
        if not self._is_current(token):
            return
        self._commit_analysis(analysis)

    def _commit_analysis(self, analysis):
        """Expose analysis actions after the source preview has released it."""
        self._analysis = analysis
        self._populate_correction_editor(analysis)
        self._refresh_analysis_display(analysis)
        self._corrections_group.show()
        self._match_btn.setEnabled(True)
        self._use_tags_btn.setEnabled(True)
        self._apply_corrections_btn.setEnabled(True)
        self._reset_corrections_btn.setEnabled(True)
        self._to_midi_btn.setEnabled(True)
        # The editor is revealed dynamically; refresh focus metadata so the
        # newly visible table and correction controls join the tab order.
        install_accessibility(self, tr("reference.accessibility.name"))
