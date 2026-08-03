"""
Slunder Studio — Reference Panel
Reference track analysis UI: drag-drop audio, view analysis results,
"Match This" one-click generation, and reference library management.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QFileDialog, QListWidget, QListWidgetItem, QScrollArea,
    QGroupBox, QGridLayout,
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from ui.theme import Palette
from ui.accessibility import install_accessibility
from ui.waveform_widget import WaveformWidget


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
        self._label.setStyleSheet(f"color: {Palette.SUBTEXT0}; font-size: 10px; font-weight: bold;")
        layout.addWidget(self._label)

        self._value = QLabel(value)
        self._value.setStyleSheet(f"color: {Palette.TEXT}; font-size: 14px; font-weight: bold;")
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._analysis = None
        self._worker = None
        self._analysis_workers = set()
        # Monotonic token so a result from a superseded file is discarded.
        self._analysis_token = 0
        self._pending_path = ""
        self.setAcceptDrops(True)
        self._setup_ui()
        install_accessibility(
            self,
            "Reference track",
            named_controls=[
                (self._browse_btn, "Browse reference track", "Selects an audio reference track for analysis."),
                (self._match_btn, "Match reference track", "Uses the analysis to configure matching generation."),
                (self._use_tags_btn, "Use reference tags", "Sends extracted reference tags to the generation form."),
                (self._cancel_btn, "Cancel reference analysis", "Cancels the running reference analysis."),
                (self._asset_list if hasattr(self, "_asset_list") else None, "Reference library", "Selects a saved reference track."),
            ],
        )

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title = QLabel("Reference Track")
        title.setStyleSheet(f"color: {Palette.TEXT}; font-weight: bold; font-size: 13px;")
        header.addWidget(title)
        header.addStretch()

        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.setFixedHeight(26)
        self._browse_btn.setProperty("class", "secondary")
        self._browse_btn.clicked.connect(self._browse_file)
        header.addWidget(self._browse_btn)

        layout.addLayout(header)

        # Drop zone / file info
        self._drop_zone = QLabel("Drop an audio file here\nor click Browse")
        self._drop_zone.setAlignment(Qt.AlignCenter)
        self._drop_zone.setFixedHeight(60)
        self._drop_zone.setStyleSheet(
            f"QLabel {{ background: {Palette.MANTLE}; border: 2px dashed {Palette.SURFACE1}; border-radius: 8px; "
            f"color: {Palette.OVERLAY0}; font-size: 12px; }}"
        )
        layout.addWidget(self._drop_zone)

        # Mini waveform
        self._waveform = WaveformWidget(show_controls=False)
        self._waveform.setFixedHeight(60)
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
        self._metrics_group = QGroupBox("Analysis")
        self._metrics_group.setStyleSheet(
            f"QGroupBox {{ color: {Palette.SUBTEXT0}; border: 1px solid {Palette.SURFACE0}; border-radius: 6px; "
            f"margin-top: 8px; padding-top: 14px; font-size: 11px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; }}"
        )
        metrics_grid = QGridLayout(self._metrics_group)
        metrics_grid.setSpacing(6)

        self._bpm_card = AnalysisCard("BPM")
        self._key_card = AnalysisCard("Key")
        self._energy_card = AnalysisCard("Energy")
        self._brightness_card = AnalysisCard("Brightness")
        self._density_card = AnalysisCard("Onset Density")
        self._duration_card = AnalysisCard("Duration")

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
        self._tags_label.setStyleSheet(f"color: {Palette.TEAL}; font-size: 11px; padding: 4px;")
        self._tags_label.hide()
        self._results_layout.addWidget(self._tags_label)

        self._clap_label = QLabel("")
        self._clap_label.setWordWrap(True)
        self._clap_label.setStyleSheet(f"color: {Palette.BLUE}; font-size: 11px; padding: 4px;")
        self._clap_label.hide()
        self._results_layout.addWidget(self._clap_label)

        # Sections
        self._sections_label = QLabel("")
        self._sections_label.setWordWrap(True)
        self._sections_label.setStyleSheet(f"color: {Palette.SUBTEXT0}; font-size: 11px; padding: 4px;")
        self._sections_label.hide()
        self._results_layout.addWidget(self._sections_label)

        self._results_layout.addStretch()

        scroll.setWidget(results_widget)
        layout.addWidget(scroll, 1)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._match_btn = QPushButton("Match This")
        self._match_btn.setFixedHeight(32)
        self._match_btn.setEnabled(False)
        self._match_btn.clicked.connect(self._on_match)
        btn_row.addWidget(self._match_btn)

        self._use_tags_btn = QPushButton("Use Tags")
        self._use_tags_btn.setFixedHeight(32)
        self._use_tags_btn.setProperty("class", "secondary")
        self._use_tags_btn.setEnabled(False)
        self._use_tags_btn.clicked.connect(self._on_use_tags)
        btn_row.addWidget(self._use_tags_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(32)
        self._cancel_btn.setProperty("class", "secondary")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self.cancel_analysis)
        btn_row.addWidget(self._cancel_btn)

        layout.addLayout(btn_row)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 11px;"
        )
        layout.addWidget(self._progress_label)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Track", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)",
        )
        if path:
            self._analyze_file(path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
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

        self._pending_path = str(file_path)
        self._analysis_token += 1
        token = self._analysis_token

        self._drop_zone.setText(f"Analyzing: {Path(file_path).name}...")
        self._drop_zone.setStyleSheet(
            f"QLabel {{ background: {Palette.MANTLE}; border: 2px solid {Palette.BLUE}; border-radius: 8px; "
            f"color: {Palette.BLUE}; font-size: 12px; }}"
        )
        self._cancel_btn.setVisible(True)

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
            self._progress_label.setText("Cancelling...")
            return

    def _release_worker_later(self, worker):
        """Retain a QThread wrapper until its native thread has stopped."""
        if worker is None:
            return
        if worker.isRunning():
            QTimer.singleShot(10, lambda: self._release_worker_later(worker))
            return
        self._analysis_workers.discard(worker)
        if self._worker is worker:
            self._worker = None
            self._cancel_btn.setEnabled(True)
            self._cancel_btn.setVisible(False)

    def _is_current(self, token: int) -> bool:
        """A result from a superseded selection must never be applied."""
        return token == self._analysis_token

    def _on_analysis_step(self, token: int, text: str):
        if self._is_current(token):
            self._progress_label.setText(text)

    def _on_analysis_progress(self, token: int, percent: int):
        if self._is_current(token):
            self._progress_label.setText(
                f"{self._progress_label.text().split(' - ')[0]} - {percent}%"
            )

    def _on_analysis_done(self, token: int, file_path: str, analysis, worker=None):
        from pathlib import Path

        self._release_worker_later(worker)
        if not self._is_current(token) or analysis is None:
            return
        self._cancel_btn.setVisible(False)
        self._progress_label.setText("")
        self._display_analysis(analysis, Path(file_path).name)

    def _on_analysis_error(self, token: int, message: str, worker=None):
        self._release_worker_later(worker)
        if not self._is_current(token):
            return
        self._cancel_btn.setVisible(False)
        self._progress_label.setText("")
        if "librosa" in message.lower() or "import" in message.lower():
            self._drop_zone.setText("Audio analysis unavailable - install librosa")
        else:
            self._drop_zone.setText(f"Analysis failed: {message[:60]}")

    def _on_analysis_cancelled(self, token: int, worker=None):
        self._release_worker_later(worker)
        if self._is_current(token):
            self._cancel_btn.setVisible(False)
            self._progress_label.setText("")
            self._drop_zone.setText("Analysis cancelled")

    def _display_analysis(self, analysis, filename: str):
        """Show analysis results in the panel."""
        from engines.audio_analyzer import AudioAnalysis
        self._analysis = analysis

        # Update drop zone
        self._drop_zone.setText(filename)
        self._drop_zone.setStyleSheet(
            f"QLabel {{ background: {Palette.MANTLE}; border: 2px solid {Palette.GREEN}; border-radius: 8px; "
            f"color: {Palette.GREEN}; font-size: 12px; font-weight: bold; }}"
        )

        # Show waveform
        try:
            self._waveform.load_audio(analysis.file_path)
            self._waveform.show()
        except Exception:
            pass

        # Update metrics
        self._bpm_card.set_value(f"{analysis.bpm:.0f}")
        self._key_card.set_value(analysis.key)
        self._energy_card.set_value(f"{analysis.energy_mean:.2f}")
        self._brightness_card.set_value(f"{analysis.brightness_mean:.0f} Hz")
        self._density_card.set_value(f"{analysis.onset_density:.1f}/s")
        self._duration_card.set_value(f"{analysis.duration:.1f}s")
        self._metrics_group.show()

        # Suggested tags
        if analysis.suggested_tags:
            tag_str = ", ".join(analysis.suggested_tags)
            if analysis.suggested_tempo_tag:
                tag_str += f", {analysis.suggested_tempo_tag}"
            self._tags_label.setText(f"Suggested tags: {tag_str}")
            self._tags_label.show()

        clap_tags = getattr(analysis, "clap_style_tags", [])
        if clap_tags:
            backend = getattr(analysis, "clap_backend", "audio-clap")
            self._clap_label.setText(
                f"Audio-CLAP conditioning ({backend}): {', '.join(clap_tags)}"
            )
            self._clap_label.show()

        # Sections
        if analysis.sections:
            parts = [f"{s['label']} ({s['start']:.0f}s-{s['end']:.0f}s)" for s in analysis.sections[:6]]
            self._sections_label.setText("Structure: " + " | ".join(parts))
            self._sections_label.show()

        # Enable buttons
        self._match_btn.setEnabled(True)
        self._use_tags_btn.setEnabled(True)

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
        self._drop_zone.setText("Drop an audio file here\nor click Browse")
        self._drop_zone.setStyleSheet(
            f"QLabel {{ background: {Palette.MANTLE}; border: 2px dashed {Palette.SURFACE1}; border-radius: 8px; "
            f"color: {Palette.OVERLAY0}; font-size: 12px; }}"
        )
        self._waveform.hide()
        self._metrics_group.hide()
        self._tags_label.hide()
        self._clap_label.hide()
        self._sections_label.hide()
        self._match_btn.setEnabled(False)
        self._use_tags_btn.setEnabled(False)
