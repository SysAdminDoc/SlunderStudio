"""
Slunder Studio v0.0.2 — Reference Panel
Reference track analysis UI: drag-drop audio, view analysis results,
"Match This" one-click generation, and reference library management.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFrame, QFileDialog, QListWidget, QListWidgetItem, QScrollArea,
    QGroupBox, QGridLayout,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from ui.waveform_widget import WaveformWidget


class AnalysisCard(QFrame):
    """Displays a single analysis metric."""
    def __init__(self, label: str, value: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame { background: #1E1E2E; border: 1px solid #313244; border-radius: 6px; padding: 8px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self._label = QLabel(label)
        self._label.setStyleSheet("color: #A6ADC8; font-size: 10px; font-weight: bold;")
        layout.addWidget(self._label)

        self._value = QLabel(value)
        self._value.setStyleSheet("color: #CDD6F4; font-size: 14px; font-weight: bold;")
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
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        title = QLabel("Reference Track")
        title.setStyleSheet("color: #CDD6F4; font-weight: bold; font-size: 13px;")
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
            "QLabel { background: #181825; border: 2px dashed #45475A; border-radius: 8px; "
            "color: #6C7086; font-size: 12px; }"
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
            "QGroupBox { color: #A6ADC8; border: 1px solid #313244; border-radius: 6px; "
            "margin-top: 8px; padding-top: 14px; font-size: 11px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
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
        self._tags_label.setStyleSheet("color: #94E2D5; font-size: 11px; padding: 4px;")
        self._tags_label.hide()
        self._results_layout.addWidget(self._tags_label)

        # Sections
        self._sections_label = QLabel("")
        self._sections_label.setWordWrap(True)
        self._sections_label.setStyleSheet("color: #A6ADC8; font-size: 11px; padding: 4px;")
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

        layout.addLayout(btn_row)

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

    def _analyze_file(self, file_path: str):
        """Run analysis on the dropped/selected file."""
        from pathlib import Path
        self._drop_zone.setText(f"Analyzing: {Path(file_path).name}...")
        self._drop_zone.setStyleSheet(
            "QLabel { background: #181825; border: 2px solid #89B4FA; border-radius: 8px; "
            "color: #89B4FA; font-size: 12px; }"
        )

        # Run analysis (would use InferenceWorker in production)
        try:
            from core.deps import ensure
            ensure("librosa")
            from engines.audio_analyzer import analyze_track
            analysis = analyze_track(file_path)
            self._display_analysis(analysis, Path(file_path).name)
        except ImportError:
            self._drop_zone.setText("Audio analysis unavailable — restart to retry")
        except Exception as e:
            self._drop_zone.setText(f"Analysis failed: {str(e)[:50]}")

    def _display_analysis(self, analysis, filename: str):
        """Show analysis results in the panel."""
        from engines.audio_analyzer import AudioAnalysis
        self._analysis = analysis

        # Update drop zone
        self._drop_zone.setText(filename)
        self._drop_zone.setStyleSheet(
            "QLabel { background: #181825; border: 2px solid #A6E3A1; border-radius: 8px; "
            "color: #A6E3A1; font-size: 12px; font-weight: bold; }"
        )

        # Show waveform
        try:
            self._waveform.load_file(analysis.file_path)
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
            "QLabel { background: #181825; border: 2px dashed #45475A; border-radius: 8px; "
            "color: #6C7086; font-size: 12px; }"
        )
        self._waveform.hide()
        self._metrics_group.hide()
        self._tags_label.hide()
        self._sections_label.hide()
        self._match_btn.setEnabled(False)
        self._use_tags_btn.setEnabled(False)
