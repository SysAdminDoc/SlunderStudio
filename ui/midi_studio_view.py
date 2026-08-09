"""
Slunder Studio — MIDI Studio View
Main MIDI Studio page: text-to-MIDI generation, piano roll editor,
per-track mixer, .mid import/export, FluidSynth rendering, and
cross-module routing (Song Forge, Vocal Suite).
"""
from typing import Optional
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QTabWidget,
    QFrame, QSplitter, QGroupBox, QLineEdit, QStackedWidget, QCheckBox,
)
from PySide6.QtCore import Qt, Signal

from ui.theme import Palette, ThemeEngine
from ui.accessibility import install_accessibility
from ui.widgets import EmptyStateWidget, OperationProgressWidget
from core.i18n import tr, user_facing_readiness
from ui.piano_roll import PianoRollWidget
from ui.midi_mixer import MidiMixer
from ui.waveform_widget import WaveformWidget
from core.midi_utils import (
    MidiData, TrackData, NoteData, load_midi, save_midi, get_program_name,
)
from core.chord_chart import save_chord_chart
from core.engine_contract import (
    ArtifactKind,
    CAP_MIDI_GENERATE,
    CAP_MIDI_RENDER,
    EngineArtifact,
    EngineRunResult,
    RunMode,
    adapt_engine_result,
)
from core.model_manager import ModelManager
from ui.file_dialogs import ensure_extension, open_midi_file, save_file, save_midi_file
from core.settings import Settings
from core.provenance import sidecar_path_for
from core.workers import CancelledJobError, InferenceWorker
from engines.midi_llm_engine import (
    DRUM_GROOVE_NAMES, MidiGenParams, MidiGenResult, generate_midi,
)


# ── Key options ────────────────────────────────────────────────────────────────

KEYS = [
    "C major", "C minor", "C# major", "C# minor", "D major", "D minor",
    "D# major", "D# minor", "E major", "E minor", "F major", "F minor",
    "F# major", "F# minor", "G major", "G minor", "G# major", "G# minor",
    "A major", "A minor", "A# major", "A# minor", "B major", "B minor",
]

INSTRUMENT_PRESETS = {
    "Piano": ["Piano"],
    "Band (4-piece)": ["Piano", "Bass", "Drums", "Melody"],
    "Orchestra": ["Strings", "Brass", "Woodwinds", "Percussion", "Harp"],
    "Electronic": ["Synth Lead", "Synth Bass", "Synth Pad", "Drums"],
    "Jazz Trio": ["Piano", "Upright Bass", "Drums"],
    "Rock Band": ["Lead Guitar", "Rhythm Guitar", "Bass", "Drums"],
    "String Quartet": ["Violin 1", "Violin 2", "Viola", "Cello"],
}

CHORD_PROGRESSIONS = [
    "Auto",
    "I-V-vi-IV",
    "ii-V-I",
    "I-vi-IV-V",
    "12-bar blues",
    "i-VI-III-VII",
    "i-VI-iv-v",
]


class MidiStudioView(QWidget):
    """Main MIDI Studio page."""

    send_to_forge = Signal(str)     # audio file path -> Song Forge
    send_to_vocals = Signal(str)    # audio file path -> Vocal Suite

    def __init__(self, parent=None, toast_mgr=None):
        super().__init__(parent)
        self.toast_mgr = toast_mgr
        self._midi_data: Optional[MidiData] = None
        self._rendered_audio = None
        self._current_audio_path: Optional[str] = None
        self._rendered_output_kind = ""
        self._model_mgr = ModelManager()
        self._generation_worker: Optional[InferenceWorker] = None
        self._render_worker: Optional[InferenceWorker] = None
        self._contract_result: Optional[EngineRunResult] = None
        self._settings = Settings()

        t = ThemeEngine.get_colors()
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # ── Left: Controls ─────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        # Generation panel
        gen_frame = QFrame()
        gen_frame.setStyleSheet(f"""
            QFrame {{
                background: {t['surface']};
                border: 1px solid {t['border']};
                border-radius: 8px;
            }}
        """)
        gen_layout = QVBoxLayout(gen_frame)
        gen_layout.setContentsMargins(12, 10, 12, 10)
        gen_layout.setSpacing(6)

        gen_title = QLabel(tr("midi.title"))
        gen_title.setStyleSheet(f"color: {t['accent']}; font-weight: bold; font-size: 9.75pt; border: none;")
        gen_layout.addWidget(gen_title)

        # Prompt
        self._prompt = QTextEdit()
        self._prompt.setPlaceholderText(tr("midi.prompt_placeholder"))
        self._prompt.setMaximumHeight(70)
        self._prompt.setStyleSheet(f"""
            QTextEdit {{
                background: {t['background']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 6px;
                font-size: 9pt;
            }}
        """)
        self._prompt.textChanged.connect(self._refresh_capability_state)
        gen_layout.addWidget(self._prompt)

        # Style
        style_row = QHBoxLayout()
        style_label = QLabel(tr("midi.style_label"))
        style_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt; border:none;")
        style_label.setMinimumWidth(40)
        self._style_input = QLineEdit()
        self._style_input.setPlaceholderText(tr("midi.style_placeholder"))
        self._style_input.setStyleSheet(f"""
            QLineEdit {{
                background: {t['background']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px; padding: 4px 8px; font-size: 8.25pt;
            }}
        """)
        style_row.addWidget(style_label)
        style_row.addWidget(self._style_input)
        gen_layout.addLayout(style_row)

        # Parameters grid
        param_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {t['background']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 3px; padding: 3px 6px; font-size: 8.25pt;
            }}
            QLabel {{ color: {t['text_secondary']}; font-size: 8.25pt; border:none; }}
        """

        # Row 1: Key + Tempo
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        key_l = QLabel(tr("midi.key_label"))
        key_l.setMinimumWidth(34)
        key_l.setStyleSheet(param_style)
        self._key_combo = QComboBox()
        self._key_combo.addItems(KEYS)
        self._key_combo.setCurrentText("C major")
        self._key_combo.setStyleSheet(param_style)

        tempo_l = QLabel(tr("midi.tempo_label"))
        tempo_l.setMinimumWidth(30)
        tempo_l.setStyleSheet(param_style)
        self._tempo_spin = QSpinBox()
        self._tempo_spin.setRange(40, 300)
        self._tempo_spin.setValue(int(self._settings.get("midi_studio.default_bpm", 120)))
        self._tempo_spin.setStyleSheet(param_style)

        row1.addWidget(key_l)
        row1.addWidget(self._key_combo)
        row1.addWidget(tempo_l)
        row1.addWidget(self._tempo_spin)
        gen_layout.addLayout(row1)

        # Row 2: Bars + Time Sig
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        bars_l = QLabel(tr("midi.bars_label"))
        bars_l.setMinimumWidth(34)
        bars_l.setStyleSheet(param_style)
        self._bars_spin = QSpinBox()
        self._bars_spin.setRange(4, 128)
        self._bars_spin.setValue(16)
        self._bars_spin.setStyleSheet(param_style)

        ts_l = QLabel(tr("midi.time_label"))
        ts_l.setMinimumWidth(30)
        ts_l.setStyleSheet(param_style)
        self._time_sig = QComboBox()
        self._time_sig.addItems(["4/4", "3/4", "6/8", "2/4", "5/4", "7/8"])
        self._time_sig.setStyleSheet(param_style)

        row2.addWidget(bars_l)
        row2.addWidget(self._bars_spin)
        row2.addWidget(ts_l)
        row2.addWidget(self._time_sig)
        gen_layout.addLayout(row2)

        # Row 3: Instruments preset
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        inst_l = QLabel(tr("midi.preset_label"))
        inst_l.setMinimumWidth(40)
        inst_l.setStyleSheet(param_style)
        self._inst_combo = QComboBox()
        self._inst_combo.addItems(INSTRUMENT_PRESETS.keys())
        self._inst_combo.setCurrentText("Band (4-piece)")
        self._inst_combo.setStyleSheet(param_style)
        row3.addWidget(inst_l)
        row3.addWidget(self._inst_combo)
        gen_layout.addLayout(row3)

        # Row 4: Chord progression prior
        row4 = QHBoxLayout()
        row4.setSpacing(6)
        prog_l = QLabel(tr("midi.chords_label"))
        prog_l.setMinimumWidth(48)
        prog_l.setStyleSheet(param_style)
        self._progression_combo = QComboBox()
        self._progression_combo.addItems(CHORD_PROGRESSIONS)
        self._progression_combo.setStyleSheet(param_style)
        row4.addWidget(prog_l)
        row4.addWidget(self._progression_combo)
        gen_layout.addLayout(row4)

        # Row 5: Drum groove template
        row5 = QHBoxLayout()
        row5.setSpacing(6)
        groove_l = QLabel(tr("midi.groove_label"))
        groove_l.setMinimumWidth(48)
        groove_l.setStyleSheet(param_style)
        self._groove_combo = QComboBox()
        self._groove_combo.addItems(DRUM_GROOVE_NAMES)
        self._groove_combo.setStyleSheet(param_style)
        row5.addWidget(groove_l)
        row5.addWidget(self._groove_combo)
        gen_layout.addLayout(row5)

        self._chart_lyrics = QTextEdit()
        self._chart_lyrics.setPlaceholderText(tr("midi.chart_lyrics_placeholder"))
        self._chart_lyrics.setMaximumHeight(54)
        self._chart_lyrics.setStyleSheet(f"""
            QTextEdit {{
                background: {t['background']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 5px;
                font-size: 8.25pt;
            }}
        """)
        gen_layout.addWidget(self._chart_lyrics)

        self._demo_checkbox = QCheckBox(tr("midi.demo_checkbox"))
        self._demo_checkbox.setToolTip(
            tr("midi.demo_tooltip")
        )
        self._demo_checkbox.toggled.connect(self._refresh_capability_state)
        gen_layout.addWidget(self._demo_checkbox)

        # Generate button
        self._gen_btn = QPushButton(tr("midi.generate"))
        self._gen_btn.setMinimumHeight(36)
        self._gen_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']};
                color: {t['background']}; border: none; border-radius: 5px;
                font-weight: bold; font-size: 9.75pt;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
            QPushButton:disabled {{ background: {t['border']}; color: {t['muted']}; }}
        """)
        self._gen_btn.clicked.connect(self._on_generate)
        gen_layout.addWidget(self._gen_btn)

        # Status
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(28)
        self._status.setStyleSheet(f"color: {t['text_secondary']}; font-size: 7.5pt; border:none;")
        gen_layout.addWidget(self._status)

        self._operation_progress = OperationProgressWidget()
        self._operation_progress.cancel_requested.connect(
            self._cancel_active_operation
        )
        gen_layout.addWidget(self._operation_progress)

        left.addWidget(gen_frame)

        # ── Mixer ──────────────────────────────────────────────────────────
        self._mixer = MidiMixer()
        self._mixer.track_selected.connect(self._on_track_selected)
        self._mixer.mix_changed.connect(self._on_mix_changed)
        self._mixer.empty_action_requested.connect(self._gen_btn.click)
        left.addWidget(self._mixer, 1)

        # ── Action buttons ─────────────────────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(6)

        btn_style = f"""
            QPushButton {{
                background: {t['surface']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 5px; padding: 6px 12px;
                font-size: 8.25pt; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """

        self._import_btn = QPushButton(tr("midi.import"))
        self._import_btn.setStyleSheet(btn_style)
        self._import_btn.clicked.connect(self._on_import)

        self._export_btn = QPushButton(tr("midi.export"))
        self._export_btn.setStyleSheet(btn_style)
        self._export_btn.clicked.connect(self._on_export)

        self._chart_btn = QPushButton(tr("midi.export_chart"))
        self._chart_btn.setStyleSheet(btn_style)
        self._chart_btn.clicked.connect(self._on_export_chart)

        self._render_btn = QPushButton(tr("midi.render_audio"))
        self._render_btn.setProperty("class", "success")
        self._render_btn.clicked.connect(self._on_render)

        action_row.addWidget(self._import_btn)
        action_row.addWidget(self._export_btn)
        action_row.addWidget(self._chart_btn)
        action_row.addWidget(self._render_btn)
        left.addLayout(action_row)

        # Route buttons
        route_row = QHBoxLayout()
        route_row.setSpacing(6)

        self._to_forge_btn = QPushButton(tr("midi.send_to_forge"))
        self._to_forge_btn.setStyleSheet(btn_style)
        self._to_forge_btn.setEnabled(False)
        self._to_forge_btn.clicked.connect(self._on_send_to_forge)

        self._to_vocals_btn = QPushButton(tr("midi.add_vocals"))
        self._to_vocals_btn.setStyleSheet(btn_style)
        self._to_vocals_btn.setEnabled(False)
        self._to_vocals_btn.clicked.connect(self._on_send_to_vocals)

        route_row.addWidget(self._to_forge_btn)
        route_row.addWidget(self._to_vocals_btn)
        left.addLayout(route_row)

        # ── Right: Piano Roll + Waveform ───────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        # Tabs: Piano Roll | Rendered Audio
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background: {t['background']};
                border: 1px solid {t['border']};
                border-radius: 6px;
            }}
            QTabBar::tab {{
                background: {t['surface']};
                color: {t['text_secondary']};
                border: 1px solid {t['border']};
                border-bottom: none;
                padding: 6px 16px;
                font-size: 8.25pt;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{
                background: {t['background']};
                color: {t['text']};
            }}
        """)

        # Piano roll tab
        self._piano_roll = PianoRollWidget()
        self._piano_roll.notes_changed.connect(self._on_notes_changed)
        self._tabs.addTab(self._piano_roll, tr("midi.tab_piano_roll"))

        # Rendered audio tab
        self._waveform = WaveformWidget()
        self._waveform.empty_action_requested.connect(self._import_btn.click)
        self._tabs.addTab(self._waveform, tr("midi.tab_rendered_audio"))

        self._workspace_empty = EmptyStateWidget(
            tr("midi.empty_title"),
            tr("midi.empty_description"),
            tr("midi.generate"),
        )
        self._workspace_empty.action_requested.connect(self._gen_btn.click)
        self._workspace_stack = QStackedWidget()
        self._workspace_stack.addWidget(self._tabs)
        self._workspace_stack.addWidget(self._workspace_empty)
        self._workspace_stack.setCurrentWidget(self._workspace_empty)
        right.addWidget(self._workspace_stack, 1)

        # Info bar
        self._info = QLabel(tr("midi.info_empty"))
        self._info.setStyleSheet(f"""
            color: {t['text_secondary']};
            font-size: 8.25pt;
            padding: 4px 8px;
            background: {t['surface']};
            border: 1px solid {t['border']};
            border-radius: 4px;
        """)
        right.addWidget(self._info)

        # Layout: left panel (fixed 320) | right panel (stretches)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(340)

        right_widget = QWidget()
        right_widget.setLayout(right)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget, 1)
        self._model_mgr.status_changed.connect(self._on_model_status_changed)
        self._refresh_capability_state()
        install_accessibility(
            self,
            tr("midi.accessibility.name"),
            named_controls=[
                (self._prompt, tr("midi.accessibility.prompt_name"), tr("midi.accessibility.prompt_description")),
                (self._style_input, tr("midi.accessibility.style_name"), tr("midi.accessibility.style_description")),
                (self._key_combo, tr("midi.accessibility.key_name"), tr("midi.accessibility.key_description")),
                (self._tempo_spin, tr("midi.accessibility.tempo_name"), tr("midi.accessibility.tempo_description")),
                (self._bars_spin, tr("midi.accessibility.bars_name"), tr("midi.accessibility.bars_description")),
                (self._time_sig, tr("midi.accessibility.time_name"), tr("midi.accessibility.time_description")),
                (self._inst_combo, tr("midi.accessibility.preset_name"), tr("midi.accessibility.preset_description")),
                (self._progression_combo, tr("midi.accessibility.chords_name"), tr("midi.accessibility.chords_description")),
                (self._groove_combo, tr("midi.accessibility.groove_name"), tr("midi.accessibility.groove_description")),
                (self._chart_lyrics, tr("midi.accessibility.chart_lyrics_name"), tr("midi.accessibility.chart_lyrics_description")),
                (self._demo_checkbox, tr("midi.accessibility.demo_name"), tr("midi.accessibility.demo_description")),
                (self._gen_btn, tr("midi.accessibility.generate_name"), tr("midi.accessibility.generate_description")),
                (self._import_btn, tr("midi.accessibility.import_name"), tr("midi.accessibility.import_description")),
                (self._export_btn, tr("midi.accessibility.export_name"), tr("midi.accessibility.export_description")),
                (self._chart_btn, tr("midi.accessibility.chart_name"), tr("midi.accessibility.chart_description")),
                (self._render_btn, tr("midi.accessibility.render_name"), tr("midi.accessibility.render_description")),
                (self._operation_progress.cancel_button, tr("midi.accessibility.cancel_name"), tr("midi.accessibility.cancel_description")),
                (self._to_forge_btn, tr("midi.accessibility.forge_name"), tr("midi.accessibility.forge_description")),
                (self._to_vocals_btn, tr("midi.accessibility.vocals_name"), tr("midi.accessibility.vocals_description")),
                (self._tabs, tr("midi.accessibility.tabs_name"), tr("midi.accessibility.tabs_description")),
            ],
        )
        self._settings.on_change(self._on_settings_change)

    def _on_settings_change(self, key: str, value, _old_value):
        """Apply the configured MIDI default to the live composition form."""
        if key == "midi_studio.default_bpm" and self._generation_worker is None:
            self._tempo_spin.setValue(int(value))

    # ── Generation ─────────────────────────────────────────────────────────────

    def _cancel_active_operation(self):
        """Request cancellation for whichever MIDI worker is running."""
        worker = self._generation_worker or self._render_worker
        if worker is None:
            self._operation_progress.finish()
            return
        self._operation_progress.mark_cancelling()
        worker.cancel()
        if worker is self._generation_worker:
            self._gen_btn.setEnabled(False)
            self._status.setText(tr("midi.status.cancelling_generation"))
        else:
            self._render_btn.setEnabled(False)
            self._status.setText(tr("midi.status.cancelling_render"))

    def _build_params(self) -> MidiGenParams:
        ts_text = self._time_sig.currentText()
        ts_parts = ts_text.split("/")
        time_sig = (int(ts_parts[0]), int(ts_parts[1]))

        preset_name = self._inst_combo.currentText()
        instruments = INSTRUMENT_PRESETS.get(preset_name, ["Piano"])

        return MidiGenParams(
            prompt=self._prompt.toPlainText().strip(),
            style=self._style_input.text().strip(),
            key=self._key_combo.currentText(),
            tempo=self._tempo_spin.value(),
            time_signature=time_sig,
            duration_bars=self._bars_spin.value(),
            instruments=instruments,
            chord_progression=self._progression_combo.currentText(),
            drum_groove=self._groove_combo.currentText(),
            allow_demo_output=self._demo_checkbox.isChecked(),
        )

    def _on_generate(self):
        """Generate with the active model or an explicitly enabled demo."""
        if self._generation_worker is not None:
            self._cancel_active_operation()
            return

        readiness = self._model_mgr.get_capability_readiness(
            CAP_MIDI_GENERATE,
            allow_demo=self._demo_checkbox.isChecked(),
        )
        if not readiness.can_run:
            self._status.setText(self._readiness_message(readiness))
            self._refresh_capability_state()
            return

        params = self._build_params()
        self._status.setText(
            tr("midi.status.generating_demo")
            if readiness.mode == RunMode.DEMO
            else tr("midi.status.generating_model")
        )
        self._generation_worker = InferenceWorker(
            self._run_generation,
            params,
            readiness.model_id,
            job_kind="midi_generation",
            job_label=tr("midi.jobs.generation"),
            job_inputs={
                "prompt_chars": len(params.prompt),
                "duration_bars": params.duration_bars,
                "tempo": params.tempo,
                "demo": params.allow_demo_output,
            },
            job_metadata={
                "module": "midi_studio",
                "capability_id": CAP_MIDI_GENERATE,
            },
        )
        self._generation_worker.progress.connect(
            self._on_generation_progress
        )
        self._generation_worker.step_info.connect(self._on_generation_step)
        self._generation_worker.finished.connect(self._on_generation_finished)
        self._generation_worker.error.connect(self._on_generation_error)
        self._generation_worker.cancelled.connect(self._on_generation_cancelled)
        self._operation_progress.start(tr("midi.operation_generation"), determinate=True)
        self._generation_worker.start()
        self._refresh_capability_state()

    def _on_generation_progress(self, percent: int):
        self._operation_progress.set_progress(percent, tr("midi.operation_generation"))
        self._status.setText(tr("midi.status.generation_progress", percent=percent))

    def _on_generation_step(self, message: str):
        self._operation_progress.set_step(message)
        self._status.setText(message)

    def _run_generation(
        self,
        params: MidiGenParams,
        model_id: str,
        progress_cb=None,
        step_cb=None,
        log_cb=None,
        cancel_event=None,
    ) -> EngineRunResult:
        if cancel_event and cancel_event.is_set():
            raise CancelledJobError("MIDI generation cancelled")

        def progress(value: float, message: str = ""):
            if progress_cb:
                progress_cb(max(0, min(100, int(round(value * 100)))))
            if step_cb and message:
                step_cb(message)

        result = generate_midi(
            params,
            progress_callback=progress,
            model_id=model_id,
            cancel_event=cancel_event,
        )
        if cancel_event and cancel_event.is_set():
            raise CancelledJobError("MIDI generation cancelled")
        artifacts = []
        if result.midi_data is not None:
            artifacts.append(EngineArtifact(
                kind=ArtifactKind.MIDI,
                payload=result.midi_data,
                provenance_path=result.provenance_path,
                routable=result.can_route,
            ))
        return adapt_engine_result(
            CAP_MIDI_GENERATE,
            result,
            artifacts,
            model_id=model_id,
        )

    def _on_generation_finished(self, run: EngineRunResult):
        self._generation_worker = None
        self._operation_progress.finish()
        self._contract_result = run
        if not run.is_success:
            self._report_error(tr("midi.status.generation_failed", error=run.error))
            self._refresh_capability_state()
            return
        result = run.source_result
        midi_data = result.midi_data
        self._load_midi_data(midi_data)
        prefix = tr("midi.status.demo_generated") if run.is_demo else tr("midi.status.model_generated")
        self._status.setText(
            tr(
                "midi.status.generated_summary",
                prefix=prefix,
                tracks=midi_data.track_count,
                notes=midi_data.total_notes,
                seconds=midi_data.duration,
            )
        )
        self._refresh_capability_state()

    def _on_generation_error(self, error: str):
        self._generation_worker = None
        self._operation_progress.finish()
        self._contract_result = EngineRunResult.failure(
            CAP_MIDI_GENERATE,
            error,
            model_id="midi-llm-1b",
        )
        self._report_error(tr("midi.status.generation_failed", error=error))
        self._refresh_capability_state()

    def _report_error(self, message: str):
        """Keep inline status and the application notification log in sync."""
        self._status.setText(message)
        if self.toast_mgr is not None:
            self.toast_mgr.error(message)

    def _on_generation_cancelled(self):
        self._generation_worker = None
        self._operation_progress.finish()
        self._contract_result = EngineRunResult.cancelled(
            CAP_MIDI_GENERATE,
            "MIDI generation cancelled",
            model_id="midi-llm-1b",
        )
        self._status.setText(tr("midi.status.generation_cancelled"))
        self._refresh_capability_state()

    def _readiness_message(self, readiness) -> str:
        info = self._model_mgr.get_model_info(readiness.model_id)
        return user_facing_readiness(
            readiness,
            model_name=info.name if info is not None else "",
        )

    def _on_model_status_changed(self, model_id: str, _status: str):
        if model_id == "midi-llm-1b":
            self._refresh_capability_state()

    def _refresh_capability_state(self):
        if not hasattr(self, "_gen_btn"):
            return
        readiness = self._model_mgr.get_capability_readiness(
            CAP_MIDI_GENERATE,
            allow_demo=self._demo_checkbox.isChecked(),
        )
        if self._generation_worker is not None:
            self._gen_btn.setText(tr("midi.cancel_generation"))
            self._gen_btn.setEnabled(True)
            self._gen_btn.setToolTip(tr("runtime.cancel_generation"))
            return
        has_prompt = bool(self._prompt.toPlainText().strip())
        self._gen_btn.setText(tr("midi.generate"))
        self._gen_btn.setEnabled(readiness.can_run and has_prompt)
        self._gen_btn.setToolTip(
            (
                tr("runtime.ready")
                if readiness.can_run and has_prompt
                else tr("runtime.enter_composition")
                if not has_prompt
                else self._readiness_message(readiness)
            )
        )
        if not readiness.can_run and not self._status.text():
            self._status.setText(self._readiness_message(readiness))

    def _load_midi_data(self, midi_data: MidiData):
        """Load MidiData into all views."""
        if self._render_worker is not None:
            self._render_worker.cancel()
        self._midi_data = midi_data
        self._workspace_stack.setCurrentWidget(self._tabs)

        # Load mixer
        self._mixer.load_midi(midi_data)

        # Load first track into piano roll
        if midi_data.tracks:
            self._piano_roll.load_track(
                midi_data.tracks[0],
                midi_data.tempo,
                max(4, int(midi_data.duration / (60.0 / midi_data.tempo * 4)) + 1),
            )

        self._update_info()
        self._rendered_audio = None
        self._current_audio_path = None
        self._rendered_output_kind = ""
        self._to_forge_btn.setEnabled(False)
        self._to_vocals_btn.setEnabled(False)

    def _on_track_selected(self, idx: int):
        """Switch piano roll to selected track."""
        if self._midi_data and 0 <= idx < len(self._midi_data.tracks):
            bars = max(4, int(self._midi_data.duration / (60.0 / self._midi_data.tempo * 4)) + 1)
            self._piano_roll.load_track(
                self._midi_data.tracks[idx],
                self._midi_data.tempo,
                bars,
            )

    def _on_notes_changed(self):
        """Handle piano roll edits."""
        self._update_info()
        if self._render_worker is not None:
            self._render_worker.cancel()
            self._status.setText(tr("midi.status.changed_cancelling_render"))
        elif self._rendered_audio is not None:
            self._rendered_audio = None
            self._current_audio_path = None
            self._rendered_output_kind = ""
            self._to_forge_btn.setEnabled(False)
            self._to_vocals_btn.setEnabled(False)
            self._status.setText(tr("midi.status.changed_render_again"))

    def _update_info(self):
        if self._midi_data:
            self._info.setText(
                tr(
                    "midi.info_summary",
                    tracks=self._midi_data.track_count,
                    notes=self._midi_data.total_notes,
                    tempo=self._midi_data.tempo,
                    seconds=self._midi_data.duration,
                    time_signature=(
                        f"{self._midi_data.time_signature[0]}"
                        f"/{self._midi_data.time_signature[1]}"
                    ),
                )
            )

    # ── Import / Export ────────────────────────────────────────────────────────

    def _on_import(self):
        path, _ = open_midi_file(
            self,
            tr("midi.dialog_import"),
            operation_kind="midi_import",
            dialog=QFileDialog,
        )
        if path:
            try:
                midi_data = load_midi(path)
                self._load_midi_data(midi_data)
                self._status.setText(tr("midi.status.imported", path=path))
            except Exception as e:
                self._report_error(tr("midi.status.import_error", error=e))

    def _on_export(self):
        if not self._midi_data:
            self._report_error(tr("midi.status.nothing_to_export"))
            return

        path, selected_filter = save_midi_file(
            self,
            tr("midi.dialog_export"),
            "composition.mid",
            operation_kind="midi_export",
            dialog=QFileDialog,
        )
        if path:
            path = ensure_extension(path, selected_filter, default="mid")
            try:
                save_midi(self._midi_data, path)
                self._status.setText(tr("midi.status.exported", path=path))
            except Exception as e:
                self._report_error(tr("midi.status.export_error", error=e))

    # ── Render ─────────────────────────────────────────────────────────────────

    def _on_export_chart(self):
        if not self._midi_data:
            self._report_error(tr("midi.status.nothing_to_export"))
            return

        path, selected_filter = save_file(
            self,
            tr("midi.dialog_chart"),
            "chord_chart.chordpro",
            tr("shell.dialogs.chordpro_filter"),
            "midi_chart_export",
            dialog=QFileDialog,
        )
        if not path:
            return

        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".crd" if "crd" in selected_filter.lower() else ".chordpro")

        try:
            output = save_chord_chart(
                self._midi_data,
                str(target),
                lyrics=self._chart_lyrics.toPlainText().strip(),
                title=target.stem,
            )
            self._status.setText(tr("midi.status.chart_exported", path=output))
        except Exception as e:
            self._report_error(tr("midi.status.chart_export_error", error=e))

    def _on_render(self):
        """Render MIDI to audio via a cancellable worker job."""
        if self._render_worker is not None:
            self._cancel_active_operation()
            self._render_btn.setEnabled(False)
            return
        if not self._midi_data:
            self._report_error(tr("midi.status.nothing_to_render"))
            return

        self._rendered_audio = None
        self._current_audio_path = None
        self._rendered_output_kind = ""
        self._to_forge_btn.setEnabled(False)
        self._to_vocals_btn.setEnabled(False)
        self._status.setText(tr("midi.status.rendering_audio"))

        import os
        import time as time_mod
        from core.settings import get_configured_output_dir

        output_dir = os.path.join(get_configured_output_dir(), "generations", "midi_renders")
        os.makedirs(output_dir, exist_ok=True)
        ts = time_mod.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"render_{ts}_{time_mod.time_ns()}.wav")
        muted = self._mixer.get_muted_tracks()
        solo_tracks = self._mixer.get_solo_tracks()
        track_mix = self._mixer.get_track_mix()

        self._render_worker = InferenceWorker(
            self._run_render,
            self._midi_data,
            output_path,
            muted,
            solo_tracks,
            track_mix,
            job_kind="midi_render",
            job_label=tr("midi.jobs.render"),
            job_inputs={
                "track_count": self._midi_data.track_count,
                "total_notes": self._midi_data.total_notes,
                "muted_tracks": sorted(muted),
                "solo_tracks": sorted(solo_tracks),
            },
            job_metadata={
                "module": "midi_studio",
                "capability_id": CAP_MIDI_RENDER,
            },
        )
        self._render_worker.progress.connect(
            self._on_render_progress
        )
        self._render_worker.step_info.connect(self._on_render_step)
        self._render_worker.finished.connect(self._on_render_finished)
        self._render_worker.error.connect(self._on_render_error)
        self._render_worker.cancelled.connect(self._on_render_cancelled)
        self._operation_progress.start(tr("midi.operation_render"), determinate=True)
        self._render_worker.start()
        self._render_btn.setText(tr("midi.cancel_render"))

    def _on_render_progress(self, percent: int):
        self._operation_progress.set_progress(percent, tr("midi.operation_render"))
        self._status.setText(tr("midi.status.render_progress", percent=percent))

    def _on_render_step(self, message: str):
        self._operation_progress.set_step(message)
        self._status.setText(message)

    def _run_render(
        self,
        midi_data: MidiData,
        output_path: str,
        muted: set[int],
        solo_tracks: set[int],
        track_mix: dict[int, dict[str, float]],
        progress_cb=None,
        step_cb=None,
        log_cb=None,
        cancel_event=None,
    ) -> EngineRunResult:
        """Render an immutable MIDI/mixer snapshot outside the Qt GUI thread."""
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledJobError("MIDI render cancelled", outputs=[output_path])

        from engines.fluidsynth_engine import (
            MidiRenderCancelled,
            MidiRenderResult,
            render_midi_to_audio,
        )

        def progress(value: float, message: str = ""):
            if progress_cb:
                progress_cb(max(0, min(100, int(round(value * 100)))))
            if step_cb and message:
                step_cb(message)

        owned_outputs = [output_path, str(sidecar_path_for(output_path))]

        try:
            render_result = render_midi_to_audio(
                midi_data,
                output_path=output_path,
                mute_tracks=set(muted),
                solo_tracks=set(solo_tracks),
                track_mix={idx: dict(values) for idx, values in track_mix.items()},
                progress_callback=progress,
                return_metadata=True,
                cancel_event=cancel_event,
            )
        except MidiRenderCancelled as exc:
            raise CancelledJobError(
                "MIDI render cancelled",
                outputs=owned_outputs,
            ) from exc
        if isinstance(render_result, MidiRenderResult):
            result = render_result
        else:
            # Keep compatibility with older third-party render adapters.
            result = MidiRenderResult(audio=render_result)

        if cancel_event is not None and cancel_event.is_set():
            raise CancelledJobError("MIDI render cancelled", outputs=owned_outputs)
        if result.audio is None:
            raise RuntimeError("MIDI renderer returned no audio")

        artifact = EngineArtifact(
            kind=ArtifactKind.AUDIO,
            path=output_path,
            payload=result.audio,
            provenance_path=str(sidecar_path_for(output_path)),
            routable=not result.is_demo,
            metadata={
                "output_kind": result.output_kind,
                "fallback_reason": result.fallback_reason,
                "muted_tracks": sorted(muted),
                "solo_tracks": sorted(solo_tracks),
                "track_mix": track_mix,
            },
        )
        return adapt_engine_result(CAP_MIDI_RENDER, result, [artifact])

    def _on_render_finished(self, run: EngineRunResult):
        self._render_worker = None
        self._operation_progress.finish()
        self._render_btn.setText(tr("midi.render_audio"))
        self._render_btn.setEnabled(True)
        self._contract_result = run
        if not run.is_success:
            self._report_error(tr("midi.status.render_failed", error=run.error))
            return

        result = run.source_result
        audio = result.audio
        artifact = run.first_artifact(ArtifactKind.AUDIO)
        output_path = artifact.path if artifact else getattr(result, "output_path", "")
        self._rendered_audio = audio
        self._current_audio_path = output_path
        self._rendered_output_kind = result.output_kind

        if audio is not None and len(audio) > 0:
            self._waveform.load_audio(audio, 44100)
            self._tabs.setCurrentIndex(1)

        is_demo = run.is_demo
        self._to_forge_btn.setEnabled(not is_demo)
        self._to_vocals_btn.setEnabled(not is_demo)
        if is_demo:
            reason = result.fallback_reason or tr("midi.status.fluidsynth_unavailable")
            self._status.setText(
                tr("midi.status.preview_render", reason=reason)
            )
        else:
            self._status.setText(tr("midi.status.rendered", path=output_path))

    def _on_render_error(self, error: str):
        self._render_worker = None
        self._operation_progress.finish()
        self._render_btn.setText(tr("midi.render_audio"))
        self._render_btn.setEnabled(True)
        self._contract_result = EngineRunResult.failure(CAP_MIDI_RENDER, error)
        self._report_error(tr("midi.status.render_error", error=error))

    def _on_render_cancelled(self):
        self._render_worker = None
        self._operation_progress.finish()
        self._render_btn.setText(tr("midi.render_audio"))
        self._render_btn.setEnabled(True)
        self._contract_result = EngineRunResult.cancelled(
            CAP_MIDI_RENDER,
            "MIDI render cancelled",
        )
        self._status.setText(tr("midi.status.render_cancelled"))

    def _on_mix_changed(self):
        """Invalidate stale audio when a mixer control changes."""
        if self._render_worker is not None:
            self._cancel_active_operation()
            self._render_btn.setEnabled(False)
            self._status.setText(tr("midi.status.mix_changed_cancelling"))
            return
        if self._rendered_audio is not None:
            self._rendered_audio = None
            self._current_audio_path = None
            self._rendered_output_kind = ""
            self._to_forge_btn.setEnabled(False)
            self._to_vocals_btn.setEnabled(False)
            self._status.setText(tr("midi.status.mix_changed_render_again"))

    # ── Cross-Module Routing ───────────────────────────────────────────────────

    def _on_send_to_forge(self):
        if self._current_audio_path and self._rendered_output_kind != "demo":
            self.send_to_forge.emit(self._current_audio_path)

    def _on_send_to_vocals(self):
        if self._current_audio_path and self._rendered_output_kind != "demo":
            self.send_to_vocals.emit(self._current_audio_path)

    def closeEvent(self, event):
        """Request cancellation before the view is torn down."""
        for worker in (self._generation_worker, self._render_worker):
            if worker is not None:
                worker.cancel()
        super().closeEvent(event)

    # ── External API ───────────────────────────────────────────────────────────

    def set_midi_data(self, midi_data: MidiData):
        """Load MIDI from external source (e.g. another module)."""
        self._load_midi_data(midi_data)

    def controller_quantize(self) -> bool:
        """Apply the Piano Roll quantize action from an external controller."""
        return self._piano_roll.controller_quantize()

    def controller_swing(self) -> bool:
        """Apply the Piano Roll swing action from an external controller."""
        return self._piano_roll.controller_swing()

    def controller_humanize(self) -> bool:
        """Apply the Piano Roll humanize action from an external controller."""
        return self._piano_roll.controller_humanize()
