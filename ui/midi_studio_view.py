"""
Slunder Studio v0.0.2 — MIDI Studio View
Main MIDI Studio page: text-to-MIDI generation, piano roll editor,
per-track mixer, .mid import/export, FluidSynth rendering, and
cross-module routing (Song Forge, Vocal Suite).
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QTabWidget,
    QFrame, QSplitter, QGroupBox, QLineEdit, QStackedWidget,
)
from PySide6.QtCore import Qt, Signal

from ui.theme import ThemeEngine
from ui.piano_roll import PianoRollWidget
from ui.midi_mixer import MidiMixer
from ui.waveform_widget import WaveformWidget
from core.midi_utils import (
    MidiData, TrackData, NoteData, load_midi, save_midi, get_program_name,
)
from engines.midi_llm_engine import MidiGenParams, MidiGenResult, generate_demo_midi


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


class MidiStudioView(QWidget):
    """Main MIDI Studio page."""

    send_to_forge = Signal(str)     # audio file path -> Song Forge
    send_to_vocals = Signal(str)    # audio file path -> Vocal Suite

    def __init__(self, parent=None):
        super().__init__(parent)
        self._midi_data: Optional[MidiData] = None
        self._rendered_audio = None
        self._current_audio_path: Optional[str] = None

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
                background: {t.get('surface', '#161b22')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 8px;
            }}
        """)
        gen_layout = QVBoxLayout(gen_frame)
        gen_layout.setContentsMargins(12, 10, 12, 10)
        gen_layout.setSpacing(6)

        gen_title = QLabel("Text-to-MIDI")
        gen_title.setStyleSheet(f"color: {t.get('accent', '#58a6ff')}; font-weight: bold; font-size: 13px; border: none;")
        gen_layout.addWidget(gen_title)

        # Prompt
        self._prompt = QTextEdit()
        self._prompt.setPlaceholderText("Describe the composition...\ne.g. 'A melancholy jazz ballad with walking bass and soft piano chords'")
        self._prompt.setMaximumHeight(70)
        self._prompt.setStyleSheet(f"""
            QTextEdit {{
                background: {t.get('background', '#0d1117')};
                color: {t.get('text', '#e6edf3')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 4px;
                padding: 6px;
                font-size: 12px;
            }}
        """)
        gen_layout.addWidget(self._prompt)

        # Style
        style_row = QHBoxLayout()
        style_label = QLabel("Style:")
        style_label.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')}; font-size: 11px; border:none;")
        style_label.setFixedWidth(40)
        self._style_input = QLineEdit()
        self._style_input.setPlaceholderText("jazz piano ballad, soft, legato")
        self._style_input.setStyleSheet(f"""
            QLineEdit {{
                background: {t.get('background', '#0d1117')};
                color: {t.get('text', '#e6edf3')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 4px; padding: 4px 8px; font-size: 11px;
            }}
        """)
        style_row.addWidget(style_label)
        style_row.addWidget(self._style_input)
        gen_layout.addLayout(style_row)

        # Parameters grid
        param_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {t.get('background', '#0d1117')};
                color: {t.get('text', '#e6edf3')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 3px; padding: 3px 6px; font-size: 11px;
            }}
            QLabel {{ color: {t.get('text_secondary', '#8b949e')}; font-size: 11px; border:none; }}
        """

        # Row 1: Key + Tempo
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        key_l = QLabel("Key:")
        key_l.setFixedWidth(34)
        key_l.setStyleSheet(param_style)
        self._key_combo = QComboBox()
        self._key_combo.addItems(KEYS)
        self._key_combo.setCurrentText("C major")
        self._key_combo.setStyleSheet(param_style)

        tempo_l = QLabel("BPM:")
        tempo_l.setFixedWidth(30)
        tempo_l.setStyleSheet(param_style)
        self._tempo_spin = QSpinBox()
        self._tempo_spin.setRange(40, 300)
        self._tempo_spin.setValue(120)
        self._tempo_spin.setStyleSheet(param_style)

        row1.addWidget(key_l)
        row1.addWidget(self._key_combo)
        row1.addWidget(tempo_l)
        row1.addWidget(self._tempo_spin)
        gen_layout.addLayout(row1)

        # Row 2: Bars + Time Sig
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        bars_l = QLabel("Bars:")
        bars_l.setFixedWidth(34)
        bars_l.setStyleSheet(param_style)
        self._bars_spin = QSpinBox()
        self._bars_spin.setRange(4, 128)
        self._bars_spin.setValue(16)
        self._bars_spin.setStyleSheet(param_style)

        ts_l = QLabel("Time:")
        ts_l.setFixedWidth(30)
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
        inst_l = QLabel("Preset:")
        inst_l.setFixedWidth(40)
        inst_l.setStyleSheet(param_style)
        self._inst_combo = QComboBox()
        self._inst_combo.addItems(INSTRUMENT_PRESETS.keys())
        self._inst_combo.setCurrentText("Band (4-piece)")
        self._inst_combo.setStyleSheet(param_style)
        row3.addWidget(inst_l)
        row3.addWidget(self._inst_combo)
        gen_layout.addLayout(row3)

        # Generate button
        self._gen_btn = QPushButton("Generate MIDI")
        self._gen_btn.setFixedHeight(36)
        self._gen_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {t.get('accent', '#58a6ff')}, stop:1 #a371f7);
                color: white; border: none; border-radius: 6px;
                font-weight: bold; font-size: 13px;
            }}
            QPushButton:hover {{ background: {t.get('accent_hover', '#79c0ff')}; }}
            QPushButton:disabled {{ background: {t.get('border', '#1e2733')}; color: #555; }}
        """)
        self._gen_btn.clicked.connect(self._on_generate)
        gen_layout.addWidget(self._gen_btn)

        # Status
        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')}; font-size: 10px; border:none;")
        gen_layout.addWidget(self._status)

        left.addWidget(gen_frame)

        # ── Mixer ──────────────────────────────────────────────────────────
        self._mixer = MidiMixer()
        self._mixer.track_selected.connect(self._on_track_selected)
        left.addWidget(self._mixer, 1)

        # ── Action buttons ─────────────────────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(6)

        btn_style = f"""
            QPushButton {{
                background: {t.get('surface', '#161b22')};
                color: {t.get('text', '#e6edf3')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 5px; padding: 6px 12px;
                font-size: 11px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t.get('surface_hover', '#1c2333')}; }}
        """

        self._import_btn = QPushButton("Import .mid")
        self._import_btn.setStyleSheet(btn_style)
        self._import_btn.clicked.connect(self._on_import)

        self._export_btn = QPushButton("Export .mid")
        self._export_btn.setStyleSheet(btn_style)
        self._export_btn.clicked.connect(self._on_export)

        self._render_btn = QPushButton("Render Audio")
        self._render_btn.setStyleSheet(f"""
            QPushButton {{
                background: #238636; color: white; border: none;
                border-radius: 5px; padding: 6px 12px;
                font-size: 11px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #2ea043; }}
        """)
        self._render_btn.clicked.connect(self._on_render)

        action_row.addWidget(self._import_btn)
        action_row.addWidget(self._export_btn)
        action_row.addWidget(self._render_btn)
        left.addLayout(action_row)

        # Route buttons
        route_row = QHBoxLayout()
        route_row.setSpacing(6)

        self._to_forge_btn = QPushButton("Send to Song Forge")
        self._to_forge_btn.setStyleSheet(btn_style)
        self._to_forge_btn.setEnabled(False)
        self._to_forge_btn.clicked.connect(self._on_send_to_forge)

        self._to_vocals_btn = QPushButton("Add Vocals")
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
                background: {t.get('background', '#0d1117')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 6px;
            }}
            QTabBar::tab {{
                background: {t.get('surface', '#161b22')};
                color: {t.get('text_secondary', '#8b949e')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-bottom: none;
                padding: 6px 16px;
                font-size: 11px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }}
            QTabBar::tab:selected {{
                background: {t.get('background', '#0d1117')};
                color: {t.get('text', '#e6edf3')};
            }}
        """)

        # Piano roll tab
        self._piano_roll = PianoRollWidget()
        self._piano_roll.notes_changed.connect(self._on_notes_changed)
        self._tabs.addTab(self._piano_roll, "Piano Roll")

        # Rendered audio tab
        self._waveform = WaveformWidget()
        self._tabs.addTab(self._waveform, "Rendered Audio")

        right.addWidget(self._tabs, 1)

        # Info bar
        self._info = QLabel("No MIDI loaded. Generate or import a file.")
        self._info.setStyleSheet(f"""
            color: {t.get('text_secondary', '#8b949e')};
            font-size: 11px;
            padding: 4px 8px;
            background: {t.get('surface', '#161b22')};
            border: 1px solid {t.get('border', '#1e2733')};
            border-radius: 4px;
        """)
        right.addWidget(self._info)

        # Layout: left panel (fixed 320) | right panel (stretches)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(340)

        right_widget = QWidget()
        right_widget.setLayout(right)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget, 1)

    # ── Generation ─────────────────────────────────────────────────────────────

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
        )

    def _on_generate(self):
        """Generate MIDI (uses model if available, else demo fallback)."""
        params = self._build_params()
        self._gen_btn.setEnabled(False)
        self._status.setText("Generating...")

        # Use demo generator directly for now (model integration via InferenceWorker)
        try:
            midi_data = generate_demo_midi(params)
            self._load_midi_data(midi_data)
            self._status.setText(
                f"Generated: {midi_data.track_count} tracks, "
                f"{midi_data.total_notes} notes, "
                f"{midi_data.duration:.1f}s"
            )
        except Exception as e:
            self._status.setText(f"Error: {e}")
        finally:
            self._gen_btn.setEnabled(True)

    def _load_midi_data(self, midi_data: MidiData):
        """Load MidiData into all views."""
        self._midi_data = midi_data

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

    def _update_info(self):
        if self._midi_data:
            self._info.setText(
                f"Tracks: {self._midi_data.track_count}  |  "
                f"Notes: {self._midi_data.total_notes}  |  "
                f"Tempo: {self._midi_data.tempo:.0f} BPM  |  "
                f"Duration: {self._midi_data.duration:.1f}s  |  "
                f"Time Sig: {self._midi_data.time_signature[0]}/{self._midi_data.time_signature[1]}"
            )

    # ── Import / Export ────────────────────────────────────────────────────────

    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import MIDI File", "", "MIDI Files (*.mid *.midi)"
        )
        if path:
            try:
                midi_data = load_midi(path)
                self._load_midi_data(midi_data)
                self._status.setText(f"Imported: {path}")
            except Exception as e:
                self._status.setText(f"Import error: {e}")

    def _on_export(self):
        if not self._midi_data:
            self._status.setText("Nothing to export")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export MIDI File", "composition.mid", "MIDI Files (*.mid)"
        )
        if path:
            try:
                save_midi(self._midi_data, path)
                self._status.setText(f"Exported: {path}")
            except Exception as e:
                self._status.setText(f"Export error: {e}")

    # ── Render ─────────────────────────────────────────────────────────────────

    def _on_render(self):
        """Render MIDI to audio via FluidSynth or fallback."""
        if not self._midi_data:
            self._status.setText("Nothing to render")
            return

        self._render_btn.setEnabled(False)
        self._status.setText("Rendering audio...")

        try:
            import os
            import time as time_mod
            from engines.fluidsynth_engine import render_midi_to_audio
            from core.settings import get_config_dir

            output_dir = os.path.join(get_config_dir(), "generations", "midi_renders")
            os.makedirs(output_dir, exist_ok=True)
            ts = time_mod.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"render_{ts}.wav")

            muted = self._mixer.get_muted_tracks()
            solo = self._mixer.get_solo_track()

            audio = render_midi_to_audio(
                self._midi_data,
                output_path=output_path,
            )

            self._rendered_audio = audio
            self._current_audio_path = output_path

            # Load into waveform view
            import numpy as np
            if audio is not None and len(audio) > 0:
                mono = audio[:, 0] if audio.ndim == 2 else audio
                self._waveform.load_audio(mono, 44100)
                self._tabs.setCurrentIndex(1)  # Switch to rendered audio tab

            self._to_forge_btn.setEnabled(True)
            self._to_vocals_btn.setEnabled(True)
            self._status.setText(f"Rendered: {output_path}")

        except Exception as e:
            self._status.setText(f"Render error: {e}")
        finally:
            self._render_btn.setEnabled(True)

    # ── Cross-Module Routing ───────────────────────────────────────────────────

    def _on_send_to_forge(self):
        if self._current_audio_path:
            self.send_to_forge.emit(self._current_audio_path)

    def _on_send_to_vocals(self):
        if self._current_audio_path:
            self.send_to_vocals.emit(self._current_audio_path)

    # ── External API ───────────────────────────────────────────────────────────

    def set_midi_data(self, midi_data: MidiData):
        """Load MIDI from external source (e.g. another module)."""
        self._load_midi_data(midi_data)
