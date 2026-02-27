"""
Slunder Studio v0.0.2 — Vocal Suite View
Main Vocal Suite page combining singing synthesis (DiffSinger),
voice conversion (RVC), voice cloning (GPT-SoVITS), stem separation (Demucs),
and stem remix/export.
"""
import os
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QTabWidget,
    QFrame, QLineEdit, QSlider, QGroupBox, QStackedWidget,
)
from PySide6.QtCore import Qt, Signal

from ui.theme import ThemeEngine
from ui.waveform_widget import WaveformWidget
from ui.stem_mixer import StemMixer
from core.voice_bank import VoiceBank, VoiceProfile


class VocalSuiteView(QWidget):
    """Main Vocal Suite page with tabbed sub-views."""

    send_to_forge = Signal(str)    # audio path -> Song Forge
    send_to_mixer = Signal(str)    # audio path -> Mixer (Phase 6)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_audio_path: Optional[str] = None

        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Tab bar for sub-views
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
                padding: 7px 18px;
                font-size: 11px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background: {t['background']};
                color: {t['text']};
                font-weight: bold;
            }}
        """)

        # Tab 1: Singing Synthesis (DiffSinger)
        self._tabs.addTab(self._build_singing_tab(), "Singing Synthesis")

        # Tab 2: Voice Conversion (RVC)
        self._tabs.addTab(self._build_rvc_tab(), "Voice Conversion")

        # Tab 3: Voice Cloning (GPT-SoVITS)
        self._tabs.addTab(self._build_clone_tab(), "Voice Cloning")

        # Tab 4: Stem Separation (Demucs)
        self._tabs.addTab(self._build_stems_tab(), "Stem Separation")

        layout.addWidget(self._tabs, 1)

        # Bottom action bar
        action_bar = QHBoxLayout()
        action_bar.setSpacing(6)

        btn_style = f"""
            QPushButton {{
                background: {t['surface']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 5px; padding: 6px 14px;
                font-size: 11px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
            QPushButton:disabled {{ color: {t['muted']}; }}
        """

        self._to_forge_btn = QPushButton("Send to Song Forge")
        self._to_forge_btn.setStyleSheet(btn_style)
        self._to_forge_btn.setEnabled(False)
        self._to_forge_btn.clicked.connect(self._on_send_to_forge)

        self._to_mixer_btn = QPushButton("Send to Mixer")
        self._to_mixer_btn.setStyleSheet(btn_style)
        self._to_mixer_btn.setEnabled(False)
        self._to_mixer_btn.clicked.connect(self._on_send_to_mixer)

        self._export_btn = QPushButton("Export WAV")
        self._export_btn.setStyleSheet(f"""
            QPushButton {{
                background: #238636; color: white; border: none;
                border-radius: 5px; padding: 6px 14px;
                font-size: 11px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #2ea043; }}
            QPushButton:disabled {{ background: {t['border']}; color: #555; }}
        """)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)

        # Status
        self._status = QLabel("Select a tab to begin")
        self._status.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")

        action_bar.addWidget(self._status, 1)
        action_bar.addWidget(self._to_forge_btn)
        action_bar.addWidget(self._to_mixer_btn)
        action_bar.addWidget(self._export_btn)
        layout.addLayout(action_bar)

    # ── Tab Builders ───────────────────────────────────────────────────────────

    def _build_singing_tab(self) -> QWidget:
        """DiffSinger singing synthesis tab."""
        t = ThemeEngine.get_colors()
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: Controls
        left = QVBoxLayout()
        left.setSpacing(6)

        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(12, 10, 12, 10)
        ctrl_layout.setSpacing(6)

        title = QLabel("DiffSinger")
        title.setStyleSheet(f"color: {t['accent']}; font-weight: bold; font-size: 13px; border: none;")
        ctrl_layout.addWidget(title)

        # Lyrics input
        self._sing_lyrics = QTextEdit()
        self._sing_lyrics.setPlaceholderText("Enter lyrics to sing...\nEach line = one phrase")
        self._sing_lyrics.setMaximumHeight(80)
        self._sing_lyrics.setStyleSheet(f"""
            QTextEdit {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 6px; font-size: 12px;
            }}
        """)
        ctrl_layout.addWidget(self._sing_lyrics)

        param_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 3px 6px; font-size: 11px;
            }}
            QLabel {{ color: {t['text_secondary']}; font-size: 11px; border: none; }}
        """

        # Voice selector
        voice_row = QHBoxLayout()
        vl = QLabel("Voice:")
        vl.setFixedWidth(50)
        vl.setStyleSheet(param_style)
        self._sing_voice = QComboBox()
        self._sing_voice.setStyleSheet(param_style)
        self._sing_voice.addItem("(No voices loaded)")
        voice_row.addWidget(vl)
        voice_row.addWidget(self._sing_voice)
        ctrl_layout.addLayout(voice_row)

        # Tempo + Key
        row1 = QHBoxLayout()
        tl = QLabel("BPM:")
        tl.setFixedWidth(30)
        tl.setStyleSheet(param_style)
        self._sing_tempo = QSpinBox()
        self._sing_tempo.setRange(40, 300)
        self._sing_tempo.setValue(120)
        self._sing_tempo.setStyleSheet(param_style)

        kl = QLabel("Key:")
        kl.setFixedWidth(24)
        kl.setStyleSheet(param_style)
        self._sing_key = QComboBox()
        self._sing_key.addItems(["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"])
        self._sing_key.setStyleSheet(param_style)

        row1.addWidget(tl)
        row1.addWidget(self._sing_tempo)
        row1.addWidget(kl)
        row1.addWidget(self._sing_key)
        ctrl_layout.addLayout(row1)

        # Expression controls
        for name, default, range_max in [
            ("Breathiness", 0, 100), ("Tension", 50, 100),
            ("Vibrato", 50, 100), ("Gender", 50, 100),
        ]:
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl = QLabel(f"{name}:")
            lbl.setFixedWidth(70)
            lbl.setStyleSheet(param_style)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, range_max)
            slider.setValue(default)
            slider.setFixedHeight(16)
            val = QLabel(str(default))
            val.setFixedWidth(24)
            val.setStyleSheet(param_style)
            slider.valueChanged.connect(lambda v, l=val: l.setText(str(v)))
            row.addWidget(lbl)
            row.addWidget(slider)
            row.addWidget(val)
            ctrl_layout.addLayout(row)
            setattr(self, f"_sing_{name.lower()}", slider)

        # Generate button
        self._sing_gen_btn = QPushButton("Synthesize Vocals")
        self._sing_gen_btn.setFixedHeight(34)
        self._sing_gen_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #f38ba8, stop:1 #cba6f7);
                color: white; border: none; border-radius: 6px;
                font-weight: bold; font-size: 12px;
            }}
            QPushButton:hover {{ opacity: 0.9; }}
        """)
        self._sing_gen_btn.clicked.connect(self._on_sing_generate)
        ctrl_layout.addWidget(self._sing_gen_btn)

        left.addWidget(ctrl_frame)
        left.addStretch()

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(320)
        layout.addWidget(left_w)

        # Right: Waveform output
        self._sing_waveform = WaveformWidget()
        layout.addWidget(self._sing_waveform, 1)

        return widget

    def _build_rvc_tab(self) -> QWidget:
        """RVC voice conversion tab."""
        t = ThemeEngine.get_colors()
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: Controls
        left = QVBoxLayout()
        left.setSpacing(6)

        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(12, 10, 12, 10)
        ctrl_layout.setSpacing(6)

        title = QLabel("RVC Voice Conversion")
        title.setStyleSheet(f"color: {t['accent']}; font-weight: bold; font-size: 13px; border: none;")
        ctrl_layout.addWidget(title)

        param_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 3px 6px; font-size: 11px;
            }}
            QLabel {{ color: {t['text_secondary']}; font-size: 11px; border: none; }}
        """

        # Input audio
        input_row = QHBoxLayout()
        self._rvc_input_label = QLabel("No file selected")
        self._rvc_input_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px; border: none;")
        self._rvc_browse_btn = QPushButton("Browse")
        self._rvc_browse_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 4px 10px; font-size: 10px;
            }}
        """)
        self._rvc_browse_btn.clicked.connect(self._on_rvc_browse)
        input_row.addWidget(QLabel("Input:"))
        input_row.addWidget(self._rvc_input_label, 1)
        input_row.addWidget(self._rvc_browse_btn)
        for w in [input_row.itemAt(0).widget()]:
            if w:
                w.setStyleSheet(param_style)
        ctrl_layout.addLayout(input_row)

        # Voice model selector
        voice_row = QHBoxLayout()
        vl = QLabel("Voice:")
        vl.setFixedWidth(50)
        vl.setStyleSheet(param_style)
        self._rvc_voice = QComboBox()
        self._rvc_voice.addItem("(No RVC models loaded)")
        self._rvc_voice.setStyleSheet(param_style)
        voice_row.addWidget(vl)
        voice_row.addWidget(self._rvc_voice)
        ctrl_layout.addLayout(voice_row)

        # Pitch shift
        pitch_row = QHBoxLayout()
        pl = QLabel("Pitch:")
        pl.setFixedWidth(50)
        pl.setStyleSheet(param_style)
        self._rvc_pitch = QSpinBox()
        self._rvc_pitch.setRange(-24, 24)
        self._rvc_pitch.setValue(0)
        self._rvc_pitch.setSuffix(" st")
        self._rvc_pitch.setStyleSheet(param_style)
        pitch_row.addWidget(pl)
        pitch_row.addWidget(self._rvc_pitch)
        ctrl_layout.addLayout(pitch_row)

        # F0 method
        f0_row = QHBoxLayout()
        fl = QLabel("F0:")
        fl.setFixedWidth(50)
        fl.setStyleSheet(param_style)
        self._rvc_f0 = QComboBox()
        self._rvc_f0.addItems(["rmvpe", "pm", "harvest", "crepe"])
        self._rvc_f0.setStyleSheet(param_style)
        f0_row.addWidget(fl)
        f0_row.addWidget(self._rvc_f0)
        ctrl_layout.addLayout(f0_row)

        # Index rate
        idx_row = QHBoxLayout()
        il = QLabel("Index:")
        il.setFixedWidth(50)
        il.setStyleSheet(param_style)
        self._rvc_index = QSlider(Qt.Horizontal)
        self._rvc_index.setRange(0, 100)
        self._rvc_index.setValue(75)
        self._rvc_index.setFixedHeight(16)
        self._rvc_idx_val = QLabel("0.75")
        self._rvc_idx_val.setFixedWidth(30)
        self._rvc_idx_val.setStyleSheet(param_style)
        self._rvc_index.valueChanged.connect(
            lambda v: self._rvc_idx_val.setText(f"{v / 100:.2f}"))
        idx_row.addWidget(il)
        idx_row.addWidget(self._rvc_index)
        idx_row.addWidget(self._rvc_idx_val)
        ctrl_layout.addLayout(idx_row)

        # Convert button
        self._rvc_convert_btn = QPushButton("Convert Voice")
        self._rvc_convert_btn.setFixedHeight(34)
        self._rvc_convert_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {t['accent']}, stop:1 #a371f7);
                color: white; border: none; border-radius: 6px;
                font-weight: bold; font-size: 12px;
            }}
        """)
        self._rvc_convert_btn.clicked.connect(self._on_rvc_convert)
        ctrl_layout.addWidget(self._rvc_convert_btn)

        left.addWidget(ctrl_frame)
        left.addStretch()

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(320)
        layout.addWidget(left_w)

        # Right: Before/After waveforms
        right = QVBoxLayout()
        right.setSpacing(4)
        ol = QLabel("Original")
        ol.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px;")
        self._rvc_original_wf = WaveformWidget()
        cl = QLabel("Converted")
        cl.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px;")
        self._rvc_converted_wf = WaveformWidget()
        right.addWidget(ol)
        right.addWidget(self._rvc_original_wf, 1)
        right.addWidget(cl)
        right.addWidget(self._rvc_converted_wf, 1)

        right_w = QWidget()
        right_w.setLayout(right)
        layout.addWidget(right_w, 1)

        return widget

    def _build_clone_tab(self) -> QWidget:
        """GPT-SoVITS voice cloning tab."""
        t = ThemeEngine.get_colors()
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Controls
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(12, 10, 12, 10)
        ctrl_layout.setSpacing(6)

        title = QLabel("GPT-SoVITS Voice Cloning")
        title.setStyleSheet(f"color: {t['accent']}; font-weight: bold; font-size: 13px; border: none;")
        ctrl_layout.addWidget(title)

        param_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTextEdit {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 3px 6px; font-size: 11px;
            }}
            QLabel {{ color: {t['text_secondary']}; font-size: 11px; border: none; }}
        """

        # Reference audio
        ref_row = QHBoxLayout()
        self._clone_ref_label = QLabel("No reference audio")
        self._clone_ref_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px; border: none;")
        self._clone_ref_btn = QPushButton("Browse")
        self._clone_ref_btn.setStyleSheet(f"""
            QPushButton {{ background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 4px 10px; font-size: 10px; }}
        """)
        self._clone_ref_btn.clicked.connect(self._on_clone_browse_ref)
        ref_lbl = QLabel("Ref:")
        ref_lbl.setFixedWidth(30)
        ref_lbl.setStyleSheet(param_style)
        ref_row.addWidget(ref_lbl)
        ref_row.addWidget(self._clone_ref_label, 1)
        ref_row.addWidget(self._clone_ref_btn)
        ctrl_layout.addLayout(ref_row)

        # Reference transcript
        self._clone_ref_text = QLineEdit()
        self._clone_ref_text.setPlaceholderText("Transcript of reference audio...")
        self._clone_ref_text.setStyleSheet(param_style)
        ctrl_layout.addWidget(self._clone_ref_text)

        # Text to generate
        self._clone_text = QTextEdit()
        self._clone_text.setPlaceholderText("Text to speak in cloned voice...")
        self._clone_text.setMaximumHeight(80)
        self._clone_text.setStyleSheet(param_style)
        ctrl_layout.addWidget(self._clone_text)

        # Language
        lang_row = QHBoxLayout()
        ll = QLabel("Lang:")
        ll.setFixedWidth(36)
        ll.setStyleSheet(param_style)
        self._clone_lang = QComboBox()
        self._clone_lang.addItems(["English", "Chinese", "Japanese"])
        self._clone_lang.setStyleSheet(param_style)
        lang_row.addWidget(ll)
        lang_row.addWidget(self._clone_lang)
        ctrl_layout.addLayout(lang_row)

        # Speed + Temperature
        st_row = QHBoxLayout()
        sl = QLabel("Speed:")
        sl.setFixedWidth(42)
        sl.setStyleSheet(param_style)
        self._clone_speed = QDoubleSpinBox()
        self._clone_speed.setRange(0.5, 2.0)
        self._clone_speed.setValue(1.0)
        self._clone_speed.setSingleStep(0.1)
        self._clone_speed.setStyleSheet(param_style)

        tl = QLabel("Temp:")
        tl.setFixedWidth(36)
        tl.setStyleSheet(param_style)
        self._clone_temp = QDoubleSpinBox()
        self._clone_temp.setRange(0.1, 1.5)
        self._clone_temp.setValue(0.7)
        self._clone_temp.setSingleStep(0.05)
        self._clone_temp.setStyleSheet(param_style)

        st_row.addWidget(sl)
        st_row.addWidget(self._clone_speed)
        st_row.addWidget(tl)
        st_row.addWidget(self._clone_temp)
        ctrl_layout.addLayout(st_row)

        # Clone button
        self._clone_gen_btn = QPushButton("Clone Voice")
        self._clone_gen_btn.setFixedHeight(34)
        self._clone_gen_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #cba6f7, stop:1 #f5c2e7);
                color: white; border: none; border-radius: 6px;
                font-weight: bold; font-size: 12px;
            }}
        """)
        self._clone_gen_btn.clicked.connect(self._on_clone_generate)
        ctrl_layout.addWidget(self._clone_gen_btn)

        ctrl_frame.setFixedWidth(320)
        layout.addWidget(ctrl_frame)

        # Output waveform
        self._clone_waveform = WaveformWidget()
        layout.addWidget(self._clone_waveform, 1)

        return widget

    def _build_stems_tab(self) -> QWidget:
        """Demucs stem separation tab."""
        t = ThemeEngine.get_colors()
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Top: Input controls
        top = QHBoxLayout()

        self._stem_input_label = QLabel("Drop or browse an audio file to separate stems")
        self._stem_input_label.setStyleSheet(f"""
            color: {t['text_secondary']}; font-size: 12px;
            padding: 12px 20px;
            background: {t['surface']};
            border: 2px dashed {t['border']};
            border-radius: 8px;
        """)

        self._stem_browse_btn = QPushButton("Browse Audio")
        self._stem_browse_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']}; color: white; border: none;
                border-radius: 5px; padding: 8px 16px;
                font-weight: bold; font-size: 11px;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
        """)
        self._stem_browse_btn.clicked.connect(self._on_stems_browse)

        self._stem_model = QComboBox()
        self._stem_model.addItems(["htdemucs", "htdemucs_ft", "htdemucs_6s"])
        self._stem_model.setStyleSheet(f"""
            QComboBox {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 6px 10px; font-size: 11px;
            }}
        """)

        self._stem_separate_btn = QPushButton("Separate Stems")
        self._stem_separate_btn.setStyleSheet(f"""
            QPushButton {{
                background: #238636; color: white; border: none;
                border-radius: 5px; padding: 8px 16px;
                font-weight: bold; font-size: 11px;
            }}
            QPushButton:hover {{ background: #2ea043; }}
            QPushButton:disabled {{ background: {t['border']}; color: #555; }}
        """)
        self._stem_separate_btn.setEnabled(False)
        self._stem_separate_btn.clicked.connect(self._on_separate)

        top.addWidget(self._stem_input_label, 1)
        top.addWidget(self._stem_browse_btn)
        top.addWidget(self._stem_model)
        top.addWidget(self._stem_separate_btn)
        layout.addLayout(top)

        # Stem mixer
        self._stem_mixer = StemMixer()
        self._stem_mixer.remix_requested.connect(self._on_remix_export)
        layout.addWidget(self._stem_mixer, 1)

        return widget

    # ── Event Handlers ─────────────────────────────────────────────────────────

    def _on_sing_generate(self):
        self._status.setText("DiffSinger synthesis requires a loaded model (see Model Hub)")

    def _on_rvc_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio", "", "Audio (*.wav *.flac *.mp3 *.ogg)"
        )
        if path:
            self._rvc_input_label.setText(os.path.basename(path))
            self._rvc_input_label.setProperty("path", path)
            # Load waveform preview
            try:
                self._load_waveform_preview(self._rvc_original_wf, path)
            except Exception:
                pass

    def _on_rvc_convert(self):
        self._status.setText("RVC conversion requires a loaded voice model (see Model Hub)")

    def _on_clone_browse_ref(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", "", "Audio (*.wav *.flac *.mp3)"
        )
        if path:
            self._clone_ref_label.setText(os.path.basename(path))
            self._clone_ref_label.setProperty("path", path)

    def _on_clone_generate(self):
        self._status.setText("GPT-SoVITS cloning requires a loaded model (see Model Hub)")

    def _on_stems_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio to Separate", "", "Audio (*.wav *.flac *.mp3 *.ogg)"
        )
        if path:
            self._stem_input_label.setText(os.path.basename(path))
            self._stem_input_label.setProperty("path", path)
            self._stem_separate_btn.setEnabled(True)

    def _on_separate(self):
        path = self._stem_input_label.property("path")
        if not path:
            return
        self._status.setText("Starting stem separation (installing dependencies if needed)...")

    def _on_remix_export(self):
        audio = self._stem_mixer.get_remix_audio()
        if audio is None:
            self._status.setText("No stems to remix")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Remix", "remix.wav", "WAV (*.wav)"
        )
        if path:
            import wave
            int_audio = (audio * 32767).clip(-32768, 32767).astype(__import__("numpy").int16)
            with wave.open(path, "w") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(int_audio.tobytes())
            self._current_audio_path = path
            self._status.setText(f"Remix exported: {path}")
            self._enable_routing()

    def _on_export(self):
        if self._current_audio_path:
            self._status.setText(f"Audio available at: {self._current_audio_path}")

    def _on_send_to_forge(self):
        if self._current_audio_path:
            self.send_to_forge.emit(self._current_audio_path)

    def _on_send_to_mixer(self):
        if self._current_audio_path:
            self.send_to_mixer.emit(self._current_audio_path)

    def _enable_routing(self):
        self._to_forge_btn.setEnabled(True)
        self._to_mixer_btn.setEnabled(True)
        self._export_btn.setEnabled(True)

    def _load_waveform_preview(self, wf_widget: WaveformWidget, path: str):
        """Load audio file into waveform widget for preview."""
        try:
            import librosa
            audio, sr = librosa.load(path, sr=44100, mono=True, duration=120)
            wf_widget.load_audio(audio, sr)
        except ImportError:
            pass

    # ── External API ───────────────────────────────────────────────────────────

    def set_audio(self, audio_path: str):
        """Receive audio from another module (Song Forge, MIDI Studio)."""
        self._stem_input_label.setText(os.path.basename(audio_path))
        self._stem_input_label.setProperty("path", audio_path)
        self._stem_separate_btn.setEnabled(True)
        self._tabs.setCurrentIndex(3)  # Switch to stems tab
        self._status.setText(f"Audio received: {os.path.basename(audio_path)}")

    def refresh_voice_bank(self):
        """Refresh voice model selectors from VoiceBank."""
        bank = VoiceBank()

        # DiffSinger voices
        self._sing_voice.clear()
        ds_voices = bank.list_by_engine("diffsinger")
        if ds_voices:
            for v in ds_voices:
                self._sing_voice.addItem(v.name, v.id)
        else:
            self._sing_voice.addItem("(No DiffSinger voices)")

        # RVC voices
        self._rvc_voice.clear()
        rvc_voices = bank.list_by_engine("rvc")
        if rvc_voices:
            for v in rvc_voices:
                self._rvc_voice.addItem(v.name, v.id)
        else:
            self._rvc_voice.addItem("(No RVC models)")
