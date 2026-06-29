"""
Slunder Studio v0.1.16 — Vocal Suite View
Main Vocal Suite page combining singing synthesis (DiffSinger),
voice conversion (RVC), voice cloning (GPT-SoVITS), stem separation (Demucs),
and stem remix/export.
"""
import os
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QTabWidget,
    QFrame, QLineEdit, QSlider, QGroupBox, QStackedWidget, QCheckBox,
)
from PySide6.QtCore import Qt, Signal

from ui.theme import ThemeEngine
from ui.waveform_widget import WaveformWidget
from ui.stem_mixer import StemMixer
from ui.accessibility import install_accessibility
from core.voice_bank import VOICE_OPERATION_CLONE, VOICE_OPERATION_CONVERSION, VoiceBank, VoiceProfile
from core.workers import InferenceWorker


class VocalSuiteView(QWidget):
    """Main Vocal Suite page with tabbed sub-views."""

    send_to_forge = Signal(str)    # audio path -> Song Forge
    send_to_mixer = Signal(str)    # audio path -> Mixer (Phase 6)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_audio_path: Optional[str] = None
        self._clone_quality_report = None
        self._clone_worker: Optional[InferenceWorker] = None

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
        self.refresh_voice_bank()
        install_accessibility(
            self,
            "Vocal Suite",
            named_controls=[
                (self._tabs, "Vocal Suite mode", "Switches between singing synthesis, voice conversion, voice cloning, and stem separation."),
                (self._sing_lyrics, "Singing lyrics", "Lyrics to synthesize with DiffSinger."),
                (self._sing_voice, "DiffSinger voice", "Selects the singing voice model."),
                (self._sing_tempo, "Singing tempo", "Sets the vocal synthesis tempo in beats per minute."),
                (self._sing_key, "Singing key", "Sets the starting singing pitch."),
                (self._sing_breathiness, "Singing breathiness", "Adjusts breathiness expression."),
                (self._sing_tension, "Singing tension", "Adjusts vocal tension expression."),
                (self._sing_vibrato, "Singing vibrato", "Adjusts vibrato expression."),
                (self._sing_gender, "Singing gender", "Adjusts vocal formant character."),
                (self._sing_gen_btn, "Synthesize vocals", "Starts DiffSinger vocal synthesis."),
                (self._rvc_browse_btn, "Browse RVC input", "Selects input audio for voice conversion."),
                (self._rvc_voice, "RVC voice", "Selects the target RVC voice model."),
                (self._rvc_pitch, "RVC pitch shift", "Adjusts pitch shift in semitones."),
                (self._rvc_f0, "RVC pitch detector", "Selects the F0 pitch extraction method."),
                (self._rvc_index, "RVC index strength", "Adjusts retrieval index blend strength."),
                (self._rvc_convert_btn, "Convert voice", "Starts RVC voice conversion."),
                (self._clone_voice, "Voice clone profile", "Selects a saved GPT-SoVITS voice profile."),
                (self._clone_profile_name, "Voice profile name", "Names a new voice profile."),
                (self._clone_owner_name, "Voice owner", "Records the voice owner or rights holder for consent provenance."),
                (self._clone_consent_source, "Voice consent source", "Records how consent or use rights were obtained."),
                (self._clone_use_scope, "Voice permitted use", "Records the operations allowed by the voice consent."),
                (self._clone_consent_confirm, "Voice consent confirmation", "Confirms ownership or permission before saving a voice profile."),
                (self._clone_ref_btn, "Browse clone reference", "Selects reference audio for voice cloning."),
                (self._clone_ref_text, "Reference transcript", "Transcript for the reference voice sample."),
                (self._clone_save_profile_btn, "Save voice profile", "Saves a validated GPT-SoVITS reference profile."),
                (self._clone_text, "Clone text", "Text to speak in the cloned voice."),
                (self._clone_lang, "Clone language", "Selects the GPT-SoVITS language."),
                (self._clone_speed, "Clone speed", "Adjusts cloned speech speed."),
                (self._clone_temp, "Clone temperature", "Adjusts generation variation."),
                (self._clone_gen_btn, "Clone voice", "Starts GPT-SoVITS voice cloning."),
                (self._stem_browse_btn, "Browse stem input", "Selects audio for stem separation."),
                (self._stem_model, "Stem separation model", "Selects the Demucs model."),
                (self._stem_separate_btn, "Separate stems", "Starts stem separation."),
                (self._to_forge_btn, "Send vocals to Song Forge", "Routes current vocal output to Song Forge."),
                (self._to_mixer_btn, "Send vocals to Mixer", "Routes current vocal output to Mixer."),
                (self._export_btn, "Export vocal audio", "Shows the current vocal output path."),
            ],
            tab_order=[
                self._tabs,
                self._sing_lyrics,
                self._sing_voice,
                self._sing_tempo,
                self._sing_key,
                self._sing_breathiness,
                self._sing_tension,
                self._sing_vibrato,
                self._sing_gender,
                self._sing_gen_btn,
                self._rvc_browse_btn,
                self._rvc_voice,
                self._rvc_pitch,
                self._rvc_f0,
                self._rvc_index,
                self._rvc_convert_btn,
                self._clone_voice,
                self._clone_profile_name,
                self._clone_owner_name,
                self._clone_consent_source,
                self._clone_use_scope,
                self._clone_consent_confirm,
                self._clone_ref_btn,
                self._clone_ref_text,
                self._clone_save_profile_btn,
                self._clone_text,
                self._clone_lang,
                self._clone_speed,
                self._clone_temp,
                self._clone_gen_btn,
                self._stem_browse_btn,
                self._stem_model,
                self._stem_separate_btn,
                self._to_forge_btn,
                self._to_mixer_btn,
                self._export_btn,
            ],
        )

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
        self._rvc_voice.currentIndexChanged.connect(self._on_rvc_profile_changed)
        voice_row.addWidget(vl)
        voice_row.addWidget(self._rvc_voice)
        ctrl_layout.addLayout(voice_row)

        self._rvc_consent_label = QLabel("Consent guardrails: select a voice profile.")
        self._rvc_consent_label.setWordWrap(True)
        self._rvc_consent_label.setStyleSheet(
            f"color: {t['text_secondary']}; background: {t['background']}; "
            f"border: 1px solid {t['border']}; border-radius: 4px; padding: 6px; font-size: 10px;"
        )
        ctrl_layout.addWidget(self._rvc_consent_label)

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

        # Onboarded voice profile
        profile_row = QHBoxLayout()
        profile_label = QLabel("Profile:")
        profile_label.setFixedWidth(50)
        profile_label.setStyleSheet(param_style)
        self._clone_voice = QComboBox()
        self._clone_voice.setStyleSheet(param_style)
        self._clone_voice.currentIndexChanged.connect(self._on_clone_profile_changed)
        profile_row.addWidget(profile_label)
        profile_row.addWidget(self._clone_voice)
        ctrl_layout.addLayout(profile_row)

        self._clone_profile_name = QLineEdit()
        self._clone_profile_name.setPlaceholderText("New voice profile name...")
        self._clone_profile_name.setStyleSheet(param_style)
        self._clone_profile_name.textChanged.connect(self._update_clone_profile_ready)
        ctrl_layout.addWidget(self._clone_profile_name)

        self._clone_owner_name = QLineEdit()
        self._clone_owner_name.setPlaceholderText("Voice owner / rights holder...")
        self._clone_owner_name.setStyleSheet(param_style)
        self._clone_owner_name.textChanged.connect(self._update_clone_profile_ready)
        ctrl_layout.addWidget(self._clone_owner_name)

        consent_row = QHBoxLayout()
        self._clone_consent_source = QComboBox()
        self._clone_consent_source.addItems([
            "Self-recorded / my voice",
            "Licensed dataset or model",
            "Third-party permission",
            "Public model with license",
        ])
        self._clone_consent_source.setStyleSheet(param_style)
        self._clone_consent_source.currentIndexChanged.connect(self._update_clone_profile_ready)
        consent_row.addWidget(self._clone_consent_source)

        self._clone_use_scope = QComboBox()
        self._clone_use_scope.addItems([
            "Clone + conversion",
            "Clone only",
            "Research/demo only",
        ])
        self._clone_use_scope.setStyleSheet(param_style)
        self._clone_use_scope.currentIndexChanged.connect(self._update_clone_profile_ready)
        consent_row.addWidget(self._clone_use_scope)
        ctrl_layout.addLayout(consent_row)

        self._clone_consent_confirm = QCheckBox("Consent confirmed")
        self._clone_consent_confirm.setStyleSheet(
            f"QCheckBox {{ color: {t['text']}; font-size: 11px; border: none; }}"
        )
        self._clone_consent_confirm.stateChanged.connect(self._update_clone_profile_ready)
        ctrl_layout.addWidget(self._clone_consent_confirm)

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
        self._clone_ref_text.textChanged.connect(self._update_clone_profile_ready)
        ctrl_layout.addWidget(self._clone_ref_text)

        self._clone_quality_label = QLabel("Reference guardrails: select a clean 10-30s voice sample.")
        self._clone_quality_label.setWordWrap(True)
        self._clone_quality_label.setStyleSheet(
            f"color: {t['text_secondary']}; background: {t['background']}; "
            f"border: 1px solid {t['border']}; border-radius: 4px; padding: 6px; font-size: 10px;"
        )
        ctrl_layout.addWidget(self._clone_quality_label)

        self._clone_consent_label = QLabel("Consent guardrails: owner, source, permitted use, and confirmation are required.")
        self._clone_consent_label.setWordWrap(True)
        self._clone_consent_label.setStyleSheet(
            f"color: {t['text_secondary']}; background: {t['background']}; "
            f"border: 1px solid {t['border']}; border-radius: 4px; padding: 6px; font-size: 10px;"
        )
        ctrl_layout.addWidget(self._clone_consent_label)

        self._clone_save_profile_btn = QPushButton("Save Voice Profile")
        self._clone_save_profile_btn.setFixedHeight(28)
        self._clone_save_profile_btn.setEnabled(False)
        self._clone_save_profile_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['surface_hover']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                font-weight: bold; font-size: 11px;
            }}
            QPushButton:hover {{ background: {t['accent']}; color: white; }}
            QPushButton:disabled {{ color: {t['muted']}; }}
        """)
        self._clone_save_profile_btn.clicked.connect(self._on_clone_save_profile)
        ctrl_layout.addWidget(self._clone_save_profile_btn)

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
        profile_id = self._rvc_voice.currentData()
        profile = VoiceBank().get(profile_id) if profile_id else None
        if not profile:
            self._status.setText("Select a consent-ready RVC voice profile before conversion")
            return
        issues = VoiceBank().validate_profile(profile, VOICE_OPERATION_CONVERSION)
        if issues:
            self._status.setText("RVC profile blocked: " + "; ".join(issues[:2]))
            self._rvc_consent_label.setText(self._format_profile_consent(profile, VOICE_OPERATION_CONVERSION))
            return
        self._status.setText("RVC conversion requires a loaded voice model (see Model Hub)")

    def _on_rvc_profile_changed(self, _index: int):
        if not hasattr(self, "_rvc_consent_label"):
            return
        profile_id = self._rvc_voice.currentData()
        profile = VoiceBank().get(profile_id) if profile_id else None
        if not profile:
            self._rvc_consent_label.setText("Consent guardrails: select a voice profile.")
            return
        self._rvc_consent_label.setText(self._format_profile_consent(profile, VOICE_OPERATION_CONVERSION))

    def _on_clone_browse_ref(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", "", "Audio (*.wav *.flac *.mp3)"
        )
        if path:
            self._clone_ref_label.setText(os.path.basename(path))
            self._clone_ref_label.setProperty("path", path)
            if not self._clone_profile_name.text().strip():
                self._clone_profile_name.setText(os.path.splitext(os.path.basename(path))[0])
            try:
                self._load_waveform_preview(self._clone_waveform, path)
            except Exception:
                pass
            self._update_clone_reference_quality(path)

    def _on_clone_generate(self):
        text = self._clone_text.toPlainText().strip()
        if not text:
            self._status.setText("Enter text to synthesize before cloning")
            return

        profile_id = self._clone_voice.currentData()
        profile = VoiceBank().get(profile_id) if profile_id else None
        if not profile:
            self._status.setText("Save or select a validated GPT-SoVITS voice profile first")
            return

        issues = VoiceBank().validate_profile(profile, VOICE_OPERATION_CLONE)
        if issues:
            self._status.setText("Voice profile blocked: " + "; ".join(issues[:2]))
            self._clone_consent_label.setText(self._format_profile_consent(profile, VOICE_OPERATION_CLONE))
            return

        from engines.rvc_engine import VoiceCloneParams, assess_clone_reference, get_sovits

        quality = assess_clone_reference(profile.ref_audio_path)
        if not quality.can_onboard:
            self._status.setText("Reference failed guardrails: " + "; ".join(quality.issues[:2]))
            return

        engine = get_sovits()
        if not engine.is_loaded:
            self._status.setText("Load a GPT-SoVITS base model in Model Hub before cloning")
            return

        params = VoiceCloneParams(
            text=text,
            ref_audio_path=profile.ref_audio_path,
            ref_text=profile.ref_text,
            language=self._clone_language_code(),
            speed=self._clone_speed.value(),
            temperature=self._clone_temp.value(),
        )
        self._clone_gen_btn.setEnabled(False)
        self._status.setText("Starting GPT-SoVITS clone...")
        self._clone_worker = InferenceWorker(self._run_clone_generation, params, profile)
        self._clone_worker.step_info.connect(self._status.setText)
        self._clone_worker.finished.connect(self._on_clone_generated)
        self._clone_worker.error.connect(self._on_clone_error)
        self._clone_worker.start()

    def _on_clone_profile_changed(self, _index: int):
        if not hasattr(self, "_clone_voice"):
            return
        profile_id = self._clone_voice.currentData()
        profile = VoiceBank().get(profile_id) if profile_id else None
        if not profile:
            return
        self._clone_profile_name.setText(profile.name)
        self._clone_owner_name.setText(profile.owner_name)
        self._set_combo_text(self._clone_consent_source, profile.consent_source)
        self._set_combo_text(self._clone_use_scope, profile.consent_scope)
        self._clone_consent_confirm.setChecked(profile.consent_confirmed)
        self._set_clone_language(profile.language)
        self._clone_ref_text.setText(profile.ref_text)
        self._clone_ref_label.setText(os.path.basename(profile.ref_audio_path) or "No reference audio")
        self._clone_ref_label.setProperty("path", profile.ref_audio_path)
        if profile.ref_audio_path:
            self._update_clone_reference_quality(profile.ref_audio_path)
        self._clone_consent_label.setText(self._format_profile_consent(profile, VOICE_OPERATION_CLONE))

    def _update_clone_reference_quality(self, path: str):
        from engines.rvc_engine import assess_clone_reference

        report = assess_clone_reference(path)
        self._clone_quality_report = report
        self._clone_quality_label.setText(self._format_clone_quality(report))
        self._update_clone_profile_ready()

    def _format_clone_quality(self, report) -> str:
        status = report.status.upper()
        details = report.issues[:2] if report.issues else ["Ready for GPT-SoVITS onboarding."]
        return (
            f"{status} {report.score}/100 - {report.metrics_summary()}\n"
            + "\n".join(details)
        )

    def _update_clone_profile_ready(self):
        if not hasattr(self, "_clone_save_profile_btn"):
            return
        has_text = bool(self._clone_ref_text.text().strip())
        has_name = bool(self._clone_profile_name.text().strip())
        has_owner = bool(self._clone_owner_name.text().strip())
        has_consent = self._clone_consent_confirm.isChecked()
        can_save = bool(
            self._clone_quality_report
            and self._clone_quality_report.can_onboard
            and has_text
            and has_name
            and has_owner
            and has_consent
        )
        self._clone_save_profile_btn.setEnabled(can_save)
        if hasattr(self, "_clone_consent_label"):
            status = "ready" if can_save else "owner, source, use scope, and confirmation are required"
            self._clone_consent_label.setText(f"Consent guardrails: {status}.")

    def _on_clone_save_profile(self):
        path = self._clone_ref_label.property("path")
        if not path or not self._clone_quality_report:
            self._status.setText("Select reference audio before saving a voice profile")
            return
        if not self._clone_quality_report.can_onboard:
            self._status.setText("Reference audio must pass guardrails before onboarding")
            return

        name = self._clone_profile_name.text().strip()
        owner_name = self._clone_owner_name.text().strip()
        ref_text = self._clone_ref_text.text().strip()
        if not name or not ref_text or not owner_name:
            self._status.setText("Profile name, voice owner, and reference transcript are required")
            return
        if not self._clone_consent_confirm.isChecked():
            self._status.setText("Confirm voice ownership or permission before saving")
            return

        report = self._clone_quality_report
        profile = VoiceProfile(
            name=name,
            engine="gpt_sovits",
            ref_audio_path=path,
            ref_text=ref_text,
            owner_name=owner_name,
            consent_status="confirmed",
            consent_source=self._clone_consent_source.currentText(),
            consent_scope=self._clone_use_scope.currentText(),
            language=self._clone_language_code(),
            permitted_uses=self._clone_permitted_uses(),
            consent_note="Confirmed in Vocal Suite voice profile form.",
            source="reference audio",
            license="user-confirmed",
            tags=["onboarded", report.status, f"{report.duration:.0f}s"],
            notes=f"Reference quality {report.status} {report.score}/100; {report.metrics_summary()}",
        )
        bank = VoiceBank()
        bank.add(profile)
        self.refresh_voice_bank()
        idx = self._clone_voice.findData(profile.id)
        if idx >= 0:
            self._clone_voice.setCurrentIndex(idx)
        self._status.setText(f"GPT-SoVITS voice profile saved: {profile.name}")

    def _clone_language_code(self) -> str:
        return {"English": "en", "Chinese": "zh", "Japanese": "ja"}.get(
            self._clone_lang.currentText(), "en"
        )

    def _set_clone_language(self, language: str):
        target = {"en": "English", "zh": "Chinese", "ja": "Japanese"}.get(language, "")
        if target:
            self._set_combo_text(self._clone_lang, target)

    def _set_combo_text(self, combo: QComboBox, text: str):
        if not text:
            return
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _clone_permitted_uses(self) -> list[str]:
        scope = self._clone_use_scope.currentText()
        if scope == "Clone + conversion":
            return [VOICE_OPERATION_CLONE, VOICE_OPERATION_CONVERSION]
        if scope == "Clone only":
            return [VOICE_OPERATION_CLONE]
        return [VOICE_OPERATION_CLONE, "research"]

    def _format_profile_consent(self, profile: VoiceProfile, operation: str) -> str:
        issues = VoiceBank().validate_profile(profile, operation)
        if issues:
            return "Consent guardrails blocked: " + "; ".join(issues[:3])
        uses = ", ".join(profile.permitted_uses)
        source = profile.consent_source or profile.source
        return (
            f"Consent confirmed: {profile.owner_name}; {profile.language}; "
            f"{source}; uses: {uses}."
        )

    def _run_clone_generation(self, params, profile, progress_cb=None, step_cb=None, log_cb=None, cancel_event=None):
        from engines.rvc_engine import get_sovits

        def progress(fraction: float, message: str):
            if progress_cb:
                progress_cb(int(fraction * 100))
            if step_cb:
                step_cb(message)

        if cancel_event and cancel_event.is_set():
            return None
        engine = get_sovits()
        result = engine.clone(params, progress)
        if result.error:
            raise RuntimeError(result.error)
        path = engine.save_output(result, profile=profile)
        return {"result": result, "path": path}

    def _on_clone_generated(self, payload):
        self._clone_worker = None
        self._clone_gen_btn.setEnabled(True)
        if not payload or not payload.get("path"):
            self._status.setText("GPT-SoVITS clone finished without an output file")
            return
        result = payload["result"]
        path = payload["path"]
        if result.audio is not None:
            self._clone_waveform.load_audio(result.audio, result.sample_rate)
        self._current_audio_path = path
        self._status.setText(f"Cloned voice generated: {os.path.basename(path)} ({result.duration:.1f}s)")
        self._enable_routing()

    def _on_clone_error(self, error: str):
        self._clone_worker = None
        self._clone_gen_btn.setEnabled(True)
        self._status.setText(f"GPT-SoVITS clone failed: {error}")

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
            self._rvc_voice.setCurrentIndex(0)
            self._on_rvc_profile_changed(0)
        else:
            self._rvc_voice.addItem("(No RVC models)")
            if hasattr(self, "_rvc_consent_label"):
                self._rvc_consent_label.setText("Consent guardrails: select a voice profile.")

        # GPT-SoVITS clone profiles
        self._clone_voice.blockSignals(True)
        self._clone_voice.clear()
        clone_voices = bank.list_by_engine("gpt_sovits")
        if clone_voices:
            for v in clone_voices:
                self._clone_voice.addItem(v.name, v.id)
        else:
            self._clone_voice.addItem("(No GPT-SoVITS profiles)", "")
        self._clone_voice.blockSignals(False)
        if clone_voices:
            self._clone_voice.setCurrentIndex(0)
            self._on_clone_profile_changed(0)
        elif hasattr(self, "_clone_consent_label"):
            self._clone_consent_label.setText("Consent guardrails: save a consent-ready voice profile.")
