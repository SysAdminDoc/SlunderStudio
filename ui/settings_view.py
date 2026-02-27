"""
Slunder Studio v0.0.2 — Settings View
Two-tier settings: Simple Mode (essentials) and Advanced Mode (full controls).
All changes apply immediately with toast feedback.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QComboBox, QLineEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider,
    QFileDialog, QGroupBox, QFormLayout, QTabWidget,
)
from PySide6.QtCore import Qt

from ui.theme import Palette
from core.settings import Settings, APP_VERSION


class SettingRow(QHBoxLayout):
    """A labeled setting control with optional description."""

    def __init__(self, label: str, widget: QWidget, description: str = ""):
        super().__init__()
        self.setSpacing(12)

        label_container = QVBoxLayout()
        label_container.setSpacing(2)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {Palette.TEXT};")
        label_container.addWidget(lbl)

        if description:
            desc = QLabel(description)
            desc.setStyleSheet(f"font-size: 11px; color: {Palette.OVERLAY0};")
            desc.setWordWrap(True)
            label_container.addWidget(desc)

        self.addLayout(label_container, 1)
        self.addWidget(widget)


class SettingsView(QWidget):
    """Settings page with Simple and Advanced tabs."""

    def __init__(self, toast_mgr=None, parent=None):
        super().__init__(parent)
        self.toast_mgr = toast_mgr
        self._settings = Settings()
        self._build_ui()
        self._load_values()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Title
        header = QHBoxLayout()
        title = QLabel("Settings")
        title.setObjectName("heading")
        header.addWidget(title)
        header.addStretch()

        version_label = QLabel(f"Slunder Studio v{APP_VERSION}")
        version_label.setObjectName("caption")
        header.addWidget(version_label)
        layout.addLayout(header)

        # Tab widget for Simple / Advanced
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_simple_tab(), "Simple")
        self._tabs.addTab(self._build_advanced_tab(), "Advanced")
        layout.addWidget(self._tabs, 1)

        # Bottom bar
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        reset_btn = QPushButton("Reset All to Defaults")
        reset_btn.setObjectName("dangerBtn")
        reset_btn.setFixedHeight(36)
        reset_btn.clicked.connect(self._reset_all)
        bottom.addWidget(reset_btn)

        bottom.addStretch()

        open_dir_btn = QPushButton("Open Config Folder")
        open_dir_btn.setObjectName("secondaryBtn")
        open_dir_btn.setFixedHeight(36)
        open_dir_btn.clicked.connect(self._open_config_dir)
        bottom.addWidget(open_dir_btn)

        layout.addLayout(bottom)

    def _build_simple_tab(self) -> QWidget:
        """Simple settings — one page of essentials."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(20)

        # ── Output ──
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("Select output directory...")
        self._output_dir.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("secondaryBtn")
        browse_btn.setFixedWidth(80)
        browse_btn.setFixedHeight(34)
        browse_btn.clicked.connect(self._browse_output_dir)

        dir_row = QHBoxLayout()
        dir_row.addWidget(self._output_dir, 1)
        dir_row.addWidget(browse_btn)
        output_layout.addLayout(SettingRow("Output Directory", QWidget()))
        output_layout.addLayout(dir_row)

        self._format_combo = QComboBox()
        self._format_combo.addItems(["WAV", "FLAC", "MP3"])
        self._format_combo.setFixedWidth(120)
        self._format_combo.currentTextChanged.connect(
            lambda v: self._save("general.audio_format", v.lower()))
        output_layout.addLayout(SettingRow("Default Audio Format", self._format_combo))

        self._sample_rate_combo = QComboBox()
        self._sample_rate_combo.addItems(["44100", "48000"])
        self._sample_rate_combo.setFixedWidth(120)
        self._sample_rate_combo.currentTextChanged.connect(
            lambda v: self._save("general.sample_rate", int(v)) if v else None)
        output_layout.addLayout(SettingRow("Sample Rate", self._sample_rate_combo, "44.1kHz for CD, 48kHz for modern production"))

        layout.addWidget(output_group)

        # ── GPU & Models ──
        gpu_group = QGroupBox("GPU & Models")
        gpu_layout = QVBoxLayout(gpu_group)

        self._gpu_device = QSpinBox()
        self._gpu_device.setRange(0, 7)
        self._gpu_device.setFixedWidth(80)
        self._gpu_device.valueChanged.connect(
            lambda v: self._save("general.gpu_device", v))
        gpu_layout.addLayout(SettingRow("GPU Device Index", self._gpu_device, "Usually 0 for single-GPU systems"))

        self._offline_mode = QCheckBox("Offline Mode")
        self._offline_mode.toggled.connect(
            lambda v: self._save("model_hub.offline_mode", v))
        gpu_layout.addLayout(SettingRow("Disable internet for model hub", self._offline_mode))

        self._hf_token = QLineEdit()
        self._hf_token.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxx")
        self._hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self._hf_token.setFixedWidth(280)
        self._hf_token.editingFinished.connect(
            lambda: self._save("model_hub.hf_token", self._hf_token.text().strip()))
        gpu_layout.addLayout(SettingRow(
            "HuggingFace Token",
            self._hf_token,
            "Required for gated models (Stable Audio Open). Get yours at huggingface.co/settings/tokens"
        ))

        layout.addWidget(gpu_group)

        # ── Appearance ──
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout(appearance_group)

        self._experience_combo = QComboBox()
        self._experience_combo.addItems(["Beginner", "Intermediate", "Advanced"])
        self._experience_combo.setFixedWidth(160)
        self._experience_combo.currentTextChanged.connect(
            lambda v: self._save("general.experience_level", v.lower()))
        appearance_layout.addLayout(SettingRow("Experience Level", self._experience_combo, "Controls default UI complexity across all modules"))

        layout.addWidget(appearance_group)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_advanced_tab(self) -> QWidget:
        """Advanced settings — full parameter controls per module."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(20)

        # ── Lyrics ──
        lyrics_group = QGroupBox("Lyrics Engine")
        lyrics_layout = QVBoxLayout(lyrics_group)

        self._lyrics_model = QComboBox()
        self._lyrics_model.addItem("LLaMA 3.1 8B (Recommended)", "llama-3.1-8b-q4")
        self._lyrics_model.addItem("LLaMA 3.2 3B (Fast)", "llama-3.2-3b-q4")
        self._lyrics_model.addItem("Qwen 2.5 14B (Premium)", "qwen-2.5-14b-q4")
        self._lyrics_model.setFixedWidth(240)
        self._lyrics_model.currentIndexChanged.connect(
            lambda: self._save("lyrics.model_id", self._lyrics_model.currentData()))
        lyrics_layout.addLayout(SettingRow("Lyrics Model", self._lyrics_model))

        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.1, 2.0)
        self._temperature.setSingleStep(0.05)
        self._temperature.setFixedWidth(100)
        self._temperature.valueChanged.connect(
            lambda v: self._save("lyrics.temperature", v))
        lyrics_layout.addLayout(SettingRow("Temperature", self._temperature, "Higher = more creative, lower = more predictable"))

        self._top_p = QDoubleSpinBox()
        self._top_p.setRange(0.1, 1.0)
        self._top_p.setSingleStep(0.05)
        self._top_p.setFixedWidth(100)
        self._top_p.valueChanged.connect(
            lambda v: self._save("lyrics.top_p", v))
        lyrics_layout.addLayout(SettingRow("Top P", self._top_p))

        self._max_tokens = QSpinBox()
        self._max_tokens.setRange(256, 8192)
        self._max_tokens.setSingleStep(256)
        self._max_tokens.setFixedWidth(120)
        self._max_tokens.valueChanged.connect(
            lambda v: self._save("lyrics.max_tokens", v))
        lyrics_layout.addLayout(SettingRow("Max Tokens", self._max_tokens))

        layout.addWidget(lyrics_group)

        # ── Song Forge ──
        forge_group = QGroupBox("Song Forge")
        forge_layout = QVBoxLayout(forge_group)

        self._cfg_scale = QDoubleSpinBox()
        self._cfg_scale.setRange(1.0, 15.0)
        self._cfg_scale.setSingleStep(0.5)
        self._cfg_scale.setFixedWidth(100)
        self._cfg_scale.valueChanged.connect(
            lambda v: self._save("song_forge.cfg_scale", v))
        forge_layout.addLayout(SettingRow("CFG Scale", self._cfg_scale, "Higher = stronger prompt adherence"))

        self._inference_steps = QSpinBox()
        self._inference_steps.setRange(10, 200)
        self._inference_steps.setSingleStep(5)
        self._inference_steps.setFixedWidth(100)
        self._inference_steps.valueChanged.connect(
            lambda v: self._save("song_forge.inference_steps", v))
        forge_layout.addLayout(SettingRow("Inference Steps", self._inference_steps, "More steps = higher quality, slower"))

        self._batch_count = QSpinBox()
        self._batch_count.setRange(1, 16)
        self._batch_count.setFixedWidth(80)
        self._batch_count.valueChanged.connect(
            lambda v: self._save("song_forge.batch_count", v))
        forge_layout.addLayout(SettingRow("Batch Count", self._batch_count, "Number of variations to generate"))

        self._default_duration = QSpinBox()
        self._default_duration.setRange(10, 600)
        self._default_duration.setSuffix(" sec")
        self._default_duration.setFixedWidth(120)
        self._default_duration.valueChanged.connect(
            lambda v: self._save("song_forge.default_duration", v))
        forge_layout.addLayout(SettingRow("Default Duration", self._default_duration))

        layout.addWidget(forge_group)

        # ── MIDI Studio ──
        midi_group = QGroupBox("MIDI Studio")
        midi_layout = QVBoxLayout(midi_group)

        self._default_bpm = QSpinBox()
        self._default_bpm.setRange(40, 300)
        self._default_bpm.setSuffix(" BPM")
        self._default_bpm.setFixedWidth(120)
        self._default_bpm.valueChanged.connect(
            lambda v: self._save("midi_studio.default_bpm", v))
        midi_layout.addLayout(SettingRow("Default BPM", self._default_bpm))

        layout.addWidget(midi_group)

        # ── Production ──
        prod_group = QGroupBox("Production / Mastering")
        prod_layout = QVBoxLayout(prod_group)

        self._mastering_target = QComboBox()
        self._mastering_target.addItem("Spotify (-14 LUFS)", "spotify")
        self._mastering_target.addItem("YouTube (-13 LUFS)", "youtube")
        self._mastering_target.addItem("Apple Music (-16 LUFS)", "apple")
        self._mastering_target.addItem("CD (-9 LUFS)", "cd")
        self._mastering_target.addItem("Broadcast (-24 LUFS)", "broadcast")
        self._mastering_target.setFixedWidth(220)
        self._mastering_target.currentIndexChanged.connect(
            lambda: self._save("production.mastering_target", self._mastering_target.currentData()))
        prod_layout.addLayout(SettingRow("Mastering Target", self._mastering_target))

        self._auto_eq = QCheckBox("Auto EQ")
        self._auto_eq.toggled.connect(
            lambda v: self._save("production.mastering_auto_eq", v))
        prod_layout.addLayout(SettingRow("Apply spectral correction during mastering", self._auto_eq))

        self._auto_compress = QCheckBox("Auto Compression")
        self._auto_compress.toggled.connect(
            lambda v: self._save("production.mastering_auto_compress", v))
        prod_layout.addLayout(SettingRow("Apply bus compression during mastering", self._auto_compress))

        layout.addWidget(prod_group)

        # ── Cache ──
        cache_group = QGroupBox("Cache & Storage")
        cache_layout = QVBoxLayout(cache_group)

        self._max_cache = QDoubleSpinBox()
        self._max_cache.setRange(1.0, 500.0)
        self._max_cache.setSuffix(" GB")
        self._max_cache.setSingleStep(5.0)
        self._max_cache.setFixedWidth(120)
        self._max_cache.valueChanged.connect(
            lambda v: self._save("general.max_cache_gb", v))
        cache_layout.addLayout(SettingRow("Max Cache Size", self._max_cache, "Auto-cleanup old generations beyond this limit"))

        self._autosave_interval = QSpinBox()
        self._autosave_interval.setRange(10, 600)
        self._autosave_interval.setSuffix(" sec")
        self._autosave_interval.setFixedWidth(120)
        self._autosave_interval.valueChanged.connect(
            lambda v: self._save("general.auto_save_interval", v))
        cache_layout.addLayout(SettingRow("Auto-Save Interval", self._autosave_interval))

        layout.addWidget(cache_group)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _load_values(self):
        """Load current settings into UI controls."""
        s = self._settings

        # Block signals on all save-connected widgets to prevent
        # cascading saves during programmatic value changes
        _widgets = [
            self._format_combo, self._sample_rate_combo, self._gpu_device,
            self._offline_mode, self._hf_token, self._experience_combo,
            self._lyrics_model, self._temperature, self._top_p,
            self._max_tokens, self._cfg_scale, self._inference_steps,
            self._batch_count, self._default_duration, self._default_bpm,
            self._mastering_target, self._auto_eq, self._auto_compress,
            self._max_cache, self._autosave_interval,
        ]
        for w in _widgets:
            w.blockSignals(True)

        try:
            # Simple tab
            self._output_dir.setText(s.get("general.output_dir", ""))
            fmt = s.get("general.audio_format", "wav").upper()
            idx = self._format_combo.findText(fmt)
            if idx >= 0:
                self._format_combo.setCurrentIndex(idx)

            sr = str(s.get("general.sample_rate", 48000))
            idx = self._sample_rate_combo.findText(sr)
            if idx >= 0:
                self._sample_rate_combo.setCurrentIndex(idx)

            self._gpu_device.setValue(s.get("general.gpu_device", 0))
            self._offline_mode.setChecked(s.get("model_hub.offline_mode", False))
            self._hf_token.setText(s.get("model_hub.hf_token", ""))

            exp = s.get("general.experience_level", "beginner").capitalize()
            idx = self._experience_combo.findText(exp)
            if idx >= 0:
                self._experience_combo.setCurrentIndex(idx)

            # Advanced tab
            model_id = s.get("lyrics.model_id", "llama-3.1-8b-q4")
            for i in range(self._lyrics_model.count()):
                if self._lyrics_model.itemData(i) == model_id:
                    self._lyrics_model.setCurrentIndex(i)
                    break

            self._temperature.setValue(s.get("lyrics.temperature", 0.8))
            self._top_p.setValue(s.get("lyrics.top_p", 0.92))
            self._max_tokens.setValue(s.get("lyrics.max_tokens", 2048))
            self._cfg_scale.setValue(s.get("song_forge.cfg_scale", 7.0))
            self._inference_steps.setValue(s.get("song_forge.inference_steps", 50))
            self._batch_count.setValue(s.get("song_forge.batch_count", 4))
            self._default_duration.setValue(s.get("song_forge.default_duration", 180))
            self._default_bpm.setValue(s.get("midi_studio.default_bpm", 120))

            target = s.get("production.mastering_target", "spotify")
            for i in range(self._mastering_target.count()):
                if self._mastering_target.itemData(i) == target:
                    self._mastering_target.setCurrentIndex(i)
                    break

            self._auto_eq.setChecked(s.get("production.mastering_auto_eq", True))
            self._auto_compress.setChecked(s.get("production.mastering_auto_compress", True))
            self._max_cache.setValue(s.get("general.max_cache_gb", 20.0))
            self._autosave_interval.setValue(s.get("general.auto_save_interval", 60))

        finally:
            for w in _widgets:
                w.blockSignals(False)

    def _save(self, key: str, value):
        """Save a setting and show toast."""
        self._settings.set(key, value)
        # Toast for important changes only
        if self.toast_mgr and key in ("general.audio_format", "general.sample_rate", "lyrics.model_id"):
            self.toast_mgr.success(f"Setting updated")

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._output_dir.setText(path)
            self._save("general.output_dir", path)

    def _reset_all(self):
        self._settings.reset_all()
        self._load_values()
        if self.toast_mgr:
            self.toast_mgr.warning("All settings reset to defaults")

    def _open_config_dir(self):
        import subprocess, sys
        config_dir = str(self._settings._config_path.parent)
        if sys.platform == "win32":
            subprocess.Popen(f'explorer "{config_dir}"')
        elif sys.platform == "darwin":
            subprocess.Popen(["open", config_dir])
        else:
            subprocess.Popen(["xdg-open", config_dir])
