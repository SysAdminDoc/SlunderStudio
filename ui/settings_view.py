"""
Slunder Studio — Settings View
Two-tier settings: Simple Mode (essentials) and Advanced Mode (full controls).
All changes apply immediately with toast feedback.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QComboBox, QLineEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QListWidget,
    QFileDialog, QGroupBox, QFormLayout, QTabWidget,
)
from PySide6.QtCore import Qt

from ui.theme import Palette
from ui.accessibility import install_accessibility
from core.diagnostics import export_health_report
from core.i18n import (
    language_code_from_label,
    language_combo_items,
    language_label,
    set_locale,
    tr,
    ui_locale_options,
)
from core.mastering import LUFS_TARGETS
from core.credentials import CredentialError
from core.settings import Settings, APP_VERSION, SECRET_SETTING_KEYS
from core.audio_engine import (
    AudioEngine,
    enumerate_output_devices,
    format_output_device_identity,
)


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
        self._audio = AudioEngine()
        self._recovery = None
        self._build_ui()
        self._audio.output_device_status.connect(self._on_audio_device_status)
        self._load_values()
        self._install_accessibility()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # Version context
        header = QHBoxLayout()
        header.addStretch()

        version_label = QLabel(tr("settings.version_label", version=APP_VERSION))
        version_label.setObjectName("caption")
        header.addWidget(version_label)
        layout.addLayout(header)

        self._repair_label = QLabel("")
        self._repair_label.setWordWrap(True)
        self._repair_label.setStyleSheet(
            f"color: {Palette.YELLOW}; font-size: 12px; padding: 4px 0;"
        )
        layout.addWidget(self._repair_label)

        # Tab widget for Simple / Advanced
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_simple_tab(), tr("settings.tabs.simple"))
        self._tabs.addTab(self._build_advanced_tab(), tr("settings.tabs.advanced"))
        layout.addWidget(self._tabs, 1)

        # Bottom bar
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        self._reset_btn = QPushButton(tr("settings.actions.reset_defaults"))
        self._reset_btn.setObjectName("dangerBtn")
        self._reset_btn.setFixedHeight(36)
        self._reset_btn.clicked.connect(self._reset_all)
        bottom.addWidget(self._reset_btn)

        bottom.addStretch()

        self._health_private_inputs = QCheckBox(tr("settings.actions.include_private_inputs"))
        bottom.addWidget(self._health_private_inputs)

        self._export_health_btn = QPushButton(tr("settings.actions.export_health"))
        self._export_health_btn.setObjectName("secondaryBtn")
        self._export_health_btn.setFixedHeight(36)
        self._export_health_btn.clicked.connect(self._export_health_report)
        bottom.addWidget(self._export_health_btn)

        self._open_dir_btn = QPushButton(tr("settings.actions.open_config"))
        self._open_dir_btn.setObjectName("secondaryBtn")
        self._open_dir_btn.setFixedHeight(36)
        self._open_dir_btn.clicked.connect(self._open_config_dir)
        bottom.addWidget(self._open_dir_btn)

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
        output_group = QGroupBox(tr("settings.output.group"))
        output_layout = QVBoxLayout(output_group)

        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText(tr("settings.output.placeholder"))
        self._output_dir.setReadOnly(True)
        self._browse_output_btn = QPushButton(tr("settings.output.browse"))
        self._browse_output_btn.setObjectName("secondaryBtn")
        self._browse_output_btn.setFixedWidth(80)
        self._browse_output_btn.setFixedHeight(34)
        self._browse_output_btn.clicked.connect(self._browse_output_dir)

        dir_row = QHBoxLayout()
        dir_row.addWidget(self._output_dir, 1)
        dir_row.addWidget(self._browse_output_btn)
        output_layout.addLayout(SettingRow(tr("settings.output.directory"), QWidget()))
        output_layout.addLayout(dir_row)

        self._format_combo = QComboBox()
        self._format_combo.addItems(["WAV", "FLAC", "MP3"])
        self._format_combo.setFixedWidth(120)
        self._format_combo.currentTextChanged.connect(
            lambda v: self._save("general.audio_format", v.lower()))
        output_layout.addLayout(SettingRow(tr("settings.output.format"), self._format_combo))

        self._sample_rate_combo = QComboBox()
        self._sample_rate_combo.addItems(["44100", "48000"])
        self._sample_rate_combo.setFixedWidth(120)
        self._sample_rate_combo.currentTextChanged.connect(
            lambda v: self._save("general.sample_rate", int(v)) if v else None)
        output_layout.addLayout(SettingRow(
            tr("settings.output.sample_rate"),
            self._sample_rate_combo,
            tr("settings.output.sample_rate_help"),
        ))

        self._audio_device_combo = QComboBox()
        self._audio_device_combo.setMinimumWidth(300)
        self._audio_device_combo.currentIndexChanged.connect(
            self._on_audio_device_selected
        )

        self._refresh_audio_devices_btn = QPushButton("Refresh")
        self._refresh_audio_devices_btn.setObjectName("secondaryBtn")
        self._refresh_audio_devices_btn.setFixedHeight(34)
        self._refresh_audio_devices_btn.clicked.connect(self._refresh_audio_devices)

        device_controls = QWidget()
        device_controls_layout = QHBoxLayout(device_controls)
        device_controls_layout.setContentsMargins(0, 0, 0, 0)
        device_controls_layout.setSpacing(8)
        device_controls_layout.addWidget(self._audio_device_combo, 1)
        device_controls_layout.addWidget(self._refresh_audio_devices_btn)
        output_layout.addLayout(SettingRow(
            "Audio output device",
            device_controls,
            "Uses PortAudio and shows the host API so similarly named devices are distinguishable.",
        ))

        self._audio_device_status = QLabel("")
        self._audio_device_status.setWordWrap(True)
        self._audio_device_status.setStyleSheet(
            f"color: {Palette.YELLOW}; font-size: 11px; padding: 2px 0;"
        )
        self._audio_device_status.setVisible(False)
        output_layout.addWidget(self._audio_device_status)

        layout.addWidget(output_group)

        # ── GPU and Models ──
        gpu_group = QGroupBox(tr("settings.gpu.group"))
        gpu_layout = QVBoxLayout(gpu_group)

        self._gpu_device = QSpinBox()
        self._gpu_device.setRange(0, 7)
        self._gpu_device.setFixedWidth(80)
        self._gpu_device.valueChanged.connect(
            lambda v: self._save("general.gpu_device", v))
        gpu_layout.addLayout(SettingRow(
            tr("settings.gpu.device_index"),
            self._gpu_device,
            tr("settings.gpu.device_index_help"),
        ))

        self._offline_mode = QCheckBox(tr("settings.gpu.offline_mode"))
        self._offline_mode.toggled.connect(
            lambda v: self._save("model_hub.offline_mode", v))
        gpu_layout.addLayout(SettingRow(tr("settings.gpu.disable_internet"), self._offline_mode))

        self._hf_token = QLineEdit()
        self._hf_token.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxx")
        self._hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self._hf_token.setFixedWidth(280)
        self._hf_token.editingFinished.connect(
            lambda: self._save("model_hub.hf_token", self._hf_token.text().strip()))
        gpu_layout.addLayout(SettingRow(
            tr("settings.gpu.hf_token"),
            self._hf_token,
            tr("settings.gpu.hf_token_help"),
        ))

        self._credential_status = QLabel()
        self._credential_status.setWordWrap(True)
        self._credential_status.setObjectName("credentialStatus")
        gpu_layout.addWidget(self._credential_status)

        layout.addWidget(gpu_group)

        # ── Appearance ──
        appearance_group = QGroupBox(tr("settings.appearance.group"))
        appearance_layout = QVBoxLayout(appearance_group)

        self._experience_combo = QComboBox()
        self._experience_combo.addItems(["Beginner", "Intermediate", "Advanced"])
        self._experience_combo.setFixedWidth(160)
        self._experience_combo.currentTextChanged.connect(
            lambda v: self._save("general.experience_level", v.lower()))
        appearance_layout.addLayout(SettingRow(
            tr("settings.appearance.experience_level"),
            self._experience_combo,
            tr("settings.appearance.experience_help"),
        ))

        self._ui_locale_combo = QComboBox()
        for code, label in ui_locale_options():
            self._ui_locale_combo.addItem(label, code)
        self._ui_locale_combo.setFixedWidth(220)
        self._ui_locale_combo.currentIndexChanged.connect(self._on_ui_locale_changed)
        appearance_layout.addLayout(SettingRow(
            tr("settings.appearance.ui_language"),
            self._ui_locale_combo,
            tr("settings.appearance.ui_language_help"),
        ))

        self._default_language = QComboBox()
        self._default_language.addItems(language_combo_items())
        self._default_language.setFixedWidth(200)
        self._default_language.currentTextChanged.connect(
            lambda v: self._save("lyrics.default_language", language_code_from_label(v)))
        appearance_layout.addLayout(SettingRow(
            tr("settings.appearance.default_lyrics_language"),
            self._default_language,
            tr("settings.appearance.default_lyrics_language_help"),
        ))

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
        lyrics_group = QGroupBox(tr("settings.lyrics.group"))
        lyrics_layout = QVBoxLayout(lyrics_group)

        self._lyrics_model = QComboBox()
        self._lyrics_model.addItem("LLaMA 3.1 8B (Recommended)", "llama-3.1-8b-q4")
        self._lyrics_model.addItem("LLaMA 3.2 3B (Fast)", "llama-3.2-3b-q4")
        self._lyrics_model.addItem("Qwen 2.5 14B (Premium)", "qwen-2.5-14b-q4")
        self._lyrics_model.setFixedWidth(240)
        self._lyrics_model.currentIndexChanged.connect(
            lambda: self._save("lyrics.model_id", self._lyrics_model.currentData()))
        lyrics_layout.addLayout(SettingRow(tr("settings.lyrics.model"), self._lyrics_model))

        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.1, 2.0)
        self._temperature.setSingleStep(0.05)
        self._temperature.setFixedWidth(100)
        self._temperature.valueChanged.connect(
            lambda v: self._save("lyrics.temperature", v))
        lyrics_layout.addLayout(SettingRow(
            tr("settings.lyrics.temperature"),
            self._temperature,
            tr("settings.lyrics.temperature_help"),
        ))

        self._top_p = QDoubleSpinBox()
        self._top_p.setRange(0.1, 1.0)
        self._top_p.setSingleStep(0.05)
        self._top_p.setFixedWidth(100)
        self._top_p.valueChanged.connect(
            lambda v: self._save("lyrics.top_p", v))
        lyrics_layout.addLayout(SettingRow(tr("settings.lyrics.top_p"), self._top_p))

        self._max_tokens = QSpinBox()
        self._max_tokens.setRange(256, 8192)
        self._max_tokens.setSingleStep(256)
        self._max_tokens.setFixedWidth(120)
        self._max_tokens.valueChanged.connect(
            lambda v: self._save("lyrics.max_tokens", v))
        lyrics_layout.addLayout(SettingRow(tr("settings.lyrics.max_tokens"), self._max_tokens))

        layout.addWidget(lyrics_group)

        # ── Song Forge ──
        forge_group = QGroupBox("Song Forge")
        forge_layout = QVBoxLayout(forge_group)

        self._timestep_shift = QDoubleSpinBox()
        self._timestep_shift.setRange(1.0, 3.0)
        self._timestep_shift.setSingleStep(1.0)
        self._timestep_shift.setDecimals(1)
        self._timestep_shift.setFixedWidth(100)
        self._timestep_shift.valueChanged.connect(
            lambda v: self._save("song_forge.timestep_shift", v))
        forge_layout.addLayout(SettingRow(
            "Timestep Shift",
            self._timestep_shift,
            "ACE-Step XL Turbo schedule shift (1, 2, or 3)",
        ))

        self._inference_steps = QSpinBox()
        self._inference_steps.setRange(1, 100)
        self._inference_steps.setSingleStep(1)
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
        for target in LUFS_TARGETS.values():
            self._mastering_target.addItem(target.label, target.key)
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
        cache_group = QGroupBox("Cache and Storage")
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

        self._autosave_enabled = QCheckBox("Autosave the open project on the interval")
        self._autosave_enabled.toggled.connect(
            lambda v: self._save("general.auto_save_enabled", v))
        cache_layout.addLayout(SettingRow(
            "Autosave",
            self._autosave_enabled,
            "Saves and versions the open project only when it has unsaved changes.",
        ))

        self._max_versions = QSpinBox()
        self._max_versions.setRange(1, 200)
        self._max_versions.setFixedWidth(120)
        self._max_versions.valueChanged.connect(
            lambda v: self._save("general.max_project_versions", v))
        cache_layout.addLayout(SettingRow(
            "Kept Project Versions",
            self._max_versions,
            "Oldest autosaves are pruned first; pre-restore versions are always kept.",
        ))

        layout.addWidget(cache_group)
        layout.addWidget(self._build_recovery_center())

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_recovery_center(self) -> QGroupBox:
        """One screen for every recovery artifact: inspect, preview, clean."""
        group = QGroupBox("Recovery Center")
        layout = QVBoxLayout(group)

        intro = QLabel(
            "Jobs, logs, crash reports, settings backups, and project versions "
            "are bounded by age, count, and size. Preview always runs first; "
            "active, recoverable, and pre-restore records are never removed."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {Palette.SUBTEXT0}; font-size: 11px;")
        layout.addWidget(intro)

        self._recovery_list = QListWidget()
        self._recovery_list.setMinimumHeight(140)
        layout.addWidget(self._recovery_list)

        self._recovery_status = QLabel("")
        self._recovery_status.setWordWrap(True)
        self._recovery_status.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 11px;"
        )
        layout.addWidget(self._recovery_status)

        row = QHBoxLayout()
        self._recovery_refresh_btn = QPushButton("Refresh")
        self._recovery_refresh_btn.setObjectName("secondaryBtn")
        self._recovery_refresh_btn.clicked.connect(self._refresh_recovery_center)

        self._recovery_preview_btn = QPushButton("Preview Cleanup")
        self._recovery_preview_btn.setObjectName("secondaryBtn")
        self._recovery_preview_btn.clicked.connect(self._preview_recovery_cleanup)

        self._recovery_clean_btn = QPushButton("Clean Now")
        self._recovery_clean_btn.setObjectName("dangerBtn")
        self._recovery_clean_btn.setEnabled(False)
        self._recovery_clean_btn.clicked.connect(self._run_recovery_cleanup)

        self._recovery_reveal_btn = QPushButton("Open Config Folder")
        self._recovery_reveal_btn.setObjectName("secondaryBtn")
        self._recovery_reveal_btn.clicked.connect(self._open_config_dir)

        for button in (
            self._recovery_refresh_btn, self._recovery_preview_btn,
            self._recovery_clean_btn, self._recovery_reveal_btn,
        ):
            button.setFixedHeight(30)
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        return group

    def _recovery_center(self):
        from core.retention import RecoveryCenter

        if getattr(self, "_recovery", None) is None:
            self._recovery = RecoveryCenter(self._settings)
        return self._recovery

    def _refresh_recovery_center(self):
        from core.retention import CATEGORIES, CATEGORY_LABELS, load_policy

        self._recovery_list.clear()
        center = self._recovery_center()
        total_bytes = 0
        for category in CATEGORIES:
            try:
                items = center.collect(category)
            except Exception as exc:
                self._recovery_list.addItem(
                    f"{CATEGORY_LABELS[category]}: unavailable ({exc})"
                )
                continue
            size = sum(item.size_bytes for item in items)
            total_bytes += size
            protected = sum(1 for item in items if item.protected)
            policy = load_policy(category, self._settings)
            self._recovery_list.addItem(
                f"{CATEGORY_LABELS[category]}: {len(items)} item(s), "
                f"{size / 1e6:.1f} MB, {protected} protected - keeps "
                f"{policy.describe()}"
            )
        self._recovery_status.setText(
            f"Recovery artifacts use {total_bytes / 1e6:.1f} MB."
        )
        self._recovery_clean_btn.setEnabled(False)

    def _preview_recovery_cleanup(self):
        center = self._recovery_center()
        self._recovery_list.clear()
        removable = 0
        freed = 0
        for category, plan in center.preview_all().items():
            self._recovery_list.addItem(plan.summary())
            removable += len(plan.remove)
            freed += plan.removed_bytes
        self._recovery_status.setText(
            f"Preview only - nothing removed. {removable} item(s) would be "
            f"removed, freeing {freed / 1e6:.1f} MB."
            if removable else "Preview only - nothing needs removing."
        )
        self._recovery_clean_btn.setEnabled(removable > 0)

    def _run_recovery_cleanup(self):
        from core.retention import CATEGORY_LABELS

        center = self._recovery_center()
        removed = center.clean_all()
        lines = [
            f"{CATEGORY_LABELS[category]}: removed {len(items)}"
            for category, items in removed.items() if items
        ]
        total = sum(len(items) for items in removed.values())
        self._refresh_recovery_center()
        self._recovery_status.setText(
            f"Removed {total} item(s). " + " | ".join(lines)
            if total else "Nothing needed removing."
        )
        if self.toast_mgr:
            self.toast_mgr.success(f"Recovery cleanup removed {total} item(s).")

    def _set_audio_device_status(self, message: str):
        """Show an audio-device warning without hiding the selected setting."""
        text = str(message or "").strip()
        self._audio_device_status.setText(text)
        self._audio_device_status.setVisible(bool(text))

    def _on_audio_device_status(self, message: str):
        """Display runtime PortAudio fallback messages in Settings when open."""
        self._set_audio_device_status(message)

    def _refresh_audio_devices(self):
        """Refresh PortAudio devices without restarting the application."""
        saved_identity = str(
            self._settings.get("general.audio_output_device", "") or ""
        ).strip()
        devices, error = enumerate_output_devices()

        self._audio_device_combo.blockSignals(True)
        try:
            self._audio_device_combo.clear()
            self._audio_device_combo.addItem("System default", "")
            for device in devices:
                self._audio_device_combo.addItem(device.label, device.identity)

            selected_index = self._audio_device_combo.findData(saved_identity)
            saved_unavailable = bool(saved_identity and selected_index < 0)
            if saved_unavailable:
                selected_index = self._audio_device_combo.count()
                self._audio_device_combo.addItem(
                    f"Unavailable: {format_output_device_identity(saved_identity)}",
                    saved_identity,
                )
            self._audio_device_combo.setCurrentIndex(max(0, selected_index))
        finally:
            self._audio_device_combo.blockSignals(False)

        # Keep the singleton transport in sync with the persisted selection,
        # including an unavailable identity so a hot-plugged device can recover
        # on a later refresh without losing the user's preference.
        self._audio.set_output_device(saved_identity)

        if error:
            self._set_audio_device_status(
                "Could not enumerate audio output devices; playback will use "
                f"the system default until the list is refreshed. ({error})"
            )
        elif saved_unavailable:
            self._set_audio_device_status(
                f"Saved output device '{format_output_device_identity(saved_identity)}' "
                "is unavailable; playback will use the system default until it returns."
            )
        elif not devices:
            self._set_audio_device_status(
                "No PortAudio output devices were found; playback will use the system default."
            )
        else:
            self._set_audio_device_status("")

    def _on_audio_device_selected(self, _index: int):
        """Persist a device identity and apply it to the shared transport."""
        identity = str(self._audio_device_combo.currentData() or "").strip()
        self._save("general.audio_output_device", identity)
        self._audio.set_output_device(identity)
        if identity:
            self._set_audio_device_status("")

    def _load_values(self):
        """Load current settings into UI controls."""
        s = self._settings

        # Block signals on all save-connected widgets to prevent
        # cascading saves during programmatic value changes
        _widgets = [
            self._format_combo, self._sample_rate_combo, self._gpu_device,
            self._offline_mode, self._hf_token, self._experience_combo,
            self._ui_locale_combo,
            self._default_language,
            self._lyrics_model, self._temperature, self._top_p,
            self._max_tokens, self._timestep_shift, self._inference_steps,
            self._batch_count, self._default_duration, self._default_bpm,
            self._mastering_target, self._auto_eq, self._auto_compress,
            self._max_cache, self._autosave_interval,
            self._autosave_enabled, self._max_versions,
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

            language = language_label(s.get("lyrics.default_language", "en"))
            idx = self._default_language.findText(language)
            if idx >= 0:
                self._default_language.setCurrentIndex(idx)

            ui_locale = s.get("general.ui_locale", "en")
            idx = self._ui_locale_combo.findData(ui_locale)
            if idx >= 0:
                self._ui_locale_combo.setCurrentIndex(idx)

            # Advanced tab
            model_id = s.get("lyrics.model_id", "llama-3.1-8b-q4")
            for i in range(self._lyrics_model.count()):
                if self._lyrics_model.itemData(i) == model_id:
                    self._lyrics_model.setCurrentIndex(i)
                    break

            self._temperature.setValue(s.get("lyrics.temperature", 0.8))
            self._top_p.setValue(s.get("lyrics.top_p", 0.92))
            self._max_tokens.setValue(s.get("lyrics.max_tokens", 2048))
            self._timestep_shift.setValue(
                s.get("song_forge.timestep_shift", 3.0)
            )
            self._inference_steps.setValue(
                s.get("song_forge.inference_steps", 8)
            )
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
            self._autosave_enabled.setChecked(s.get("general.auto_save_enabled", True))
            self._max_versions.setValue(s.get("general.max_project_versions", 20))

        finally:
            for w in _widgets:
                w.blockSignals(False)
        self._refresh_audio_devices()
        self._update_repair_status()
        self._refresh_credential_status()
        self._refresh_recovery_center()

    def _save(self, key: str, value):
        """Save a setting and show toast."""
        try:
            self._settings.set(key, value)
        except CredentialError as exc:
            if self.toast_mgr:
                self.toast_mgr.error(f"{key} was not saved: {exc}")
            self._update_repair_status()
            self._refresh_credential_status()
            return
        self._update_repair_status()
        if key in SECRET_SETTING_KEYS:
            self._refresh_credential_status()
            if self.toast_mgr:
                store = self._settings.credential_store
                self.toast_mgr.success(
                    f"Saved to {store.backend_name}."
                    if value else "Stored secret cleared."
                )
            return
        # Toast for important changes only
        if self.toast_mgr and key in (
            "general.audio_format",
            "general.sample_rate",
            "general.audio_output_device",
            "lyrics.default_language",
            "lyrics.model_id",
        ):
            self.toast_mgr.success(tr("settings.messages.setting_updated"))

    def _on_ui_locale_changed(self, _index: int):
        """Persist the interface locale and apply layout direction immediately."""
        code = str(self._ui_locale_combo.currentData() or "en")
        selected = set_locale(code, persist=True)
        if self.toast_mgr:
            self.toast_mgr.info(tr("settings.messages.locale_changed", locale=selected))

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, tr("settings.dialogs.select_output_directory"))
        if path:
            self._output_dir.setText(path)
            self._save("general.output_dir", path)

    def _reset_all(self):
        self._settings.reset_all()
        self._load_values()
        self._update_repair_status()
        if self.toast_mgr:
            self.toast_mgr.warning(tr("settings.messages.reset"))

    def _refresh_credential_status(self):
        """Name the credential service in use, or state plainly that there is none."""
        status = self._settings.credential_backend_status()
        if status.get("available"):
            text = (
                f"Secrets are stored in {status.get('name')}. "
                "They are never written to config JSON, backups, or diagnostics."
            )
            self._credential_status.setProperty("state", "ok")
        else:
            detail = (status.get("detail") or "").strip()
            lead = "No OS credential service is available, so secrets cannot be stored."
            text = lead if detail in ("", lead) else f"{lead} {detail}"
            self._credential_status.setProperty("state", "warning")
        self._credential_status.setText(text)
        color = Palette.SUBTEXT0 if status.get("available") else Palette.YELLOW
        self._credential_status.setStyleSheet(
            f"color: {color}; font-size: 11px; padding: 2px 0;"
        )

    def _update_repair_status(self):
        status = self._settings.repair_status
        state = status.get("status", "ok")
        if state == "ok":
            self._repair_label.setVisible(False)
            self._repair_label.setText("")
            return

        messages = status.get("messages") or []
        backups = status.get("backup_paths") or []
        text = f"Config {state}: " + (" ".join(messages) if messages else "Review the config file.")
        if backups:
            text += f" Backup: {backups[-1]}"
        self._repair_label.setText(text)
        self._repair_label.setVisible(True)

    def _install_accessibility(self):
        install_accessibility(
            self,
            "Settings",
            named_controls=[
                (self._tabs, "Settings sections", "Switches between simple and advanced settings."),
                (self._output_dir, "Output directory", "Current default render output directory."),
                (self._browse_output_btn, "Browse output directory", "Chooses the default render output directory."),
                (self._format_combo, "Default audio format", "Selects the default export format."),
                (self._sample_rate_combo, "Sample rate", "Selects the default audio sample rate."),
                (self._audio_device_combo, "Audio output device", "Selects the PortAudio output device and shows its host API."),
                (self._refresh_audio_devices_btn, "Refresh audio output devices", "Refreshes the PortAudio output-device list without restarting."),
                (self._gpu_device, "GPU device index", "Selects the GPU device index."),
                (self._offline_mode, "Offline mode", "Disables internet access for Model Hub."),
                (self._hf_token, "HuggingFace token", "Stores a token for gated model downloads."),
                (self._experience_combo, "Experience level", "Controls default UI complexity."),
                (self._ui_locale_combo, "Interface language", "Selects the interface language and layout direction."),
                (self._default_language, "Default lyrics language", "Sets the default language metadata for lyrics and new voice profiles."),
                (self._lyrics_model, "Lyrics model", "Selects the local lyrics model."),
                (self._temperature, "Lyrics temperature", "Controls creative variation."),
                (self._top_p, "Lyrics top-p", "Controls nucleus sampling."),
                (self._max_tokens, "Max lyrics tokens", "Controls maximum lyrics generation length."),
                (self._timestep_shift, "Song Forge timestep shift", "Controls the ACE-Step XL Turbo timestep schedule."),
                (self._inference_steps, "Song Forge inference steps", "Controls generation quality and speed."),
                (self._batch_count, "Song Forge batch count", "Controls number of variations."),
                (self._default_duration, "Default song duration", "Controls default generation duration."),
                (self._default_bpm, "Default MIDI tempo", "Controls default MIDI BPM."),
                (self._mastering_target, "Mastering target", "Selects the loudness target."),
                (self._auto_eq, "Automatic mastering EQ", "Toggles automatic EQ during mastering."),
                (self._auto_compress, "Automatic mastering compression", "Toggles automatic bus compression."),
                (self._max_cache, "Maximum cache size", "Controls cache cleanup threshold."),
                (self._autosave_interval, "Auto-save interval", "Controls project auto-save frequency."),
                (self._autosave_enabled, "Autosave enabled", "Enables interval autosave for the open project."),
                (self._max_versions, "Kept project versions", "Limits how many project versions are retained."),
                (self._reset_btn, "Reset settings", "Resets all settings to defaults."),
                (self._health_private_inputs, "Include private job inputs", "Includes job prompt and input fields in the health report."),
                (self._export_health_btn, "Export health report", "Saves a redacted diagnostics bundle."),
                (self._open_dir_btn, "Open config folder", "Opens the settings folder in the file manager."),
            ],
            tab_order=[
                self._tabs,
                self._output_dir,
                self._browse_output_btn,
                self._format_combo,
                self._sample_rate_combo,
                self._audio_device_combo,
                self._refresh_audio_devices_btn,
                self._gpu_device,
                self._offline_mode,
                self._hf_token,
                self._experience_combo,
                self._ui_locale_combo,
                self._default_language,
                self._lyrics_model,
                self._temperature,
                self._top_p,
                self._max_tokens,
                self._timestep_shift,
                self._inference_steps,
                self._batch_count,
                self._default_duration,
                self._default_bpm,
                self._mastering_target,
                self._auto_eq,
                self._auto_compress,
                self._max_cache,
                self._autosave_interval,
                self._autosave_enabled,
                self._max_versions,
                self._reset_btn,
                self._health_private_inputs,
                self._export_health_btn,
                self._open_dir_btn,
            ],
        )

    def _export_health_report(self):
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            tr("settings.dialogs.export_health"),
            "slunderstudio-health-report.zip",
            "Health Report (*.zip)",
        )
        if not path:
            return
        try:
            output = export_health_report(
                path,
                include_private=self._health_private_inputs.isChecked(),
            )
        except Exception as exc:
            if self.toast_mgr:
                self.toast_mgr.error(tr("settings.messages.health_export_failed", error=exc))
            return
        if self.toast_mgr:
            self.toast_mgr.success(tr("settings.messages.health_exported", filename=output.name))

    def _open_config_dir(self):
        import subprocess, sys
        config_dir = str(self._settings._config_path.parent)
        if sys.platform == "win32":
            subprocess.Popen(f'explorer "{config_dir}"')
        elif sys.platform == "darwin":
            subprocess.Popen(["open", config_dir])
        else:
            subprocess.Popen(["xdg-open", config_dir])
