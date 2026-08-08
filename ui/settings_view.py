"""
Slunder Studio — Settings View
Two-tier settings: Simple Mode (essentials) and Advanced Mode (full controls).
All changes apply immediately with toast feedback.
"""
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QComboBox, QLineEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QListWidget,
    QFileDialog, QGroupBox, QFormLayout, QTabWidget, QDialog, QGridLayout,
)
from PySide6.QtCore import Qt, QTimer

from ui.theme import Palette
from ui.accessibility import install_accessibility
from ui.widgets import OperationProgressWidget
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
from core.stem_export import STEM_EXPORT_TEMPLATES
from core.credentials import CredentialError
from core.midi_controller import (
    DEFAULT_MIDI_BINDINGS,
    MIDI_BINDING_MODES,
    MIDI_CHANNEL_OMNI,
    MidiBinding,
    MIDI_ACTION_LABELS,
    bindings_to_settings,
    normalized_bindings,
)
from core.midi_input import list_midi_input_ports
from core.settings import Settings, APP_VERSION, SECRET_SETTING_KEYS
from core.audio_engine import (
    AudioEngine,
    enumerate_output_devices,
    format_output_device_identity,
)
from core.workers import CancelledJobError, InferenceWorker
from ui.file_dialogs import choose_directory, open_file, save_file


def _recovery_cleanup_task(
    center,
    progress_cb=None,
    step_cb=None,
    log_cb=None,
    cancel_event=None,
):
    """Clean recovery categories with a cancellation boundary per category."""
    from core.retention import CATEGORIES, CATEGORY_LABELS

    removed = {}
    total = len(CATEGORIES)
    for index, category in enumerate(CATEGORIES):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledJobError("Recovery cleanup cancelled")
        label = CATEGORY_LABELS[category]
        if step_cb:
            step_cb(f"Cleaning {label}...")
        removed[category] = center.clean(category)
        if progress_cb:
            progress_cb(int(round((index + 1) * 100 / total)))
    return removed


def _recovery_policy_text(policy) -> str:
    """Render retention limits as user guidance instead of internal prose."""
    details = []
    if policy.max_age_days:
        details.append(
            tr("settings.recovery.policy_age", days=f"{policy.max_age_days:g}")
        )
    if policy.max_count:
        details.append(
            tr("settings.recovery.policy_count", count=policy.max_count)
        )
    if policy.max_total_mb:
        details.append(
            tr("settings.recovery.policy_size", size=f"{policy.max_total_mb:g}")
        )
    if not details:
        return tr("settings.recovery.policy_none")
    return tr("settings.recovery.policy_join", details=" · ".join(details))


class SettingRow(QHBoxLayout):
    """A labeled setting control with optional description."""

    def __init__(self, label: str, widget: QWidget, description: str = ""):
        super().__init__()
        self.setSpacing(12)

        label_container = QVBoxLayout()
        label_container.setSpacing(2)

        lbl = QLabel(label)
        lbl.setStyleSheet(f"font-size: 9.75pt; font-weight: 600; color: {Palette.TEXT};")
        label_container.addWidget(lbl)

        if description:
            desc = QLabel(description)
            desc.setStyleSheet(f"font-size: 8.25pt; color: {Palette.OVERLAY0};")
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
        self._health_report_worker: Optional[InferenceWorker] = None
        self._recovery_worker: Optional[InferenceWorker] = None
        self._midi_mapping_rows = []
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
            f"color: {Palette.YELLOW}; font-size: 9pt; padding: 4px 0;"
        )
        layout.addWidget(self._repair_label)

        self._operation_progress = OperationProgressWidget()
        self._operation_progress.cancel_requested.connect(
            self._cancel_active_operation
        )
        layout.addWidget(self._operation_progress)

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
        self._reset_btn.setMinimumHeight(36)
        self._reset_btn.clicked.connect(self._reset_all)
        bottom.addWidget(self._reset_btn)

        bottom.addStretch()

        self._health_private_inputs = QCheckBox(tr("settings.actions.include_private_inputs"))
        bottom.addWidget(self._health_private_inputs)

        self._export_health_btn = QPushButton(tr("settings.actions.export_health"))
        self._export_health_btn.setObjectName("secondaryBtn")
        self._export_health_btn.setMinimumHeight(36)
        self._export_health_btn.clicked.connect(self._export_health_report)
        bottom.addWidget(self._export_health_btn)

        self._open_dir_btn = QPushButton(tr("settings.actions.open_config"))
        self._open_dir_btn.setObjectName("secondaryBtn")
        self._open_dir_btn.setMinimumHeight(36)
        self._open_dir_btn.clicked.connect(self._open_config_dir)
        bottom.addWidget(self._open_dir_btn)

        self._onboarding_btn = QPushButton(tr("settings.actions.open_onboarding"))
        self._onboarding_btn.setObjectName("secondaryBtn")
        self._onboarding_btn.setMinimumHeight(36)
        self._onboarding_btn.clicked.connect(self._open_onboarding)
        bottom.addWidget(self._onboarding_btn)

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
        self._browse_output_btn.setMinimumWidth(80)
        self._browse_output_btn.setMinimumHeight(34)
        self._browse_output_btn.clicked.connect(self._browse_output_dir)

        dir_row = QHBoxLayout()
        dir_row.addWidget(self._output_dir, 1)
        dir_row.addWidget(self._browse_output_btn)
        output_layout.addLayout(SettingRow(tr("settings.output.directory"), QWidget()))
        output_layout.addLayout(dir_row)

        self._format_combo = QComboBox()
        self._format_combo.addItems(["WAV", "FLAC", "MP3"])
        self._format_combo.setMinimumWidth(120)
        self._format_combo.currentTextChanged.connect(
            lambda v: self._save("general.audio_format", v.lower()))
        output_layout.addLayout(SettingRow(tr("settings.output.format"), self._format_combo))

        self._sample_rate_combo = QComboBox()
        self._sample_rate_combo.addItems(["44100", "48000"])
        self._sample_rate_combo.setMinimumWidth(120)
        self._sample_rate_combo.currentTextChanged.connect(
            lambda v: self._save("general.sample_rate", int(v)) if v else None)
        output_layout.addLayout(SettingRow(
            tr("settings.output.sample_rate"),
            self._sample_rate_combo,
            tr("settings.output.sample_rate_help"),
        ))

        self._stem_export_template_combo = QComboBox()
        for template in STEM_EXPORT_TEMPLATES:
            self._stem_export_template_combo.addItem(
                template.label,
                template.id,
            )
        self._stem_export_template_combo.setMinimumWidth(220)
        self._stem_export_template_combo.currentIndexChanged.connect(
            self._on_stem_export_template_changed
        )
        output_layout.addLayout(SettingRow(
            "Stem export naming",
            self._stem_export_template_combo,
            "Chooses the filename template used when exporting separated stems.",
        ))

        self._audio_device_combo = QComboBox()
        self._audio_device_combo.setMinimumWidth(300)
        self._audio_device_combo.currentIndexChanged.connect(
            self._on_audio_device_selected
        )

        self._refresh_audio_devices_btn = QPushButton(tr("settings.actions.refresh"))
        self._refresh_audio_devices_btn.setObjectName("secondaryBtn")
        self._refresh_audio_devices_btn.setMinimumHeight(34)
        self._refresh_audio_devices_btn.clicked.connect(self._refresh_audio_devices)

        device_controls = QWidget()
        device_controls_layout = QHBoxLayout(device_controls)
        device_controls_layout.setContentsMargins(0, 0, 0, 0)
        device_controls_layout.setSpacing(8)
        device_controls_layout.addWidget(self._audio_device_combo, 1)
        device_controls_layout.addWidget(self._refresh_audio_devices_btn)
        output_layout.addLayout(SettingRow(
            tr("settings.output.audio_device"),
            device_controls,
            tr("settings.output.audio_device_help"),
        ))

        self._audio_device_status = QLabel("")
        self._audio_device_status.setWordWrap(True)
        self._audio_device_status.setStyleSheet(
            f"color: {Palette.YELLOW}; font-size: 8.25pt; padding: 2px 0;"
        )
        self._audio_device_status.setVisible(False)
        output_layout.addWidget(self._audio_device_status)

        # ── Optional Content Credentials ──
        self._c2pa_enabled = QCheckBox(tr("settings.output.c2pa_enabled"))
        self._c2pa_enabled.toggled.connect(self._on_c2pa_toggled)
        output_layout.addLayout(SettingRow(
            tr("settings.output.c2pa_label"),
            self._c2pa_enabled,
            tr("settings.output.c2pa_help"),
        ))

        self._c2pa_certificate_path = QLineEdit()
        self._c2pa_certificate_path.setReadOnly(True)
        self._c2pa_certificate_path.setPlaceholderText(
            tr("settings.output.c2pa_certificate_placeholder")
        )
        self._c2pa_certificate_browse = QPushButton(tr("settings.output.browse"))
        self._c2pa_certificate_browse.setObjectName("secondaryBtn")
        self._c2pa_certificate_browse.setMinimumWidth(80)
        self._c2pa_certificate_browse.setMinimumHeight(34)
        self._c2pa_certificate_browse.clicked.connect(
            lambda: self._browse_c2pa_file(
                "general.c2pa_certificate_path",
                "c2pa_certificate",
                tr("settings.dialogs.select_c2pa_certificate"),
                "PEM Certificates (*.pem *.crt *.cer);;All Files (*)",
                self._c2pa_certificate_path,
            )
        )
        certificate_controls = QWidget()
        certificate_layout = QHBoxLayout(certificate_controls)
        certificate_layout.setContentsMargins(0, 0, 0, 0)
        certificate_layout.setSpacing(8)
        certificate_layout.addWidget(self._c2pa_certificate_path, 1)
        certificate_layout.addWidget(self._c2pa_certificate_browse)
        output_layout.addLayout(SettingRow(
            tr("settings.output.c2pa_certificate"),
            certificate_controls,
        ))

        self._c2pa_private_key_path = QLineEdit()
        self._c2pa_private_key_path.setReadOnly(True)
        self._c2pa_private_key_path.setPlaceholderText(
            tr("settings.output.c2pa_private_key_placeholder")
        )
        self._c2pa_private_key_browse = QPushButton(tr("settings.output.browse"))
        self._c2pa_private_key_browse.setObjectName("secondaryBtn")
        self._c2pa_private_key_browse.setMinimumWidth(80)
        self._c2pa_private_key_browse.setMinimumHeight(34)
        self._c2pa_private_key_browse.clicked.connect(
            lambda: self._browse_c2pa_file(
                "general.c2pa_private_key_path",
                "c2pa_private_key",
                tr("settings.dialogs.select_c2pa_private_key"),
                "PEM Private Keys (*.pem *.key);;All Files (*)",
                self._c2pa_private_key_path,
            )
        )
        key_controls = QWidget()
        key_layout = QHBoxLayout(key_controls)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.setSpacing(8)
        key_layout.addWidget(self._c2pa_private_key_path, 1)
        key_layout.addWidget(self._c2pa_private_key_browse)
        output_layout.addLayout(SettingRow(
            tr("settings.output.c2pa_private_key"),
            key_controls,
        ))

        self._c2pa_timestamp_url = QLineEdit()
        self._c2pa_timestamp_url.setPlaceholderText(
            tr("settings.output.c2pa_timestamp_placeholder")
        )
        self._c2pa_timestamp_url.editingFinished.connect(
            lambda: self._save(
                "general.c2pa_timestamp_url",
                self._c2pa_timestamp_url.text().strip(),
            )
        )
        self._c2pa_timestamp_url.textChanged.connect(
            lambda _value: self._refresh_c2pa_state()
        )
        output_layout.addLayout(SettingRow(
            tr("settings.output.c2pa_timestamp"),
            self._c2pa_timestamp_url,
            tr("settings.output.c2pa_timestamp_help"),
        ))

        self._c2pa_status = QLabel()
        self._c2pa_status.setWordWrap(True)
        self._c2pa_status.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8.25pt; padding: 2px 0;"
        )
        output_layout.addWidget(self._c2pa_status)

        layout.addWidget(output_group)

        # ── OSC Control ──
        osc_group = QGroupBox(tr("settings.osc.group"))
        osc_layout = QVBoxLayout(osc_group)

        self._osc_enabled = QCheckBox(tr("settings.osc.enabled"))
        self._osc_enabled.toggled.connect(
            lambda enabled: self._save("osc.enabled", bool(enabled))
        )
        osc_layout.addLayout(SettingRow(
            tr("settings.osc.control"),
            self._osc_enabled,
            tr("settings.osc.control_help"),
        ))

        self._osc_port = QSpinBox()
        self._osc_port.setRange(1024, 65535)
        self._osc_port.setValue(9000)
        self._osc_port.valueChanged.connect(
            lambda value: self._save("osc.port", int(value))
        )
        osc_layout.addLayout(SettingRow(
            tr("settings.osc.port"),
            self._osc_port,
            tr("settings.osc.port_help"),
        ))

        self._osc_allow_lan = QCheckBox(tr("settings.osc.allow_lan"))
        self._osc_allow_lan.toggled.connect(self._on_osc_lan_toggled)
        osc_layout.addLayout(SettingRow(
            tr("settings.osc.lan_access"),
            self._osc_allow_lan,
            tr("settings.osc.lan_access_help"),
        ))

        self._osc_allowed_hosts = QLineEdit()
        self._osc_allowed_hosts.setPlaceholderText(
            tr("settings.osc.allowed_hosts_placeholder")
        )
        self._osc_allowed_hosts.editingFinished.connect(
            self._save_osc_allowed_hosts
        )
        osc_layout.addLayout(SettingRow(
            tr("settings.osc.allowed_hosts"),
            self._osc_allowed_hosts,
            tr("settings.osc.allowed_hosts_help"),
        ))

        self._osc_packet_bytes = QSpinBox()
        self._osc_packet_bytes.setRange(256, 65507)
        self._osc_packet_bytes.valueChanged.connect(
            lambda value: self._save("osc.max_packet_bytes", int(value))
        )
        osc_layout.addLayout(SettingRow(
            tr("settings.osc.packet_limit"),
            self._osc_packet_bytes,
            tr("settings.osc.packet_limit_help"),
        ))

        self._osc_rate = QSpinBox()
        self._osc_rate.setRange(1, 10000)
        self._osc_rate.valueChanged.connect(
            lambda value: self._save("osc.max_messages_per_second", int(value))
        )
        osc_layout.addLayout(SettingRow(
            tr("settings.osc.rate_limit"),
            self._osc_rate,
            tr("settings.osc.rate_limit_help"),
        ))

        osc_note = QLabel(tr("settings.osc.note"))
        osc_note.setWordWrap(True)
        osc_note.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8.25pt; padding: 2px 0;"
        )
        osc_layout.addWidget(osc_note)
        layout.addWidget(osc_group)

        # ── MIDI Controller ──
        midi_controller_group = QGroupBox("MIDI Controller")
        midi_controller_layout = QVBoxLayout(midi_controller_group)

        self._midi_enabled = QCheckBox("Enable live MIDI controller input")
        self._midi_enabled.toggled.connect(
            lambda enabled: self._save("midi_controller.enabled", bool(enabled))
        )
        midi_controller_layout.addLayout(SettingRow(
            "MIDI input",
            self._midi_enabled,
            "Opt in to one local MIDI input port. Input stays off until enabled.",
        ))

        self._midi_port_combo = QComboBox()
        self._midi_port_combo.setMinimumWidth(260)
        self._midi_port_combo.currentIndexChanged.connect(self._on_midi_port_changed)
        self._midi_refresh_btn = QPushButton(tr("settings.actions.refresh"))
        self._midi_refresh_btn.setObjectName("secondaryBtn")
        self._midi_refresh_btn.setMinimumHeight(34)
        self._midi_refresh_btn.clicked.connect(self._refresh_midi_ports)
        midi_port_controls = QWidget()
        midi_port_layout = QHBoxLayout(midi_port_controls)
        midi_port_layout.setContentsMargins(0, 0, 0, 0)
        midi_port_layout.setSpacing(8)
        midi_port_layout.addWidget(self._midi_port_combo, 1)
        midi_port_layout.addWidget(self._midi_refresh_btn)
        midi_controller_layout.addLayout(SettingRow(
            "Input port",
            midi_port_controls,
            "Select a port supplied by the optional mido backend; System default uses the backend default.",
        ))

        self._midi_status = QLabel("")
        self._midi_status.setWordWrap(True)
        self._midi_status.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8.25pt; padding: 2px 0;"
        )
        midi_controller_layout.addWidget(self._midi_status)

        mapping_note = QLabel(
            "Bindings target the selected Mixer track. CC values use normalized faders; "
            "trigger and toggle controls fire once per press."
        )
        mapping_note.setWordWrap(True)
        mapping_note.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8.25pt; padding: 2px 0;"
        )
        midi_controller_layout.addWidget(mapping_note)

        mapping_frame = QWidget()
        mapping_grid = QGridLayout(mapping_frame)
        mapping_grid.setContentsMargins(0, 4, 0, 0)
        mapping_grid.setHorizontalSpacing(8)
        mapping_grid.setVerticalSpacing(4)
        for column, label in enumerate(("Action", "Type", "Channel", "Number", "Mode")):
            heading = QLabel(label)
            heading.setStyleSheet(
                f"color: {Palette.SUBTEXT0}; font-size: 7.5pt; font-weight: bold;"
            )
            mapping_grid.addWidget(heading, 0, column)

        for row_index, raw_binding in enumerate(DEFAULT_MIDI_BINDINGS, start=1):
            binding = MidiBinding.from_dict(raw_binding)
            action_label = QLabel(MIDI_ACTION_LABELS.get(binding.action, binding.action))
            action_label.setToolTip(binding.action)
            mapping_grid.addWidget(action_label, row_index, 0)

            type_combo = QComboBox()
            type_combo.setObjectName(f"midiBindingType{row_index}")
            type_combo.addItem("CC", "cc")
            type_combo.addItem("Note", "note")
            type_combo.setCurrentIndex(type_combo.findData(binding.message_type))

            channel_combo = QComboBox()
            channel_combo.setObjectName(f"midiBindingChannel{row_index}")
            channel_combo.addItem("All", MIDI_CHANNEL_OMNI)
            for channel in range(16):
                channel_combo.addItem(str(channel + 1), channel)
            channel_combo.setCurrentIndex(channel_combo.findData(binding.channel))

            number_spin = QSpinBox()
            number_spin.setObjectName(f"midiBindingNumber{row_index}")
            number_spin.setRange(0, 127)
            number_spin.setValue(binding.number)
            number_spin.setMinimumWidth(68)

            mode_combo = QComboBox()
            mode_combo.setObjectName(f"midiBindingMode{row_index}")
            for mode in MIDI_BINDING_MODES:
                mode_combo.addItem(mode.capitalize(), mode)
            mode_combo.setCurrentIndex(mode_combo.findData(binding.mode))

            row_controls = {
                "action": binding.action,
                "type": type_combo,
                "channel": channel_combo,
                "number": number_spin,
                "mode": mode_combo,
            }
            self._midi_mapping_rows.append(row_controls)
            self._midi_mapping_controls = getattr(self, "_midi_mapping_controls", [])
            self._midi_mapping_controls.extend((type_combo, channel_combo, number_spin, mode_combo))
            for control in (type_combo, channel_combo, number_spin, mode_combo):
                if isinstance(control, QSpinBox):
                    control.valueChanged.connect(self._on_midi_mapping_changed)
                else:
                    control.currentIndexChanged.connect(self._on_midi_mapping_changed)
            mapping_grid.addWidget(type_combo, row_index, 1)
            mapping_grid.addWidget(channel_combo, row_index, 2)
            mapping_grid.addWidget(number_spin, row_index, 3)
            mapping_grid.addWidget(mode_combo, row_index, 4)
        midi_controller_layout.addWidget(mapping_frame)
        layout.addWidget(midi_controller_group)

        # ── GPU and Models ──
        gpu_group = QGroupBox(tr("settings.gpu.group"))
        gpu_layout = QVBoxLayout(gpu_group)

        self._gpu_device = QSpinBox()
        self._gpu_device.setRange(0, 7)
        self._gpu_device.setMinimumWidth(80)
        self._gpu_device.valueChanged.connect(
            lambda v: self._save("general.gpu_device", v))
        gpu_layout.addLayout(SettingRow(
            tr("settings.gpu.device_index"),
            self._gpu_device,
            tr("settings.gpu.device_index_help"),
        ))

        self._offline_mode = QCheckBox(tr("settings.gpu.offline_mode"))
        self._offline_mode.toggled.connect(self._on_offline_mode_toggled)
        gpu_layout.addLayout(SettingRow(tr("settings.gpu.disable_internet"), self._offline_mode))

        self._hf_token = QLineEdit()
        self._hf_token.setPlaceholderText("hf_xxxxxxxxxxxxxxxxxxxx")
        self._hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self._hf_token.setMinimumWidth(280)
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
        for code, key in (
            ("beginner", "settings.appearance.experience_beginner"),
            ("intermediate", "settings.appearance.experience_intermediate"),
            ("advanced", "settings.appearance.experience_advanced"),
        ):
            self._experience_combo.addItem(tr(key), code)
        self._experience_combo.setMinimumWidth(160)
        self._experience_combo.currentIndexChanged.connect(
            lambda: self._save(
                "general.experience_level",
                self._experience_combo.currentData() or "beginner",
            ))
        appearance_layout.addLayout(SettingRow(
            tr("settings.appearance.experience_level"),
            self._experience_combo,
            tr("settings.appearance.experience_help"),
        ))

        self._ui_locale_combo = QComboBox()
        for code, label in ui_locale_options():
            self._ui_locale_combo.addItem(label, code)
        self._ui_locale_combo.setMinimumWidth(220)
        self._ui_locale_combo.currentIndexChanged.connect(self._on_ui_locale_changed)
        appearance_layout.addLayout(SettingRow(
            tr("settings.appearance.ui_language"),
            self._ui_locale_combo,
            tr("settings.appearance.ui_language_help"),
        ))

        self._default_language = QComboBox()
        self._default_language.addItems(language_combo_items())
        self._default_language.setMinimumWidth(200)
        self._default_language.currentTextChanged.connect(
            lambda v: self._save("lyrics.default_language", language_code_from_label(v)))
        appearance_layout.addLayout(SettingRow(
            tr("settings.appearance.default_lyrics_language"),
            self._default_language,
            tr("settings.appearance.default_lyrics_language_help"),
        ))

        self._reduced_motion = QCheckBox(tr("settings.appearance.reduced_motion"))
        self._reduced_motion.toggled.connect(
            lambda v: self._save("general.reduced_motion", v)
        )
        appearance_layout.addLayout(SettingRow(
            tr("settings.appearance.reduced_motion"),
            self._reduced_motion,
            tr("settings.appearance.reduced_motion_help"),
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
        self._lyrics_model.addItem(tr("settings.lyrics.model_recommended"), "llama-3.1-8b-q4")
        self._lyrics_model.addItem(tr("settings.lyrics.model_fast"), "llama-3.2-3b-q4")
        self._lyrics_model.addItem(tr("settings.lyrics.model_premium"), "qwen-2.5-14b-q4")
        self._lyrics_model.setMinimumWidth(240)
        self._lyrics_model.currentIndexChanged.connect(
            lambda: self._save("lyrics.model_id", self._lyrics_model.currentData()))
        lyrics_layout.addLayout(SettingRow(tr("settings.lyrics.model"), self._lyrics_model))

        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.1, 2.0)
        self._temperature.setSingleStep(0.05)
        self._temperature.setMinimumWidth(100)
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
        self._top_p.setMinimumWidth(100)
        self._top_p.valueChanged.connect(
            lambda v: self._save("lyrics.top_p", v))
        lyrics_layout.addLayout(SettingRow(tr("settings.lyrics.top_p"), self._top_p))

        self._max_tokens = QSpinBox()
        self._max_tokens.setRange(256, 8192)
        self._max_tokens.setSingleStep(256)
        self._max_tokens.setMinimumWidth(120)
        self._max_tokens.valueChanged.connect(
            lambda v: self._save("lyrics.max_tokens", v))
        lyrics_layout.addLayout(SettingRow(tr("settings.lyrics.max_tokens"), self._max_tokens))

        layout.addWidget(lyrics_group)

        # ── Song Forge ──
        forge_group = QGroupBox(tr("settings.song_forge.group"))
        forge_layout = QVBoxLayout(forge_group)

        self._timestep_shift = QDoubleSpinBox()
        self._timestep_shift.setRange(1.0, 3.0)
        self._timestep_shift.setSingleStep(1.0)
        self._timestep_shift.setDecimals(1)
        self._timestep_shift.setMinimumWidth(100)
        self._timestep_shift.valueChanged.connect(
            lambda v: self._save("song_forge.timestep_shift", v))
        forge_layout.addLayout(SettingRow(
            tr("settings.song_forge.timestep_shift"),
            self._timestep_shift,
            tr("settings.song_forge.timestep_shift_help"),
        ))

        self._inference_steps = QSpinBox()
        self._inference_steps.setRange(1, 100)
        self._inference_steps.setSingleStep(1)
        self._inference_steps.setMinimumWidth(100)
        self._inference_steps.valueChanged.connect(
            lambda v: self._save("song_forge.inference_steps", v))
        forge_layout.addLayout(SettingRow(
            tr("settings.song_forge.inference_steps"),
            self._inference_steps,
            tr("settings.song_forge.inference_steps_help"),
        ))

        self._batch_count = QSpinBox()
        self._batch_count.setRange(1, 16)
        self._batch_count.setMinimumWidth(80)
        self._batch_count.valueChanged.connect(
            lambda v: self._save("song_forge.batch_count", v))
        forge_layout.addLayout(SettingRow(
            tr("settings.song_forge.batch_count"),
            self._batch_count,
            tr("settings.song_forge.batch_count_help"),
        ))

        self._default_duration = QSpinBox()
        self._default_duration.setRange(10, 600)
        self._default_duration.setSuffix(f" {tr('settings.units.seconds')}")
        self._default_duration.setMinimumWidth(120)
        self._default_duration.valueChanged.connect(
            lambda v: self._save("song_forge.default_duration", v))
        forge_layout.addLayout(SettingRow(tr("settings.song_forge.default_duration"), self._default_duration))

        layout.addWidget(forge_group)

        # ── MIDI Studio ──
        midi_group = QGroupBox(tr("settings.midi.group"))
        midi_layout = QVBoxLayout(midi_group)

        self._default_bpm = QSpinBox()
        self._default_bpm.setRange(40, 300)
        self._default_bpm.setSuffix(f" {tr('settings.units.bpm')}")
        self._default_bpm.setMinimumWidth(120)
        self._default_bpm.valueChanged.connect(
            lambda v: self._save("midi_studio.default_bpm", v))
        midi_layout.addLayout(SettingRow(tr("settings.midi.default_bpm"), self._default_bpm))

        layout.addWidget(midi_group)

        # ── Production ──
        prod_group = QGroupBox(tr("settings.production.group"))
        prod_layout = QVBoxLayout(prod_group)

        self._mastering_target = QComboBox()
        for target in LUFS_TARGETS.values():
            self._mastering_target.addItem(target.label, target.key)
        self._mastering_target.setMinimumWidth(220)
        self._mastering_target.currentIndexChanged.connect(
            lambda: self._save("production.mastering_target", self._mastering_target.currentData()))
        prod_layout.addLayout(SettingRow(tr("settings.production.mastering_target"), self._mastering_target))

        self._auto_eq = QCheckBox(tr("settings.production.auto_eq"))
        self._auto_eq.toggled.connect(
            lambda v: self._save("production.mastering_auto_eq", v))
        prod_layout.addLayout(SettingRow(tr("settings.production.auto_eq_help"), self._auto_eq))

        self._auto_compress = QCheckBox(tr("settings.production.auto_compress"))
        self._auto_compress.toggled.connect(
            lambda v: self._save("production.mastering_auto_compress", v))
        prod_layout.addLayout(SettingRow(tr("settings.production.auto_compress_help"), self._auto_compress))

        layout.addWidget(prod_group)

        # ── Cache ──
        cache_group = QGroupBox(tr("settings.cache.group"))
        cache_layout = QVBoxLayout(cache_group)

        self._max_cache = QDoubleSpinBox()
        self._max_cache.setRange(1.0, 500.0)
        self._max_cache.setSuffix(f" {tr('settings.units.gb')}")
        self._max_cache.setSingleStep(5.0)
        self._max_cache.setMinimumWidth(120)
        self._max_cache.valueChanged.connect(
            lambda v: self._save("general.max_cache_gb", v))
        cache_layout.addLayout(SettingRow(
            tr("settings.cache.max_size"),
            self._max_cache,
            tr("settings.cache.max_size_help"),
        ))

        self._autosave_interval = QSpinBox()
        self._autosave_interval.setRange(10, 600)
        self._autosave_interval.setSuffix(f" {tr('settings.units.seconds')}")
        self._autosave_interval.setMinimumWidth(120)
        self._autosave_interval.valueChanged.connect(
            lambda v: self._save("general.auto_save_interval", v))
        cache_layout.addLayout(SettingRow(tr("settings.cache.autosave_interval"), self._autosave_interval))

        self._autosave_enabled = QCheckBox(tr("settings.cache.autosave_enabled"))
        self._autosave_enabled.toggled.connect(
            lambda v: self._save("general.auto_save_enabled", v))
        cache_layout.addLayout(SettingRow(
            tr("settings.cache.autosave"),
            self._autosave_enabled,
            tr("settings.cache.autosave_help"),
        ))

        self._max_versions = QSpinBox()
        self._max_versions.setRange(1, 200)
        self._max_versions.setMinimumWidth(120)
        self._max_versions.valueChanged.connect(
            lambda v: self._save("general.max_project_versions", v))
        cache_layout.addLayout(SettingRow(
            tr("settings.cache.max_versions"),
            self._max_versions,
            tr("settings.cache.max_versions_help"),
        ))

        layout.addWidget(cache_group)
        layout.addWidget(self._build_recovery_center())

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_recovery_center(self) -> QGroupBox:
        """One screen for every recovery artifact: inspect, preview, clean."""
        group = QGroupBox(tr("settings.recovery.group"))
        layout = QVBoxLayout(group)

        intro = QLabel(
            tr("settings.recovery.intro")
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {Palette.SUBTEXT0}; font-size: 8.25pt;")
        layout.addWidget(intro)

        self._recovery_list = QListWidget()
        self._recovery_list.setMinimumHeight(140)
        layout.addWidget(self._recovery_list)

        self._recovery_status = QLabel("")
        self._recovery_status.setWordWrap(True)
        self._recovery_status.setStyleSheet(
            f"color: {Palette.SUBTEXT0}; font-size: 8.25pt;"
        )
        layout.addWidget(self._recovery_status)

        row = QHBoxLayout()
        self._recovery_refresh_btn = QPushButton(tr("settings.actions.refresh"))
        self._recovery_refresh_btn.setObjectName("secondaryBtn")
        self._recovery_refresh_btn.clicked.connect(self._refresh_recovery_center)

        self._recovery_preview_btn = QPushButton(tr("settings.recovery.preview"))
        self._recovery_preview_btn.setObjectName("secondaryBtn")
        self._recovery_preview_btn.clicked.connect(self._preview_recovery_cleanup)

        self._recovery_clean_btn = QPushButton(tr("settings.recovery.clean"))
        self._recovery_clean_btn.setObjectName("dangerBtn")
        self._recovery_clean_btn.setEnabled(False)
        self._recovery_clean_btn.clicked.connect(self._run_recovery_cleanup)

        self._recovery_reveal_btn = QPushButton(tr("settings.actions.open_config"))
        self._recovery_reveal_btn.setObjectName("secondaryBtn")
        self._recovery_reveal_btn.clicked.connect(self._open_config_dir)

        for button in (
            self._recovery_refresh_btn, self._recovery_preview_btn,
            self._recovery_clean_btn, self._recovery_reveal_btn,
        ):
            button.setMinimumHeight(30)
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
                    tr(
                        "settings.recovery.category_unavailable",
                        category=CATEGORY_LABELS[category],
                        error=exc,
                    )
                )
                continue
            size = sum(item.size_bytes for item in items)
            total_bytes += size
            protected = sum(1 for item in items if item.protected)
            policy = load_policy(category, self._settings)
            self._recovery_list.addItem(
                tr(
                    "settings.recovery.category_summary",
                    category=CATEGORY_LABELS[category],
                    count=len(items),
                    size=f"{size / 1e6:.1f}",
                    protected=protected,
                    policy=_recovery_policy_text(policy),
                )
            )
        self._recovery_status.setText(
            tr("settings.recovery.usage", size=f"{total_bytes / 1e6:.1f}")
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
            tr(
                "settings.recovery.preview_summary",
                count=removable,
                size=f"{freed / 1e6:.1f}",
            )
            if removable else tr("settings.recovery.preview_empty")
        )
        self._recovery_clean_btn.setEnabled(removable > 0)

    def _run_recovery_cleanup(self):
        if self._recovery_worker is not None:
            self._cancel_active_operation()
            return
        center = self._recovery_center()
        self._recovery_clean_btn.setEnabled(False)
        self._recovery_refresh_btn.setEnabled(False)
        self._recovery_preview_btn.setEnabled(False)
        self._operation_progress.start(
            tr("settings.recovery.cleaning"), determinate=True
        )
        worker = InferenceWorker(
            _recovery_cleanup_task,
            center,
            job_kind="recovery_cleanup",
            job_label="Recovery file cleanup",
            job_metadata={"module": "settings"},
        )
        worker.progress.connect(self._on_recovery_cleanup_progress)
        worker.step_info.connect(self._on_recovery_cleanup_step)
        worker.finished.connect(self._on_recovery_cleanup_finished)
        worker.error.connect(self._on_recovery_cleanup_error)
        worker.cancelled.connect(self._on_recovery_cleanup_cancelled)
        self._recovery_worker = worker
        worker.start()

    def _on_recovery_cleanup_progress(self, percent: int):
        self._operation_progress.set_progress(
            percent, tr("settings.recovery.cleaning")
        )

    def _on_recovery_cleanup_step(self, message: str):
        self._operation_progress.set_step(message)
        self._recovery_status.setText(message)

    def _on_recovery_cleanup_finished(self, removed):
        from core.retention import CATEGORY_LABELS

        self._recovery_worker = None
        self._operation_progress.finish()
        lines = [
            f"{CATEGORY_LABELS[category]}: removed {len(items)}"
            for category, items in removed.items() if items
        ]
        total = sum(len(items) for items in removed.values())
        self._refresh_recovery_center()
        self._recovery_status.setText(
            tr(
                "settings.recovery.cleanup_summary",
                count=total,
                details=" | ".join(lines),
            )
            if total else tr("settings.recovery.cleanup_empty")
        )
        if self.toast_mgr:
            self.toast_mgr.success(tr("settings.recovery.cleanup_toast", count=total))

        self._set_recovery_operation_controls(enabled=True)

    def _on_recovery_cleanup_error(self, message: str):
        self._recovery_worker = None
        self._operation_progress.finish()
        self._set_recovery_operation_controls(enabled=True)
        self._recovery_status.setText(f"Recovery cleanup failed: {message}")
        if self.toast_mgr:
            self.toast_mgr.error(f"Recovery cleanup failed: {message}")

    def _on_recovery_cleanup_cancelled(self):
        self._recovery_worker = None
        self._operation_progress.finish()
        self._set_recovery_operation_controls(enabled=True)
        self._refresh_recovery_center()
        self._recovery_status.setText("Recovery cleanup cancelled")

    def _set_recovery_operation_controls(self, *, enabled: bool):
        self._recovery_refresh_btn.setEnabled(enabled)
        self._recovery_preview_btn.setEnabled(enabled)
        if enabled:
            self._recovery_clean_btn.setEnabled(False)

    def _cancel_active_operation(self):
        """Request cancellation for health export or recovery cleanup."""
        worker = self._health_report_worker or self._recovery_worker
        if worker is None:
            self._operation_progress.finish()
            return
        self._operation_progress.mark_cancelling()
        worker.cancel()
        if worker is self._health_report_worker:
            self._export_health_btn.setEnabled(False)
        else:
            self._recovery_status.setText("Cancelling recovery cleanup...")

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
            self._audio_device_combo.addItem(tr("settings.output.system_default"), "")
            for device in devices:
                self._audio_device_combo.addItem(device.label, device.identity)

            selected_index = self._audio_device_combo.findData(saved_identity)
            saved_unavailable = bool(saved_identity and selected_index < 0)
            if saved_unavailable:
                selected_index = self._audio_device_combo.count()
                self._audio_device_combo.addItem(
                    tr(
                        "settings.output.unavailable",
                        identity=format_output_device_identity(saved_identity),
                    ),
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
                tr("settings.output.enumeration_failed", error=error)
            )
        elif saved_unavailable:
            self._set_audio_device_status(
                tr(
                    "settings.output.saved_unavailable",
                    identity=format_output_device_identity(saved_identity),
                )
            )
        elif not devices:
            self._set_audio_device_status(
                tr("settings.output.no_devices")
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
            self._format_combo, self._sample_rate_combo,
            self._stem_export_template_combo, self._gpu_device,
            self._c2pa_enabled, self._c2pa_timestamp_url,
            self._offline_mode, self._hf_token, self._experience_combo,
            self._ui_locale_combo,
            self._default_language,
            self._reduced_motion,
            self._osc_enabled, self._osc_port, self._osc_allow_lan,
            self._osc_allowed_hosts, self._osc_packet_bytes, self._osc_rate,
            self._midi_enabled, self._midi_port_combo,
            self._lyrics_model, self._temperature, self._top_p,
            self._max_tokens, self._timestep_shift, self._inference_steps,
            self._batch_count, self._default_duration, self._default_bpm,
            self._mastering_target, self._auto_eq, self._auto_compress,
            self._max_cache, self._autosave_interval,
            self._autosave_enabled, self._max_versions,
        ]
        _widgets.extend(getattr(self, "_midi_mapping_controls", []))
        for w in _widgets:
            w.blockSignals(True)

        try:
            # Simple tab
            self._output_dir.setText(s.get("general.output_dir", ""))
            self._c2pa_enabled.setChecked(
                bool(s.get("general.c2pa_enabled", False))
            )
            self._c2pa_certificate_path.setText(
                str(s.get("general.c2pa_certificate_path", "") or "")
            )
            self._c2pa_private_key_path.setText(
                str(s.get("general.c2pa_private_key_path", "") or "")
            )
            self._c2pa_timestamp_url.setText(
                str(s.get("general.c2pa_timestamp_url", "") or "")
            )
            fmt = s.get("general.audio_format", "wav").upper()
            idx = self._format_combo.findText(fmt)
            if idx >= 0:
                self._format_combo.setCurrentIndex(idx)

            sr = str(s.get("general.sample_rate", 48000))
            idx = self._sample_rate_combo.findText(sr)
            if idx >= 0:
                self._sample_rate_combo.setCurrentIndex(idx)

            template_id = str(s.get("general.stem_export_template", "generic") or "generic")
            idx = self._stem_export_template_combo.findData(template_id)
            self._stem_export_template_combo.setCurrentIndex(idx if idx >= 0 else 0)

            self._osc_enabled.setChecked(bool(s.get("osc.enabled", False)))
            self._osc_port.setValue(int(s.get("osc.port", 9000) or 9000))
            self._osc_allow_lan.setChecked(bool(s.get("osc.allow_lan", False)))
            allowed_hosts = s.get("osc.allowed_hosts", ["127.0.0.1"])
            if isinstance(allowed_hosts, (list, tuple)):
                allowed_hosts = ", ".join(str(host) for host in allowed_hosts)
            self._osc_allowed_hosts.setText(str(allowed_hosts or ""))
            self._osc_packet_bytes.setValue(
                int(s.get("osc.max_packet_bytes", 4096) or 4096)
            )
            self._osc_rate.setValue(
                int(s.get("osc.max_messages_per_second", 60) or 60)
            )
            self._midi_enabled.setChecked(
                bool(s.get("midi_controller.enabled", False))
            )
            configured_bindings = {
                binding.action: binding
                for binding in normalized_bindings(
                    s.get("midi_controller.bindings", None)
                )
            }
            defaults_by_action = {
                binding.action: binding
                for binding in normalized_bindings(list(DEFAULT_MIDI_BINDINGS))
            }
            for row in self._midi_mapping_rows:
                binding = configured_bindings.get(
                    row["action"], defaults_by_action[row["action"]]
                )
                row["type"].setCurrentIndex(row["type"].findData(binding.message_type))
                row["channel"].setCurrentIndex(row["channel"].findData(binding.channel))
                row["number"].setValue(binding.number)
                row["mode"].setCurrentIndex(row["mode"].findData(binding.mode))

            self._gpu_device.setValue(s.get("general.gpu_device", 0))
            self._offline_mode.setChecked(s.get("model_hub.offline_mode", False))
            self._hf_token.setText(s.get("model_hub.hf_token", ""))

            exp = s.get("general.experience_level", "beginner")
            idx = self._experience_combo.findData(exp)
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
            self._tabs.setCurrentIndex(1 if s.get("general.ui_mode", "simple") == "advanced" else 0)

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

            target = s.get("production.mastering_target", "streaming")
            if target == "spotify":
                target = "streaming"
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
            self._reduced_motion.setChecked(s.get("general.reduced_motion", False))

        finally:
            for w in _widgets:
                w.blockSignals(False)
        self._refresh_audio_devices()
        self._refresh_midi_ports()
        self._refresh_c2pa_state()
        self._update_repair_status()
        self._refresh_credential_status()
        self._refresh_recovery_center()

    def _save(self, key: str, value):
        """Save a setting and show toast."""
        try:
            self._settings.set(key, value)
        except CredentialError as exc:
            if self.toast_mgr:
                self.toast_mgr.error(
                    tr("settings.messages.setting_save_failed", key=key, error=exc)
                )
            self._update_repair_status()
            self._refresh_credential_status()
            return
        self._update_repair_status()
        if key in SECRET_SETTING_KEYS:
            self._refresh_credential_status()
            if self.toast_mgr:
                store = self._settings.credential_store
                self.toast_mgr.success(
                    tr("settings.messages.secret_saved", backend=store.backend_name)
                    if value else tr("settings.messages.secret_cleared")
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

    def _on_stem_export_template_changed(self, index: int):
        """Persist the selected target-DAW stem filename convention."""
        template_id = self._stem_export_template_combo.itemData(index)
        if template_id:
            self._save("general.stem_export_template", str(template_id))

    def _on_midi_port_changed(self, index: int):
        """Persist the selected MIDI input identity, not its display label."""
        if index < 0:
            return
        self._save(
            "midi_controller.port_name",
            str(self._midi_port_combo.itemData(index) or ""),
        )

    def _refresh_midi_ports(self):
        """Refresh optional MIDI input ports without requiring a backend."""
        saved = str(self._settings.get("midi_controller.port_name", "") or "")
        names, status = list_midi_input_ports()
        self._midi_port_combo.blockSignals(True)
        try:
            self._midi_port_combo.clear()
            self._midi_port_combo.addItem("System default", "")
            for name in names:
                self._midi_port_combo.addItem(name, name)
            if saved and self._midi_port_combo.findData(saved) < 0:
                self._midi_port_combo.addItem(f"Missing: {saved}", saved)
            selected = self._midi_port_combo.findData(saved)
            self._midi_port_combo.setCurrentIndex(max(0, selected))
        finally:
            self._midi_port_combo.blockSignals(False)

        if status:
            self._midi_status.setText(status)
        elif not names:
            self._midi_status.setText(
                "No MIDI input ports detected. Connect a device, then refresh."
            )
        else:
            self._midi_status.setText(
                "MIDI input is optional and remains disabled until enabled above."
            )

    def _on_midi_mapping_changed(self, *_args):
        """Persist the complete validated mapping after one row changes."""
        bindings = []
        for row in self._midi_mapping_rows:
            try:
                bindings.append(MidiBinding(
                    action=row["action"],
                    message_type=str(row["type"].currentData()),
                    channel=int(row["channel"].currentData()),
                    number=int(row["number"].value()),
                    mode=str(row["mode"].currentData()),
                ))
            except (TypeError, ValueError):
                return
        self._save("midi_controller.bindings", bindings_to_settings(bindings))

    def _save_osc_allowed_hosts(self):
        """Persist a comma-separated IPv4 host/CIDR allowlist."""
        hosts = [
            host.strip()
            for host in self._osc_allowed_hosts.text().replace(";", ",").split(",")
            if host.strip()
        ]
        self._save("osc.allowed_hosts", hosts)

    def _on_osc_lan_toggled(self, enabled: bool):
        """Persist LAN opt-in and keep the allowlist visibly editable."""
        self._save("osc.allow_lan", bool(enabled))
        self._osc_allowed_hosts.setEnabled(True)

    def _on_ui_locale_changed(self, _index: int):
        """Persist the interface locale and apply layout direction immediately."""
        code = str(self._ui_locale_combo.currentData() or "en")
        selected = set_locale(code, persist=True)
        if self.toast_mgr:
            self.toast_mgr.info(tr("settings.messages.locale_changed", locale=selected))

    def _browse_output_dir(self):
        path = choose_directory(
            self,
            tr("settings.dialogs.select_output_directory"),
            operation_kind="settings_output_directory",
            dialog=QFileDialog,
        )
        if path:
            self._output_dir.setText(path)
            self._save("general.output_dir", path)

    def _on_c2pa_toggled(self, enabled: bool):
        self._save("general.c2pa_enabled", bool(enabled))
        self._refresh_c2pa_state()

    def _on_offline_mode_toggled(self, enabled: bool):
        self._save("model_hub.offline_mode", bool(enabled))
        if hasattr(self, "_c2pa_status"):
            self._refresh_c2pa_state()

    def _refresh_c2pa_state(self):
        """Explain the opt-in state without reading or exposing private keys."""
        if not self._c2pa_enabled.isChecked():
            text = tr("settings.output.c2pa_off")
            color = Palette.SUBTEXT0
        elif self._offline_mode.isChecked() and self._c2pa_timestamp_url.text().strip():
            text = tr("settings.output.c2pa_offline_timestamp")
            color = Palette.YELLOW
        elif not self._c2pa_certificate_path.text().strip() or not self._c2pa_private_key_path.text().strip():
            text = tr("settings.output.c2pa_missing_credentials")
            color = Palette.YELLOW
        elif self._c2pa_timestamp_url.text().strip():
            text = tr("settings.output.c2pa_ready_timestamp")
            color = Palette.YELLOW
        else:
            text = tr("settings.output.c2pa_ready")
            color = Palette.SUBTEXT0
        self._c2pa_status.setText(text)
        self._c2pa_status.setStyleSheet(
            f"color: {color}; font-size: 8.25pt; padding: 2px 0;"
        )

    def _browse_c2pa_file(
        self,
        setting_key: str,
        operation_kind: str,
        title: str,
        file_filter: str,
        field: QLineEdit,
    ):
        path, _selected_filter = open_file(
            self,
            title,
            file_filter,
            operation_kind=operation_kind,
            dialog=QFileDialog,
        )
        if path:
            field.setText(path)
            self._save(setting_key, path)
            self._refresh_c2pa_state()

    def _reset_all(self):
        try:
            snapshot = self._settings.snapshot()
            self._settings.reset_all()
        except Exception as exc:
            if self.toast_mgr:
                self.toast_mgr.error(f"Settings reset failed: {exc}")
            return
        self._load_values()
        self._update_repair_status()
        if self.toast_mgr:
            self.toast_mgr.info(
                tr("settings.messages.reset"),
                duration_ms=8000,
                action_label="Undo",
                action_callback=lambda item=snapshot: self._restore_settings_snapshot(item),
            )

    def _restore_settings_snapshot(self, snapshot: dict):
        """Restore the pre-reset settings, including OS-backed secrets."""
        try:
            self._settings.restore_snapshot(snapshot)
            self._load_values()
            self._update_repair_status()
        except Exception as exc:
            if self.toast_mgr:
                self.toast_mgr.error(f"Settings restore failed: {exc}")
            return
        if self.toast_mgr:
            self.toast_mgr.success(tr("settings.messages.restored"))

    def _open_onboarding(self):
        """Reopen onboarding without changing completion until it is accepted."""
        from ui.onboarding import OnboardingWizard

        wizard = OnboardingWizard(self)
        if wizard.exec() == QDialog.DialogCode.Accepted:
            handoff = wizard.model_handoff()
            window = self.window()
            if handoff and hasattr(window, "open_model_hub_for_onboarding"):
                QTimer.singleShot(
                    0,
                    lambda value=handoff: window.open_model_hub_for_onboarding(
                        value["model_id"], value["action"]
                    ),
                )

    def _refresh_credential_status(self):
        """Name the credential service in use, or state plainly that there is none."""
        status = self._settings.credential_backend_status()
        if status.get("available"):
            text = tr("settings.credentials.available", backend=status.get("name"))
            self._credential_status.setProperty("state", "ok")
        else:
            detail = (status.get("detail") or "").strip()
            lead = tr("settings.credentials.unavailable")
            text = lead if detail in ("", lead) else f"{lead} {detail}"
            self._credential_status.setProperty("state", "warning")
        self._credential_status.setText(text)
        color = Palette.SUBTEXT0 if status.get("available") else Palette.YELLOW
        self._credential_status.setStyleSheet(
            f"color: {color}; font-size: 8.25pt; padding: 2px 0;"
        )

    def _update_repair_status(self):
        status = self._settings.repair_status
        state = status.get("status", "ok")
        if state == "ok":
            self._repair_label.setVisible(False)
            self._repair_label.setText("")
            return

        backups = status.get("backup_paths") or []
        text = tr(
            {
                "migrated": "settings.config.updated",
                "repaired": "settings.config.repaired",
                "error": "settings.config.attention",
            }.get(state, "settings.config.attention")
        )
        if backups:
            text += f" {tr('settings.config.backup_saved')}"
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
                (self._stem_export_template_combo, "Stem export naming", "Selects the filename template used for separated stem exports."),
                (self._audio_device_combo, "Audio output device", "Selects the PortAudio output device and shows its host API."),
                (self._refresh_audio_devices_btn, "Refresh audio output devices", "Refreshes the PortAudio output-device list without restarting."),
                (self._c2pa_enabled, "C2PA Content Credentials", "Opt-in embedding of a signed C2PA manifest on supported audio exports."),
                (self._c2pa_certificate_path, "C2PA certificate path", "Selects the user-managed PEM claim-signing certificate chain."),
                (self._c2pa_certificate_browse, "Browse C2PA certificate", "Selects the C2PA claim-signing certificate chain."),
                (self._c2pa_private_key_path, "C2PA private key path", "Selects the user-managed PEM private key; the key contents are never copied into settings."),
                (self._c2pa_private_key_browse, "Browse C2PA private key", "Selects the C2PA claim-signing private key."),
                (self._c2pa_timestamp_url, "C2PA timestamp URL", "Optional RFC 3161 timestamp authority. A configured URL makes signing a network operation and is blocked by Offline Mode."),
                (self._osc_enabled, "OSC control", "Enables the versioned OSC transport for local control."),
                (self._osc_port, "OSC port", "Selects the UDP port used by OSC control."),
                (self._osc_allow_lan, "Allow OSC LAN access", "Explicitly enables non-loopback OSC sources."),
                (self._osc_allowed_hosts, "OSC allowed hosts", "Limits LAN OSC to the listed IPv4 hosts or CIDR networks."),
                (self._osc_packet_bytes, "OSC packet size limit", "Rejects OSC datagrams larger than this many bytes."),
                (self._osc_rate, "OSC rate limit", "Limits accepted OSC datagrams per source per second."),
                (self._midi_enabled, "MIDI controller input", "Enables the optional local MIDI input service."),
                (self._midi_port_combo, "MIDI input port", "Selects the local MIDI input port."),
                (self._midi_refresh_btn, "Refresh MIDI input ports", "Refreshes the available MIDI input ports."),
                *[
                    (control, f"MIDI binding {row['action']}", "Edits the MIDI message mapped to this action.")
                    for row in self._midi_mapping_rows
                    for control in (row["type"], row["channel"], row["number"], row["mode"])
                ],
                (self._gpu_device, "GPU device index", "Selects the GPU device index."),
                (self._offline_mode, "Offline mode", "Disables internet access for Model Hub."),
                (self._hf_token, "HuggingFace token", "Stores a token for gated model downloads."),
                (self._experience_combo, "Experience level", "Controls default UI complexity."),
                (self._ui_locale_combo, "Interface language", "Selects the interface language and layout direction."),
                (self._default_language, "Default lyrics language", "Sets the default language metadata for lyrics and new voice profiles."),
                (self._reduced_motion, "Reduced motion", "Disables sliding and repositioning animations for notifications."),
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
                (self._health_private_inputs, "Include private task inputs", "Includes task prompts and input fields in the health report."),
                (self._export_health_btn, "Export health report", "Saves a redacted diagnostics bundle."),
                (self._open_dir_btn, "Open config folder", "Opens the settings folder in the file manager."),
                (self._onboarding_btn, "Open onboarding", "Reopens the first-run setup wizard."),
            ],
            tab_order=[
                self._tabs,
                self._output_dir,
                self._browse_output_btn,
                self._format_combo,
                self._sample_rate_combo,
                self._stem_export_template_combo,
                self._audio_device_combo,
                self._refresh_audio_devices_btn,
                self._c2pa_enabled,
                self._c2pa_certificate_path,
                self._c2pa_certificate_browse,
                self._c2pa_private_key_path,
                self._c2pa_private_key_browse,
                self._c2pa_timestamp_url,
                self._osc_enabled,
                self._osc_port,
                self._osc_allow_lan,
                self._osc_allowed_hosts,
                self._osc_packet_bytes,
                self._osc_rate,
                self._midi_enabled,
                self._midi_port_combo,
                self._midi_refresh_btn,
                *self._midi_mapping_controls,
                self._gpu_device,
                self._offline_mode,
                self._hf_token,
                self._experience_combo,
                self._ui_locale_combo,
                self._default_language,
                self._reduced_motion,
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
                self._operation_progress.cancel_button,
                self._open_dir_btn,
                self._onboarding_btn,
            ],
        )

    def _export_health_report(self):
        if self._health_report_worker is not None:
            self._cancel_active_operation()
            return
        path, _selected_filter = save_file(
            self,
            tr("settings.dialogs.export_health"),
            "slunderstudio-health-report.zip",
            "Health Report (*.zip)",
            "health_report_export",
            dialog=QFileDialog,
        )
        if not path:
            return
        self._export_health_btn.setEnabled(False)
        self._operation_progress.start("Exporting health report", determinate=True)
        worker = InferenceWorker(
            export_health_report,
            path,
            include_private=self._health_private_inputs.isChecked(),
            job_kind="health_report_export",
            job_label="Health report export",
            job_metadata={"module": "settings"},
        )
        worker.progress.connect(self._on_health_report_progress)
        worker.step_info.connect(self._on_health_report_step)
        worker.finished.connect(self._on_health_report_finished)
        worker.error.connect(self._on_health_report_error)
        worker.cancelled.connect(self._on_health_report_cancelled)
        self._health_report_worker = worker
        worker.start()

    def _on_health_report_progress(self, percent: int):
        self._operation_progress.set_progress(percent, "Exporting health report")

    def _on_health_report_step(self, message: str):
        self._operation_progress.set_step(message)

    def _on_health_report_finished(self, output):
        self._health_report_worker = None
        self._operation_progress.finish()
        self._export_health_btn.setEnabled(True)
        if self.toast_mgr:
            self.toast_mgr.success(
                tr("settings.messages.health_exported", filename=output.name)
            )

    def _on_health_report_error(self, message: str):
        self._health_report_worker = None
        self._operation_progress.finish()
        self._export_health_btn.setEnabled(True)
        if self.toast_mgr:
            self.toast_mgr.error(
                tr("settings.messages.health_export_failed", error=message)
            )

    def _on_health_report_cancelled(self):
        self._health_report_worker = None
        self._operation_progress.finish()
        self._export_health_btn.setEnabled(True)
        if self.toast_mgr and hasattr(self.toast_mgr, "info"):
            self.toast_mgr.info("Health report export cancelled")

    def _open_config_dir(self):
        import subprocess, sys
        config_dir = str(self._settings._config_path.parent)
        if sys.platform == "win32":
            subprocess.Popen(f'explorer "{config_dir}"')
        elif sys.platform == "darwin":
            subprocess.Popen(["open", config_dir])
        else:
            subprocess.Popen(["xdg-open", config_dir])
