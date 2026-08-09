"""
Slunder Studio — Model Hub UI
Grid view of all models with live download progress, speed tracking,
partial download detection, and one-click download/delete.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QScrollArea, QGridLayout, QPushButton, QProgressBar,
    QLineEdit, QComboBox, QCheckBox, QDialog, QSizePolicy, QStackedWidget,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl

from ui.theme import Palette
from ui.accessibility import install_accessibility, set_accessible
from ui.widgets import EmptyStateWidget
from core.i18n import tr, user_facing_readiness
from core.model_manager import (
    EXECUTABLE_MODEL_WARNING,
    ModelCategory,
    ModelInfo,
    ModelManager,
    ModelSecurityError,
    ModelStatus,
    ModelUpdate,
    OfflineModeError,
    model_hardware_fit,
    model_supports_task,
    model_tasks,
    recommend_model_for_task,
)
from core.credentials import CredentialError
from core.settings import Settings
from core.workers import DownloadWorker, InferenceWorker
from core.job_state import JobStatus, JobStore


_CATEGORY_LABEL_KEYS = {
    ModelCategory.SONG_FORGE: "model_hub_ui.categories.song_forge",
    ModelCategory.LYRICS: "model_hub_ui.categories.lyrics",
    ModelCategory.MIDI: "model_hub_ui.categories.midi",
    ModelCategory.VOCAL: "model_hub_ui.categories.vocal",
    ModelCategory.SEPARATION: "model_hub_ui.categories.separation",
    ModelCategory.SFX: "model_hub_ui.categories.sfx",
    ModelCategory.ALIGNMENT: "model_hub_ui.categories.alignment",
    ModelCategory.EXTRAS: "model_hub_ui.categories.extras",
}


class HFTokenDialog(QDialog):
    """Inline dialog to paste a HuggingFace token for gated model downloads."""

    def __init__(self, model_name: str, repo_id: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("model_hub_ui.dialogs.token_title"))
        self.setMinimumSize(480, 240)
        self.token = ""
        self._build_ui(model_name, repo_id)
        self.resize(480, 240)

    def _build_ui(self, model_name: str, repo_id: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel(
            tr("model_hub_ui.dialogs.token_required", model=model_name)
        )
        title.setStyleSheet(f"font-size: 11.25pt; color: {Palette.TEXT};")
        title.setWordWrap(True)
        layout.addWidget(title)

        link_row = QHBoxLayout()
        link_row.setSpacing(8)
        open_btn = QPushButton(tr("model_hub_ui.dialogs.get_token"))
        open_btn.setMinimumHeight(34)
        open_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://huggingface.co/settings/tokens"))
        )
        link_row.addWidget(open_btn)
        link_row.addStretch()
        layout.addLayout(link_row)

        self._token_input = QLineEdit()
        self._token_input.setPlaceholderText(tr("model_hub_ui.dialogs.token_placeholder"))
        self._token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_input.setMinimumHeight(38)
        layout.addWidget(self._token_input)

        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(f"color: {Palette.RED};")
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton(tr("model_hub_ui.dialogs.cancel"))
        cancel_btn.setMinimumSize(100, 36)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._save_btn = QPushButton(tr("model_hub_ui.dialogs.save_download"))
        self._save_btn.setMinimumSize(160, 36)
        self._save_btn.setObjectName("accentBtn")
        self._save_btn.clicked.connect(self._accept)
        btn_row.addWidget(self._save_btn)
        layout.addLayout(btn_row)

    def _accept(self):
        t = self._token_input.text().strip()
        if t and t.startswith("hf_"):
            self.token = t
            self.accept()
        else:
            self._error_label.setText(tr("model_hub_ui.dialogs.token_error"))
            self._error_label.setVisible(True)
            self._token_input.setStyleSheet(f"border: 1px solid {Palette.RED};")
            self.adjustSize()


class ExecutableModelConsentDialog(QDialog):
    """Explicit review gate for one pinned custom-code or pickle revision."""

    def __init__(self, info: ModelInfo, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("model_hub_ui.dialogs.review_title"))
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel(tr("model_hub_ui.dialogs.review_model", model=info.name))
        title.setStyleSheet(f"font-size: 12pt; font-weight: 700; color: {Palette.TEXT};")
        layout.addWidget(title)

        warning = QLabel(EXECUTABLE_MODEL_WARNING)
        warning.setWordWrap(True)
        warning.setStyleSheet(f"color: {Palette.YELLOW};")
        layout.addWidget(warning)

        details = QLabel(
            tr(
                "model_hub_ui.dialogs.review_details",
                source=info.source,
                revision=info.revision,
                remote_code=tr("model_hub_ui.dialogs.yes" if info.requires_remote_code else "model_hub_ui.dialogs.no"),
                pickle_weights=tr("model_hub_ui.dialogs.yes" if info.allows_unsafe_weights else "model_hub_ui.dialogs.no"),
            )
        )
        details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details.setStyleSheet(f"color: {Palette.SUBTEXT0}; font-family: Consolas;")
        layout.addWidget(details)
        self._details = details

        self._ack = QCheckBox(
            tr("model_hub_ui.dialogs.review_ack")
        )
        layout.addWidget(self._ack)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton(tr("model_hub_ui.dialogs.cancel"))
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self._approve = QPushButton(tr("model_hub_ui.dialogs.allow_revision"))
        self._approve.setObjectName("danger")
        self._approve.setEnabled(False)
        self._approve.clicked.connect(self.accept)
        self._ack.toggled.connect(self._approve.setEnabled)
        buttons.addWidget(self._approve)
        layout.addLayout(buttons)

        install_accessibility(
            self,
            tr("model_hub_ui.accessibility.review_name", model=info.name),
            named_controls=[
                (
                    self._ack,
                    tr("model_hub_ui.accessibility.ack_name"),
                    tr("model_hub_ui.accessibility.ack_description"),
                ),
                (
                    self._approve,
                    tr("model_hub_ui.accessibility.allow_name", model=info.name),
                    tr("model_hub_ui.accessibility.allow_description"),
                ),
            ],
            tab_order=[self._ack, cancel, self._approve],
        )


class ModelCard(QFrame):
    """A single model card with integrated download panel."""

    download_requested = Signal(str)
    cancel_requested = Signal(str)
    delete_requested = Signal(str)
    consent_requested = Signal(str)
    activation_requested = Signal(str)
    activation_cancel_requested = Signal(str)
    deactivation_requested = Signal(str)
    update_requested = Signal(str)
    rollback_requested = Signal(str)

    def __init__(self, info: ModelInfo, parent=None):
        super().__init__(parent)
        self.model_id = info.model_id
        self.info = info
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setMinimumWidth(320)
        self.setMinimumHeight(140)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        # -- Header: name + badge --
        header = QHBoxLayout()
        header.setSpacing(8)

        self._name_label = QLabel(self.info.name)
        self._name_label.setStyleSheet(
            f"font-size: 10.5pt; font-weight: 700; color: "
            f"{Palette.BLUE if self.info.is_core else Palette.TEXT};"
        )
        self._name_label.setWordWrap(True)
        header.addWidget(self._name_label, 1)

        self._core_badge = QLabel(
            tr("model_hub_ui.card.core") if self.info.is_core else ""
        )
        self._core_badge.setVisible(bool(self.info.is_core))
        self._core_badge.setAccessibleName(
            tr("model_hub_ui.card.core_accessible_name", model=self.info.name)
        )
        self._core_badge.setAccessibleDescription(
            tr("model_hub_ui.card.core_description") if self.info.is_core else ""
        )
        self._core_badge.setToolTip(
            tr("model_hub_ui.card.core_tooltip") if self.info.is_core else ""
        )
        self._core_badge.setStyleSheet(
            f"font-size: 8pt; font-weight: 700; color: {Palette.CRUST}; "
            f"background: {Palette.BLUE}; border-radius: 4px; padding: 2px 6px;"
        )
        header.addWidget(self._core_badge)

        self._status_badge = QLabel()
        self._status_badge.setMinimumHeight(22)
        header.addWidget(self._status_badge)
        layout.addLayout(header)

        # -- Description --
        desc = QLabel(self.info.description)
        desc.setWordWrap(True)
        desc.setToolTip(self.info.description)
        desc.setAccessibleName(
            tr("model_hub_ui.accessibility.description_name", model=self.info.name)
        )
        desc.setAccessibleDescription(self.info.description)
        desc.setStyleSheet(f"font-size: 9pt; color: {Palette.SUBTEXT0};")
        layout.addWidget(desc)
        self._description_label = desc

        # -- Stats row --
        stats = QHBoxLayout()
        stats.setSpacing(12)
        stat_texts = [
            tr("model_hub_ui.card.vram", value=self.info.vram_gb),
            tr("model_hub_ui.card.vram_tier", tier=self.info.advertised_vram_tier),
            tr("model_hub_ui.card.disk", value=self.info.disk_gb),
            self.info.license,
        ]
        if self.info.quantization:
            stat_texts.insert(0, self.info.variant_label)
        if self.info.quality_label:
            stat_texts.append(
                tr("model_hub_ui.card.quality", quality=self.info.quality_label)
            )
        if self.info.has_local_benchmark:
            stat_texts.append(
                tr(
                    "model_hub_ui.card.benchmark",
                    value=self.info.benchmark_latency_tokens_per_second,
                )
            )
        for text in stat_texts:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"font-size: 8.25pt; color: {Palette.OVERLAY0};")
            stats.addWidget(lbl)
        if getattr(self.info, "gated", False):
            g = QLabel(tr("model_hub_ui.card.token_required"))
            g.setStyleSheet(f"font-size: 8.25pt; color: {Palette.OVERLAY0};")
            stats.addWidget(g)
        stats.addStretch()
        layout.addLayout(stats)

        task_text = ", ".join(self.info.task_labels) or tr("model_hub_ui.card.no_task_guidance")
        task_label = QLabel(tr("model_hub_ui.card.tasks", tasks=task_text))
        task_label.setWordWrap(True)
        task_label.setToolTip(task_text)
        task_label.setStyleSheet(f"font-size: 8.25pt; color: {Palette.BLUE};")
        layout.addWidget(task_label)
        self._task_label = task_label

        rights = QLabel(
            tr(
                "model_hub_ui.card.rights",
                license=self.info.license,
                commercial=self.info.commercial_use_label,
                access=self.info.access_label,
            )
        )
        rights.setWordWrap(True)
        rights.setStyleSheet(f"font-size: 8.25pt; color: {Palette.SUBTEXT0};")
        layout.addWidget(rights)
        self._rights_label = rights

        warning_text = self.info.license_warning
        if warning_text:
            warning = QLabel(warning_text)
            warning.setWordWrap(True)
            warning.setStyleSheet(f"font-size: 7.5pt; color: {Palette.PEACH};")
            layout.addWidget(warning)
            self._license_warning = warning
        else:
            self._license_warning = None

        trust_status = (
            tr("model_hub_ui.card.trust_reviewed")
            if self.info.trusted_source
            else tr("model_hub_ui.card.trust_untrusted")
        )
        if self.info.revision:
            trust_text = (
                tr(
                    "model_hub_ui.card.trust_pinned",
                    revision=self.info.revision[:12],
                    status=trust_status,
                )
            )
        elif self.info.pip_managed:
            trust_text = (
                tr("model_hub_ui.card.trust_package", status=trust_status)
            )
        else:
            trust_text = tr("model_hub_ui.card.trust_unpinned", status=trust_status)
        self._base_trust_text = trust_text
        trust = QLabel(trust_text)
        trust.setWordWrap(True)
        trust.setStyleSheet(f"font-size: 7.5pt; color: {Palette.SUBTEXT0};")
        layout.addWidget(trust)
        self._trust_label = trust
        self._refresh_signature_label()

        self._update_label = QLabel("")
        self._update_label.setWordWrap(True)
        self._update_label.setVisible(False)
        self._update_label.setStyleSheet(
            f"background: rgba(137, 180, 250, 24); color: {Palette.BLUE}; "
            "border: 1px solid rgba(137, 180, 250, 70); border-radius: 5px; "
            "padding: 5px 7px; font-size: 7.75pt;"
        )
        layout.addWidget(self._update_label)

        measurement_text = self.info.measurement_basis or tr("model_hub_ui.card.no_measurement")
        if self.info.benchmark_method:
            measurement_text = f"{measurement_text} {self.info.benchmark_method}"
        measurement_date = self.info.measurement_date or tr("model_hub_ui.card.undated")
        measurement = QLabel(
            tr(
                "model_hub_ui.card.measurement",
                date=measurement_date,
                basis=measurement_text,
            )
        )
        measurement.setWordWrap(True)
        measurement.setToolTip(
            tr(
                "model_hub_ui.card.measurement_source",
                source=self.info.measurement_source or tr("model_hub_ui.card.not_recorded"),
            )
        )
        measurement.setStyleSheet(f"font-size: 7.5pt; color: {Palette.SUBTEXT0};")
        layout.addWidget(measurement)
        self._measurement_label = measurement

        self._hardware_label = QLabel(tr("model_hub_ui.card.hardware_checking"))
        self._hardware_label.setWordWrap(True)
        self._hardware_label.setStyleSheet(f"font-size: 7.5pt; color: {Palette.YELLOW};")
        layout.addWidget(self._hardware_label)

        # -- Download panel (hidden by default, expands inline) --
        self._dl_panel = QFrame()
        self._dl_panel.setVisible(False)
        dl_layout = QVBoxLayout(self._dl_panel)
        dl_layout.setContentsMargins(0, 6, 0, 2)
        dl_layout.setSpacing(4)

        # Progress bar — gradient fill, rounded
        self._progress = QProgressBar()
        self._progress.setMinimumHeight(14)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {Palette.SURFACE0};
                border: none;
                border-radius: 7px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {Palette.BLUE}, stop:1 {Palette.MAUVE}
                );
                border-radius: 7px;
            }}
        """)
        dl_layout.addWidget(self._progress)

        # Info row: percent | size | speed | cancel btn
        info_row = QHBoxLayout()
        info_row.setSpacing(8)

        self._pct_label = QLabel("0%")
        self._pct_label.setStyleSheet(
            f"font-size: 9pt; font-weight: 700; color: {Palette.BLUE};"
        )
        self._pct_label.setMinimumWidth(40)
        info_row.addWidget(self._pct_label)

        self._size_label = QLabel("")
        self._size_label.setStyleSheet(
            f"font-size: 8.25pt; color: {Palette.SUBTEXT0};"
        )
        info_row.addWidget(self._size_label)

        info_row.addStretch()

        self._speed_label = QLabel("")
        self._speed_label.setStyleSheet(
            f"font-size: 8.25pt; color: {Palette.OVERLAY0};"
        )
        info_row.addWidget(self._speed_label)

        self._cancel_btn = QPushButton(tr("model_hub_ui.card.cancel"))
        self._cancel_btn.setMinimumSize(60, 24)
        self._cancel_btn.setStyleSheet(f"""
            QPushButton {{
                font-size: 8.25pt; padding: 2px 8px;
                background: {Palette.SURFACE1}; color: {Palette.RED};
                border: 1px solid {Palette.SURFACE2}; border-radius: 4px;
            }}
            QPushButton:hover {{ background: {Palette.SURFACE2}; }}
        """)
        self._cancel_btn.clicked.connect(
            lambda: self.cancel_requested.emit(self.model_id)
        )
        info_row.addWidget(self._cancel_btn)

        dl_layout.addLayout(info_row)
        layout.addWidget(self._dl_panel)

        # -- Action button --
        self._action_btn = QPushButton(tr("model_hub_ui.card.download"))
        self._action_btn.setMinimumHeight(32)
        self._action_btn.clicked.connect(self._on_action)

        self._delete_btn = QPushButton(tr("model_hub_ui.card.remove"))
        self._delete_btn.setMinimumHeight(32)
        self._delete_btn.setVisible(False)
        self._delete_btn.clicked.connect(
            lambda: self.delete_requested.emit(self.model_id)
        )

        self._update_btn = QPushButton(tr("model_hub_ui.card.install_update"))
        self._update_btn.setVisible(False)
        self._update_btn.clicked.connect(
            lambda: self.update_requested.emit(self.model_id)
        )

        self._rollback_btn = QPushButton(tr("model_hub_ui.card.rollback"))
        self._rollback_btn.setVisible(False)
        self._rollback_btn.clicked.connect(
            lambda: self.rollback_requested.emit(self.model_id)
        )

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        action_row.addWidget(self._action_btn, 1)
        action_row.addWidget(self._delete_btn)
        layout.addLayout(action_row)

        update_row = QHBoxLayout()
        update_row.setSpacing(6)
        update_row.addWidget(self._update_btn, 1)
        update_row.addWidget(self._rollback_btn)
        layout.addLayout(update_row)

        self._consent_btn = QPushButton(tr("model_hub_ui.card.review_executable"))
        self._consent_btn.setVisible(False)
        self._consent_btn.clicked.connect(
            lambda: self.consent_requested.emit(self.model_id)
        )
        layout.addWidget(self._consent_btn)
        install_accessibility(
            self,
            tr("model_hub_ui.accessibility.card_name", model=self.info.name),
            named_controls=[
                (self._status_badge, tr("model_hub_ui.accessibility.status_name", model=self.info.name), tr("model_hub_ui.accessibility.status_description")),
                (self._rights_label, tr("model_hub_ui.accessibility.rights_name", model=self.info.name), tr("model_hub_ui.accessibility.rights_description")),
                (self._progress, tr("model_hub_ui.accessibility.progress_name", model=self.info.name), tr("model_hub_ui.accessibility.progress_description")),
                (self._cancel_btn, tr("model_hub_ui.accessibility.cancel_name", model=self.info.name), tr("model_hub_ui.accessibility.cancel_description")),
                (self._action_btn, tr("model_hub_ui.accessibility.action_name", model=self.info.name), tr("model_hub_ui.accessibility.action_description")),
                (self._delete_btn, tr("model_hub_ui.accessibility.remove_name", model=self.info.name), tr("model_hub_ui.accessibility.remove_description")),
                (self._update_btn, tr("model_hub_ui.accessibility.update_name", model=self.info.name), tr("model_hub_ui.accessibility.update_description")),
                (self._rollback_btn, tr("model_hub_ui.accessibility.rollback_name", model=self.info.name), tr("model_hub_ui.accessibility.rollback_description")),
                (
                    self._consent_btn,
                    tr("model_hub_ui.accessibility.review_name", model=self.info.name),
                    tr("model_hub_ui.accessibility.review_description"),
                ),
            ],
            tab_order=[
                self._action_btn,
                self._delete_btn,
                self._update_btn,
                self._rollback_btn,
                self._consent_btn,
                self._cancel_btn,
            ],
        )

        self._model_update: ModelUpdate | None = None

    def _refresh_signature_label(self):
        """Keep the card's trust copy explicit about OMS signature state."""
        metadata = ModelManager().get_model_signature_metadata(self.model_id)
        label = metadata.get("label", tr("model_hub_ui.card.oms_unsigned"))
        self._trust_label.setText(f"{self._base_trust_text} - {label}")
        self._trust_label.setToolTip(metadata.get("signature_reason", ""))

    def _on_action(self):
        mgr = ModelManager()
        status = mgr.get_status(self.model_id)
        if status in (
            ModelStatus.NOT_DOWNLOADED, ModelStatus.ERROR, ModelStatus.PARTIAL
        ):
            readiness = mgr.get_model_readiness(self.model_id)
            if self.info.pip_managed or readiness.installed:
                self.activation_requested.emit(self.model_id)
            else:
                self.download_requested.emit(self.model_id)
        elif status == ModelStatus.DOWNLOADED:
            self.activation_requested.emit(self.model_id)
        elif status == ModelStatus.LOADED:
            self.deactivation_requested.emit(self.model_id)
        elif status == ModelStatus.LOADING:
            self.activation_cancel_requested.emit(self.model_id)

    def update_status(self, status: ModelStatus):
        """Update the card visual state based on model status."""
        btn = self._action_btn
        readiness = ModelManager().get_model_readiness(self.model_id)
        self._refresh_signature_label()
        self._delete_btn.setVisible(False)
        self._delete_btn.setEnabled(False)

        if status == ModelStatus.NOT_DOWNLOADED:
            self._set_badge(
                tr("model_hub_ui.status.engine_missing" if self.info.pip_managed else "model_hub_ui.status.not_downloaded"),
                Palette.OVERLAY0,
            )
            btn.setText(
                tr("model_hub_ui.card.install_activate" if self.info.pip_managed else "model_hub_ui.card.download")
            )
            btn.setEnabled(True)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)
            btn.setToolTip(
                user_facing_readiness(readiness, model_name=self.info.name)
            )

        elif status == ModelStatus.PARTIAL:
            self._set_badge(tr("model_hub_ui.status.incomplete"), Palette.PEACH)
            btn.setText(tr("model_hub_ui.card.resume_download"))
            btn.setEnabled(True)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)

        elif status == ModelStatus.DOWNLOADING:
            self._set_badge(tr("model_hub_ui.status.downloading"), Palette.BLUE)
            btn.setVisible(False)
            self._dl_panel.setVisible(True)
            self._cancel_btn.setText(tr("model_hub_ui.card.cancel"))
            self._cancel_btn.setEnabled(True)
            self._progress.setValue(0)
            self._pct_label.setText("0%")
            self._size_label.setText(tr("model_hub_ui.card.starting"))
            self._speed_label.setText("")

        elif status == ModelStatus.DOWNLOADED:
            self._set_badge(
                tr("model_hub_ui.status.runtime_missing" if readiness.missing_packages else "model_hub_ui.status.installed"),
                Palette.PEACH if readiness.missing_packages else Palette.GREEN,
            )
            btn.setText(
                tr("model_hub_ui.card.install_activate")
                if readiness.missing_packages
                else tr("model_hub_ui.card.activate")
            )
            btn.setEnabled(True)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)
            btn.setToolTip(
                user_facing_readiness(readiness, model_name=self.info.name)
                if readiness.missing_packages
                else tr("model_hub_ui.card.verify_activate", model=self.info.name)
            )
            self._delete_btn.setVisible(not self.info.pip_managed)
            self._delete_btn.setEnabled(not self.info.pip_managed)

        elif status == ModelStatus.LOADED:
            self._set_badge(tr("model_hub_ui.status.active"), Palette.BLUE)
            btn.setText(tr("model_hub_ui.card.deactivate"))
            btn.setEnabled(True)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)
            btn.setToolTip(tr("model_hub_ui.card.release", model=self.info.name))

        elif status == ModelStatus.LOADING:
            self._set_badge(tr("model_hub_ui.status.activating"), Palette.YELLOW)
            btn.setText(tr("model_hub_ui.card.cancel_activation"))
            btn.setEnabled(True)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)

        elif status == ModelStatus.ERROR:
            self._set_badge(tr("model_hub_ui.status.error"), Palette.RED)
            btn.setText(
                tr("model_hub_ui.card.retry_activation")
                if readiness.installed or self.info.pip_managed
                else tr("model_hub_ui.card.retry_download")
            )
            btn.setEnabled(True)
            btn.setVisible(True)
            self._dl_panel.setVisible(False)
            btn.setToolTip(
                user_facing_readiness(readiness, model_name=self.info.name)
            )
            self._delete_btn.setVisible(
                readiness.installed and not self.info.pip_managed
            )
            self._delete_btn.setEnabled(self._delete_btn.isVisible())

        requires_consent = bool(
            self.info.requires_remote_code or self.info.allows_unsafe_weights
        )
        consent_visible = requires_consent and status in {
            ModelStatus.DOWNLOADED,
            ModelStatus.LOADED,
        }
        self._consent_btn.setVisible(consent_visible)
        if consent_visible:
            approved = ModelManager().has_executable_model_consent(self.model_id)
            self._consent_btn.setText(
                tr("model_hub_ui.card.approved_for", revision=self.info.revision[:12])
                if approved
                else tr("model_hub_ui.card.review_executable")
            )
            self._consent_btn.setEnabled(not approved)
        self._refresh_update_controls(status)

    def set_model_update(self, update: ModelUpdate | None):
        """Show the checked target and its explicit upstream release notes."""
        self._model_update = update
        if update is not None and update.available:
            notes = " ".join(update.release_notes[:2])
            self._update_label.setText(
                tr(
                    "model_hub_ui.update.available",
                    revision=update.target_revision[:12],
                    notes=notes,
                )
            )
            self._update_label.setToolTip(
                "\n".join(update.release_notes) + f"\n\n{update.source_url}"
            )
            set_accessible(
                self._update_label,
                tr("model_hub_ui.accessibility.update_available_name", model=self.info.name),
                self._update_label.text(),
            )
        elif update is not None and update.error:
            self._update_label.setText(
                tr("model_hub_ui.update.check_error", error=update.error)
            )
            self._update_label.setToolTip(update.error)
            set_accessible(
                self._update_label,
                tr("model_hub_ui.accessibility.update_status_name", model=self.info.name),
                update.error,
            )
        self._refresh_update_controls(ModelManager().get_status(self.model_id))

    def _refresh_update_controls(self, status: ModelStatus):
        """Keep update actions consistent with lifecycle state and rollback availability."""
        update = self._model_update
        available = bool(update is not None and update.available)
        can_update = available and status == ModelStatus.DOWNLOADED
        rollback = ModelManager().get_model_rollback(self.model_id)
        can_rollback = bool(rollback) and status in {
            ModelStatus.DOWNLOADED,
            ModelStatus.LOADED,
        }
        self._update_label.setVisible(available or bool(update and update.error))
        self._update_btn.setVisible(can_update)
        self._update_btn.setEnabled(can_update)
        self._rollback_btn.setVisible(can_rollback)
        self._rollback_btn.setEnabled(can_rollback)

    def _set_badge(self, text: str, color: str):
        self._status_badge.setText(text)
        set_accessible(
            self._status_badge,
            tr("model_hub_ui.accessibility.status_value_name", model=self.info.name, status=text),
            tr("model_hub_ui.accessibility.status_value_description", status=text),
        )
        self._status_badge.setStyleSheet(
            f"background: rgba({self._hex_to_rgba(color)},40); "
            f"color: {color}; padding: 2px 10px; border-radius: 11px; "
            f"font-size: 8.25pt; font-weight: 600;"
        )

    @staticmethod
    def _hex_to_rgba(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        if len(h) == 6:
            return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"
        return "128,128,128"

    def update_download_progress(self, pct: int, size_text: str = "",
                                 speed_text: str = ""):
        """Update the download panel with live metrics."""
        self._progress.setValue(pct)
        self._pct_label.setText(f"{pct}%")
        if size_text:
            self._size_label.setText(size_text)
        if speed_text:
            self._speed_label.setText(speed_text)

    def set_progress(self, pct: int):
        """Legacy compat — route through new method."""
        self.update_download_progress(pct)

    def set_download_stopping(self):
        """Show that cancellation was requested while the worker drains."""
        self._set_badge(tr("model_hub_ui.status.stopping"), Palette.YELLOW)
        self._dl_panel.setVisible(True)
        self._cancel_btn.setText(tr("model_hub_ui.status.stopping"))
        self._cancel_btn.setEnabled(False)
        self._size_label.setText(tr("model_hub_ui.card.finishing_transfer"))

    def update_hardware_status(self, hardware: dict):
        """Show whether this model fits the currently detected execution tier."""
        fit = model_hardware_fit(self.info, hardware)
        if fit.status in {"cuda", "mps", "cpu"}:
            color = Palette.GREEN
            prefix = tr("model_hub_ui.hardware.fits")
        elif fit.status == "cpu-fallback":
            color = Palette.YELLOW
            prefix = tr("model_hub_ui.hardware.cpu_fallback")
        else:
            color = Palette.RED
            prefix = tr("model_hub_ui.hardware.unavailable")
        self._hardware_label.setText(
            tr(
                "model_hub_ui.hardware.summary",
                prefix=prefix,
                reason=fit.reason,
                tier=fit.tier,
            )
        )
        self._hardware_label.setStyleSheet(f"font-size: 7.5pt; color: {color};")
        set_accessible(
            self._hardware_label,
            tr("model_hub_ui.accessibility.hardware_name", model=self.info.name),
            self._hardware_label.text(),
        )


class ModelHubView(QWidget):
    """Model Hub page with grid of model cards, search/filter, and disk usage."""

    def __init__(self, toast_mgr=None, parent=None):
        super().__init__(parent)
        self.toast_mgr = toast_mgr
        self._cards: dict[str, ModelCard] = {}
        self._workers: dict[str, DownloadWorker] = {}
        self._stopping_downloads: set[str] = set()
        self._activation_workers: dict[str, InferenceWorker] = {}
        self._update_worker: InferenceWorker | None = None
        self._update_workers: dict[str, InferenceWorker] = {}
        self._mgr = ModelManager()
        self._hardware_profile = {
            "available": False,
            "backend": "cpu",
            "name": tr("model_hub_ui.hardware.detecting"),
            "total_gb": 0,
        }
        self._job_store = JobStore()
        self._build_ui()
        self._connect_signals()
        # Only the initial page load may classify active records as interrupted.
        self._job_store.recover_stale_jobs()
        self._refresh_all_cards()
        self._filter_cards()
        # Let the shell paint before the first GPU probe touches torch.
        QTimer.singleShot(0, self._update_gpu_display)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        subtitle = QLabel(tr("model_hub_ui.subtitle"))
        subtitle.setObjectName("caption")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self._recovery_label = QLabel("")
        self._recovery_label.setWordWrap(True)
        self._recovery_label.setVisible(False)
        self._recovery_label.setStyleSheet(
            f"background: rgba(249, 226, 175, 28); color: {Palette.YELLOW}; "
            f"border: 1px solid rgba(249, 226, 175, 70); border-radius: 6px; "
            "padding: 8px 10px; font-size: 9pt;"
        )
        layout.addWidget(self._recovery_label)

        # GPU status bar
        self._gpu_bar = QFrame()
        self._gpu_bar.setObjectName("accentCard")
        gpu_layout = QHBoxLayout(self._gpu_bar)
        gpu_layout.setContentsMargins(14, 10, 14, 10)

        self._gpu_label = QLabel(tr("model_hub_ui.gpu_detecting"))
        self._gpu_label.setStyleSheet(
            f"font-size: 9.75pt; font-weight: 600; color: {Palette.BLUE};"
        )
        gpu_layout.addWidget(self._gpu_label)
        gpu_layout.addStretch()

        self._disk_label = QLabel(tr("model_hub_ui.disk_calculating"))
        self._disk_label.setStyleSheet(
            f"font-size: 9pt; color: {Palette.SUBTEXT0};"
        )
        gpu_layout.addWidget(self._disk_label)
        layout.addWidget(self._gpu_bar)

        self._recommendation_label = QLabel(tr("model_hub_ui.recommendations_pending"))
        self._recommendation_label.setWordWrap(True)
        self._recommendation_label.setStyleSheet(
            f"background: rgba(137, 180, 250, 24); color: {Palette.BLUE}; "
            "border: 1px solid rgba(137, 180, 250, 70); border-radius: 6px; "
            "padding: 8px 10px; font-size: 9pt;"
        )
        layout.addWidget(self._recommendation_label)

        # Filter bar
        filter_bar = QHBoxLayout()
        filter_bar.setSpacing(12)

        self._search = QLineEdit()
        self._search.setPlaceholderText(tr("model_hub_ui.search_placeholder"))
        self._search.setMinimumHeight(36)
        self._search.textChanged.connect(self._filter_cards)
        filter_bar.addWidget(self._search, 1)

        self._category_filter = QComboBox()
        self._category_filter.addItem(tr("model_hub_ui.filters.all_categories"), "all")
        for cat in ModelCategory:
            self._category_filter.addItem(
                tr(_CATEGORY_LABEL_KEYS.get(cat, "model_hub_ui.categories.unknown")),
                cat.value,
            )
        self._category_filter.setMinimumHeight(36)
        self._category_filter.currentIndexChanged.connect(self._filter_cards)
        filter_bar.addWidget(self._category_filter)

        self._task_filter = QComboBox()
        self._task_filter.addItem(tr("model_hub_ui.filters.all_tasks"), "all")
        for task in model_tasks(self._mgr.registry):
            # Task identifiers are registry taxonomy data; filtering keeps their raw value.
            self._task_filter.addItem(task.title().replace("Vram", "VRAM"), task)
        self._task_filter.setMinimumHeight(36)
        self._task_filter.currentIndexChanged.connect(self._filter_cards)
        filter_bar.addWidget(self._task_filter)

        self._sort_combo = QComboBox()
        self._sort_combo.addItem(tr("model_hub_ui.filters.name_asc"), "name_asc")
        self._sort_combo.addItem(tr("model_hub_ui.filters.name_desc"), "name_desc")
        self._sort_combo.addItem(tr("model_hub_ui.filters.date_newest"), "date_desc")
        self._sort_combo.addItem(tr("model_hub_ui.filters.date_oldest"), "date_asc")
        self._sort_combo.setMinimumHeight(36)
        self._sort_combo.currentIndexChanged.connect(self._filter_cards)
        filter_bar.addWidget(self._sort_combo)

        layout.addLayout(filter_bar)

        hardware_filter_bar = QHBoxLayout()
        hardware_filter_bar.setSpacing(12)
        self._hardware_filter = QComboBox()
        self._hardware_filter.addItem(tr("model_hub_ui.filters.fits_hardware"), "fit")
        self._hardware_filter.addItem(tr("model_hub_ui.filters.all_models"), "all")
        self._hardware_filter.setMinimumHeight(36)
        self._hardware_filter.currentIndexChanged.connect(self._filter_cards)
        hardware_filter_bar.addWidget(self._hardware_filter)

        self._downloaded_only = QCheckBox(tr("model_hub_ui.filters.downloaded_only"))
        self._downloaded_only.stateChanged.connect(self._filter_cards)
        hardware_filter_bar.addWidget(self._downloaded_only)
        hardware_filter_bar.addStretch()
        layout.addLayout(hardware_filter_bar)

        update_bar = QHBoxLayout()
        update_bar.setSpacing(10)
        self._update_status_label = QLabel(
            tr("model_hub_ui.updates.manual_only")
        )
        self._update_status_label.setWordWrap(True)
        self._update_status_label.setStyleSheet(
            f"font-size: 8.5pt; color: {Palette.SUBTEXT0};"
        )
        update_bar.addWidget(self._update_status_label, 1)
        self._check_updates_btn = QPushButton(tr("model_hub_ui.updates.check"))
        self._check_updates_btn.setMinimumHeight(34)
        self._check_updates_btn.clicked.connect(self._start_update_check)
        update_bar.addWidget(self._check_updates_btn)
        layout.addLayout(update_bar)

        # Scrollable grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(16)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        for column in range(3):
            self._grid_layout.setColumnStretch(column, 1)
            self._grid_layout.setColumnMinimumWidth(column, 320)

        col = 0
        row = 0
        for model_id, info in self._mgr.registry.items():
            card = ModelCard(info)
            card.download_requested.connect(self._start_download)
            card.cancel_requested.connect(self._cancel_download)
            card.delete_requested.connect(self._delete_model)
            card.consent_requested.connect(self._review_execution_consent)
            card.activation_requested.connect(self._start_activation)
            card.activation_cancel_requested.connect(self._cancel_activation)
            card.deactivation_requested.connect(self._deactivate_model)
            card.update_requested.connect(self._start_model_update)
            card.rollback_requested.connect(self._start_model_rollback)
            self._cards[model_id] = card
            self._grid_layout.addWidget(card, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1

        self._grid_layout.setRowStretch(row + 1, 1)
        self._grid_empty = EmptyStateWidget(
            tr("model_hub_ui.empty.no_matches_title"),
            tr("model_hub_ui.empty.no_matches_description"),
            tr("model_hub_ui.empty.clear_filters"),
        )
        self._grid_empty.action_requested.connect(self._clear_filters)
        self._grid_stack = QStackedWidget()
        self._grid_stack.addWidget(self._grid_container)
        self._grid_stack.addWidget(self._grid_empty)
        scroll.setWidget(self._grid_stack)
        layout.addWidget(scroll, 1)
        install_accessibility(
            self,
            tr("model_hub_ui.accessibility.name"),
            named_controls=[
                (self._search, tr("model_hub_ui.accessibility.search_name"), tr("model_hub_ui.accessibility.search_description")),
                (self._category_filter, tr("model_hub_ui.accessibility.category_name"), tr("model_hub_ui.accessibility.category_description")),
                (self._task_filter, tr("model_hub_ui.accessibility.task_name"), tr("model_hub_ui.accessibility.task_description")),
                (self._sort_combo, tr("model_hub_ui.accessibility.sort_name"), tr("model_hub_ui.accessibility.sort_description")),
                (self._hardware_filter, tr("model_hub_ui.accessibility.hardware_filter_name"), tr("model_hub_ui.accessibility.hardware_filter_description")),
                (self._downloaded_only, tr("model_hub_ui.accessibility.downloaded_name"), tr("model_hub_ui.accessibility.downloaded_description")),
                (self._gpu_label, tr("model_hub_ui.accessibility.gpu_name"), tr("model_hub_ui.accessibility.gpu_description")),
                (self._disk_label, tr("model_hub_ui.accessibility.disk_name"), tr("model_hub_ui.accessibility.disk_description")),
                (self._recommendation_label, tr("model_hub_ui.accessibility.recommendation_name"), tr("model_hub_ui.accessibility.recommendation_description")),
                (self._update_status_label, tr("model_hub_ui.accessibility.update_status_name_generic"), tr("model_hub_ui.accessibility.update_status_description")),
                (self._check_updates_btn, tr("model_hub_ui.accessibility.check_updates_name"), tr("model_hub_ui.accessibility.check_updates_description")),
            ],
            tab_order=[
                self._search,
                self._category_filter,
                self._task_filter,
                self._sort_combo,
                self._hardware_filter,
                self._downloaded_only,
                self._check_updates_btn,
                *[card._action_btn for card in self._cards.values()],
            ],
        )

    def _connect_signals(self):
        self._mgr.status_changed.connect(self._on_status_changed)
        self._mgr.gpu_status_changed.connect(self._on_gpu_changed)

    def _start_update_check(self):
        """Check upstream revisions off the UI thread; never poll on page load."""
        if self._update_worker is not None:
            return
        self._check_updates_btn.setEnabled(False)
        self._update_status_label.setText(tr("model_hub_ui.updates.checking"))
        worker = InferenceWorker(
            self._mgr.check_for_updates,
            job_kind="model_update_check",
            job_label=tr("model_hub_ui.jobs.update_check"),
            job_store=self._job_store,
            job_metadata={"module": "model_hub"},
        )
        worker.finished.connect(self._on_update_check_finished)
        worker.error.connect(self._on_update_check_error)
        worker.cancelled.connect(self._on_update_check_cancelled)
        self._update_worker = worker
        worker.start()

    def _finish_update_check(self):
        self._update_worker = None
        self._check_updates_btn.setEnabled(True)

    def _on_update_check_finished(self, results):
        self._finish_update_check()
        available = 0
        for model_id, update in (results.items() if isinstance(results, dict) else ()):
            if model_id in self._cards:
                self._cards[model_id].set_model_update(update)
            if getattr(update, "available", False):
                available += 1
        self._update_status_label.setText(
            tr("model_hub_ui.updates.complete", count=available)
        )
        if self.toast_mgr:
            if available:
                self.toast_mgr.info(
                    tr("model_hub_ui.updates.available_toast", count=available)
                )
            else:
                self.toast_mgr.success(tr("model_hub_ui.updates.up_to_date"))

    def _on_update_check_error(self, error: str):
        self._finish_update_check()
        self._update_status_label.setText(tr("model_hub_ui.updates.failed", error=error))
        if self.toast_mgr:
            self.toast_mgr.error(tr("model_hub_ui.updates.failed", error=error))

    def _on_update_check_cancelled(self):
        self._finish_update_check()
        self._update_status_label.setText(tr("model_hub_ui.updates.cancelled"))

    def _start_model_update(self, model_id: str):
        """Install a checked revision through the persistent worker contract."""
        if model_id in self._update_workers:
            return
        update = self._mgr.get_model_update(model_id)
        if update is None or not update.available:
            return
        info = self._mgr.get_model_info(model_id)
        if info is None:
            return
        worker = InferenceWorker(
            self._mgr.install_model_update,
            model_id,
            job_kind="model_update",
            job_label=tr("model_hub_ui.jobs.update", model=info.name),
            job_inputs={"model_id": model_id, "target_revision": update.target_revision},
            job_metadata={"model_id": model_id, "target_revision": update.target_revision},
            job_store=self._job_store,
        )
        worker.finished.connect(
            lambda result, mid=model_id: self._on_model_update_finished(mid, result)
        )
        worker.error.connect(
            lambda error, mid=model_id: self._on_model_update_error(mid, error)
        )
        worker.cancelled.connect(
            lambda mid=model_id: self._on_model_update_cancelled(mid)
        )
        self._update_workers[model_id] = worker
        self._cards[model_id].set_model_update(update)
        self._update_status_label.setText(
            tr("model_hub_ui.updates.installing", model=info.name)
        )
        worker.start()

    def _on_model_update_finished(self, model_id: str, result):
        self._update_workers.pop(model_id, None)
        if model_id in self._cards:
            self._cards[model_id].set_model_update(self._mgr.get_model_update(model_id))
            self._cards[model_id].update_status(self._mgr.get_status(model_id))
        if self.toast_mgr:
            self.toast_mgr.success(
                tr(
                    "model_hub_ui.updates.installed_toast",
                    model=self._mgr.get_model_info(model_id).name,
                )
            )
        self._update_status_label.setText(tr("model_hub_ui.updates.installed"))

    def _on_model_update_error(self, model_id: str, error: str):
        self._update_workers.pop(model_id, None)
        if model_id in self._cards:
            self._cards[model_id].update_status(self._mgr.get_status(model_id))
        self._update_status_label.setText(tr("model_hub_ui.updates.install_failed", error=error))
        if self.toast_mgr:
            self.toast_mgr.error(tr("model_hub_ui.updates.install_failed", error=error))

    def _on_model_update_cancelled(self, model_id: str):
        self._update_workers.pop(model_id, None)
        if model_id in self._cards:
            self._cards[model_id].update_status(self._mgr.get_status(model_id))
        self._update_status_label.setText(tr("model_hub_ui.updates.install_cancelled"))

    def _start_model_rollback(self, model_id: str):
        """Restore a retained last-good revision through a cancellable job."""
        if model_id in self._update_workers:
            return
        info = self._mgr.get_model_info(model_id)
        if info is None or self._mgr.get_model_rollback(model_id) is None:
            return
        worker = InferenceWorker(
            self._mgr.rollback_model_update,
            model_id,
            job_kind="model_rollback",
            job_label=tr("model_hub_ui.jobs.rollback", model=info.name),
            job_inputs={"model_id": model_id},
            job_metadata={"model_id": model_id},
            job_store=self._job_store,
        )
        worker.finished.connect(
            lambda result, mid=model_id: self._on_model_rollback_finished(mid, result)
        )
        worker.error.connect(
            lambda error, mid=model_id: self._on_model_rollback_error(mid, error)
        )
        worker.cancelled.connect(
            lambda mid=model_id: self._on_model_rollback_error(
                mid, tr("model_hub_ui.updates.rollback_cancelled")
            )
        )
        self._update_workers[model_id] = worker
        self._update_status_label.setText(
            tr("model_hub_ui.updates.restoring", model=info.name)
        )
        worker.start()

    def _on_model_rollback_finished(self, model_id: str, _result):
        self._update_workers.pop(model_id, None)
        if model_id in self._cards:
            self._cards[model_id].update_status(self._mgr.get_status(model_id))
        self._update_status_label.setText(tr("model_hub_ui.updates.restored"))
        if self.toast_mgr:
            self.toast_mgr.success(tr("model_hub_ui.updates.restored_toast"))

    def _on_model_rollback_error(self, model_id: str, error: str):
        self._update_workers.pop(model_id, None)
        if model_id in self._cards:
            self._cards[model_id].update_status(self._mgr.get_status(model_id))
        self._update_status_label.setText(tr("model_hub_ui.updates.rollback_failed", error=error))
        if self.toast_mgr:
            self.toast_mgr.error(tr("model_hub_ui.updates.rollback_failed", error=error))

    def prepare_onboarding_model(self, model_id: str, action: str = "open") -> bool:
        """Select the model handed off by onboarding and optionally start it."""
        if model_id not in self._cards:
            return False
        if action not in {"open", "download"}:
            action = "open"
        self._search.clear()
        self._category_filter.setCurrentIndex(0)
        task_filter = getattr(self, "_task_filter", None)
        if task_filter is not None:
            task_filter.setCurrentIndex(0)
        hardware_filter = getattr(self, "_hardware_filter", None)
        if hardware_filter is not None:
            hardware_filter.setCurrentIndex(1)
        self._downloaded_only.setChecked(False)
        self._filter_cards()
        card = self._cards[model_id]
        card.setVisible(True)
        card.setAccessibleDescription(
            tr("model_hub_ui.accessibility.onboarding_selected", model=card.info.name)
        )
        if action == "download":
            status = self._mgr.get_status(model_id)
            handler = (
                self._start_activation
                if status in {ModelStatus.DOWNLOADED, ModelStatus.LOADED}
                else self._start_download
            )
            QTimer.singleShot(0, lambda mid=model_id: handler(mid))
        return True

    def _refresh_all_cards(self):
        for record in self._recoverable_download_records():
            model_id = record.inputs.get("model_id") or record.metadata.get("model_id")
            if model_id in self._mgr.registry and self._mgr.has_partial_download(model_id):
                self._mgr._set_status(model_id, ModelStatus.PARTIAL)

        for model_id, card in self._cards.items():
            status = self._mgr.get_status(model_id)
            card.update_status(status)
            card.update_hardware_status(self._hardware_profile)
        self._update_disk_display()
        self._update_recovery_banner()
        self._update_recommendation_label()

        # Alert user about partial downloads on startup
        partials = [
            mid for mid in self._mgr.registry
            if self._mgr.get_status(mid) == ModelStatus.PARTIAL
        ]
        if partials and self.toast_mgr:
            names = ", ".join(
                self._mgr.get_model_info(m).name for m in partials
            )
            self.toast_mgr.warning(
                tr("model_hub_ui.recovery.incomplete_toast", models=names)
            )

    def _update_recovery_banner(self):
        names: list[str] = []
        for record in self._recoverable_download_records():
            model_id = record.inputs.get("model_id") or record.metadata.get("model_id")
            if not model_id or model_id not in self._mgr.registry:
                continue
            if self._mgr.get_status(model_id) != ModelStatus.PARTIAL:
                continue
            info = self._mgr.get_model_info(model_id)
            names.append(info.name if info else model_id)

        if names:
            self._recovery_label.setText(
                tr(
                    "model_hub_ui.recovery.banner",
                    models=", ".join(names[:4]),
                    suffix=(
                        tr("model_hub_ui.recovery.more", count=len(names) - 4)
                        if len(names) > 4 else ""
                    ),
                )
            )
            self._recovery_label.setVisible(True)
        else:
            self._recovery_label.setVisible(False)

    def _recoverable_download_records(self):
        return [
            record
            for record in self._job_store.list_records(kind="model_download")
            if record.status == JobStatus.RECOVERABLE or record.recoverable
        ]

    def _on_status_changed(self, model_id: str, status_str: str):
        if model_id in self._cards:
            status = ModelStatus(status_str)
            self._cards[model_id].update_status(status)
        self._update_disk_display()

    def _on_gpu_changed(self, gpu_info: dict):
        self._update_gpu_display(gpu_info)

    def _update_gpu_display(self, gpu_info: dict = None):
        if gpu_info is None:
            gpu_info = self._mgr.get_gpu_status()
        self._hardware_profile = dict(gpu_info)
        if gpu_info.get("available"):
            name = gpu_info["name"]
            total = float(gpu_info.get("total_gb", 0) or 0)
            used = float(gpu_info.get("used_gb", 0) or 0)
            current = gpu_info.get(
                "current_model_name", tr("model_hub_ui.gpu.none")
            )
            if gpu_info.get("backend") == "mps":
                self._gpu_label.setText(
                    tr(
                        "model_hub_ui.gpu.mps",
                        name=name,
                        active=current or tr("model_hub_ui.gpu.none"),
                    )
                )
            else:
                self._gpu_label.setText(
                    tr(
                        "model_hub_ui.gpu.standard",
                        name=name,
                        used=used,
                        total=total,
                        active=current or tr("model_hub_ui.gpu.none"),
                    )
                )
            self._gpu_label.setStyleSheet(
                f"font-size: 9.75pt; font-weight: 600; color: {Palette.BLUE};"
            )
        else:
            self._gpu_label.setText(tr("model_hub_ui.gpu.cpu_only"))
            self._gpu_label.setStyleSheet(
                f"font-size: 9.75pt; font-weight: 600; color: {Palette.YELLOW};"
            )
        self._update_hardware_filter_label()
        for card in self._cards.values():
            card.update_hardware_status(self._hardware_profile)
        self._update_recommendation_label()
        self._filter_cards()

    def _update_hardware_filter_label(self):
        if self._hardware_profile.get("available"):
            backend = self._hardware_profile.get("backend", "cuda").upper()
            total = float(self._hardware_profile.get("total_gb", 0) or 0)
            suffix = (
                tr("model_hub_ui.hardware.memory_gb", value=total)
                if total else tr("model_hub_ui.hardware.shared_memory")
            )
            text = tr("model_hub_ui.hardware.filter_fit", backend=backend, memory=suffix)
        else:
            text = tr("model_hub_ui.hardware.filter_cpu")
        self._hardware_filter.setItemText(0, text)

    def _update_recommendation_label(self):
        selected_task = self._task_filter.currentData()
        tasks = (
            [selected_task]
            if selected_task and selected_task != "all"
            else ["best vocal isolation", "fastest", "lowest vram"]
        )
        hardware_name = self._hardware_profile.get(
            "name", tr("model_hub_ui.hardware.detected")
        )
        recommendations = []
        for task in tasks:
            info = recommend_model_for_task(
                task,
                registry=self._mgr.registry,
                hardware=self._hardware_profile,
            )
            if info is None:
                continue
            fit = model_hardware_fit(info, self._hardware_profile)
            mode = tr("model_hub_ui.hardware.gpu_fit") if fit.fits and fit.status in {"cuda", "mps"} else (
                tr("model_hub_ui.hardware.cpu_fallback") if fit.status == "cpu-fallback" else fit.status
            )
            recommendations.append(
                tr(
                    "model_hub_ui.recommendation.item",
                    task=task.title().replace("Vram", "VRAM"),
                    model=info.name,
                    mode=mode,
                )
            )
        if recommendations:
            self._recommendation_label.setText(
                tr(
                    "model_hub_ui.recommendation.summary",
                    hardware=hardware_name,
                    recommendations="; ".join(recommendations),
                )
            )
        else:
            self._recommendation_label.setText(
                tr("model_hub_ui.recommendation.none")
            )

    def _update_disk_display(self):
        usage = self._mgr.get_total_disk_usage()
        downloaded = sum(
            1 for s in self._mgr._status.values()
            if s in (ModelStatus.DOWNLOADED, ModelStatus.LOADED)
        )
        total = len(self._mgr.registry)
        self._disk_label.setText(
            tr(
                "model_hub_ui.disk_summary",
                usage=usage,
                downloaded=downloaded,
                total=total,
            )
        )

    @staticmethod
    def _sort_model_ids(registry: dict[str, ModelInfo], sort_key: str | None) -> list[str]:
        """Return registry IDs in the explicit user-selected order."""
        sort_key = sort_key or "name_asc"
        reverse = sort_key in {"name_desc", "date_desc"}
        if sort_key in {"date_asc", "date_desc"}:
            ordered = sorted(
                registry.items(),
                key=lambda item: (
                    item[1].measurement_date or "",
                    item[1].name.casefold(),
                    item[0],
                ),
                reverse=reverse,
            )
        else:
            ordered = sorted(
                registry.items(),
                key=lambda item: (item[1].name.casefold(), item[0]),
                reverse=reverse,
            )
        return [model_id for model_id, _info in ordered]

    def _arrange_cards(self):
        """Rebuild grid positions without recreating cards or losing state."""
        for card in self._cards.values():
            self._grid_layout.removeWidget(card)
        for row in range(self._grid_layout.rowCount() + 2):
            self._grid_layout.setRowStretch(row, 0)

        registry = {
            model_id: self._mgr.registry.get(model_id, card.info)
            for model_id, card in self._cards.items()
        }
        for index, model_id in enumerate(
            self._sort_model_ids(registry, self._sort_combo.currentData())
        ):
            row, column = divmod(index, 3)
            self._grid_layout.addWidget(self._cards[model_id], row, column)
        self._grid_layout.setRowStretch((len(self._cards) + 2) // 3, 1)

    def _filter_cards(self):
        self._arrange_cards()
        search = self._search.text().lower()
        cat_filter = self._category_filter.currentData()
        task_filter = self._task_filter.currentData()
        hardware_filter = self._hardware_filter.currentData()
        downloaded_only = self._downloaded_only.isChecked()

        visible_count = 0
        for model_id, card in self._cards.items():
            info = self._mgr.get_model_info(model_id)
            status = self._mgr.get_status(model_id)
            if info is None:
                # A registry refresh can remove a card between layout and
                # filtering. Treat that transient state as unavailable rather
                # than dereferencing a stale model record in the Qt callback.
                card.setVisible(False)
                continue
            visible = True
            if search and search not in info.name.lower() \
                    and search not in info.description.lower():
                visible = False
            if cat_filter != "all" and info.category.value != cat_filter:
                visible = False
            if task_filter != "all" and not model_supports_task(info, task_filter):
                visible = False
            if hardware_filter == "fit" and not model_hardware_fit(
                info, self._hardware_profile
            ).fits:
                visible = False
            if downloaded_only and status not in (
                ModelStatus.DOWNLOADED, ModelStatus.LOADED
            ):
                visible = False
            card.setVisible(visible)
            visible_count += int(visible)
        if visible_count:
            self._grid_stack.setCurrentWidget(self._grid_container)
        elif search or cat_filter != "all" or task_filter != "all" or hardware_filter != "fit" or downloaded_only:
            filters = []
            if search:
                filters.append(f'“{search}”')
            if cat_filter != "all":
                filters.append(cat_filter.replace("_", " "))
            if task_filter != "all":
                filters.append(task_filter)
            if hardware_filter == "fit":
                filters.append(tr("model_hub_ui.filters.detected_hardware"))
            if downloaded_only:
                filters.append(tr("model_hub_ui.filters.downloaded_models"))
            self._grid_empty.set_no_matches(
                tr("model_hub_ui.empty.filtered", filters=", ".join(filters)),
                tr("model_hub_ui.empty.clear_filters"),
            )
            self._grid_stack.setCurrentWidget(self._grid_empty)
        else:
            self._grid_empty.set_state(
                tr("model_hub_ui.empty.no_models_title"),
                tr("model_hub_ui.empty.no_models_description"),
                tr("model_hub_ui.empty.refresh_catalog"),
            )
            self._grid_stack.setCurrentWidget(self._grid_empty)
        self._update_recommendation_label()

    def _clear_filters(self):
        self._search.clear()
        self._category_filter.setCurrentIndex(0)
        self._task_filter.setCurrentIndex(0)
        self._hardware_filter.setCurrentIndex(1)
        self._downloaded_only.setChecked(False)
        self._filter_cards()

    # -- Download Management -----------------------------------------------

    def _start_download(self, model_id: str):
        """Start or resume downloading a model in a background thread."""
        if model_id in self._workers:
            if model_id in self._stopping_downloads and self.toast_mgr:
                info = self._mgr.get_model_info(model_id)
                self.toast_mgr.info(
                    tr("model_hub_ui.download.still_stopping", model=info.name)
                )
            return

        if self._mgr.is_offline:
            if self.toast_mgr:
                self.toast_mgr.error(
                    tr("model_hub_ui.download.offline_disabled")
                )
            return

        info = self._mgr.get_model_info(model_id)

        # Gated model check
        if getattr(info, "gated", False):
            token = self._mgr._get_hf_token()
            if not token:
                dlg = HFTokenDialog(info.name, info.source, parent=self)
                if dlg.exec() == QDialog.DialogCode.Accepted and dlg.token:
                    try:
                        Settings().set("model_hub.hf_token", dlg.token)
                    except CredentialError as exc:
                        if self.toast_mgr:
                            self.toast_mgr.error(
                                tr("model_hub_ui.download.token_not_saved", error=exc)
                            )
                        return
                    if self.toast_mgr:
                        store = Settings().credential_store
                        self.toast_mgr.success(
                            tr("model_hub_ui.download.token_saved", backend=store.backend_name)
                        )
                else:
                    return

        card = self._cards[model_id]
        card.update_status(ModelStatus.DOWNLOADING)

        worker = DownloadWorker(self._mgr.download_model, model_id, model_name=info.name)

        # Wire all progress signals into the card
        worker.progress.connect(
            lambda pct, mid=model_id: self._on_dl_progress(mid, pct)
        )
        worker.speed.connect(
            lambda s, mid=model_id: self._on_dl_speed(mid, s)
        )
        worker.downloaded.connect(
            lambda s, mid=model_id: self._on_dl_size(mid, s)
        )
        worker.finished.connect(self._on_download_finished)
        worker.cancelled.connect(self._on_download_cancelled)
        worker.error.connect(
            lambda err, mid=model_id: self._on_download_error(mid, err)
        )

        self._workers[model_id] = worker
        worker.start()

        if self.toast_mgr:
            self.toast_mgr.info(tr("model_hub_ui.download.started", model=info.name))

    def _on_dl_progress(self, model_id: str, pct: int):
        if model_id in self._cards:
            self._cards[model_id].update_download_progress(pct)

    def _on_dl_speed(self, model_id: str, speed: str):
        if model_id in self._cards:
            self._cards[model_id]._speed_label.setText(speed)

    def _on_dl_size(self, model_id: str, size: str):
        if model_id in self._cards:
            self._cards[model_id]._size_label.setText(size)

    def _cancel_download(self, model_id: str):
        """Request cancellation and keep the card in a stopping state."""
        worker = self._workers.get(model_id)
        if worker is None:
            return
        self._stopping_downloads.add(model_id)
        worker.cancel()
        if model_id in self._cards:
            self._cards[model_id].set_download_stopping()
        if self.toast_mgr:
            info = self._mgr.get_model_info(model_id)
            self.toast_mgr.info(tr("model_hub_ui.download.stopping", model=info.name))

    def _on_download_cancelled(self, model_id: str):
        self._stopping_downloads.discard(model_id)
        if model_id in self._workers:
            del self._workers[model_id]
        self._mgr._set_status(model_id, ModelStatus.PARTIAL)
        if model_id in self._cards:
            self._cards[model_id].update_status(ModelStatus.PARTIAL)
        self._update_recovery_banner()

    def _on_download_finished(self, model_id: str):
        self._stopping_downloads.discard(model_id)
        if model_id in self._workers:
            del self._workers[model_id]
        if model_id in self._cards:
            self._cards[model_id].update_status(self._mgr.get_status(model_id))
        self._update_recovery_banner()
        if self.toast_mgr:
            info = self._mgr.get_model_info(model_id)
            self.toast_mgr.success(tr("model_hub_ui.download.success", model=info.name))

    def _on_download_error(self, model_id: str, error: str):
        was_stopping = model_id in self._stopping_downloads
        self._stopping_downloads.discard(model_id)
        if model_id in self._workers:
            del self._workers[model_id]
        if was_stopping:
            self._mgr._set_status(model_id, ModelStatus.PARTIAL)
            if model_id in self._cards:
                self._cards[model_id].update_status(ModelStatus.PARTIAL)
        self._update_recovery_banner()
        if self.toast_mgr:
            info = self._mgr.get_model_info(model_id)
            self.toast_mgr.error(
                tr("model_hub_ui.download.failed", model=info.name, error=error)
            )

    # -- Activation Management ---------------------------------------------

    def _start_activation(self, model_id: str):
        if model_id in self._activation_workers:
            return
        info = self._mgr.get_model_info(model_id)
        if info is None:
            return

        self._mgr._set_status(model_id, ModelStatus.LOADING)
        worker = InferenceWorker(
            self._mgr.activate_model,
            model_id,
            job_kind="model_activation",
            job_label=tr("model_hub_ui.jobs.activation", model=info.name),
            job_inputs={"model_id": model_id},
            job_metadata={"model_id": model_id},
        )
        worker.finished.connect(self._on_activation_finished)
        worker.error.connect(
            lambda error, mid=model_id: self._on_activation_error(mid, error)
        )
        worker.cancelled.connect(
            lambda mid=model_id: self._on_activation_cancelled(mid)
        )
        self._activation_workers[model_id] = worker
        worker.start()
        if self.toast_mgr:
            self.toast_mgr.info(
                tr("model_hub_ui.activation.started", model=info.name)
            )

    def _cancel_activation(self, model_id: str):
        worker = self._activation_workers.get(model_id)
        if worker:
            worker.cancel()
            if self.toast_mgr:
                info = self._mgr.get_model_info(model_id)
                self.toast_mgr.info(
                    tr("model_hub_ui.activation.cancelling", model=info.name)
                )

    def _on_activation_finished(self, result):
        model_id = getattr(result, "model_id", "")
        if model_id:
            self._activation_workers.pop(model_id, None)
            self._cards[model_id].update_status(self._mgr.get_status(model_id))
        cancelled = getattr(result, "cancelled", False)
        if callable(cancelled):
            cancelled = cancelled()
        is_cancelled = getattr(result, "is_cancelled", False)
        if callable(is_cancelled):
            is_cancelled = is_cancelled()
        if cancelled or is_cancelled:
            self._on_activation_cancelled(model_id)
            return
        if getattr(result, "is_success", False):
            if self.toast_mgr:
                self.toast_mgr.success(result.message)
        elif self.toast_mgr:
            self.toast_mgr.error(
                result.error or tr("model_hub_ui.activation.failed")
            )
        self._update_gpu_display()

    def _on_activation_error(self, model_id: str, error: str):
        self._activation_workers.pop(model_id, None)
        self._mgr._set_status(model_id, ModelStatus.ERROR)
        if self.toast_mgr:
            info = self._mgr.get_model_info(model_id)
            self.toast_mgr.error(
                tr("model_hub_ui.activation.error", model=info.name, error=error)
            )

    def _on_activation_cancelled(self, model_id: str):
        self._activation_workers.pop(model_id, None)
        if self._mgr.current_model_id == model_id:
            result = self._mgr.deactivate_model(model_id)
            if not result.is_success:
                if self.toast_mgr:
                    self.toast_mgr.error(
                        result.error
                        or tr("model_hub_ui.activation.release_after_cancel_failed", model=model_id)
                    )
                self._update_gpu_display()
                return
        status = (
            ModelStatus.DOWNLOADED
            if self._mgr.get_model_readiness(model_id).installed
            else ModelStatus.NOT_DOWNLOADED
        )
        self._mgr._set_status(model_id, status)
        if self.toast_mgr:
            info = self._mgr.get_model_info(model_id)
            self.toast_mgr.info(
                tr("model_hub_ui.activation.cancelled", model=info.name)
            )

    def _deactivate_model(self, model_id: str):
        result = self._mgr.deactivate_model(model_id)
        if self.toast_mgr:
            if result.is_success:
                info = self._mgr.get_model_info(model_id)
                self.toast_mgr.success(
                    tr("model_hub_ui.activation.deactivated", model=info.name)
                )
            else:
                self.toast_mgr.error(result.error)
        self._update_gpu_display()

    def _delete_model(self, model_id: str):
        """Delete a downloaded model's cache."""
        info = self._mgr.get_model_info(model_id)
        if info and getattr(info, "pip_managed", False):
            if self.toast_mgr:
                self.toast_mgr.info(
                    tr("model_hub_ui.delete.pip_managed", model=info.name)
                )
            return

        entry = self._mgr.delete_model_cache(model_id)
        if not entry:
            if self.toast_mgr:
                self.toast_mgr.error(
                    self._mgr.get_model_error(model_id)
                    or tr("model_hub_ui.delete.failed", model=info.name)
                )
            return

        self._cards[model_id].update_status(ModelStatus.NOT_DOWNLOADED)
        self._update_disk_display()
        if self.toast_mgr:
            self.toast_mgr.info(
                tr("model_hub_ui.delete.moved_to_trash", model=info.name),
                duration_ms=8000,
                action_label=tr("model_hub_ui.delete.undo"),
                action_callback=lambda entry_id=entry.id, mid=model_id: self._restore_model(mid, entry_id),
            )

    def _restore_model(self, model_id: str, trash_entry_id: str):
        if self._mgr.restore_model_cache(trash_entry_id):
            self._cards[model_id].update_status(self._mgr.get_status(model_id))
            self._update_disk_display()
            if self.toast_mgr:
                info = self._mgr.get_model_info(model_id)
                self.toast_mgr.success(
                    tr("model_hub_ui.delete.restored", model=info.name)
                )
        elif self.toast_mgr:
            info = self._mgr.get_model_info(model_id)
            self.toast_mgr.error(
                tr("model_hub_ui.delete.restore_failed", model=info.name)
            )

    def _review_execution_consent(self, model_id: str):
        info = self._mgr.get_model_info(model_id)
        if info is None:
            return
        dialog = ExecutableModelConsentDialog(info, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self._mgr.approve_executable_model(
                model_id,
                info.revision,
                acknowledged=dialog._ack.isChecked(),
            )
        except ModelSecurityError as exc:
            if self.toast_mgr:
                self.toast_mgr.error(str(exc))
            return
        self._cards[model_id].update_status(self._mgr.get_status(model_id))
        if self.toast_mgr:
            self.toast_mgr.warning(
                tr(
                    "model_hub_ui.consent.approved",
                    model=info.name,
                    revision=info.revision[:12],
                )
            )
