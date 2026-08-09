"""
Slunder Studio — Onboarding Wizard
First-run experience: welcome, system check, model download prompt,
quick start guide, and preference setup.
"""
import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QStackedWidget, QWidget, QCheckBox, QComboBox,
    QProgressBar, QLineEdit, QFileDialog,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont

from core.settings import APP_VERSION
from core.device import configured_cuda_index
from core.engine_contract import ModelReadiness
from core.model_manager import (
    ModelManager,
    model_hardware_fit,
    recommend_model_for_task,
)
from ui.accessibility import install_accessibility
from ui.theme import Palette, ThemeEngine
from ui.widgets import ElidedLabel
from core.workers import InferenceWorker
from core.i18n import tr
from ui.file_dialogs import choose_directory


# ── System Check ───────────────────────────────────────────────────────────────

def run_dependency_setup(progress_cb=None, **_kwargs) -> str:
    """Install the source checkout's runtime requirements for a failed check."""
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    if not requirements.is_file():
        raise RuntimeError(f"Requirements file not found: {requirements}")
    if progress_cb:
        progress_cb(10)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        cwd=str(requirements.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout or "setup command failed").strip()
        raise RuntimeError(detail[-1200:])
    if progress_cb:
        progress_cb(100)
    return tr("onboarding.system.setup_complete")

def check_system() -> dict:
    """Run system compatibility checks."""
    checks = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_ok": sys.version_info >= (3, 10),
        "arch": platform.machine(),
        "cpu": platform.processor() or tr("onboarding.system.unknown"),
        "setup_command": f'"{sys.executable}" -m pip install -r requirements.txt',
    }

    # RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        checks["ram_gb"] = round(ram.total / (1024**3), 1)
        checks["ram_ok"] = checks["ram_gb"] >= 8
    except Exception as exc:
        checks["ram_gb"] = 0
        checks["ram_ok"] = False
        checks["ram_error"] = tr("onboarding.system.ram_error", error=type(exc).__name__)

    # Core Python dependencies
    try:
        from core.deps import CORE_RUNTIME_PACKAGES, dependency_status
        missing_deps = dependency_status(CORE_RUNTIME_PACKAGES)
        checks["deps_missing"] = [pip_name for _, pip_name in missing_deps]
        checks["deps_ok"] = not missing_deps
    except Exception as exc:
        checks["deps_missing"] = [
            tr("onboarding.system.dependencies_error_detail", error=type(exc).__name__)
        ]
        checks["deps_ok"] = False
        checks["deps_error"] = tr("onboarding.system.dependencies_error")

    # GPU / CUDA
    checks["cuda"] = False
    checks["gpu_name"] = tr("onboarding.system.gpu_none")
    checks["vram_gb"] = 0
    try:
        import torch
        checks["cuda"] = torch.cuda.is_available()
        if checks["cuda"]:
            gpu_index = configured_cuda_index(torch)
            checks["gpu_index"] = gpu_index
            checks["gpu_name"] = torch.cuda.get_device_name(gpu_index)
            checks["vram_gb"] = round(
                torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3), 1
            )
    except (ImportError, RuntimeError, AttributeError) as exc:
        checks["gpu_error"] = tr("onboarding.system.gpu_error", error=type(exc).__name__)

    # Disk space
    try:
        import shutil
        from core.settings import get_config_dir
        usage = shutil.disk_usage(get_config_dir())
        checks["disk_free_gb"] = round(usage.free / (1024**3), 1)
        checks["disk_ok"] = checks["disk_free_gb"] >= 10
    except Exception as exc:
        checks["disk_free_gb"] = 0
        checks["disk_ok"] = False
        checks["disk_error"] = tr("onboarding.system.disk_error", error=type(exc).__name__)

    return checks


# ── Wizard Pages ───────────────────────────────────────────────────────────────


def model_readiness_label(readiness: ModelReadiness, offline: bool = False) -> str:
    """Map lifecycle evidence to the state shown during onboarding."""
    if readiness.active:
        return tr("onboarding.readiness.loaded")
    if readiness.status == "error":
        return tr("onboarding.readiness.error")
    if offline and not readiness.installed:
        return tr("onboarding.readiness.offline")
    if readiness.installed and readiness.loadable:
        return tr("onboarding.readiness.downloaded")
    if readiness.installed:
        return tr("onboarding.readiness.installed_unloadable")
    return tr("onboarding.readiness.not_downloaded")

class WelcomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        logo = QLabel("SLUNDER STUDIO")
        logo.setAlignment(Qt.AlignCenter)
        logo.setFont(QFont("Segoe UI", 28, QFont.Bold))
        logo.setStyleSheet(f"color: {t['accent']}; font-size: 21pt; font-weight: bold;")
        layout.addWidget(logo)

        version = QLabel(f"v{APP_VERSION}")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10.5pt;")
        layout.addWidget(version)

        tagline = QLabel(tr("onboarding.welcome.tagline"))
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setStyleSheet(f"color: {t['text']}; font-size: 12pt;")
        layout.addWidget(tagline)

        desc = QLabel(
            tr("onboarding.welcome.description")
        )
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {t['text_secondary']}; font-size: 9pt; line-height: 1.6;")
        layout.addWidget(desc)
        install_accessibility(self, tr("onboarding.accessibility.welcome_name"))


class SystemCheckPage(QWidget):
    remediation_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel(tr("onboarding.system.title"))
        title.setStyleSheet(f"color: {t['text']}; font-size: 13.5pt; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(tr("onboarding.system.subtitle"))
        subtitle.setStyleSheet(f"color: {t['text_secondary']}; font-size: 9pt;")
        layout.addWidget(subtitle)

        self._checks_frame = QFrame()
        self._checks_frame.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        self._checks_layout = QVBoxLayout(self._checks_frame)
        self._checks_layout.setContentsMargins(16, 12, 16, 12)
        self._checks_layout.setSpacing(8)
        layout.addWidget(self._checks_frame)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt;")
        layout.addWidget(self._summary)
        layout.addStretch()
        self._check_worker = None
        self._check_workers = set()
        self._setup_worker = None
        self._setup_workers = set()
        install_accessibility(self, tr("onboarding.accessibility.system_name"))

    def run_checks(self):
        if self._check_worker is not None and self._check_worker.isRunning():
            return
        self._clear_check_rows()
        self._summary.setText(tr("onboarding.system.checking"))
        worker = InferenceWorker(check_system)
        self._check_workers.add(worker)
        self._check_worker = worker
        worker.finished.connect(self._on_checks_finished)
        worker.error.connect(self._on_checks_error)
        worker.start()

    def _clear_check_rows(self):
        """Remove the previous result rows before rendering a fresh check."""
        while self._checks_layout.count():
            item = self._checks_layout.takeAt(0)
            child_layout = item.layout()
            if child_layout is not None:
                while child_layout.count():
                    child = child_layout.takeAt(0)
                    if child.widget() is not None:
                        child.widget().deleteLater()
                child_layout.deleteLater()
            elif item.widget() is not None:
                item.widget().deleteLater()

    def _release_check_worker_later(self, worker):
        if worker is None:
            return
        if worker.isRunning():
            QTimer.singleShot(10, lambda: self._release_check_worker_later(worker))
            return
        self._check_workers.discard(worker)
        if self._check_worker is worker:
            self._check_worker = None

    def _on_checks_finished(self, checks: dict):
        worker = self._check_worker
        self._release_check_worker_later(worker)
        self._check_worker = None
        self._display_checks(checks)

    def _on_checks_error(self, message: str):
        worker = self._check_worker
        self._release_check_worker_later(worker)
        self._check_worker = None
        self._summary.setText(tr("onboarding.system.check_failed", error=message))

    def _start_dependency_setup(self):
        if self._setup_worker is not None and self._setup_worker.isRunning():
            return
        self._summary.setText(tr("onboarding.system.installing"))
        worker = InferenceWorker(
            run_dependency_setup,
            job_kind="onboarding_dependency_setup",
            job_label=tr("onboarding.jobs.dependency_setup"),
        )
        self._setup_workers.add(worker)
        self._setup_worker = worker
        worker.finished.connect(self._on_setup_finished)
        worker.error.connect(self._on_setup_error)
        worker.thread_stopped.connect(lambda w=worker: self._release_setup_worker_later(w))
        worker.start()

    def _release_setup_worker_later(self, worker):
        if worker is None:
            return
        if worker.isRunning():
            QTimer.singleShot(10, lambda: self._release_setup_worker_later(worker))
            return
        self._setup_workers.discard(worker)
        if self._setup_worker is worker:
            self._setup_worker = None

    def _on_setup_finished(self, message: str):
        self._summary.setText(str(message))
        worker = self._setup_worker
        self._release_setup_worker_later(worker)
        self.run_checks()

    def _on_setup_error(self, message: str):
        self._summary.setText(tr("onboarding.system.setup_failed", error=message))
        worker = self._setup_worker
        self._release_setup_worker_later(worker)

    @staticmethod
    def _remediation_label(key: str) -> str:
        return {
            "dependencies": tr("onboarding.remediation.setup"),
            "gpu": tr("onboarding.remediation.model"),
            "ram": tr("onboarding.remediation.model"),
            "disk": tr("onboarding.remediation.output"),
            "python": tr("onboarding.remediation.instructions"),
        }.get(key, tr("onboarding.remediation.details"))

    def _request_remediation(self, key: str, checks: dict):
        if key == "dependencies":
            self._start_dependency_setup()
        elif key == "python":
            self._summary.setText(
                tr("onboarding.system.python_remediation", version=checks["python"])
            )
        else:
            self.remediation_requested.emit(key)

    def _display_checks(self, checks: dict):
        t = ThemeEngine.get_colors()

        items = [
            ("python", tr("onboarding.checks.python"), checks["python"], checks["python_ok"],
             tr("onboarding.checks.python_required")),
            ("dependencies", tr("onboarding.checks.dependencies"),
             tr("onboarding.checks.ready") if checks["deps_ok"] else ", ".join(checks["deps_missing"]),
             checks["deps_ok"],
             checks.get("deps_error")
             or ("" if checks["deps_ok"] else tr("onboarding.checks.run_setup", command=checks["setup_command"]))),
            ("os", tr("onboarding.checks.operating_system"), f"{checks['os']} {checks['arch']}", True, ""),
            ("gpu", tr("onboarding.checks.gpu"), checks["gpu_name"],
             checks["cuda"],
             checks.get("gpu_error")
             or (tr("onboarding.checks.vram", vram=checks["vram_gb"]) if checks["cuda"] else tr("onboarding.checks.cpu_only"))),
            ("ram", tr("onboarding.checks.ram"), f"{checks['ram_gb']} GB",
             checks.get("ram_ok", True), checks.get("ram_error", tr("onboarding.checks.ram_recommended"))),
            ("disk", tr("onboarding.checks.disk"), f"{checks['disk_free_gb']} GB free",
             checks.get("disk_ok", True), checks.get("disk_error", tr("onboarding.checks.disk_recommended"))),
        ]

        for key, label, value, ok, note in items:
            row = QHBoxLayout()
            icon = QLabel(tr("onboarding.checks.ok") if ok else tr("onboarding.checks.warning"))
            icon.setMinimumWidth(24)
            icon.setAlignment(Qt.AlignCenter)
            icon.setStyleSheet(
                f"color: {Palette.GREEN}; font-weight: bold; font-size: 8.25pt;"
                if ok else f"color: {Palette.YELLOW}; font-weight: bold; font-size: 8.25pt;"
            )
            name = ElidedLabel(f"{label}:", minimum_width=110)
            name.setStyleSheet(f"color: {t['text']}; font-size: 9pt; border: none;")
            val = QLabel(str(value))
            val.setStyleSheet(f"color: {t['text']}; font-size: 9pt; font-weight: bold; border: none;")
            note_l = QLabel(note)
            note_l.setStyleSheet(f"color: {t['text_secondary']}; font-size: 7.5pt; border: none;")
            row.addWidget(icon)
            row.addWidget(name)
            row.addWidget(val, 1)
            row.addWidget(note_l)
            if not ok:
                action = QPushButton(self._remediation_label(key))
                action.setObjectName("remediationButton")
                action.setMinimumHeight(28)
                action.clicked.connect(
                    lambda _checked=False, check_key=key: self._request_remediation(
                        check_key, checks
                    )
                )
                row.addWidget(action)
            self._checks_layout.addLayout(row)

        if checks["cuda"]:
            self._summary.setText(
                tr(
                    "onboarding.system.gpu_available",
                    name=checks["gpu_name"],
                    vram=checks["vram_gb"],
                )
            )
        else:
            self._summary.setText(
                tr("onboarding.system.cpu_summary")
            )


class ModelGuidePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        manager = ModelManager()
        hardware = (
            manager.get_gpu_status()
            if hasattr(manager, "get_gpu_status")
            else {"available": False, "backend": "cpu", "name": "CPU"}
        )
        registry = getattr(manager, "registry", {})

        title = QLabel(tr("onboarding.models.title"))
        title.setStyleSheet(f"color: {t['text']}; font-size: 13.5pt; font-weight: bold;")
        layout.addWidget(title)

        info = QLabel(
            tr("onboarding.models.info")
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {t['text_secondary']}; font-size: 9pt;")
        layout.addWidget(info)

        hardware_name = hardware.get("name", tr("onboarding.models.detected_hardware"))
        hardware_note = QLabel(
            tr("onboarding.models.hardware_note", hardware=hardware_name)
        )
        hardware_note.setWordWrap(True)
        hardware_note.setStyleSheet(f"color: {Palette.BLUE}; font-size: 8.25pt;")
        layout.addWidget(hardware_note)

        # Recommended models
        models_frame = QFrame()
        models_frame.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        ml = QVBoxLayout(models_frame)
        ml.setContentsMargins(16, 12, 16, 12)
        ml.setSpacing(10)

        recommendation_specs = [
            ("best song generation", "onboarding.models.tasks.song"),
            ("best lyrics", "onboarding.models.tasks.lyrics"),
            ("singing voice synthesis", "onboarding.models.tasks.singing"),
            ("voice conversion", "onboarding.models.tasks.conversion"),
            ("best vocal isolation", "onboarding.models.tasks.isolation"),
            ("sfx generation", "onboarding.models.tasks.sfx"),
        ]
        models = []
        for task, description in recommendation_specs:
            info_item = recommend_model_for_task(
                task,
                registry=registry,
                hardware=hardware,
            )
            if info_item is None:
                continue
            fit = model_hardware_fit(info_item, hardware)
            mode = "GPU fit" if fit.fits and fit.status in {"cuda", "mps"} else (
                "CPU fallback" if fit.status == "cpu-fallback" else fit.status
            )
            models.append(
                (
                    info_item.name,
                    f"{tr(description)} · {tr('onboarding.models.gpu_fit') if mode == 'GPU fit' else tr('onboarding.models.cpu_fallback') if mode == 'CPU fallback' else mode} · {info_item.advertised_vram_tier}",
                    f"{info_item.disk_gb:.1f} GB",
                    True,
                )
            )

        for name, desc, size, recommended in models:
            row = QHBoxLayout()
            tag = QLabel("REC" if recommended else "OPT")
            tag.setMinimumWidth(30)
            tag.setAlignment(Qt.AlignCenter)
            tag.setStyleSheet(
                f"color: white; background: {Palette.GREEN}; border-radius: 3px; "
                f"font-size: 6pt; font-weight: bold; padding: 2px;"
                if recommended else
                f"color: {t['text_secondary']}; background: {t['border']}; "
                f"border-radius: 3px; font-size: 6pt; padding: 2px;"
            )
            n = ElidedLabel(name, minimum_width=140)
            n.setStyleSheet(f"color: {t['text']}; font-size: 9pt; font-weight: bold; border: none;")
            d = QLabel(desc)
            d.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt; border: none;")
            s = QLabel(size)
            s.setMinimumWidth(80)
            s.setAlignment(Qt.AlignRight)
            s.setStyleSheet(f"color: {t['text_secondary']}; font-size: 7.5pt; border: none;")
            row.addWidget(tag)
            row.addWidget(n)
            row.addWidget(d, 1)
            row.addWidget(s)
            ml.addLayout(row)

        layout.addWidget(models_frame)

        readiness_frame = QFrame()
        readiness_frame.setStyleSheet(f"""
            QFrame {{ background: {t['background']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        readiness_layout = QVBoxLayout(readiness_frame)
        readiness_layout.setContentsMargins(12, 8, 12, 8)
        readiness_title = QLabel(tr("onboarding.models.readiness_title"))
        readiness_title.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9pt;")
        readiness_layout.addWidget(readiness_title)
        self._manager = manager
        self._core_models = manager.get_core_models()
        for info_item in self._core_models:
            try:
                readiness = manager.get_model_readiness(info_item.model_id)
            except Exception as exc:
                readiness = ModelReadiness(
                    model_id=info_item.model_id,
                    installed=False,
                    verified=False,
                    loadable=False,
                    active=False,
                    status="error",
                    remedy=tr("onboarding.models.readiness_probe_failed", error=type(exc).__name__),
                )
            row = QHBoxLayout()
            state = QLabel(model_readiness_label(readiness, manager.is_offline))
            state.setMinimumWidth(145)
            state.setStyleSheet(f"color: {Palette.YELLOW if readiness.status == 'error' else t['text']}; font-size: 7.5pt;")
            name = QLabel(info_item.name)
            name.setStyleSheet(f"color: {t['text']}; font-size: 8.25pt; font-weight: bold;")
            estimate = QLabel(
                tr(
                    "onboarding.models.estimate",
                    disk=info_item.disk_gb,
                    vram=info_item.vram_gb,
                )
            )
            estimate.setStyleSheet(f"color: {t['text_secondary']}; font-size: 7.5pt;")
            row.addWidget(state)
            row.addWidget(name, 1)
            row.addWidget(estimate)
            readiness_layout.addLayout(row)
        layout.addWidget(readiness_frame)

        setup_frame = QFrame()
        setup_frame.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        setup_layout = QVBoxLayout(setup_frame)
        setup_layout.setContentsMargins(12, 10, 12, 10)
        setup_layout.setSpacing(8)
        setup_title = QLabel(tr("onboarding.models.setup_title"))
        setup_title.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9pt;")
        setup_layout.addWidget(setup_title)

        self._model_selector = QComboBox()
        self._model_selector.setObjectName("onboardingModelSelector")
        for info_item in self._core_models:
            self._model_selector.addItem(
                tr(
                    "onboarding.models.selector_item",
                    name=info_item.name,
                    disk=info_item.disk_gb,
                    vram=info_item.vram_gb,
                ),
                info_item.model_id,
            )
        if not self._core_models:
            self._model_selector.addItem(tr("onboarding.models.no_core_models"), "")
            self._model_selector.setEnabled(False)
        setup_layout.addWidget(self._model_selector)

        self._model_action = QComboBox()
        self._model_action.setObjectName("onboardingModelAction")
        self._model_action.addItem(tr("onboarding.models.action_open"), "open")
        self._model_action.addItem(tr("onboarding.models.action_download"), "download")
        if manager.is_offline:
            self._model_action.setItemText(1, tr("onboarding.models.action_offline"))
            self._model_action.model().item(1).setEnabled(False)
        setup_layout.addWidget(self._model_action)

        action_note = QLabel(
            tr("onboarding.models.action_note")
        )
        action_note.setWordWrap(True)
        action_note.setStyleSheet(f"color: {t['text_secondary']}; font-size: 7.5pt;")
        setup_layout.addWidget(action_note)

        token_label = QLabel(tr("onboarding.models.token_label"))
        token_label.setStyleSheet(f"color: {t['text']}; font-size: 8.25pt;")
        setup_layout.addWidget(token_label)
        self._hf_token = QLineEdit()
        self._hf_token.setObjectName("onboardingHfToken")
        self._hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self._hf_token.setPlaceholderText(tr("onboarding.models.token_placeholder"))
        token_lookup = getattr(manager, "_get_hf_token", lambda: None)
        if token_lookup():
            self._hf_token.setPlaceholderText(tr("onboarding.models.token_saved"))
        setup_layout.addWidget(self._hf_token)
        self._token_error = QLabel("")
        self._token_error.setVisible(False)
        self._token_error.setWordWrap(True)
        self._token_error.setStyleSheet(f"color: {Palette.RED}; font-size: 7.5pt;")
        setup_layout.addWidget(self._token_error)
        layout.addWidget(setup_frame)

        note = QLabel(
            tr("onboarding.models.note")
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt;")
        layout.addWidget(note)
        layout.addStretch()
        install_accessibility(self, tr("onboarding.accessibility.models_name"))

    def selected_model_id(self) -> str:
        return str(self._model_selector.currentData() or "")

    def selected_model_action(self) -> str:
        return str(self._model_action.currentData() or "open")

    def hf_token(self) -> str:
        return self._hf_token.text().strip()

    def show_token_error(self, message: str):
        self._token_error.setText(message)
        self._token_error.setVisible(True)
        self._hf_token.setFocus()

    def clear_token_error(self):
        self._token_error.clear()
        self._token_error.setVisible(False)

    def focus_model_selector(self):
        self._model_selector.setFocus()


class QuickStartPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel(tr("onboarding.quickstart.title"))
        title.setStyleSheet(f"color: {t['text']}; font-size: 13.5pt; font-weight: bold;")
        layout.addWidget(title)

        preferences = QFrame()
        preferences.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        preferences_layout = QVBoxLayout(preferences)
        preferences_layout.setContentsMargins(12, 10, 12, 10)
        preferences_layout.setSpacing(8)
        preferences_title = QLabel(tr("onboarding.quickstart.preferences"))
        preferences_title.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9pt;")
        preferences_layout.addWidget(preferences_title)

        output_row = QHBoxLayout()
        output_label = QLabel(tr("onboarding.quickstart.output_label"))
        output_label.setStyleSheet(f"color: {t['text']}; font-size: 8.25pt;")
        output_row.addWidget(output_label)
        from core.settings import Settings, get_default_output_dir
        settings = Settings()
        self._output_dir = QLineEdit(str(settings.get("general.output_dir", "") or ""))
        self._output_dir.setObjectName("onboardingOutputDirectory")
        self._output_dir.setReadOnly(True)
        self._output_dir.setPlaceholderText(
            tr("onboarding.quickstart.output_default", path=get_default_output_dir())
        )
        output_row.addWidget(self._output_dir, 1)
        browse = QPushButton(tr("onboarding.quickstart.browse"))
        browse.setObjectName("onboardingBrowseOutput")
        browse.clicked.connect(self._browse_output_dir)
        output_row.addWidget(browse)
        preferences_layout.addLayout(output_row)

        experience_row = QHBoxLayout()
        experience_label = QLabel(tr("onboarding.quickstart.experience"))
        experience_label.setStyleSheet(f"color: {t['text']}; font-size: 8.25pt;")
        experience_row.addWidget(experience_label)
        self._experience = QComboBox()
        self._experience.setObjectName("onboardingExperience")
        for code, label in (
            ("beginner", tr("onboarding.quickstart.beginner")),
            ("intermediate", tr("onboarding.quickstart.intermediate")),
            ("advanced", tr("onboarding.quickstart.advanced")),
        ):
            self._experience.addItem(label, code)
        current_experience = settings.get("general.experience_level", "beginner")
        experience_index = self._experience.findData(current_experience)
        if experience_index >= 0:
            self._experience.setCurrentIndex(experience_index)
        experience_row.addWidget(self._experience)
        experience_row.addStretch()
        preferences_layout.addLayout(experience_row)
        layout.addWidget(preferences)

        steps = [
            ("onboarding.quickstart.steps.song_forge", "onboarding.quickstart.steps.song_forge_desc"),
            ("onboarding.quickstart.steps.lyrics", "onboarding.quickstart.steps.lyrics_desc"),
            ("onboarding.quickstart.steps.midi", "onboarding.quickstart.steps.midi_desc"),
            ("onboarding.quickstart.steps.vocal", "onboarding.quickstart.steps.vocal_desc"),
            ("onboarding.quickstart.steps.sfx", "onboarding.quickstart.steps.sfx_desc"),
            ("onboarding.quickstart.steps.mixer", "onboarding.quickstart.steps.mixer_desc"),
            ("onboarding.quickstart.steps.producer", "onboarding.quickstart.steps.producer_desc"),
        ]

        for i, (name, desc) in enumerate(steps, 1):
            row_frame = QFrame()
            row_frame.setStyleSheet(f"""
                QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                    border-radius: 6px; }}
            """)
            row_layout = QHBoxLayout(row_frame)
            row_layout.setContentsMargins(12, 8, 12, 8)
            row_layout.setSpacing(10)

            num = QLabel(str(i))
            num.setMinimumSize(24, 24)
            num.setAlignment(Qt.AlignCenter)
            num.setStyleSheet(f"""
                background: {t['accent']}; color: white; border-radius: 12px;
                font-size: 8.25pt; font-weight: bold;
            """)
            n = ElidedLabel(tr(name), minimum_width=110)
            n.setStyleSheet(f"color: {t['text']}; font-size: 9pt; font-weight: bold; border: none;")
            d = QLabel(tr(desc))
            d.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt; border: none;")

            row_layout.addWidget(num)
            row_layout.addWidget(n)
            row_layout.addWidget(d, 1)

            layout.addWidget(row_frame)

        layout.addStretch()
        install_accessibility(self, tr("onboarding.accessibility.quickstart_name"))

    def output_dir(self) -> str:
        return self._output_dir.text().strip()

    def experience_level(self) -> str:
        return str(self._experience.currentData() or "beginner")

    def _browse_output_dir(self):
        path = choose_directory(
            self,
            tr("onboarding.quickstart.select_output"),
            operation_kind="onboarding_output_directory",
            dialog=QFileDialog,
        )
        if path:
            self._output_dir.setText(path)

    def focus_output_directory(self):
        self._output_dir.setFocus()


# ── Onboarding Dialog ──────────────────────────────────────────────────────────

class OnboardingWizard(QDialog):
    """First-run onboarding wizard."""

    completed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("onboarding.window_title"))
        self.setMinimumSize(700, 520)
        self.setModal(True)
        self._model_handoff = None

        t = ThemeEngine.get_colors()
        self.setStyleSheet(f"background: {t['background']};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Page stack
        self._pages = QStackedWidget()
        self._welcome = WelcomePage()
        self._system = SystemCheckPage()
        self._models = ModelGuidePage()
        self._quickstart = QuickStartPage()
        self._system.remediation_requested.connect(self._handle_system_remediation)

        self._pages.addWidget(self._welcome)
        self._pages.addWidget(self._system)
        self._pages.addWidget(self._models)
        self._pages.addWidget(self._quickstart)

        layout.addWidget(self._pages, 1)

        # Navigation bar
        nav = QFrame()
        nav.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border-top: 1px solid {t['border']}; }}
        """)
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(20, 10, 20, 10)

        # Step indicators
        self._step_labels = []
        for i, key in enumerate(("welcome", "system", "models", "quickstart")):
            lbl = QLabel(tr(f"onboarding.steps.{key}"))
            lbl.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt;")
            self._step_labels.append(lbl)
            nav_layout.addWidget(lbl)
            if i < 3:
                sep = QLabel(tr("onboarding.steps.separator"))
                sep.setStyleSheet(f"color: {t['border']}; font-size: 8.25pt;")
                nav_layout.addWidget(sep)

        nav_layout.addStretch()

        self._back_btn = QPushButton(tr("onboarding.actions.back"))
        self._back_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 5px;
                padding: 6px 16px; font-size: 9pt;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """)
        self._back_btn.clicked.connect(self._prev_page)
        self._back_btn.setVisible(False)

        self._skip_btn = QPushButton(tr("onboarding.actions.skip"))
        self._skip_btn.setStyleSheet(f"color: {t['text_secondary']}; border: none; padding: 6px 10px;")
        self._skip_btn.clicked.connect(self._skip)

        self._next_btn = QPushButton(tr("onboarding.actions.get_started"))
        self._next_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']}; color: white; border: none;
                border-radius: 5px; padding: 6px 20px;
                font-size: 9pt; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
        """)
        self._next_btn.clicked.connect(self._next_page)

        nav_layout.addWidget(self._skip_btn)
        nav_layout.addWidget(self._back_btn)
        nav_layout.addWidget(self._next_btn)

        layout.addWidget(nav)
        self._update_nav()
        install_accessibility(
            self,
            tr("onboarding.accessibility.wizard_name"),
            named_controls=[
                (self._pages, tr("onboarding.accessibility.pages_name"), tr("onboarding.accessibility.pages_description")),
                (self._back_btn, tr("onboarding.accessibility.back_name"), tr("onboarding.accessibility.back_description")),
                (self._skip_btn, tr("onboarding.accessibility.skip_name"), tr("onboarding.accessibility.skip_description")),
                (self._next_btn, tr("onboarding.accessibility.next_name"), tr("onboarding.accessibility.next_description")),
            ],
        )

    def _next_page(self):
        idx = self._pages.currentIndex()
        if idx == 0:
            # Run system check when leaving welcome
            self._system.run_checks()

        if idx < self._pages.count() - 1:
            self._pages.setCurrentIndex(idx + 1)
        else:
            self._finish()
        self._update_nav()

    def _prev_page(self):
        idx = self._pages.currentIndex()
        if idx > 0:
            self._pages.setCurrentIndex(idx - 1)
        self._update_nav()

    def _handle_system_remediation(self, key: str):
        if key == "disk":
            self._pages.setCurrentWidget(self._quickstart)
            self._quickstart.focus_output_directory()
        elif key in {"gpu", "ram"}:
            self._pages.setCurrentWidget(self._models)
            self._models.focus_model_selector()
        self._update_nav()

    def _update_nav(self):
        t = ThemeEngine.get_colors()
        idx = self._pages.currentIndex()
        self._back_btn.setVisible(idx > 0)

        is_last = idx == self._pages.count() - 1
        self._next_btn.setText(
            tr("onboarding.actions.launch") if is_last else tr("onboarding.actions.next")
        )

        for i, lbl in enumerate(self._step_labels):
            if i == idx:
                lbl.setStyleSheet(f"color: {t['accent']}; font-size: 8.25pt; font-weight: bold;")
            elif i < idx:
                lbl.setStyleSheet(f"color: {t['text']}; font-size: 8.25pt;")
            else:
                lbl.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt;")
        install_accessibility(
            self,
            tr("onboarding.accessibility.wizard_name"),
            named_controls=[
                (self._back_btn, tr("onboarding.accessibility.back_name"), tr("onboarding.accessibility.back_description")),
                (self._skip_btn, tr("onboarding.accessibility.skip_name"), tr("onboarding.accessibility.skip_description")),
                (self._next_btn, tr("onboarding.accessibility.next_name"), tr("onboarding.accessibility.next_description")),
            ],
        )

    def _finish(self):
        from core.settings import Settings

        settings = Settings()
        token = self._models.hf_token()
        if token and not token.startswith("hf_"):
            self._pages.setCurrentWidget(self._models)
            self._models.show_token_error(tr("onboarding.models.token_invalid"))
            self._update_nav()
            return
        if token:
            try:
                settings.set("model_hub.hf_token", token)
            except Exception as exc:
                self._pages.setCurrentWidget(self._models)
                self._models.show_token_error(tr("onboarding.models.token_save_failed", error=exc))
                self._update_nav()
                return
        self._models.clear_token_error()
        output_dir = self._quickstart.output_dir()
        if output_dir:
            settings.set("general.output_dir", output_dir)
        settings.set("general.experience_level", self._quickstart.experience_level())
        model_id = self._models.selected_model_id()
        action = self._models.selected_model_action()
        if model_id:
            self._model_handoff = {"model_id": model_id, "action": action}
        settings.set("general.onboarding_complete", True)
        settings.set("general.onboarding_skipped", False)
        self.completed.emit()
        self.accept()

    def model_handoff(self) -> Optional[dict]:
        """Return the selected Model Hub handoff after an accepted wizard."""
        return dict(self._model_handoff) if self._model_handoff else None

    def _skip(self):
        from core.settings import Settings

        Settings().set("general.onboarding_skipped", True)
        self.reject()
