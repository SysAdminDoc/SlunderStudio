"""
Slunder Studio v0.1.0 — Onboarding Wizard
First-run experience: welcome, system check, model download prompt,
quick start guide, and preference setup.
"""
import os
import sys
import platform
from typing import Optional
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QStackedWidget, QWidget, QCheckBox, QComboBox,
    QProgressBar,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ui.theme import ThemeEngine


# ── System Check ───────────────────────────────────────────────────────────────

def check_system() -> dict:
    """Run system compatibility checks."""
    checks = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_ok": sys.version_info >= (3, 10),
        "arch": platform.machine(),
        "cpu": platform.processor() or "Unknown",
    }

    # RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        checks["ram_gb"] = round(ram.total / (1024**3), 1)
        checks["ram_ok"] = checks["ram_gb"] >= 8
    except ImportError:
        checks["ram_gb"] = 0
        checks["ram_ok"] = True  # assume OK if psutil missing

    # GPU / CUDA
    checks["cuda"] = False
    checks["gpu_name"] = "None detected"
    checks["vram_gb"] = 0
    try:
        import torch
        checks["cuda"] = torch.cuda.is_available()
        if checks["cuda"]:
            checks["gpu_name"] = torch.cuda.get_device_name(0)
            checks["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024**3), 1
            )
    except ImportError:
        pass

    # Disk space
    try:
        import shutil
        from core.settings import get_config_dir
        usage = shutil.disk_usage(get_config_dir())
        checks["disk_free_gb"] = round(usage.free / (1024**3), 1)
        checks["disk_ok"] = checks["disk_free_gb"] >= 10
    except Exception:
        checks["disk_free_gb"] = 0
        checks["disk_ok"] = True

    return checks


# ── Wizard Pages ───────────────────────────────────────────────────────────────

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
        logo.setStyleSheet(f"""
            color: transparent;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {t['accent']}, stop:0.5 #a371f7, stop:1 #f38ba8);
            -webkit-background-clip: text;
            background-clip: text;
        """)
        # Fallback for non-webkit
        logo.setStyleSheet(f"color: {t['accent']}; font-size: 28px; font-weight: bold;")
        layout.addWidget(logo)

        version = QLabel("v0.1.0")
        version.setAlignment(Qt.AlignCenter)
        version.setStyleSheet(f"color: {t['text_secondary']}; font-size: 14px;")
        layout.addWidget(version)

        tagline = QLabel("Offline AI Music Generation Suite")
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setStyleSheet(f"color: {t['text']}; font-size: 16px;")
        layout.addWidget(tagline)

        desc = QLabel(
            "Generate songs, compose MIDI, synthesize vocals, separate stems,\n"
            "create SFX, and master tracks — all locally on your machine.\n"
            "No cloud, no subscriptions, no limits."
        )
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {t['text_secondary']}; font-size: 12px; line-height: 1.6;")
        layout.addWidget(desc)


class SystemCheckPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("System Check")
        title.setStyleSheet(f"color: {t['text']}; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel("Checking your system compatibility...")
        subtitle.setStyleSheet(f"color: {t['text_secondary']}; font-size: 12px;")
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
        self._summary.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")
        layout.addWidget(self._summary)
        layout.addStretch()

    def run_checks(self):
        t = ThemeEngine.get_colors()
        checks = check_system()

        items = [
            ("Python", checks["python"], checks["python_ok"],
             "3.10+ required"),
            ("Operating System", f"{checks['os']} {checks['arch']}", True, ""),
            ("GPU / CUDA", checks["gpu_name"],
             checks["cuda"],
             f"{checks['vram_gb']} GB VRAM" if checks["cuda"] else "CPU-only mode"),
            ("RAM", f"{checks['ram_gb']} GB",
             checks.get("ram_ok", True), "8 GB+ recommended"),
            ("Disk Space", f"{checks['disk_free_gb']} GB free",
             checks.get("disk_ok", True), "10 GB+ recommended for models"),
        ]

        for label, value, ok, note in items:
            row = QHBoxLayout()
            icon = QLabel("OK" if ok else "!!")
            icon.setFixedWidth(24)
            icon.setAlignment(Qt.AlignCenter)
            icon.setStyleSheet(
                f"color: #238636; font-weight: bold; font-size: 11px;"
                if ok else f"color: #d29922; font-weight: bold; font-size: 11px;"
            )
            name = QLabel(f"{label}:")
            name.setFixedWidth(110)
            name.setStyleSheet(f"color: {t['text']}; font-size: 12px; border: none;")
            val = QLabel(str(value))
            val.setStyleSheet(f"color: {t['text']}; font-size: 12px; font-weight: bold; border: none;")
            note_l = QLabel(note)
            note_l.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px; border: none;")
            row.addWidget(icon)
            row.addWidget(name)
            row.addWidget(val, 1)
            row.addWidget(note_l)
            self._checks_layout.addLayout(row)

        if checks["cuda"]:
            self._summary.setText(
                f"GPU acceleration available. {checks['gpu_name']} with "
                f"{checks['vram_gb']} GB VRAM will be used for AI inference."
            )
        else:
            self._summary.setText(
                "No CUDA GPU detected. Models will run on CPU (slower). "
                "Install PyTorch with CUDA support for GPU acceleration."
            )


class ModelGuidePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("AI Models")
        title.setStyleSheet(f"color: {t['text']}; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        info = QLabel(
            "Slunder Studio uses AI models that run locally on your machine. "
            "Models are downloaded from HuggingFace and stored in your config directory. "
            "You can manage models anytime from the Model Hub."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {t['text_secondary']}; font-size: 12px;")
        layout.addWidget(info)

        # Recommended models
        models_frame = QFrame()
        models_frame.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        ml = QVBoxLayout(models_frame)
        ml.setContentsMargins(16, 12, 16, 12)
        ml.setSpacing(10)

        models = [
            ("ACE-Step", "Song generation from lyrics + tags", "~3 GB", True),
            ("Llama 3.2 1B", "Lyrics generation + MIDI composition", "~2 GB", True),
            ("DiffSinger", "Singing voice synthesis", "~500 MB", False),
            ("RVC v2", "Voice conversion", "~200 MB per voice", False),
            ("Demucs (htdemucs)", "Stem separation", "~300 MB", False),
            ("Stable Audio Open", "SFX generation", "~3 GB", False),
        ]

        for name, desc, size, recommended in models:
            row = QHBoxLayout()
            tag = QLabel("REC" if recommended else "OPT")
            tag.setFixedWidth(30)
            tag.setAlignment(Qt.AlignCenter)
            tag.setStyleSheet(
                f"color: white; background: #238636; border-radius: 3px; "
                f"font-size: 8px; font-weight: bold; padding: 2px;"
                if recommended else
                f"color: {t['text_secondary']}; background: {t['border']}; "
                f"border-radius: 3px; font-size: 8px; padding: 2px;"
            )
            n = QLabel(name)
            n.setFixedWidth(140)
            n.setStyleSheet(f"color: {t['text']}; font-size: 12px; font-weight: bold; border: none;")
            d = QLabel(desc)
            d.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px; border: none;")
            s = QLabel(size)
            s.setFixedWidth(80)
            s.setAlignment(Qt.AlignRight)
            s.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px; border: none;")
            row.addWidget(tag)
            row.addWidget(n)
            row.addWidget(d, 1)
            row.addWidget(s)
            ml.addLayout(row)

        layout.addWidget(models_frame)

        note = QLabel(
            "You can skip model downloads now and install them later from Model Hub. "
            "The app works without models — generation will use built-in fallbacks."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")
        layout.addWidget(note)
        layout.addStretch()


class QuickStartPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Quick Start")
        title.setStyleSheet(f"color: {t['text']}; font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        steps = [
            ("Song Forge", "Generate full songs from lyrics and style tags with ACE-Step"),
            ("Lyrics Engine", "Write lyrics using AI templates, rhyme tools, and LLM generation"),
            ("MIDI Studio", "Compose MIDI with piano roll editor or text-to-MIDI AI"),
            ("Vocal Suite", "Add singing voices, convert vocals, or separate stems"),
            ("SFX Generator", "Create sound effects from text descriptions"),
            ("Mixer", "Combine all tracks, apply mastering, and export your final song"),
            ("AI Producer", "Describe your vision in one prompt and let AI build the full song"),
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
            num.setFixedSize(24, 24)
            num.setAlignment(Qt.AlignCenter)
            num.setStyleSheet(f"""
                background: {t['accent']}; color: white; border-radius: 12px;
                font-size: 11px; font-weight: bold;
            """)
            n = QLabel(name)
            n.setFixedWidth(110)
            n.setStyleSheet(f"color: {t['text']}; font-size: 12px; font-weight: bold; border: none;")
            d = QLabel(desc)
            d.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px; border: none;")

            row_layout.addWidget(num)
            row_layout.addWidget(n)
            row_layout.addWidget(d, 1)

            layout.addWidget(row_frame)

        layout.addStretch()


# ── Onboarding Dialog ──────────────────────────────────────────────────────────

class OnboardingWizard(QDialog):
    """First-run onboarding wizard."""

    completed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to Slunder Studio")
        self.setMinimumSize(700, 520)
        self.setModal(True)

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
        for i, name in enumerate(["Welcome", "System", "Models", "Quick Start"]):
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")
            self._step_labels.append(lbl)
            nav_layout.addWidget(lbl)
            if i < 3:
                sep = QLabel(" > ")
                sep.setStyleSheet(f"color: {t['border']}; font-size: 11px;")
                nav_layout.addWidget(sep)

        nav_layout.addStretch()

        self._back_btn = QPushButton("Back")
        self._back_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 5px;
                padding: 6px 16px; font-size: 12px;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """)
        self._back_btn.clicked.connect(self._prev_page)
        self._back_btn.setVisible(False)

        self._next_btn = QPushButton("Get Started")
        self._next_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']}; color: white; border: none;
                border-radius: 5px; padding: 6px 20px;
                font-size: 12px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
        """)
        self._next_btn.clicked.connect(self._next_page)

        nav_layout.addWidget(self._back_btn)
        nav_layout.addWidget(self._next_btn)

        layout.addWidget(nav)
        self._update_nav()

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

    def _update_nav(self):
        t = ThemeEngine.get_colors()
        idx = self._pages.currentIndex()
        self._back_btn.setVisible(idx > 0)

        is_last = idx == self._pages.count() - 1
        self._next_btn.setText("Launch Studio" if is_last else "Next")

        for i, lbl in enumerate(self._step_labels):
            if i == idx:
                lbl.setStyleSheet(f"color: {t['accent']}; font-size: 11px; font-weight: bold;")
            elif i < idx:
                lbl.setStyleSheet(f"color: {t['text']}; font-size: 11px;")
            else:
                lbl.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")

    def _finish(self):
        from core.settings import Settings
        settings = Settings()
        settings.set("general.onboarding_complete", True)
        self.completed.emit()
        self.accept()
