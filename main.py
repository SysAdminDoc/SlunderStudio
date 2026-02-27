#!/usr/bin/env python3
"""
Slunder Studio v0.1.0
Offline AI Music Generation Suite

Run: python main.py
"""
import sys
import os
import subprocess
import importlib
import traceback

APP_VERSION = "0.1.0"


# ── Phase 1: Console Bootstrap (no GUI deps yet) ─────────────────────────────

def _phase1_bootstrap():
    """Install absolute minimum deps needed before GUI can start."""
    if sys.version_info < (3, 10):
        print(f"Slunder Studio requires Python 3.10+. Current: {sys.version}")
        input("Press Enter to exit...")
        sys.exit(1)

    # Ensure pip exists
    try:
        import pip  # noqa: F401
    except ImportError:
        print("[Slunder Studio] Installing pip...")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--default-pip"])

    # Phase 1 packages: everything needed before the main window can render.
    # These are installed in console before any GUI appears.
    phase1 = {
        "PySide6": "PySide6",
        "numpy": "numpy",
        "pyqtgraph": "pyqtgraph",
        "sounddevice": "sounddevice",
        "soundfile": "soundfile",
        "huggingface_hub": "huggingface-hub",
        "librosa": "librosa",
        "psutil": "psutil",
    }
    missing = []
    for import_name, pip_name in phase1.items():
        try:
            importlib.import_module(import_name)
        except Exception:
            missing.append((import_name, pip_name))

    if missing:
        print(f"[Slunder Studio] Installing {len(missing)} packages...")
        for import_name, pip_name in missing:
            print(f"  -> {pip_name}")
            _pip_install(pip_name)
            # Verify it actually works
            try:
                importlib.invalidate_caches()
                importlib.import_module(import_name)
            except Exception:
                print(f"  [!] {pip_name} install may have failed, will retry later")


def _pip_install(pip_name: str):
    """Robust pip install with fallback strategies and site-packages refresh."""
    strategies = [
        [sys.executable, "-m", "pip", "install", pip_name],
        [sys.executable, "-m", "pip", "install", pip_name, "--user"],
        [sys.executable, "-m", "pip", "install", pip_name, "--break-system-packages"],
        [sys.executable, "-m", "pip", "install", pip_name, "--force-reinstall"],
    ]
    for cmd in strategies:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                # Refresh import paths so newly installed packages are visible
                importlib.invalidate_caches()
                import site
                if hasattr(site, "getusersitepackages"):
                    usp = site.getusersitepackages()
                    if usp and usp not in sys.path:
                        sys.path.insert(0, usp)
                print(f"[Slunder Studio] Installed: {pip_name}")
                return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    print(f"[Slunder Studio] WARNING: Failed to install {pip_name}")


_phase1_bootstrap()


# Clean stale bytecode — prevents old .pyc from overriding updated .py files
def _clean_pycache():
    import shutil
    root = os.path.dirname(os.path.abspath(__file__))
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d == "__pycache__":
                try:
                    shutil.rmtree(os.path.join(dirpath, d))
                except OSError:
                    pass

_clean_pycache()


# ── Phase 2: GUI Splash + Remaining Dependencies ─────────────────────────────

from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar,
)
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QFont, QColor  # noqa: E402


class _SplashInstaller(QWidget):
    """Dark splash screen — safety net for any deps Phase 1 missed."""

    # All packages the app needs (Phase 1 should have gotten these already)
    PACKAGES = [
        ("PySide6", "PySide6"),
        ("numpy", "numpy"),
        ("sounddevice", "sounddevice"),
        ("soundfile", "soundfile"),
        ("huggingface_hub", "huggingface-hub"),
        ("pyqtgraph", "pyqtgraph"),
        ("librosa", "librosa"),
        ("psutil", "psutil"),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slunder Studio")
        self.setFixedSize(460, 200)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setStyleSheet("background: #1e1e2e;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(12)

        title = QLabel("Slunder Studio")
        title.setStyleSheet(
            "font-size: 22px; font-weight: 800; color: #89b4fa;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._status = QLabel("Checking dependencies...")
        self._status.setStyleSheet("font-size: 12px; color: #a6adc8;")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(8)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet("""
            QProgressBar {
                background: #313244; border: none; border-radius: 4px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #89b4fa, stop:1 #cba6f7
                );
                border-radius: 4px;
            }
        """)
        self._progress.setMaximum(len(self.PACKAGES))
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        layout.addStretch()

        self._install_index = 0
        self._all_ok = True

        # Center on screen
        self.show()
        screen = self.screen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2,
        )

        # Start installing after event loop begins
        QTimer.singleShot(100, self._install_next)

    def _install_next(self):
        """Install packages one at a time, keeping the GUI responsive."""
        if self._install_index >= len(self.PACKAGES):
            self._status.setText("Starting Slunder Studio...")
            self._progress.setValue(len(self.PACKAGES))
            QTimer.singleShot(300, self._finish)
            return

        import_name, pip_name = self.PACKAGES[self._install_index]

        try:
            importlib.import_module(import_name)
            # Already installed
        except ImportError:
            self._status.setText(f"Installing {pip_name}...")
            self.repaint()
            QApplication.processEvents()

            try:
                from core.deps import _install
                _install(pip_name, import_name)
            except ImportError as e:
                print(f"[Slunder Studio] WARNING: {e}")
                self._all_ok = False

        self._install_index += 1
        self._progress.setValue(self._install_index)
        self._status.setText(
            f"Checked {self._install_index}/{len(self.PACKAGES)} packages"
        )
        QApplication.processEvents()

        # Yield back to event loop before next package
        QTimer.singleShot(10, self._install_next)

    def _finish(self):
        self.close()
        _launch_app()


def _needs_install() -> bool:
    """Quick check if any packages are missing or broken."""
    for import_name, _ in _SplashInstaller.PACKAGES:
        try:
            importlib.import_module(import_name)
        except Exception:
            return True
    return False


# ── Crash Logging ─────────────────────────────────────────────────────────────

def _setup_crash_handler():
    """Global exception handler with crash log."""
    config_dir = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~/.config")),
        "SlunderStudio"
    )
    os.makedirs(config_dir, exist_ok=True)
    crash_file = os.path.join(config_dir, "crash.log")

    def handler(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        timestamp = __import__("datetime").datetime.now().isoformat()
        entry = (
            f"\n{'='*60}\n{timestamp}\n"
            f"Slunder Studio v{APP_VERSION}\n{'='*60}\n{msg}\n"
        )
        try:
            with open(crash_file, "a", encoding="utf-8") as f:
                f.write(entry)
        except IOError:
            pass

        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(
                    0,
                    f"Slunder Studio crashed.\n\n"
                    f"Crash log: {crash_file}\n\n{msg[:500]}",
                    "Slunder Studio — Fatal Error", 0x10,
                )
            except Exception:
                pass

        print(f"FATAL: {msg}", file=sys.stderr)
        sys.exit(1)

    sys.excepthook = handler


# ── Single Instance Lock ──────────────────────────────────────────────────────

def _acquire_lock() -> bool:
    """Prevent multiple instances via lockfile."""
    config_dir = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~/.config")),
        "SlunderStudio"
    )
    os.makedirs(config_dir, exist_ok=True)
    lock_file = os.path.join(config_dir, "studio.lock")

    try:
        if os.path.isfile(lock_file):
            age = __import__("time").time() - os.path.getmtime(lock_file)
            if age < 300:
                try:
                    with open(lock_file) as f:
                        pid = int(f.read().strip())
                    os.kill(pid, 0)
                    return False
                except (ValueError, OSError, ProcessLookupError):
                    pass

        with open(lock_file, "w") as f:
            f.write(str(os.getpid()))

        import atexit
        atexit.register(
            lambda: os.remove(lock_file) if os.path.isfile(lock_file) else None
        )
        return True
    except Exception:
        return True


# ── Application Launch ────────────────────────────────────────────────────────

def _launch_app():
    """Launch main application window (called after deps are ready)."""
    from PySide6.QtWidgets import QMessageBox

    if not _acquire_lock():
        QMessageBox.warning(
            None, "Slunder Studio", "Another instance is already running."
        )
        sys.exit(0)

    app = QApplication.instance()

    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)

    from core.settings import Settings
    from ui.theme import build_stylesheet

    settings = Settings()
    accent = settings.get("general.theme_accent", "#89b4fa")
    app.setStyleSheet(build_stylesheet(accent))

    def on_settings_change(key, new_val, old_val):
        if key == "general.theme_accent":
            app.setStyleSheet(build_stylesheet(new_val))

    settings.on_change(on_settings_change)

    if not settings.get("general.onboarding_complete", False):
        from ui.onboarding import OnboardingWizard
        wizard = OnboardingWizard()
        wizard.exec()
        settings.set("general.onboarding_complete", True)

    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()


def main():
    """Entry point."""
    _setup_crash_handler()

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Slunder Studio")
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("SysAdminDoc")

    if _needs_install():
        # Show splash and install missing deps
        splash = _SplashInstaller()
        sys.exit(app.exec())
    else:
        # Everything ready — launch directly
        _launch_app()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
