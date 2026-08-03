#!/usr/bin/env python3
"""
Slunder Studio
Offline AI Music Generation Suite

Run: python main.py
"""
import multiprocessing
multiprocessing.freeze_support()

import atexit
import errno
import json
import sys
import os
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from core.version import APP_VERSION


def _is_frozen() -> bool:
    """Return True when running from a PyInstaller executable."""
    return getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS")


# ── Phase 1: Console Bootstrap (no GUI deps yet) ─────────────────────────────

def _phase1_bootstrap():
    """Report missing core dependencies before importing PySide6."""
    if sys.version_info < (3, 10):
        sys.stderr.write(
            f"Slunder Studio requires Python 3.10+. Current: {sys.version}\n"
        )
        sys.exit(1)

    from core.dependency_profiles import (
        DependencyProfileError,
        validate_profile_registry_security,
    )

    try:
        validate_profile_registry_security()
    except DependencyProfileError as exc:
        sys.stderr.write(f"Unsafe optional dependency profile registry: {exc}\n")
        sys.exit(1)

    if _is_frozen():
        return []

    from core.deps import CORE_RUNTIME_PACKAGES, dependency_status

    missing = dependency_status(CORE_RUNTIME_PACKAGES)
    if missing and any(import_name == "PySide6" for import_name, _ in missing):
        _print_dependency_diagnostics(missing)
        _show_dependency_diagnostics_tk(missing)
        sys.exit(1)
    return missing


def _print_dependency_diagnostics(missing: Sequence[tuple[str, str]]) -> None:
    from core.deps import format_missing_dependency_message
    sys.stderr.write(format_missing_dependency_message(missing) + "\n")


def _show_dependency_diagnostics_tk(
    missing: Sequence[tuple[str, str]],
) -> None:
    """Best-effort dark diagnostics when Qt itself is missing."""
    try:
        import tkinter as tk
        from core.deps import format_missing_dependency_message

        root = tk.Tk()
        root.title("Slunder Studio - Missing Dependencies")
        root.geometry("720x420")
        root.configure(bg="#1e1e2e")

        title = tk.Label(
            root,
            text="Slunder Studio cannot start",
            bg="#1e1e2e",
            fg="#f38ba8",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(padx=24, pady=(22, 8), anchor="w")

        text = tk.Text(
            root,
            bg="#11111b",
            fg="#cdd6f4",
            insertbackground="#cdd6f4",
            relief="flat",
            wrap="word",
            font=("Consolas", 10),
        )
        text.insert("1.0", format_missing_dependency_message(missing))
        text.configure(state="disabled")
        text.pack(padx=24, pady=8, fill="both", expand=True)

        close = tk.Button(
            root,
            text="Close",
            command=root.destroy,
            bg="#89b4fa",
            fg="#11111b",
            relief="flat",
            font=("Segoe UI", 10, "bold"),
            padx=20,
            pady=8,
        )
        close.pack(padx=24, pady=(4, 20), anchor="e")
        root.mainloop()
    except Exception:
        pass


_BOOTSTRAP_MISSING = _phase1_bootstrap()


# ── Phase 2: GUI Splash + Remaining Dependencies ─────────────────────────────

from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton,
)
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402


class _DependencyDiagnostics(QWidget):
    """Dark diagnostics screen for missing runtime dependencies."""

    def __init__(self, missing: Sequence[tuple[str, str]]):
        super().__init__()
        from core.deps import (
            format_missing_dependency_message,
            package_install_command,
            setup_commands,
        )

        self._commands = setup_commands()
        self._direct_command = package_install_command(
            pip_name for _, pip_name in missing
        )
        self.setWindowTitle("Slunder Studio - Dependency Diagnostics")
        self.setMinimumSize(720, 440)
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet("""
            QWidget {
                background: #1e1e2e; color: #cdd6f4; font-family: "Segoe UI";
            }
            QLabel#Title { color: #f38ba8; font-size: 22px; font-weight: 800; }
            QLabel#Subtitle { color: #a6adc8; font-size: 12px; }
            QPlainTextEdit {
                background: #11111b; color: #cdd6f4; border: 1px solid #313244;
                border-radius: 6px; padding: 10px; font-family: Consolas, "Courier New";
                font-size: 10pt;
            }
            QPushButton {
                background: #313244; color: #cdd6f4; border: 1px solid #45475a;
                border-radius: 5px; padding: 8px 14px; font-weight: 700;
            }
            QPushButton:hover { background: #45475a; }
            QPushButton#Primary {
                background: #89b4fa; color: #11111b; border-color: #89b4fa;
            }
            QPushButton#Primary:hover { background: #74c7ec; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        title = QLabel("Missing Python dependencies")
        title.setObjectName("Title")
        layout.addWidget(title)

        subtitle = QLabel(
            "Install the dependencies below, then launch Slunder Studio again."
        )
        subtitle.setObjectName("Subtitle")
        layout.addWidget(subtitle)

        details = QPlainTextEdit()
        details.setReadOnly(True)
        details.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        details.setPlainText(format_missing_dependency_message(missing))
        details.setAccessibleName("Dependency diagnostics")
        layout.addWidget(details, 1)

        self._status = QLabel("")
        self._status.setObjectName("Subtitle")
        layout.addWidget(self._status)

        buttons = QHBoxLayout()
        buttons.addStretch()

        copy_setup = QPushButton("Copy Setup Command")
        copy_setup.setObjectName("Primary")
        copy_setup.clicked.connect(self._copy_setup_command)
        buttons.addWidget(copy_setup)

        copy_direct = QPushButton("Copy Direct Command")
        copy_direct.clicked.connect(self._copy_direct_command)
        buttons.addWidget(copy_direct)

        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)

        layout.addLayout(buttons)

    def _copy_setup_command(self):
        QApplication.clipboard().setText("\n".join(self._commands))
        self._status.setText("Setup command copied.")

    def _copy_direct_command(self):
        QApplication.clipboard().setText(self._direct_command)
        self._status.setText("Direct install command copied.")


def _missing_core_dependencies() -> list[tuple[str, str]]:
    if _is_frozen():
        return []
    from core.deps import CORE_RUNTIME_PACKAGES, dependency_status
    return dependency_status(CORE_RUNTIME_PACKAGES)


# ── Crash Logging ─────────────────────────────────────────────────────────────

def _setup_crash_handler():
    """Global exception handler with crash log."""
    config_dir = os.path.join(
        os.environ.get("APPDATA", os.path.expanduser("~/.config")),
        "SlunderStudio"
    )
    os.makedirs(config_dir, exist_ok=True)
    crash_file = os.path.join(config_dir, "crash.log")
    from core.logging_setup import configure_logging

    configure_logging(config_dir)

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

        logging.getLogger(__name__).critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        sys.exit(1)

    sys.excepthook = handler


# ── Single Instance Lock ──────────────────────────────────────────────────────

_LOCK_HANDLE = None
_LOCK_PATH: Path | None = None
_LOCK_FAILURE_REASON = ""
_LOCK_CONTENTION = False
_LOCK_ATEXIT_REGISTERED = False


def _lock_file_path() -> Path:
    """Return the stable per-user lock path used by every application launch."""
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / ".config"
    return base / "SlunderStudio" / "studio.lock"


def _acquire_file_lock(handle) -> None:
    """Acquire one non-blocking OS lock byte, or raise on contention/failure."""
    handle.seek(0)
    if handle.read(1) == b"":
        handle.seek(0)
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)

    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_file_lock(handle) -> None:
    """Release an OS lock and close its descriptor without masking shutdown."""
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError):
        pass
    finally:
        try:
            handle.close()
        except (OSError, ValueError):
            pass


def _is_lock_contention(exc: BaseException) -> bool:
    """Distinguish an occupied lock from an inaccessible lock path."""
    if isinstance(exc, (BlockingIOError, PermissionError)):
        return True
    if getattr(exc, "errno", None) in {errno.EACCES, errno.EAGAIN}:
        return True
    return getattr(exc, "winerror", None) in {33, 36}


def _lock_record() -> dict[str, object]:
    executable = Path(sys.executable).resolve()
    command = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else executable
    return {
        "schema": 2,
        "pid": os.getpid(),
        "executable": str(executable),
        "executable_name": executable.name,
        "command": str(command),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_lock_record(handle) -> None:
    payload = json.dumps(_lock_record(), sort_keys=True).encode("utf-8")
    handle.seek(0)
    handle.truncate()
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())


def _lock_failure_message() -> str:
    path = str(_LOCK_PATH or _lock_file_path())
    if _LOCK_CONTENTION:
        return (
            "Another Slunder Studio instance is already running.\n\n"
            f"Lock file: {path}\n\n"
            "The operating-system lock releases automatically when that process exits. "
            "If no Slunder Studio process remains, confirm that first, then delete the "
            "lock file and retry."
        )
    detail = _LOCK_FAILURE_REASON or "Unknown lock error."
    return (
        "Slunder Studio stopped before startup because it could not acquire its "
        "single-instance lock.\n\n"
        f"Lock file: {path}\n\n"
        f"{detail}\n\n"
        "Check the directory permissions and retry."
    )


def _release_lock() -> None:
    global _LOCK_HANDLE
    handle = _LOCK_HANDLE
    _LOCK_HANDLE = None
    if handle is not None:
        _release_file_lock(handle)


def _acquire_lock() -> bool:
    """Acquire the process-wide lock, failing closed on every error."""
    global _LOCK_HANDLE, _LOCK_PATH, _LOCK_FAILURE_REASON, _LOCK_CONTENTION
    global _LOCK_ATEXIT_REGISTERED

    if _LOCK_HANDLE is not None:
        return True

    lock_file = _lock_file_path()
    _LOCK_PATH = lock_file
    _LOCK_FAILURE_REASON = ""
    _LOCK_CONTENTION = False
    handle = None
    try:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_file.open("a+b", buffering=0)
        try:
            _acquire_file_lock(handle)
        except Exception as exc:
            if _is_lock_contention(exc):
                _LOCK_CONTENTION = True
                _LOCK_FAILURE_REASON = "The lock file is held by another process."
            else:
                _LOCK_FAILURE_REASON = f"{type(exc).__name__}: {exc}"
            return False
        try:
            _write_lock_record(handle)
        except Exception as exc:
            _LOCK_FAILURE_REASON = f"{type(exc).__name__}: {exc}"
            return False
        _LOCK_HANDLE = handle
        handle = None
        if not _LOCK_ATEXIT_REGISTERED:
            atexit.register(_release_lock)
            _LOCK_ATEXIT_REGISTERED = True
        return True
    except Exception as exc:
        _LOCK_FAILURE_REASON = f"{type(exc).__name__}: {exc}"
        return False
    finally:
        if handle is not None:
            _release_file_lock(handle)


# ── Application Launch ────────────────────────────────────────────────────────

def _launch_app():
    """Launch main application window (called after deps are ready)."""
    from PySide6.QtWidgets import QMessageBox

    if not _acquire_lock():
        QMessageBox.warning(
            None, "Slunder Studio — Single Instance", _lock_failure_message()
        )
        sys.exit(0 if _LOCK_CONTENTION else 1)

    app = QApplication.instance()

    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)

    from core.settings import Settings
    from core.i18n import set_locale
    from ui.theme import build_stylesheet

    settings = Settings()
    set_locale(settings.get("general.ui_locale", "en"), persist=False, app=app)
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
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)

    missing = _BOOTSTRAP_MISSING or _missing_core_dependencies()
    if missing:
        diagnostics = _DependencyDiagnostics(missing)
        diagnostics.show()
        sys.exit(app.exec())

    _launch_app()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
