"""
Slunder Studio — Autosave Coordinator
Drives the interval promised by Settings > General > Autosave interval: a dirty
project is saved and versioned on a timer, and never while another save runs.
"""
from __future__ import annotations

import time
from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from core.project import ProjectManager, ProjectVersion, get_project_manager
from core.settings import Settings

MIN_INTERVAL_SECONDS = 10
MAX_INTERVAL_SECONDS = 3600


def resolve_interval(settings: Optional[Settings] = None) -> int:
    """Clamp the configured autosave interval to a sane range."""
    settings = settings or Settings()
    try:
        value = int(settings.get("general.auto_save_interval", 60) or 60)
    except (TypeError, ValueError):
        value = 60
    return max(MIN_INTERVAL_SECONDS, min(MAX_INTERVAL_SECONDS, value))


class AutosaveCoordinator(QObject):
    """Saves the open project on the configured interval when it is dirty.

    Signals:
        autosaved(int, str)  - (version number, description)
        autosave_failed(str) - human-readable reason
        skipped(str)         - why a tick did nothing (clean, disabled, no project)
    """

    autosaved = Signal(int, str)
    autosave_failed = Signal(str)
    skipped = Signal(str)

    def __init__(self, project_manager: Optional[ProjectManager] = None,
                 settings: Optional[Settings] = None, parent=None):
        super().__init__(parent)
        self._projects = project_manager or get_project_manager()
        self._settings = settings or Settings()
        self._running = False
        self._last_autosave = 0.0
        self._interval = resolve_interval(self._settings)
        self._timer = QTimer(self)
        # Coarse is plenty for a minute-scale interval and costs less.
        self._timer.setTimerType(Qt.TimerType.CoarseTimer)
        self._timer.timeout.connect(self.tick)
        self._settings.on_change(self._on_setting_changed)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    @property
    def interval_seconds(self) -> int:
        return self._interval

    @property
    def enabled(self) -> bool:
        return bool(self._settings.get("general.auto_save_enabled", True))

    @property
    def last_autosave(self) -> float:
        return self._last_autosave

    def start(self):
        if not self.enabled:
            self.skipped.emit("Autosave is disabled in Settings.")
            return
        self._interval = resolve_interval(self._settings)
        self._timer.start(self._interval * 1000)

    def stop(self):
        self._timer.stop()

    def is_active(self) -> bool:
        return self._timer.isActive()

    def _on_setting_changed(self, key: str, new_value, old_value):
        if key not in ("general.auto_save_interval", "general.auto_save_enabled", "*"):
            return
        if not self.enabled:
            self.stop()
            return
        self._interval = resolve_interval(self._settings)
        if self._timer.isActive():
            self._timer.start(self._interval * 1000)
        else:
            self.start()

    # ── Work ───────────────────────────────────────────────────────────────────

    def tick(self) -> Optional[ProjectVersion]:
        """One autosave attempt. Safe to call directly (tests, manual flush)."""
        if self._running:
            self.skipped.emit("An autosave is already running.")
            return None
        if not self.enabled:
            self.skipped.emit("Autosave is disabled in Settings.")
            return None
        if self._projects.current is None:
            self.skipped.emit("No project is open.")
            return None
        if not self._projects.is_dirty:
            self.skipped.emit("No unsaved changes.")
            return None

        return self._persist_dirty_project()

    def flush(self) -> Optional[ProjectVersion]:
        """Persist dirty work immediately, even when interval autosave is off."""
        if self._running:
            self.skipped.emit("An autosave is already running.")
            return None
        if self._projects.current is None:
            self.skipped.emit("No project is open.")
            return None
        if not self._projects.is_dirty:
            self.skipped.emit("No unsaved changes.")
            return None
        return self._persist_dirty_project()

    def _persist_dirty_project(self) -> Optional[ProjectVersion]:
        self._running = True
        try:
            version = self._projects.autosave()
        except Exception as exc:  # noqa: BLE001 - reported, never silent
            self.autosave_failed.emit(f"{type(exc).__name__}: {exc}")
            return None
        finally:
            self._running = False

        if version is None:
            self.autosave_failed.emit("Autosave could not write the project.")
            return None
        self._last_autosave = time.time()
        self.autosaved.emit(version.version, version.description)
        return version
