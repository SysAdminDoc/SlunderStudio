"""Optional live MIDI input adapter.

The application does not make a backend a hard startup dependency.  When
``mido`` and a backend such as ``python-rtmidi`` are installed, this adapter
opens an input port and emits validated :class:`MidiMessage` objects.  The
same small adapter surface accepts a test double, keeping hardware out of
headless verification.
"""
from __future__ import annotations

import importlib
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from core.midi_controller import MidiMessage


MIDI_BACKEND_INSTALL_HINT = "Install optional mido and python-rtmidi packages."


def _load_backend(backend: Any = None) -> Any:
    if backend is not None:
        return backend
    return importlib.import_module("mido")


def list_midi_input_ports(backend: Any = None) -> tuple[list[str], str]:
    """Return input names and a user-facing backend/status message."""
    try:
        module = _load_backend(backend)
    except Exception:
        return [], MIDI_BACKEND_INSTALL_HINT
    try:
        names = module.get_input_names()
    except Exception as exc:
        return [], f"MIDI backend unavailable: {exc}"
    if not isinstance(names, (list, tuple)):
        return [], "MIDI backend returned an invalid input-port list."
    return [str(name) for name in names], ""


class MidiInputService(QObject):
    """Open one optional MIDI input and forward messages through Qt signals."""

    message_received = Signal(object)
    status_changed = Signal(str)
    error = Signal(str)

    def __init__(self, backend: Any = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._backend = backend
        self._port = None
        self._port_name = ""
        self._last_error = ""

    @property
    def is_running(self) -> bool:
        return self._port is not None

    @property
    def port_name(self) -> str:
        return self._port_name

    @property
    def last_error(self) -> str:
        return self._last_error

    def available_ports(self) -> list[str]:
        return list_midi_input_ports(self._backend)[0]

    def start(self, port_name: str = "") -> bool:
        """Start the selected port, returning false on missing hardware/backend."""
        self.stop()
        try:
            module = _load_backend(self._backend)
            selected = str(port_name or "").strip()
            self._port = module.open_input(
                selected or None,
                callback=self._on_backend_message,
            )
            self._port_name = selected
            self._last_error = ""
            label = selected or "system default"
            self.status_changed.emit(f"MIDI input listening on {label}")
            return True
        except Exception as exc:
            self._port = None
            self._port_name = ""
            self._last_error = str(exc)
            message = f"MIDI input unavailable: {exc}"
            self.error.emit(message)
            return False

    def stop(self) -> None:
        port, self._port = self._port, None
        self._port_name = ""
        if port is None:
            return
        try:
            port.close()
        except Exception as exc:
            self.error.emit(f"MIDI input close failed: {exc}")

    def _on_backend_message(self, message: Any) -> None:
        try:
            normalized = MidiMessage.from_backend(message)
        except (TypeError, ValueError):
            return
        self.message_received.emit(normalized)


__all__ = [
    "MIDI_BACKEND_INSTALL_HINT",
    "MidiInputService",
    "list_midi_input_ports",
]
