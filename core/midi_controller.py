"""Validated MIDI controller bindings and action dispatch.

The router is deliberately independent of Qt and any MIDI backend.  This
keeps settings and headless tests usable when no hardware backend is
installed, while the UI can inject messages from an optional input service.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


MIDI_CONTROLLER_SCHEMA_VERSION = 1
MIDI_CHANNEL_OMNI = -1
MIDI_MESSAGE_TYPES = ("cc", "note")
MIDI_BINDING_MODES = ("absolute", "trigger", "toggle")

MIDI_ACTION_LABELS = {
    "transport.toggle": "Transport play / pause",
    "transport.stop": "Transport stop",
    "mixer.volume": "Selected mixer volume",
    "mixer.pan": "Selected mixer pan",
    "mixer.mute": "Selected mixer mute",
    "mixer.solo": "Selected mixer solo",
    "piano.quantize": "Piano Roll quantize",
    "piano.swing": "Piano Roll swing",
    "piano.humanize": "Piano Roll humanize",
}

# The defaults follow common MIDI conventions and are intentionally inactive
# until the user enables MIDI input in Settings.
DEFAULT_MIDI_BINDINGS = (
    {
        "action": "transport.toggle",
        "message_type": "cc",
        "channel": MIDI_CHANNEL_OMNI,
        "number": 20,
        "mode": "trigger",
    },
    {
        "action": "transport.stop",
        "message_type": "cc",
        "channel": MIDI_CHANNEL_OMNI,
        "number": 21,
        "mode": "trigger",
    },
    {
        "action": "mixer.volume",
        "message_type": "cc",
        "channel": MIDI_CHANNEL_OMNI,
        "number": 7,
        "mode": "absolute",
    },
    {
        "action": "mixer.pan",
        "message_type": "cc",
        "channel": MIDI_CHANNEL_OMNI,
        "number": 10,
        "mode": "absolute",
    },
    {
        "action": "mixer.mute",
        "message_type": "cc",
        "channel": MIDI_CHANNEL_OMNI,
        "number": 29,
        "mode": "toggle",
    },
    {
        "action": "mixer.solo",
        "message_type": "cc",
        "channel": MIDI_CHANNEL_OMNI,
        "number": 30,
        "mode": "toggle",
    },
    {
        "action": "piano.quantize",
        "message_type": "note",
        "channel": MIDI_CHANNEL_OMNI,
        "number": 36,
        "mode": "trigger",
    },
    {
        "action": "piano.swing",
        "message_type": "note",
        "channel": MIDI_CHANNEL_OMNI,
        "number": 37,
        "mode": "trigger",
    },
    {
        "action": "piano.humanize",
        "message_type": "note",
        "channel": MIDI_CHANNEL_OMNI,
        "number": 38,
        "mode": "trigger",
    },
)

_ACTION_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class MidiMessage:
    """A bounded MIDI message accepted by the controller router."""

    message_type: str
    channel: int
    number: int
    value: int

    def __post_init__(self):
        if self.message_type not in {"cc", "note_on", "note_off"}:
            raise ValueError(f"Unsupported MIDI message type: {self.message_type}")
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (self.channel, self.number, self.value)
        ):
            raise ValueError("MIDI channel, number, and value must be integers")
        if not 0 <= self.channel <= 15:
            raise ValueError("MIDI channel must be between 0 and 15")
        if not 0 <= self.number <= 127:
            raise ValueError("MIDI number must be between 0 and 127")
        if not 0 <= self.value <= 127:
            raise ValueError("MIDI value must be between 0 and 127")

    @classmethod
    def from_values(
        cls,
        message_type: str,
        channel: int,
        number: int,
        value: int,
    ) -> "MidiMessage":
        """Create a message while rejecting coercion-prone values."""
        values = (channel, number, value)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
            raise ValueError("MIDI channel, number, and value must be integers")
        return cls(message_type, channel, number, value)

    @classmethod
    def from_backend(cls, message: Any) -> "MidiMessage":
        """Convert a mido-compatible message without importing mido."""
        message_type = str(getattr(message, "type", ""))
        if message_type == "control_change":
            message_type = "cc"
            number = getattr(message, "control", None)
            value = getattr(message, "value", None)
        elif message_type in {"note_on", "note_off"}:
            number = getattr(message, "note", None)
            value = getattr(message, "velocity", None)
        else:
            raise ValueError(f"Unsupported MIDI backend message: {message_type}")
        channel = getattr(message, "channel", None)
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in (channel, number, value)):
            raise ValueError("MIDI backend message has invalid numeric fields")
        return cls.from_values(message_type, channel, number, value)

    @property
    def normalized_value(self) -> float:
        return self.value / 127.0

    @property
    def is_pressed(self) -> bool:
        return self.message_type == "note_on" and self.value > 0 or (
            self.message_type == "cc" and self.value > 0
        )


@dataclass(frozen=True)
class MidiBinding:
    """One physical MIDI control mapped to a public application action."""

    action: str
    message_type: str
    channel: int
    number: int
    mode: str

    def __post_init__(self):
        if not isinstance(self.action, str) or not _ACTION_RE.fullmatch(self.action):
            raise ValueError("MIDI action must be a short dotted identifier")
        if self.message_type not in MIDI_MESSAGE_TYPES:
            raise ValueError(f"Unsupported MIDI binding type: {self.message_type}")
        if isinstance(self.channel, bool) or not isinstance(self.channel, int):
            raise ValueError("MIDI binding channel must be an integer")
        if self.channel != MIDI_CHANNEL_OMNI and not 0 <= self.channel <= 15:
            raise ValueError("MIDI binding channel must be -1 or between 0 and 15")
        if isinstance(self.number, bool) or not isinstance(self.number, int) or not 0 <= self.number <= 127:
            raise ValueError("MIDI binding number must be between 0 and 127")
        if self.mode not in MIDI_BINDING_MODES:
            raise ValueError(f"Unsupported MIDI binding mode: {self.mode}")

    @classmethod
    def from_dict(cls, value: Any) -> "MidiBinding":
        if not isinstance(value, dict):
            raise ValueError("MIDI binding must be an object")
        number = value.get("number", value.get("controller"))
        return cls(
            action=value.get("action", ""),
            message_type=value.get("message_type", value.get("type", "cc")),
            channel=value.get("channel", MIDI_CHANNEL_OMNI),
            number=number,
            mode=value.get("mode", "absolute"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MIDI_CONTROLLER_SCHEMA_VERSION,
            "action": self.action,
            "message_type": self.message_type,
            "channel": self.channel,
            "number": self.number,
            "mode": self.mode,
        }

    @property
    def label(self) -> str:
        return MIDI_ACTION_LABELS.get(self.action, self.action)


@dataclass(frozen=True)
class MidiActionEvent:
    """A normalized action produced by the router."""

    action: str
    value: float
    binding: MidiBinding
    message: MidiMessage


def normalized_bindings(value: Any) -> list[MidiBinding]:
    """Parse persisted bindings, falling back safely to the defaults."""
    source: Iterable[Any]
    if isinstance(value, list):
        source = value
    else:
        source = DEFAULT_MIDI_BINDINGS

    parsed: list[MidiBinding] = []
    seen: set[tuple[str, str, int, int]] = set()
    for item in source:
        try:
            binding = item if isinstance(item, MidiBinding) else MidiBinding.from_dict(item)
        except (TypeError, ValueError):
            continue
        key = (binding.action, binding.message_type, binding.channel, binding.number)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(binding)
    if not parsed:
        parsed = [MidiBinding.from_dict(item) for item in DEFAULT_MIDI_BINDINGS]
    return parsed


def bindings_to_settings(bindings: Iterable[MidiBinding]) -> list[dict[str, Any]]:
    """Serialize only validated bindings for Settings persistence."""
    return [binding.to_dict() for binding in normalized_bindings(list(bindings))]


class MidiControllerRouter:
    """Translate MIDI messages into debounced, normalized UI actions."""

    def __init__(self, bindings: Any = None):
        self._bindings: list[MidiBinding] = normalized_bindings(bindings)
        self._active: set[tuple[int, int]] = set()
        self._toggle_states: dict[int, bool] = {}

    @property
    def bindings(self) -> tuple[MidiBinding, ...]:
        return tuple(self._bindings)

    def set_bindings(self, bindings: Any) -> None:
        self._bindings = normalized_bindings(bindings)
        self.reset()

    def reset(self) -> None:
        self._active.clear()
        self._toggle_states.clear()

    def dispatch(self, message: MidiMessage) -> list[MidiActionEvent]:
        if not isinstance(message, MidiMessage):
            raise TypeError("MIDI router accepts MidiMessage instances")
        events: list[MidiActionEvent] = []
        for index, binding in enumerate(self._bindings):
            if not self._matches(binding, message):
                continue
            key = (index, message.number)
            pressed = message.is_pressed
            if binding.mode == "absolute":
                if message.message_type == "note_off":
                    continue
                events.append(MidiActionEvent(
                    binding.action,
                    message.normalized_value,
                    binding,
                    message,
                ))
                continue
            if pressed and key not in self._active:
                self._active.add(key)
                value = 1.0
                if binding.mode == "toggle":
                    value = 0.0 if self._toggle_states.get(index, False) else 1.0
                    self._toggle_states[index] = bool(value)
                events.append(MidiActionEvent(binding.action, value, binding, message))
            elif not pressed:
                self._active.discard(key)
        return events

    @staticmethod
    def _matches(binding: MidiBinding, message: MidiMessage) -> bool:
        if binding.message_type == "cc" and message.message_type != "cc":
            return False
        if binding.message_type == "note" and message.message_type not in {"note_on", "note_off"}:
            return False
        return (
            (binding.channel == MIDI_CHANNEL_OMNI or binding.channel == message.channel)
            and binding.number == message.number
        )


__all__ = [
    "DEFAULT_MIDI_BINDINGS",
    "MIDI_ACTION_LABELS",
    "MIDI_BINDING_MODES",
    "MIDI_CHANNEL_OMNI",
    "MIDI_CONTROLLER_SCHEMA_VERSION",
    "MIDI_MESSAGE_TYPES",
    "MidiActionEvent",
    "MidiBinding",
    "MidiControllerRouter",
    "MidiMessage",
    "bindings_to_settings",
    "normalized_bindings",
]
