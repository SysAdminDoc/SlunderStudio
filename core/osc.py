"""Bounded OSC control transport for Slunder Studio.

The application only accepts direct OSC messages in the versioned
``/slunder/v1`` namespace.  Bundles, address patterns, unsupported argument
types, oversized packets, and untrusted sources are rejected before a message
can reach the UI thread.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import ipaddress
import logging
import math
import socket
import struct
import threading
import time
from typing import Callable, Iterable


logger = logging.getLogger(__name__)

OSC_NAMESPACE = "/slunder/v1"
DEFAULT_OSC_PORT = 9000
DEFAULT_MAX_PACKET_BYTES = 4096
DEFAULT_MAX_MESSAGES_PER_SECOND = 60
_UDP_MAX_PACKET_BYTES = 65507


class OSCProtocolError(ValueError):
    """Raised when a datagram is not a supported OSC message."""


@dataclass(frozen=True)
class OSCMessage:
    """One decoded, versioned OSC message."""

    address: str
    arguments: tuple[object, ...] = ()


def _coerce_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        converted = default
    return max(minimum, min(maximum, converted))


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _normalize_allowed_hosts(value: object) -> tuple[str, ...]:
    """Return explicit IPv4 host/network entries suitable for an allowlist."""
    if isinstance(value, str):
        entries: Iterable[object] = value.replace(";", ",").split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        entries = value
    else:
        entries = ()

    normalized: list[str] = []
    for raw_entry in entries:
        entry = str(raw_entry or "").strip()
        if not entry:
            continue
        try:
            if "/" in entry:
                network = ipaddress.ip_network(entry, strict=False)
            else:
                address = ipaddress.ip_address(entry)
                network = ipaddress.ip_network(f"{address}/{address.max_prefixlen}")
        except ValueError:
            continue
        if network.version != 4 or network.prefixlen == 0 or network.is_unspecified:
            continue
        normalized_entry = str(network.network_address)
        if network.prefixlen != 32:
            normalized_entry += f"/{network.prefixlen}"
        if normalized_entry not in normalized:
            normalized.append(normalized_entry)
    return tuple(normalized)


@dataclass(frozen=True)
class OSCConfig:
    """Persisted and validated OSC server policy."""

    enabled: bool = False
    port: int = DEFAULT_OSC_PORT
    allow_lan: bool = False
    allowed_hosts: tuple[str, ...] = ("127.0.0.1",)
    max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES
    max_messages_per_second: int = DEFAULT_MAX_MESSAGES_PER_SECOND

    def __post_init__(self):
        object.__setattr__(self, "enabled", _coerce_bool(self.enabled))
        object.__setattr__(
            self,
            "port",
            _coerce_int(self.port, DEFAULT_OSC_PORT, 0, 65535),
        )
        object.__setattr__(self, "allow_lan", _coerce_bool(self.allow_lan))
        object.__setattr__(
            self,
            "allowed_hosts",
            _normalize_allowed_hosts(self.allowed_hosts),
        )
        object.__setattr__(
            self,
            "max_packet_bytes",
            _coerce_int(
                self.max_packet_bytes,
                DEFAULT_MAX_PACKET_BYTES,
                1,
                _UDP_MAX_PACKET_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "max_messages_per_second",
            _coerce_int(
                self.max_messages_per_second,
                DEFAULT_MAX_MESSAGES_PER_SECOND,
                1,
                10000,
            ),
        )

    @classmethod
    def from_settings(cls, section: dict | None) -> "OSCConfig":
        """Build a safe config from the untrusted persisted settings section."""
        values = section if isinstance(section, dict) else {}
        allowed_hosts = values.get("allowed_hosts", ("127.0.0.1",))
        if allowed_hosts is None:
            allowed_hosts = ()
        return cls(
            enabled=_coerce_bool(values.get("enabled", False)),
            port=values.get("port", DEFAULT_OSC_PORT),
            allow_lan=_coerce_bool(values.get("allow_lan", False)),
            allowed_hosts=allowed_hosts,
            max_packet_bytes=values.get("max_packet_bytes", DEFAULT_MAX_PACKET_BYTES),
            max_messages_per_second=values.get(
                "max_messages_per_second", DEFAULT_MAX_MESSAGES_PER_SECOND
            ),
        )

    @property
    def bind_host(self) -> str:
        """Return the bind address; LAN exposure is never implicit."""
        return "0.0.0.0" if self.allow_lan else "127.0.0.1"


def _network_entries(config: OSCConfig) -> tuple[ipaddress.IPv4Network, ...]:
    networks: list[ipaddress.IPv4Network] = []
    for entry in config.allowed_hosts:
        try:
            network = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            continue
        if network.version == 4 and network.prefixlen > 0 and not network.is_unspecified:
            networks.append(network)
    return tuple(networks)


def source_allowed(source_host: str, config: OSCConfig) -> bool:
    """Return whether a datagram source is permitted by the current policy.

    Loopback is always accepted because the server binds to loopback by
    default.  Any non-loopback source additionally requires both the explicit
    LAN opt-in and a matching IPv4 host or CIDR entry.
    """
    try:
        address = ipaddress.ip_address(str(source_host).strip())
    except ValueError:
        return False
    if address.version != 4:
        return False
    if address.is_loopback:
        return True
    if not config.allow_lan:
        return False
    return any(address in network for network in _network_entries(config))


class RateLimiter:
    """Per-source sliding-window datagram limiter."""

    def __init__(self, max_messages_per_second: int):
        self._limit = max(1, int(max_messages_per_second))
        self._events: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, source_key: str, now: float | None = None) -> bool:
        """Consume one allowance for ``source_key`` if its window is open."""
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            events = self._events[source_key]
            while events and current - events[0] >= 1.0:
                events.popleft()
            if len(events) >= self._limit:
                return False
            events.append(current)
            return True


def _read_osc_string(packet: bytes, offset: int, field: str) -> tuple[str, int]:
    end = packet.find(b"\x00", offset)
    if end < 0:
        raise OSCProtocolError(f"OSC {field} is not null terminated")
    try:
        value = packet[offset:end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OSCProtocolError(f"OSC {field} is not valid UTF-8") from exc
    next_offset = (end + 4) & ~3
    if next_offset > len(packet):
        raise OSCProtocolError(f"OSC {field} padding is truncated")
    return value, next_offset


def parse_osc_message(
    packet: bytes,
    max_packet_bytes: int = DEFAULT_MAX_PACKET_BYTES,
) -> OSCMessage:
    """Decode one supported OSC 1.0 message from a bounded datagram."""
    if not isinstance(packet, (bytes, bytearray, memoryview)):
        raise OSCProtocolError("OSC packet must be bytes")
    raw = bytes(packet)
    if len(raw) > int(max_packet_bytes):
        raise OSCProtocolError("OSC packet exceeds the configured size limit")
    if not raw:
        raise OSCProtocolError("OSC packet is empty")
    if raw.startswith(b"#bundle\x00"):
        raise OSCProtocolError("OSC bundles are not supported")

    address, offset = _read_osc_string(raw, 0, "address")
    if not address.startswith(f"{OSC_NAMESPACE}/"):
        raise OSCProtocolError("OSC address is outside the versioned Slunder namespace")
    if "*" in address or "?" in address or "[" in address:
        raise OSCProtocolError("OSC address patterns are not supported")

    type_tags, offset = _read_osc_string(raw, offset, "type tag")
    if not type_tags.startswith(","):
        raise OSCProtocolError("OSC type tag string must start with a comma")

    arguments: list[object] = []
    for tag in type_tags[1:]:
        if tag == "i":
            if offset + 4 > len(raw):
                raise OSCProtocolError("OSC integer argument is truncated")
            arguments.append(struct.unpack(">i", raw[offset : offset + 4])[0])
            offset += 4
        elif tag == "f":
            if offset + 4 > len(raw):
                raise OSCProtocolError("OSC float argument is truncated")
            value = struct.unpack(">f", raw[offset : offset + 4])[0]
            if not math.isfinite(value):
                raise OSCProtocolError("OSC float argument must be finite")
            arguments.append(value)
            offset += 4
        elif tag == "s":
            value, offset = _read_osc_string(raw, offset, "string argument")
            arguments.append(value)
        elif tag == "T":
            arguments.append(True)
        elif tag == "F":
            arguments.append(False)
        else:
            raise OSCProtocolError(f"Unsupported OSC type tag: {tag!r}")

    if offset != len(raw):
        raise OSCProtocolError("OSC packet has trailing bytes")
    return OSCMessage(address=address, arguments=tuple(arguments))


class OSCServer:
    """A daemon UDP listener that delivers validated messages to a callback."""

    def __init__(
        self,
        config: OSCConfig,
        message_callback: Callable[[OSCMessage, tuple[str, int]], None],
        error_callback: Callable[[str], None] | None = None,
    ):
        self.config = config
        self._message_callback = message_callback
        self._error_callback = error_callback
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._bound_address: tuple[str, int] | None = None
        self._rate_limiter = RateLimiter(config.max_messages_per_second)

    @property
    def is_running(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def bound_address(self) -> tuple[str, int] | None:
        with self._state_lock:
            return self._bound_address

    def start(self) -> bool:
        """Bind and start the listener; raise the bind error to the caller."""
        if not self.config.enabled:
            return False
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_socket.settimeout(0.2)
                server_socket.bind((self.config.bind_host, self.config.port))
            except OSError:
                server_socket.close()
                raise
            self._socket = server_socket
            self._bound_address = server_socket.getsockname()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="SlunderOSC",
                daemon=True,
            )
            self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the listener and release its UDP socket."""
        with self._state_lock:
            thread = self._thread
            server_socket = self._socket
            self._stop_event.set()
            self._socket = None
            self._bound_address = None
        if server_socket is not None:
            try:
                server_socket.close()
            except OSError:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        with self._state_lock:
            if self._thread is thread:
                self._thread = None

    def _report_error(self, message: str) -> None:
        logger.warning("OSC server stopped: %s", message)
        if self._error_callback is not None:
            try:
                self._error_callback(message)
            except Exception:
                logger.exception("OSC error callback failed")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._state_lock:
                server_socket = self._socket
            if server_socket is None:
                return
            try:
                packet, source = server_socket.recvfrom(self.config.max_packet_bytes + 1)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop_event.is_set():
                    self._report_error(str(exc))
                return

            source_host = str(source[0])
            if len(packet) > self.config.max_packet_bytes:
                logger.debug("Dropped oversized OSC packet from %s", source_host)
                continue
            if not source_allowed(source_host, self.config):
                logger.warning("Dropped OSC packet from untrusted source %s", source_host)
                continue
            if not self._rate_limiter.allow(source_host):
                logger.warning("Dropped rate-limited OSC packet from %s", source_host)
                continue
            try:
                message = parse_osc_message(packet, self.config.max_packet_bytes)
            except OSCProtocolError as exc:
                logger.debug("Dropped invalid OSC packet from %s: %s", source_host, exc)
                continue
            try:
                self._message_callback(message, (source_host, int(source[1])))
            except Exception:
                logger.exception("OSC message callback failed")
