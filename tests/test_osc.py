import socket
import struct
import threading
import unittest
from types import SimpleNamespace

from core.osc import (
    DEFAULT_MAX_PACKET_BYTES,
    OSCConfig,
    OSCMessage,
    OSC_NAMESPACE,
    OSCProtocolError,
    OSCServer,
    RateLimiter,
    parse_osc_message,
    source_allowed,
)
from ui.main_window import MainWindow


def _osc_string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    return raw + b"\x00" * ((-len(raw)) % 4)


def _packet(address: str, tags: str = "", *values) -> bytes:
    payload = _osc_string(address) + _osc_string(f",{tags}")
    for tag, value in zip(tags, values):
        if tag == "i":
            payload += struct.pack(">i", value)
        elif tag == "f":
            payload += struct.pack(">f", value)
        elif tag == "s":
            payload += _osc_string(value)
        elif tag in "TF":
            continue
        else:
            raise AssertionError(f"test helper does not encode {tag}")
    return payload


class OSCProtocolTests(unittest.TestCase):
    def test_parser_decodes_versioned_arguments(self):
        message = parse_osc_message(
            _packet(
                f"{OSC_NAMESPACE}/transport/example",
                "ifsTF",
                7,
                0.5,
                "hello",
                True,
                False,
            )
        )
        self.assertEqual(message.address, f"{OSC_NAMESPACE}/transport/example")
        self.assertEqual(message.arguments[0], 7)
        self.assertAlmostEqual(message.arguments[1], 0.5, places=5)
        self.assertEqual(message.arguments[2:], ("hello", True, False))

    def test_parser_rejects_unsupported_or_malformed_messages(self):
        cases = [
            b"",
            _packet("/other/v1/ping"),
            _osc_string(f"{OSC_NAMESPACE}/ping") + _osc_string(",d") + b"\x00" * 8,
            _packet(f"{OSC_NAMESPACE}/ping") + b"x",
            _packet(f"{OSC_NAMESPACE}/ping")[:-1],
            b"#bundle\x00" + b"\x00" * 12,
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(OSCProtocolError):
                    parse_osc_message(payload)

        with self.assertRaises(OSCProtocolError):
            parse_osc_message(
                b"x" * (DEFAULT_MAX_PACKET_BYTES + 1),
                max_packet_bytes=DEFAULT_MAX_PACKET_BYTES - 1,
            )

    def test_source_policy_requires_explicit_lan_opt_in_and_match(self):
        default = OSCConfig.from_settings({})
        self.assertEqual(default.bind_host, "127.0.0.1")
        self.assertTrue(source_allowed("127.0.0.1", default))
        self.assertFalse(source_allowed("192.168.1.25", default))

        lan_without_match = OSCConfig.from_settings(
            {"allow_lan": True, "allowed_hosts": ["127.0.0.1"]}
        )
        self.assertEqual(lan_without_match.bind_host, "0.0.0.0")
        self.assertFalse(source_allowed("192.168.1.25", lan_without_match))

        lan = OSCConfig.from_settings(
            {"allow_lan": True, "allowed_hosts": ["192.168.1.0/24"]}
        )
        self.assertTrue(source_allowed("192.168.1.25", lan))
        self.assertFalse(source_allowed("192.168.2.25", lan))

    def test_rate_limiter_is_per_source_and_expires(self):
        limiter = RateLimiter(2)
        self.assertTrue(limiter.allow("a", now=10.0))
        self.assertTrue(limiter.allow("a", now=10.1))
        self.assertFalse(limiter.allow("a", now=10.2))
        self.assertTrue(limiter.allow("b", now=10.2))
        self.assertTrue(limiter.allow("a", now=11.0))

    def test_server_receives_loopback_message_and_stops_cleanly(self):
        received = []
        ready = threading.Event()
        server = OSCServer(
            OSCConfig(enabled=True, port=0),
            lambda message, source: (received.append((message, source)), ready.set()),
        )
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.assertTrue(server.start())
            self.assertIsNotNone(server.bound_address)
            sender.sendto(
                _packet(f"{OSC_NAMESPACE}/ping"),
                server.bound_address,
            )
            self.assertTrue(ready.wait(2.0))
            self.assertEqual(received[0][0], OSCMessage(f"{OSC_NAMESPACE}/ping", ()))
            self.assertEqual(received[0][1][0], "127.0.0.1")
        finally:
            sender.close()
            server.stop()
        self.assertFalse(server.is_running)
        self.assertIsNone(server.bound_address)


class OSCDispatchTests(unittest.TestCase):
    def setUp(self):
        self.actions = []
        self.window = MainWindow.__new__(MainWindow)
        self.window._transport = SimpleNamespace(
            osc_play=lambda: self.actions.append(("play",)),
            osc_pause=lambda: self.actions.append(("pause",)),
            osc_stop=lambda: self.actions.append(("stop",)),
            osc_toggle=lambda: self.actions.append(("toggle",)),
            osc_seek=lambda value: self.actions.append(("seek", value)),
            osc_seek_relative=lambda value: self.actions.append(("relative", value)),
            osc_set_loop=lambda value: self.actions.append(("loop", value)),
            osc_set_volume=lambda value: self.actions.append(("volume", value)),
        )
        self.window._status_bar = SimpleNamespace(showMessage=lambda *_args: None)

    def test_dispatch_accepts_only_explicit_transport_commands(self):
        source = ("127.0.0.1", 9001)
        self.assertTrue(
            self.window._on_osc_message(
                OSCMessage(f"{OSC_NAMESPACE}/transport/play"), source
            )
        )
        self.assertTrue(
            self.window._on_osc_message(
                OSCMessage(f"{OSC_NAMESPACE}/transport/volume", (0.25,)), source
            )
        )
        self.assertTrue(
            self.window._on_osc_message(
                OSCMessage(f"{OSC_NAMESPACE}/transport/loop", (True,)), source
            )
        )
        self.assertEqual(self.actions, [("play",), ("volume", 0.25), ("loop", True)])

    def test_dispatch_rejects_bad_arguments_without_side_effects(self):
        source = ("127.0.0.1", 9001)
        invalid = [
            OSCMessage(f"{OSC_NAMESPACE}/transport/play", (1,)),
            OSCMessage(f"{OSC_NAMESPACE}/transport/seek", ("1",)),
            OSCMessage(f"{OSC_NAMESPACE}/transport/loop", (1,)),
            OSCMessage(f"{OSC_NAMESPACE}/transport/volume", (1.1,)),
            OSCMessage(f"{OSC_NAMESPACE}/transport/unknown"),
        ]
        for message in invalid:
            with self.subTest(address=message.address):
                self.assertFalse(self.window._on_osc_message(message, source))
        self.assertEqual(self.actions, [])


if __name__ == "__main__":
    unittest.main()
