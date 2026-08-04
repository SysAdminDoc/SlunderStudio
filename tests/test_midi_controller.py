import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import numpy as np

from core.midi_controller import (
    DEFAULT_MIDI_BINDINGS,
    MIDI_CHANNEL_OMNI,
    MidiBinding,
    MidiControllerRouter,
    MidiMessage,
    normalized_bindings,
)
from core.midi_input import MidiInputService


class _FakePort:
    def __init__(self, callback):
        self.callback = callback
        self.closed = False

    def close(self):
        self.closed = True


class _FakeBackend:
    def __init__(self):
        self.port = None

    def get_input_names(self):
        return ["Test Keyboard"]

    def open_input(self, name, callback):
        assert name == "Test Keyboard"
        self.port = _FakePort(callback)
        return self.port


class MidiControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_default_bindings_are_bounded_and_round_trip(self):
        bindings = normalized_bindings(list(DEFAULT_MIDI_BINDINGS))
        self.assertEqual(9, len(bindings))
        self.assertEqual(MIDI_CHANNEL_OMNI, bindings[0].channel)
        self.assertEqual(bindings, normalized_bindings([item.to_dict() for item in bindings]))

    def test_router_normalizes_absolute_cc_and_filters_channels(self):
        router = MidiControllerRouter([
            MidiBinding("mixer.volume", "cc", 2, 7, "absolute"),
        ])
        self.assertEqual([], router.dispatch(MidiMessage.from_values("cc", 1, 7, 64)))
        events = router.dispatch(MidiMessage.from_values("cc", 2, 7, 64))
        self.assertEqual(1, len(events))
        self.assertAlmostEqual(64 / 127, events[0].value)

    def test_trigger_and_toggle_bindings_emit_once_per_press(self):
        router = MidiControllerRouter([
            MidiBinding("transport.toggle", "cc", -1, 20, "trigger"),
            MidiBinding("mixer.mute", "cc", -1, 29, "toggle"),
        ])
        press = MidiMessage.from_values("cc", 0, 20, 127)
        release = MidiMessage.from_values("cc", 0, 20, 0)
        self.assertEqual(["transport.toggle"], [event.action for event in router.dispatch(press)])
        self.assertEqual([], router.dispatch(press))
        self.assertEqual([], router.dispatch(release))
        self.assertEqual(["transport.toggle"], [event.action for event in router.dispatch(press)])

        mute_press = MidiMessage.from_values("cc", 0, 29, 127)
        mute_release = MidiMessage.from_values("cc", 0, 29, 0)
        self.assertEqual([1.0], [event.value for event in router.dispatch(mute_press)])
        router.dispatch(mute_release)
        self.assertEqual([0.0], [event.value for event in router.dispatch(mute_press)])

    def test_note_on_zero_and_note_off_release_trigger(self):
        router = MidiControllerRouter([
            MidiBinding("piano.quantize", "note", -1, 36, "trigger"),
        ])
        self.assertEqual([], router.dispatch(MidiMessage.from_values("note_on", 0, 36, 0)))
        self.assertEqual(1, len(router.dispatch(MidiMessage.from_values("note_on", 0, 36, 100))))
        self.assertEqual([], router.dispatch(MidiMessage.from_values("note_off", 0, 36, 0)))
        self.assertEqual(1, len(router.dispatch(MidiMessage.from_values("note_on", 0, 36, 100))))

    def test_backend_adapter_emits_validated_messages_and_closes(self):
        backend = _FakeBackend()
        service = MidiInputService(backend=backend)
        received = []
        service.message_received.connect(received.append)
        self.assertTrue(service.start("Test Keyboard"))
        backend.port.callback(SimpleNamespace(
            type="control_change", channel=1, control=7, value=96
        ))
        self.app.processEvents()
        self.assertEqual(
            [MidiMessage.from_values("cc", 1, 7, 96)],
            received,
        )
        service.stop()
        self.assertTrue(backend.port.closed)

    def test_mixer_and_piano_roll_public_controller_actions(self):
        from core.midi_utils import NoteData, TrackData
        from ui.mixer_view import MixerView
        from ui.piano_roll import PianoRollWidget

        mixer = MixerView(project_sample_rate=48000)
        mixer.add_track("Controller target", np.zeros(64, dtype=np.float32), 48000)
        self.assertTrue(mixer.set_selected_volume(0.5))
        self.assertAlmostEqual(0.5, mixer._strips[0].volume)
        self.assertTrue(mixer.set_selected_pan(-0.25))
        self.assertAlmostEqual(-0.25, mixer._strips[0].pan)
        self.assertTrue(mixer.toggle_selected_mute())
        self.assertTrue(mixer._strips[0].is_muted)
        mixer.close()

        piano = PianoRollWidget()
        piano.load_track(
            TrackData(notes=[NoteData(pitch=60, start=0.13, end=0.3, velocity=90)]),
            tempo=120,
            bars=4,
        )
        self.assertTrue(piano.controller_quantize())
        self.assertAlmostEqual(0.125, piano.get_notes()[0].start)
        piano.deleteLater()


if __name__ == "__main__":
    unittest.main()
