import math
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.panning import CENTER_GAIN, CENTER_GAIN_DB, constant_power_pan, pan_gains


def db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


class ConstantPowerPanTests(unittest.TestCase):
    def test_center_is_minus_three_db_on_both_sides(self):
        left, right = constant_power_pan(0.0)
        self.assertAlmostEqual(left, CENTER_GAIN, places=12)
        self.assertAlmostEqual(right, CENTER_GAIN, places=12)
        self.assertAlmostEqual(db(left), CENTER_GAIN_DB, places=6)
        self.assertAlmostEqual(db(right), CENTER_GAIN_DB, places=6)

    def test_endpoints_fully_attenuate_the_opposite_channel(self):
        left, right = constant_power_pan(-1.0)
        self.assertAlmostEqual(left, 1.0, places=12)
        self.assertAlmostEqual(right, 0.0, places=12)

        left, right = constant_power_pan(1.0)
        self.assertAlmostEqual(left, 0.0, places=12)
        self.assertAlmostEqual(right, 1.0, places=12)

    def test_power_is_constant_across_the_whole_sweep(self):
        for position in np.linspace(-1.0, 1.0, 101):
            left, right = constant_power_pan(float(position))
            with self.subTest(pan=round(float(position), 3)):
                self.assertAlmostEqual(left ** 2 + right ** 2, 1.0, places=12)

    def test_out_of_range_positions_are_clamped(self):
        self.assertEqual(constant_power_pan(-5.0), constant_power_pan(-1.0))
        self.assertEqual(constant_power_pan(5.0), constant_power_pan(1.0))

    def test_volume_scales_both_channels(self):
        left, right = pan_gains(0.0, volume=0.5)
        self.assertAlmostEqual(left, CENTER_GAIN * 0.5, places=12)
        self.assertAlmostEqual(right, CENTER_GAIN * 0.5, places=12)

    def test_pan_is_monotonic(self):
        positions = np.linspace(-1.0, 1.0, 51)
        lefts = [constant_power_pan(float(p))[0] for p in positions]
        rights = [constant_power_pan(float(p))[1] for p in positions]
        self.assertTrue(all(a >= b for a, b in zip(lefts, lefts[1:])))
        self.assertTrue(all(a <= b for a, b in zip(rights, rights[1:])))


class BothMixersShareTheImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def _mono_energy(self, view_module_name: str) -> list[float]:
        """Summed energy of a centred mono source at several pan positions."""
        energies = []
        for position in (-1.0, -0.5, 0.0, 0.5, 1.0):
            left, right = pan_gains(position, 1.0)
            energies.append(left ** 2 + right ** 2)
        return energies

    def test_mono_energy_is_constant_within_tolerance(self):
        for energy in self._mono_energy("mixer"):
            self.assertAlmostEqual(energy, 1.0, places=10)

    def test_neither_mixer_still_computes_its_own_pan(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for rel in ("ui/mixer_view.py", "ui/stem_mixer.py"):
            source = (root / rel).read_text(encoding="utf-8")
            with self.subTest(module=rel):
                self.assertIn("from core.panning import pan_gains", source)
                self.assertNotIn("np.cos(max(0, pan)", source)
                self.assertNotIn("np.cos(max(0, -pan)", source)

    def test_mixer_and_stem_mixer_agree(self):
        import ui.mixer_view as mixer_view
        import ui.stem_mixer as stem_mixer

        self.assertIs(mixer_view.pan_gains, stem_mixer.pan_gains)

    def test_hard_panned_track_only_reaches_one_channel(self):
        from ui.mixer_view import MixerView

        view = MixerView()
        self.addCleanup(view.deleteLater)
        tone = np.column_stack([np.ones(1000, dtype=np.float32)] * 2)
        view.add_track("Hard Right", tone, 48000)
        view._strips[0]._pan_slider.setValue(100)

        mixed = view._get_mixed_audio()
        self.assertIsNotNone(mixed)
        self.assertLess(float(np.max(np.abs(mixed[:, 0]))), 1e-6)
        self.assertGreater(float(np.max(np.abs(mixed[:, 1]))), 0.0)


if __name__ == "__main__":
    unittest.main()
