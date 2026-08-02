import time
import unittest

import numpy as np

from core.mastering import (
    ABSOLUTE_GATE_LUFS,
    SILENCE_LUFS,
    _biquad_filter,
    apply_compression,
    apply_limiter,
    db_to_linear,
    measure_loudness_range,
    measure_lufs,
    measure_sample_peak_db,
    measure_true_peak_db,
)

SR = 48000

# Benchmark budget for a three-minute 48 kHz stereo fixture on the reference
# CPU. Measured locally at roughly 0.7 s / 1.3 s / 1.6 s per stage; the budget
# is deliberately loose so it fails on an algorithmic regression, not on a
# slower machine.
THREE_MINUTE_STAGE_BUDGET_SECONDS = 10.0
THREE_MINUTE_CHAIN_BUDGET_SECONDS = 25.0


def sine(freq: float, level_db: float, seconds: float = 20.0,
         channels: int = 2, phase: float = 0.0) -> np.ndarray:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    wave = db_to_linear(level_db) * np.sin(2 * np.pi * freq * t + phase)
    if channels == 1:
        return wave
    return np.column_stack([wave] * channels)


class LoudnessConformanceTests(unittest.TestCase):
    """EBU Tech 3341 / 3342 compliance cases for BS.1770-5 loudness."""

    def test_stereo_1khz_sine_reads_its_own_level(self):
        for level in (-23.0, -33.0, -10.0):
            with self.subTest(level=level):
                self.assertAlmostEqual(
                    measure_lufs(sine(1000, level), SR), level, delta=0.1
                )

    def test_mono_reads_three_db_below_the_same_stereo_signal(self):
        stereo = measure_lufs(sine(1000, -23.0), SR)
        mono = measure_lufs(sine(1000, -23.0, channels=1), SR)
        self.assertAlmostEqual(mono, stereo - 3.01, delta=0.1)

    def test_k_weighting_boosts_highs_and_attenuates_lows(self):
        reference = measure_lufs(sine(1000, -23.0), SR)
        high = measure_lufs(sine(10000, -23.0), SR)
        low = measure_lufs(sine(60, -23.0), SR)
        self.assertGreater(high, reference + 2.5)
        self.assertLess(low, reference - 2.0)

    def test_coefficients_track_the_sample_rate(self):
        """44.1 kHz must not be measured with 48 kHz coefficients."""
        for rate in (44100, 48000, 96000):
            n = int(rate * 10)
            t = np.arange(n) / rate
            wave = db_to_linear(-23.0) * np.sin(2 * np.pi * 1000 * t)
            audio = np.column_stack([wave, wave])
            with self.subTest(rate=rate):
                self.assertAlmostEqual(measure_lufs(audio, rate), -23.0, delta=0.2)

    def test_silence_reports_the_absolute_gate(self):
        self.assertEqual(measure_lufs(np.zeros((SR * 5, 2)), SR), SILENCE_LUFS)
        self.assertEqual(SILENCE_LUFS, ABSOLUTE_GATE_LUFS)

    def test_absolute_gate_ignores_very_quiet_passages(self):
        loud = sine(1000, -23.0, seconds=10.0)
        near_silence = sine(1000, -80.0, seconds=10.0)
        combined = np.concatenate([loud, near_silence])
        self.assertAlmostEqual(measure_lufs(combined, SR), -23.0, delta=0.3)

    def test_loudness_range_matches_the_tech_3342_case(self):
        combined = np.concatenate([
            sine(1000, -20.0, seconds=20.0),
            sine(1000, -30.0, seconds=20.0),
        ])
        self.assertAlmostEqual(measure_loudness_range(combined, SR), 10.0, delta=1.0)

    def test_loudness_range_of_a_constant_tone_is_near_zero(self):
        self.assertLess(measure_loudness_range(sine(1000, -23.0, seconds=30.0), SR), 1.0)


class TruePeakTests(unittest.TestCase):
    def test_true_peak_exceeds_sample_peak_for_intersample_overshoot(self):
        # A sine at fs/4 offset by 45 degrees samples at +/-0.707 but peaks at 1.0.
        audio = sine(SR / 4, 0.0, seconds=2.0, phase=np.pi / 4)
        sample_peak = measure_sample_peak_db(audio)
        true_peak = measure_true_peak_db(audio, SR)
        self.assertAlmostEqual(sample_peak, -3.01, delta=0.1)
        self.assertGreater(true_peak, sample_peak + 2.5)
        self.assertAlmostEqual(true_peak, 0.0, delta=0.5)

    def test_true_peak_is_never_below_the_sample_peak(self):
        rng = np.random.default_rng(11)
        audio = rng.standard_normal((SR, 2)) * 0.3
        self.assertGreaterEqual(
            measure_true_peak_db(audio, SR), measure_sample_peak_db(audio) - 1e-9
        )

    def test_empty_audio_reports_silence(self):
        self.assertEqual(measure_true_peak_db(np.zeros((0, 2)), SR), SILENCE_LUFS)


class VectorizedDspEquivalenceTests(unittest.TestCase):
    """The vectorized DSP must match the per-sample reference implementations."""

    @staticmethod
    def _reference_biquad(audio, b0, b1, b2, a1, a2):
        output = np.zeros_like(audio)
        channels = audio.shape[1] if audio.ndim == 2 else 1
        for ch in range(channels):
            x = audio[:, ch] if audio.ndim == 2 else audio
            y = np.zeros_like(x)
            x1 = x2 = y1 = y2 = 0.0
            for i in range(len(x)):
                y[i] = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                x2, x1 = x1, x[i]
                y2, y1 = y1, y[i]
            if audio.ndim == 2:
                output[:, ch] = y
            else:
                output = y
        return output

    @staticmethod
    def _reference_compressor(audio, sr, threshold_db, ratio, attack_ms,
                              release_ms, makeup_db):
        threshold = db_to_linear(threshold_db)
        makeup = db_to_linear(makeup_db)
        attack_coeff = np.exp(-1.0 / (attack_ms * 0.001 * sr))
        release_coeff = np.exp(-1.0 / (release_ms * 0.001 * sr))
        output = np.copy(audio)
        envelope = 0.0
        for i in range(len(audio)):
            sample = np.max(np.abs(audio[i])) if audio.ndim == 2 else abs(audio[i])
            if sample > envelope:
                envelope = attack_coeff * envelope + (1 - attack_coeff) * sample
            else:
                envelope = release_coeff * envelope + (1 - release_coeff) * sample
            if envelope > threshold:
                gain = (threshold + (envelope - threshold) / ratio) / max(envelope, 1e-10)
            else:
                gain = 1.0
            output[i] = audio[i] * gain * makeup
        return output

    @staticmethod
    def _reference_limiter(audio, ceiling_db, release_ms, sr):
        ceiling = db_to_linear(ceiling_db)
        release_coeff = np.exp(-1.0 / (release_ms * 0.001 * sr))
        output = np.copy(audio)
        gain = 1.0
        for i in range(len(audio)):
            peak = np.max(np.abs(audio[i])) if audio.ndim == 2 else abs(audio[i])
            if peak * gain > ceiling:
                gain = ceiling / max(peak, 1e-10)
            else:
                gain = min(release_coeff * gain + (1 - release_coeff), 1.0)
            output[i] = audio[i] * gain
        return output

    def setUp(self):
        rng = np.random.default_rng(3)
        self.audio = (rng.standard_normal((4000, 2)) * 0.2)

    def test_biquad_matches_the_scalar_reference(self):
        coeffs = (0.6, -0.3, 0.1, -0.4, 0.2)
        np.testing.assert_allclose(
            _biquad_filter(self.audio, *coeffs),
            self._reference_biquad(self.audio, *coeffs),
            rtol=0, atol=1e-12,
        )

    def test_biquad_keeps_channels_independent(self):
        coeffs = (0.6, -0.3, 0.1, -0.4, 0.2)
        both = _biquad_filter(self.audio, *coeffs)
        left_only = _biquad_filter(self.audio[:, :1], *coeffs)
        np.testing.assert_allclose(both[:, 0], left_only[:, 0], rtol=0, atol=1e-12)

    def test_compressor_matches_the_scalar_reference(self):
        args = (SR, -18.0, 3.0, 10.0, 100.0, 2.0)
        np.testing.assert_allclose(
            apply_compression(self.audio, *args),
            self._reference_compressor(self.audio, *args),
            rtol=0, atol=1e-10,
        )

    def test_compressor_matches_when_attack_equals_release(self):
        args = (SR, -18.0, 4.0, 50.0, 50.0, 0.0)
        np.testing.assert_allclose(
            apply_compression(self.audio, *args),
            self._reference_compressor(self.audio, *args),
            rtol=0, atol=1e-10,
        )

    def test_limiter_matches_the_scalar_reference(self):
        np.testing.assert_allclose(
            apply_limiter(self.audio, -1.0, 50.0, SR),
            self._reference_limiter(self.audio, -1.0, 50.0, SR),
            rtol=0, atol=1e-10,
        )

    def test_limiter_holds_the_ceiling(self):
        loud = self.audio * 8.0
        limited = apply_limiter(loud, -1.0, 50.0, SR)
        self.assertLessEqual(
            measure_sample_peak_db(limited), -1.0 + 0.05
        )

    def test_mono_input_stays_mono(self):
        mono = self.audio[:, 0]
        self.assertEqual(apply_compression(mono, SR, -18.0, 3.0, 10.0, 100.0, 0.0).ndim, 1)
        self.assertEqual(apply_limiter(mono, -1.0, 50.0, SR).ndim, 1)


class BenchmarkBudgetTests(unittest.TestCase):
    """A three-minute stereo render must finish inside an explicit budget."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        cls.audio = (rng.standard_normal((SR * 180, 2)) * 0.1)

    def _timed(self, label, fn):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        self.assertLess(
            elapsed, THREE_MINUTE_STAGE_BUDGET_SECONDS,
            f"{label} took {elapsed:.2f}s, budget {THREE_MINUTE_STAGE_BUDGET_SECONDS}s",
        )
        return result, elapsed

    def test_full_chain_meets_the_budget(self):
        total = 0.0
        eq, elapsed = self._timed(
            "eq", lambda: _biquad_filter(self.audio, 0.6, -0.3, 0.1, -0.4, 0.2)
        )
        total += elapsed
        compressed, elapsed = self._timed(
            "compressor",
            lambda: apply_compression(eq, SR, -18.0, 3.0, 10.0, 100.0, 2.0),
        )
        total += elapsed
        _, elapsed = self._timed(
            "limiter", lambda: apply_limiter(compressed, -1.0, 50.0, SR)
        )
        total += elapsed
        _, elapsed = self._timed("lufs", lambda: measure_lufs(self.audio, SR))
        total += elapsed
        _, elapsed = self._timed("lra", lambda: measure_loudness_range(self.audio, SR))
        total += elapsed
        _, elapsed = self._timed(
            "true_peak", lambda: measure_true_peak_db(self.audio, SR)
        )
        total += elapsed
        self.assertLess(
            total, THREE_MINUTE_CHAIN_BUDGET_SECONDS,
            f"full chain took {total:.2f}s, budget {THREE_MINUTE_CHAIN_BUDGET_SECONDS}s",
        )


if __name__ == "__main__":
    unittest.main()
