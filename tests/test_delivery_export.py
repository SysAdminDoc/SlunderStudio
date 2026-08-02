import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from core.audio_export import (
    DELIVERY_FORMATS,
    LOSSLESS_FORMATS,
    LOSSY_FORMATS,
    METADATA_STANDARDS,
    CodecAvailability,
    ExportSettings,
    deterministic_filename,
    export_from_numpy,
    probe_codecs,
    require_codec,
)

SR = 48000


def tone(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    wave = 0.2 * np.sin(2 * np.pi * 440 * t)
    return np.column_stack([wave, wave]).astype(np.float32)


class CodecProbeTests(unittest.TestCase):
    def test_every_delivery_format_is_probed(self):
        codecs = probe_codecs()
        self.assertEqual(set(codecs), set(DELIVERY_FORMATS))
        for fmt, availability in codecs.items():
            with self.subTest(fmt=fmt):
                self.assertIsInstance(availability, CodecAvailability)
                self.assertIn(
                    availability.writer,
                    ("soundfile", "ffmpeg"),
                )
                if not availability.available:
                    self.assertTrue(availability.detail, "no reason given")

    def test_lossless_formats_are_always_writable(self):
        codecs = probe_codecs()
        for fmt in LOSSLESS_FORMATS:
            self.assertTrue(codecs[fmt].available, codecs[fmt].detail)

    def test_missing_encoder_is_reported_not_silently_skipped(self):
        with mock.patch("core.audio_export._find_ffmpeg", return_value=None):
            codecs = probe_codecs()
            for fmt in LOSSY_FORMATS:
                with self.subTest(fmt=fmt):
                    self.assertFalse(codecs[fmt].available)
                    self.assertIn("ffmpeg", codecs[fmt].detail)
            with self.assertRaises(RuntimeError) as ctx:
                require_codec("mp3")
            self.assertIn("ffmpeg", str(ctx.exception))

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(ValueError):
            require_codec("aiff")


class DeterministicFilenameTests(unittest.TestCase):
    def test_same_inputs_produce_the_same_name(self):
        first = deterministic_filename("My Song!!", fmt="mp3", revision="r3", variant="master")
        second = deterministic_filename("My Song!!", fmt="mp3", revision="r3", variant="master")
        self.assertEqual(first, second)
        self.assertEqual(first, "My-Song-master-r3.mp3")

    def test_unsafe_characters_are_removed(self):
        name = deterministic_filename("a/b\\c:*?\"<>|", fmt="wav")
        self.assertNotRegex(name, r'[/\\:*?"<>|]')
        self.assertTrue(name.endswith(".wav"))

    def test_empty_base_still_produces_a_name(self):
        self.assertEqual(deterministic_filename("   ", fmt="flac"), "export.flac")


class DeliveryRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.audio = tone()
        self.codecs = probe_codecs()

    def _settings(self, fmt: str) -> ExportSettings:
        return ExportSettings(
            format=fmt,
            sample_rate=SR,
            title="Gutter Sermon",
            artist="Slunder",
            album="Test Album",
            year="2026",
            genre="trap metal",
            bpm=174.0,
            musical_key="F minor",
            language="en",
            rights="(c) 2026 Slunder",
            revision="r1",
            isrc="US-ABC-26-00001",
        )

    def _sidecar(self, path: str) -> dict:
        return json.loads(Path(path + ".provenance.json").read_text(encoding="utf-8"))

    def test_every_available_format_writes_verifies_and_records_provenance(self):
        for fmt in DELIVERY_FORMATS:
            if not self.codecs[fmt].available:
                self.skipTest(f"{fmt} unavailable: {self.codecs[fmt].detail}")
            with self.subTest(fmt=fmt):
                out = str(self.root / f"delivery.{fmt}")
                written = export_from_numpy(
                    self.audio, SR, out, self._settings(fmt),
                    module="test", operation="delivery",
                )
                self.assertTrue(os.path.isfile(written))

                delivery = self._sidecar(written)["extra"]["delivery"]
                verification = delivery["verification"]
                self.assertGreater(verification["bytes"], 0)
                self.assertEqual(len(verification["sha256"]), 64)
                self.assertTrue(verification["reopened"], verification["reopen_error"])
                self.assertEqual(
                    delivery["metadata_standard"], METADATA_STANDARDS[fmt]
                )
                self.assertEqual(delivery["writer"], self.codecs[fmt].writer)

    def test_metadata_maps_every_declared_field(self):
        settings = self._settings("wav")
        tags = settings.metadata_tags()
        self.assertEqual(tags["title"], "Gutter Sermon")
        self.assertEqual(tags["artist"], "Slunder")
        self.assertEqual(tags["date"], "2026")
        self.assertEqual(tags["genre"], "trap metal")
        self.assertEqual(tags["language"], "en")
        self.assertEqual(tags["copyright"], "(c) 2026 Slunder")
        self.assertEqual(tags["version"], "r1")
        self.assertEqual(tags["TSRC"], "US-ABC-26-00001")
        self.assertEqual(tags["TBPM"], "174")
        self.assertEqual(tags["TKEY"], "F minor")

    def test_empty_metadata_fields_are_dropped(self):
        tags = ExportSettings(format="wav", artist="").metadata_tags()
        self.assertNotIn("artist", tags)
        self.assertNotIn("TBPM", tags)
        self.assertNotIn("TKEY", tags)

    def test_hash_is_stable_for_identical_lossless_content(self):
        settings = self._settings("wav")
        first = export_from_numpy(
            self.audio, SR, str(self.root / "a.wav"), settings)
        second = export_from_numpy(
            self.audio, SR, str(self.root / "b.wav"), settings)
        self.assertEqual(
            self._sidecar(first)["extra"]["delivery"]["verification"]["sha256"],
            self._sidecar(second)["extra"]["delivery"]["verification"]["sha256"],
        )

    def test_verification_reports_the_written_duration(self):
        written = export_from_numpy(
            self.audio, SR, str(self.root / "dur.wav"), self._settings("wav"))
        verification = self._sidecar(written)["extra"]["delivery"]["verification"]
        self.assertAlmostEqual(verification["duration_sec"], 1.0, places=3)
        self.assertEqual(verification["channels"], 2)
        self.assertEqual(verification["sample_rate"], SR)

    def test_unavailable_format_fails_loudly(self):
        with mock.patch("core.audio_export._find_ffmpeg", return_value=None):
            with self.assertRaises(RuntimeError):
                export_from_numpy(
                    self.audio, SR, str(self.root / "no.mp3"),
                    ExportSettings(format="mp3", sample_rate=SR),
                )

    def test_empty_output_is_rejected(self):
        from core.audio_export import _verify_written_file

        empty = self.root / "empty.wav"
        empty.write_bytes(b"")
        with self.assertRaises(RuntimeError):
            _verify_written_file(str(empty))
        with self.assertRaises(RuntimeError):
            _verify_written_file(str(self.root / "missing.wav"))


if __name__ == "__main__":
    unittest.main()
