import os
import tempfile
import unittest
import wave
import zipfile
from xml.etree import ElementTree as ET

import numpy as np

from core.dawproject import (
    DAWProjectSpec,
    DAWProjectValidation,
    DAWTrack,
    export_dawproject,
    validate_dawproject,
)


def _write_wav(path: str, duration: float = 1.0, sr: int = 44100):
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * t) * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())


class DAWProjectExportTests(unittest.TestCase):
    def test_export_creates_valid_archive_with_required_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "vocals.wav")
            _write_wav(stem)

            spec = DAWProjectSpec(
                title="Test Song",
                artist="Slunder",
                tempo=120.0,
                tracks=[DAWTrack(name="Vocals", media_file=stem)],
            )

            out = export_dawproject(spec, os.path.join(tmp, "test.dawproject"))
            self.assertTrue(os.path.isfile(out))

            result = validate_dawproject(out)
            self.assertTrue(result.valid, result.errors)
            self.assertIn("project.xml", result.entries)
            self.assertIn("metadata.xml", result.entries)
            self.assertIn("media/vocals.wav", result.entries)
            self.assertEqual(result.errors, [])

    def test_project_xml_contains_required_elements(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "bass.wav")
            _write_wav(stem)

            spec = DAWProjectSpec(
                title="Structure Test",
                tempo=140.0,
                time_signature="3/4",
                tracks=[
                    DAWTrack(name="Bass", media_file=stem, volume=0.8, pan=-0.3),
                ],
            )

            out = export_dawproject(spec, os.path.join(tmp, "out.dawproject"))
            with zipfile.ZipFile(out) as zf:
                project_xml = zf.read("project.xml").decode("utf-8")

            root = ET.fromstring(project_xml)
            self.assertIn("Project", root.tag)
            self.assertEqual(root.get("version"), "1.0")

            tempo_els = [el for el in root.iter() if el.tag.endswith("Tempo")]
            self.assertTrue(tempo_els)
            self.assertEqual(tempo_els[0].get("value"), "140.0")

            track_els = [el for el in root.iter() if el.tag.endswith("Track")]
            self.assertTrue(track_els)
            self.assertEqual(track_els[0].get("name"), "Bass")

            audio_els = [el for el in root.iter() if el.tag.endswith("Audio")]
            self.assertTrue(audio_els)
            self.assertEqual(audio_els[0].get("file"), "media/bass.wav")

    def test_metadata_xml_contains_title_and_application(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = DAWProjectSpec(title="Meta Test", artist="TestArtist")

            out = export_dawproject(spec, os.path.join(tmp, "meta.dawproject"))
            with zipfile.ZipFile(out) as zf:
                metadata_xml = zf.read("metadata.xml").decode("utf-8")

            root = ET.fromstring(metadata_xml)
            title_el = next((el for el in root.iter() if el.tag.endswith("Title")), None)
            self.assertIsNotNone(title_el)
            self.assertEqual(title_el.text, "Meta Test")

            app_el = next((el for el in root.iter() if el.tag.endswith("Application")), None)
            self.assertIsNotNone(app_el)
            self.assertEqual(app_el.get("name"), "SlunderStudio")

    def test_validation_fails_on_missing_project_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = os.path.join(tmp, "bad.dawproject")
            with zipfile.ZipFile(bad_path, "w") as zf:
                zf.writestr("metadata.xml", "<MetaData/>")

            result = validate_dawproject(bad_path)
            self.assertFalse(result.valid)
            self.assertTrue(any("project.xml" in e for e in result.errors))

    def test_validation_fails_on_missing_media_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = os.path.join(tmp, "missing_media.dawproject")
            project_xml = (
                '<?xml version="1.0"?>'
                '<Project version="1.0">'
                '<Transport><Tempo value="120"/></Transport>'
                '<Structure><Track id="t0" name="T"/></Structure>'
                '<Arrangement><Lane trackRef="t0">'
                '<Clip time="0"><Audio file="media/ghost.wav"/></Clip>'
                '</Lane></Arrangement>'
                '</Project>'
            )
            with zipfile.ZipFile(bad_path, "w") as zf:
                zf.writestr("project.xml", project_xml)
                zf.writestr("metadata.xml", "<MetaData><Title>T</Title></MetaData>")

            result = validate_dawproject(bad_path)
            self.assertFalse(result.valid)
            self.assertTrue(any("ghost.wav" in e for e in result.errors))

    def test_multi_track_export_includes_all_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            stems = []
            for name in ["drums", "bass", "guitar"]:
                p = os.path.join(tmp, f"{name}.wav")
                _write_wav(p)
                stems.append(DAWTrack(name=name.title(), media_file=p))

            spec = DAWProjectSpec(title="Multi", tracks=stems)
            out = export_dawproject(spec, os.path.join(tmp, "multi.dawproject"))
            result = validate_dawproject(out)
            self.assertTrue(result.valid, result.errors)
            self.assertEqual(len(result.media_refs), 3)


if __name__ == "__main__":
    unittest.main()
