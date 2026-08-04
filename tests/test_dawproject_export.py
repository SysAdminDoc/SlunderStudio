import os
import stat
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np

from core.dawproject import (
    DAWProjectSpec,
    DAWProjectValidation,
    DAWProjectSecurityError,
    DAWTrack,
    extract_dawproject,
    export_dawproject,
    spec_from_project,
    validate_dawproject,
)
from ui.mixer_view import _export_mixer_dawproject_task
from ui.project_manager import _export_dawproject_task


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

    def test_duplicate_media_basenames_get_collision_safe_archive_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_dir = os.path.join(tmp, "first")
            second_dir = os.path.join(tmp, "second")
            os.makedirs(first_dir)
            os.makedirs(second_dir)
            first = os.path.join(first_dir, "vocals.wav")
            second = os.path.join(second_dir, "vocals.wav")
            _write_wav(first)
            _write_wav(second, duration=0.5, sr=48000)

            spec = DAWProjectSpec(
                tracks=[
                    DAWTrack(name="Lead", media_file=first),
                    DAWTrack(name="Backing", media_file=second),
                ]
            )
            out = export_dawproject(spec, os.path.join(tmp, "collision.dawproject"))

            with zipfile.ZipFile(out) as archive:
                media_names = sorted(
                    name for name in archive.namelist() if name.startswith("media/")
                )
                project_xml = archive.read("project.xml").decode("utf-8")
            self.assertEqual(["media/vocals-2.wav", "media/vocals.wav"], media_names)
            self.assertEqual(2, project_xml.count("<Audio"))
            validation = validate_dawproject(out)
            self.assertTrue(validation.valid, validation.errors)

    def test_spec_from_project_uses_existing_audio_assets_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = os.path.join(tmp, "mix.wav")
            _write_wav(audio)
            project = SimpleNamespace(
                name="Project Export",
                tempo=128,
                time_signature=(7, 8),
                assets=[
                    SimpleNamespace(
                        name="Mix",
                        asset_type="audio",
                        file_path=audio,
                    ),
                    SimpleNamespace(
                        name="Missing",
                        asset_type="audio",
                        file_path=os.path.join(tmp, "missing.wav"),
                    ),
                    SimpleNamespace(
                        name="Lyrics",
                        asset_type="lyrics",
                        file_path=audio,
                    ),
                ],
            )
            spec = spec_from_project(project)
            self.assertEqual("Project Export", spec.title)
            self.assertEqual(128.0, spec.tempo)
            self.assertEqual("7/8", spec.time_signature)
            self.assertEqual(["Mix"], [track.name for track in spec.tracks])

    def test_project_manager_task_exports_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "asset.wav")
            _write_wav(source)
            output = os.path.join(tmp, "project.dawproject")
            spec = DAWProjectSpec(tracks=[DAWTrack(name="Asset", media_file=source)])
            result = _export_dawproject_task(spec, output)
            self.assertEqual(output, result["path"])
            self.assertTrue(validate_dawproject(output).valid)

    def test_mixer_task_materializes_current_track_buffers(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "mixer.dawproject")
            audio = np.zeros((480, 2), dtype=np.float32)
            result = _export_mixer_dawproject_task(
                [{
                    "audio": audio,
                    "sample_rate": 48000,
                    "name": "Mix",
                    "volume": 0.75,
                    "pan": -0.2,
                    "muted": False,
                    "soloed": True,
                }],
                output,
            )
            self.assertEqual(1, result["track_count"])
            validation = validate_dawproject(output)
            self.assertTrue(validation.valid, validation.errors)

    def test_safe_extraction_rejects_traversal_absolute_symlink_and_oversize(self):
        for member_name in ("../escape.wav", "/absolute.wav", "C:/drive.wav"):
            with self.subTest(member_name=member_name), tempfile.TemporaryDirectory() as tmp:
                archive_path = os.path.join(tmp, "unsafe.dawproject")
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(member_name, b"unsafe")
                validation = validate_dawproject(archive_path)
                self.assertFalse(validation.valid)
                self.assertTrue(any("Unsafe archive entry" in error for error in validation.errors))

        with tempfile.TemporaryDirectory() as tmp:
            archive_path = os.path.join(tmp, "symlink.dawproject")
            link = zipfile.ZipInfo("media/link.wav")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(link, b"target")
            validation = validate_dawproject(archive_path)
            self.assertFalse(validation.valid)
            self.assertTrue(any("Symbolic-link" in error for error in validation.errors))

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.wav")
            _write_wav(source, duration=0.01)
            archive_path = export_dawproject(
                DAWProjectSpec(tracks=[DAWTrack(media_file=source)]),
                os.path.join(tmp, "valid.dawproject"),
            )
            with self.assertRaises(DAWProjectSecurityError):
                extract_dawproject(archive_path, os.path.join(tmp, "out"), max_total_bytes=1)

    def test_safe_extraction_preserves_exported_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source.wav")
            _write_wav(source, duration=0.01)
            archive_path = export_dawproject(
                DAWProjectSpec(tracks=[DAWTrack(media_file=source)]),
                os.path.join(tmp, "valid.dawproject"),
            )
            destination = extract_dawproject(archive_path, os.path.join(tmp, "out"))
            self.assertEqual(
                (destination / "media" / "source.wav").read_bytes(),
                Path(source).read_bytes(),
            )

    def test_validation_rejects_truncated_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = os.path.join(tmp, "truncated.dawproject")
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("project.xml", "<Project/>")
                archive.writestr("metadata.xml", "<MetaData/>")
            with open(archive_path, "rb+") as handle:
                handle.truncate(max(0, os.path.getsize(archive_path) - 12))
            validation = validate_dawproject(archive_path)
            self.assertFalse(validation.valid)
            self.assertTrue(any("Invalid ZIP" in error for error in validation.errors))


if __name__ == "__main__":
    unittest.main()
