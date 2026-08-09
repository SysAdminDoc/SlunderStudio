import json
import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.credentials import MemoryCredentialStore
from core.model_manager import ModelManager
from core.project import ProjectManager
from core.routing import (
    ARTIFACT_AUDIO,
    ARTIFACT_MIDI,
    RouteError,
    RoutedArtifact,
    build_routed_artifact,
    infer_kind,
    register_with_project,
)
from core.settings import Settings

SR = 48000


class RoutedArtifactTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.audio_path = self.root / "vocal.wav"
        import soundfile as sf

        t = np.arange(SR) / SR
        sf.write(str(self.audio_path), np.column_stack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2), SR)

    def _write_sidecar(self, payload: dict):
        sidecar = Path(str(self.audio_path) + ".provenance.json")
        sidecar.write_text(json.dumps(payload), encoding="utf-8")
        return sidecar

    def test_missing_file_refuses_to_route(self):
        with self.assertRaises(RouteError):
            build_routed_artifact(str(self.root / "nope.wav"), source_module="test")
        with self.assertRaises(RouteError):
            build_routed_artifact("", source_module="test")

    def test_kind_is_inferred_from_the_extension(self):
        self.assertEqual(infer_kind("a.mid"), ARTIFACT_MIDI)
        self.assertEqual(infer_kind("a.MIDI"), ARTIFACT_MIDI)
        self.assertEqual(infer_kind("a.flac"), ARTIFACT_AUDIO)

    def test_context_comes_from_the_provenance_sidecar(self):
        self._write_sidecar({
            "parameters": {"bpm": 174, "key": "F minor", "lyrics": "gutter sermon"},
            "extra": {},
        })
        artifact = build_routed_artifact(
            str(self.audio_path), source_module="song_forge")
        self.assertEqual(artifact.kind, ARTIFACT_AUDIO)
        self.assertEqual(artifact.tempo, 174.0)
        self.assertEqual(artifact.musical_key, "F minor")
        self.assertEqual(artifact.lyrics, "gutter sermon")
        self.assertTrue(artifact.provenance_path)
        self.assertAlmostEqual(artifact.duration_sec, 1.0, places=3)
        self.assertIn("174", artifact.context_summary())

    def test_explicit_context_wins_over_the_sidecar(self):
        self._write_sidecar({"parameters": {"bpm": 100}, "extra": {}})
        artifact = build_routed_artifact(
            str(self.audio_path), source_module="test", tempo=90.0)
        self.assertEqual(artifact.tempo, 90.0)

    def test_missing_sidecar_is_not_fatal(self):
        artifact = build_routed_artifact(str(self.audio_path), source_module="test")
        self.assertEqual(artifact.provenance, {})
        self.assertEqual(artifact.tempo, 0.0)
        self.assertTrue(artifact.exists)

    def test_payload_serializes_without_the_lyrics_body(self):
        self._write_sidecar({"parameters": {"lyrics": "secret verse"}, "extra": {}})
        payload = build_routed_artifact(
            str(self.audio_path), source_module="test").as_dict()
        self.assertTrue(payload["has_lyrics"])
        self.assertNotIn("secret verse", json.dumps(payload))


class RouteEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    @classmethod
    def _wait_for(cls, predicate, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            cls._app.processEvents()
            if predicate():
                return
            time.sleep(0.01)
        cls._app.processEvents()
        if not predicate():
            raise AssertionError("Timed out waiting for Mixer import")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config = self.root / "config"
        self.config.mkdir(parents=True, exist_ok=True)

        stack = ExitStack()
        self.addCleanup(stack.close)
        for target in (
            ("core.settings.get_config_dir", self.config),
            ("core.retention.get_config_dir", self.config),
            ("core.model_manager.get_config_dir", self.config),
            ("core.lyrics_db.get_config_dir", self.config),
        ):
            stack.enter_context(mock.patch(target[0], return_value=target[1]))
        stack.enter_context(mock.patch("core.settings.get_default_output_dir",
                                       return_value=self.root / "out"))
        stack.enter_context(mock.patch("core.settings.get_default_cache_dir",
                                       return_value=self.root / "models"))
        stack.enter_context(mock.patch("core.project.get_config_dir",
                                       return_value=str(self.config)))
        stack.enter_context(mock.patch("core.credentials.get_credential_store",
                                       return_value=MemoryCredentialStore()))
        Settings._instance = None
        ProjectManager._instance = None
        ModelManager._instance = None
        self.addCleanup(setattr, Settings, "_instance", None)
        self.addCleanup(setattr, ProjectManager, "_instance", None)
        self.addCleanup(setattr, ModelManager, "_instance", None)

        self.audio_path = self.root / "routed.wav"
        import soundfile as sf

        t = np.arange(SR) / SR
        sf.write(str(self.audio_path),
                 np.column_stack([0.2 * np.sin(2 * np.pi * 330 * t)] * 2), SR)
        Path(str(self.audio_path) + ".provenance.json").write_text(
            json.dumps({"parameters": {"bpm": 150, "key": "A minor"}, "extra": {}}),
            encoding="utf-8",
        )

        from core.lyrics_db import LyricsDB
        from ui.main_window import MainWindow

        LyricsDB._instance = None

        def _release_lyrics_db():
            # MainWindow opens the lyrics database; release it before the temp
            # directory is removed or Windows refuses to delete the file.
            instance = LyricsDB._instance
            if instance is not None:
                instance.close()
            LyricsDB._instance = None

        self.addCleanup(_release_lyrics_db)

        self.window = MainWindow()
        self.addCleanup(self._cleanup_window)
        self.projects = ProjectManager()

    def _cleanup_window(self):
        reference_panel = getattr(
            getattr(self.window, "_song_forge_view", None),
            "_ref_panel",
            None,
        )
        if reference_panel is not None:
            reference_panel.cancel_analysis()
        self.window.deleteLater()

    def test_route_to_mixer_transfers_selects_and_registers(self):
        self.projects.create("Routing Project")
        artifact = self.window._on_sfx_to_mixer(str(self.audio_path))
        self._wait_for(lambda: self.window._mixer_view._import_worker is None)

        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.tempo, 150.0)
        self.assertEqual(artifact.musical_key, "A minor")
        self.assertEqual(len(self.window._mixer_view._tracks), 1)
        self.assertEqual(self.window._mixer_view.selected_track_index, 0)
        self.assertEqual(len(self.projects.current.assets), 1)
        self.assertIn("150", self.window.toast_mgr.latest_message())

    def test_route_to_song_forge_carries_the_reference_not_just_the_page(self):
        artifact = self.window._on_vocal_to_forge(str(self.audio_path))

        self.assertIsNotNone(artifact)
        reference = self.window._song_forge_view.routed_reference
        self.assertIsNotNone(reference)
        self.assertEqual(reference.path, os.path.abspath(str(self.audio_path)))
        self.assertEqual(reference.source_module, "vocal_suite")
        self.assertEqual(reference.tempo, 150.0)

    def test_song_forge_reference_context_reaches_generation_tags(self):
        artifact = self.window._on_vocal_to_forge(str(self.audio_path))

        self.assertIsNotNone(artifact)
        tags = self.window._song_forge_view._get_tags()
        self.assertIn("150 bpm", tags.lower())
        self.assertIn("A minor", tags)

    def test_reference_constraints_route_to_midi_studio(self):
        constraints = {
            "schema_version": 1,
            "bpm": 96.0,
            "key": "A minor",
            "sections": [{"start": 0.0, "end": 30.0, "label": "Verse"}],
            "effective": {
                "bpm": 96.0,
                "key": "A minor",
                "sections": [{"start": 0.0, "end": 30.0, "label": "Verse"}],
            },
        }

        self.assertTrue(self.window._on_reference_to_midi(constraints))
        self.assertEqual(self.window._pages.currentIndex(), 2)
        self.assertEqual(self.window._midi_studio_view._tempo_spin.value(), 96)
        self.assertEqual(self.window._midi_studio_view._key_combo.currentData(), "A minor")
        self.assertEqual(
            self.window._midi_studio_view._reference_constraints["effective"]["sections"][0]["label"],
            "Verse",
        )

    def test_route_to_vocal_suite_selects_the_file(self):
        artifact = self.window._on_send_to_vocals(str(self.audio_path))
        self.assertIsNotNone(artifact)
        label = self.window._vocal_suite_view._stem_input_label
        self.assertEqual(label.property("path"), artifact.path)

    def test_midi_to_vocal_suite_route_preserves_source_module(self):
        artifact = self.window._on_midi_to_vocals(str(self.audio_path))

        self.assertIsNotNone(artifact)
        self.assertEqual("midi_studio", artifact.source_module)

    def test_a_missing_file_reports_instead_of_silently_switching_pages(self):
        missing = str(self.root / "gone.wav")
        self.assertIsNone(self.window._on_sfx_to_mixer(missing))
        self.assertEqual(len(self.window._mixer_view._tracks), 0)
        self.assertIn("could not be loaded", self.window.toast_mgr.latest_message())

    def test_routing_without_an_open_project_still_transfers(self):
        self.projects.close()
        artifact = self.window._on_vocal_to_mixer(str(self.audio_path))
        self._wait_for(lambda: self.window._mixer_view._import_worker is None)
        self.assertIsNotNone(artifact)
        self.assertEqual(len(self.window._mixer_view._tracks), 1)
        self.assertNotIn("added to the project",
                         self.window.toast_mgr.latest_message())


class ProjectRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_no_open_project_returns_none(self):
        artifact = RoutedArtifact(path=__file__, kind=ARTIFACT_AUDIO,
                                  source_module="test")
        manager = mock.Mock()
        manager.current = None
        self.assertIsNone(
            register_with_project(artifact, module="test", project_manager=manager)
        )
        manager.import_asset.assert_not_called()

    def test_midi_registers_as_a_midi_asset(self):
        artifact = RoutedArtifact(path="song.mid", kind=ARTIFACT_MIDI,
                                  source_module="midi_studio")
        manager = mock.Mock()
        manager.import_asset.return_value = "asset_1"
        self.assertEqual(
            register_with_project(artifact, module="midi", project_manager=manager),
            "asset_1",
        )
        self.assertEqual(manager.import_asset.call_args.args[1], "midi")


if __name__ == "__main__":
    unittest.main()
