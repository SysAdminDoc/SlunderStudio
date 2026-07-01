"""Regression tests for deep audit fixes."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from core.audio_export import trim_audio
from core.job_state import JobStore
from core.project import Project, ProjectAsset
from core.settings import Settings
from core.voice_bank import VoiceBank, VoiceProfile
from engines.ai_producer import AIProducer, ProducerBrief, PipelineStage


class ThreadSafeSingletonTests(unittest.TestCase):
    def tearDown(self):
        Settings._instance = None

    def test_settings_concurrent_construction_returns_same_instance(self):
        Settings._instance = None
        instances = []

        def create():
            with tempfile.TemporaryDirectory() as tmp:
                config_dir = Path(tmp) / "config"
                config_dir.mkdir()
                with mock.patch("core.settings.get_config_dir", return_value=config_dir):
                    with mock.patch("core.settings.get_default_output_dir", return_value=config_dir / "out"):
                        with mock.patch("core.settings.get_default_cache_dir", return_value=config_dir / "m"):
                            instances.append(id(Settings()))

        Settings._instance = None
        threads = [threading.Thread(target=create) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertTrue(len(set(instances)) <= 2)


class ProjectAssetIdTests(unittest.TestCase):
    def test_rapid_asset_creation_produces_unique_ids(self):
        ids = set()
        for _ in range(50):
            asset = ProjectAsset(name="test", asset_type="audio")
            self.assertNotIn(asset.id, ids)
            ids.add(asset.id)

    def test_rapid_project_creation_produces_unique_ids(self):
        ids = set()
        for _ in range(50):
            proj = Project(name="test")
            self.assertNotIn(proj.id, ids)
            ids.add(proj.id)


class VoiceBankAtomicWriteTests(unittest.TestCase):
    def tearDown(self):
        VoiceBank._instance = None

    def test_voice_bank_writes_utf8_and_survives_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            VoiceBank._instance = None
            with mock.patch("core.voice_bank.get_config_dir", return_value=Path(tmp)):
                bank = VoiceBank()
                bank._db_path = os.path.join(tmp, "voice_bank.json")
                profile = VoiceProfile(
                    name="テストボイス",
                    engine="rvc",
                    model_path="/fake/path.pth",
                    owner_name="テスト",
                    consent_status="confirmed",
                    consent_source="Self",
                    consent_scope="conversion",
                    language="ja",
                    permitted_uses=["voice-conversion"],
                )
                bank.add(profile)

                with open(bank._db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.assertTrue(any("テストボイス" in p.get("name", "") for p in data["profiles"]))
                VoiceBank._instance = None


class TrimAudioValidationTests(unittest.TestCase):
    def test_trim_rejects_inverted_range(self):
        with self.assertRaises(ValueError) as ctx:
            trim_audio("fake.wav", "out.wav", start_sec=10.0, end_sec=5.0)
        self.assertIn("start", str(ctx.exception).lower())


class AiProducerPipelineShortCircuitTests(unittest.TestCase):
    def test_pipeline_stops_after_song_gen_failure_without_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            producer = AIProducer()
            producer._output_dir = tmp

            brief = ProducerBrief(
                prompt="test",
                genre="pop",
                duration_seconds=5.0,
                demo_fallback=False,
                include_sfx=False,
                vocal_style="none",
            )

            with mock.patch.object(producer, "_generate_lyrics",
                                   return_value={"lyrics": "test"}):
                with mock.patch.object(producer, "_select_style",
                                       return_value={"tags": ["pop"], "tempo": 120, "key": "C"}):
                    result = producer.produce(brief)

            self.assertEqual(result.stage, PipelineStage.FAILED)

            mix_step = result.get_step(PipelineStage.MIXING)
            self.assertIsNone(mix_step)

            master_step = result.get_step(PipelineStage.MASTERING)
            self.assertIsNone(master_step)


class JobStoreLockInstanceTests(unittest.TestCase):
    def test_separate_stores_have_separate_locks(self):
        with tempfile.TemporaryDirectory() as tmp1:
            with tempfile.TemporaryDirectory() as tmp2:
                store1 = JobStore(Path(tmp1))
                store2 = JobStore(Path(tmp2))
                self.assertIsNot(store1._lock, store2._lock)


if __name__ == "__main__":
    unittest.main()
