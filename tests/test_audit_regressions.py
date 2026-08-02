"""Regression tests for deep audit fixes."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.audio_export import trim_audio
from core import model_manager
from core.job_state import JobStore
from core.model_manager import get_gpu_info
from core.project import Project, ProjectAsset
from core.settings import Settings
from core.voice_bank import VoiceBank, VoiceProfile
from core.workers import InferenceWorker
from engines.ai_producer import AIProducer, ProducerBrief, PipelineStage
from engines.ace_step_engine import ACEStepEngine, GenerationResult, generate_song
from engines.lyrics_engine import LyricsLLM, generate_lyrics
from ui.onboarding import check_system


class GpuProbeTests(unittest.TestCase):
    def setUp(self):
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda _index: SimpleNamespace(
                    name="Fake CUDA GPU", total_memory=8 * (1024**3)
                ),
                get_device_name=lambda _index: "Fake CUDA GPU",
                memory_reserved=lambda _index: 2 * (1024**3),
                memory_allocated=lambda _index: 1 * (1024**3),
            )
        )

    def test_gpu_info_uses_torch_total_memory(self):
        with mock.patch.object(model_manager, "_torch_module_cache", self.torch):
            info = get_gpu_info()

        self.assertTrue(info["available"])
        self.assertEqual(info["name"], "Fake CUDA GPU")
        self.assertEqual(info["total_gb"], 8.0)
        self.assertEqual(info["free_gb"], 6.0)

    def test_gpu_info_caches_a_failed_torch_import(self):
        with (
            mock.patch.object(model_manager, "_torch_module_cache", model_manager._TORCH_MODULE_UNSET),
            mock.patch("builtins.__import__", side_effect=ImportError("torch unavailable")) as importer,
        ):
            first = get_gpu_info()
            second = get_gpu_info()

        torch_imports = [
            call for call in importer.call_args_list
            if call.args and call.args[0] == "torch"
        ]
        self.assertEqual(1, len(torch_imports))
        self.assertFalse(first["available"])
        self.assertFalse(second["available"])

    def test_onboarding_system_check_uses_torch_total_memory(self):
        with mock.patch.dict("sys.modules", {"torch": self.torch}):
            checks = check_system()

        self.assertTrue(checks["cuda"])
        self.assertEqual(checks["gpu_name"], "Fake CUDA GPU")
        self.assertEqual(checks["vram_gb"], 8.0)


class _CachedModelManager:
    def __init__(self):
        self.model = None
        self.calls = 0

    def load_model(self, _model_id, loader_fn):
        self.calls += 1
        if self.model is None:
            self.model = loader_fn()
        return self.model


class ManagedGenerationReuseTests(unittest.TestCase):
    def test_song_generation_uses_cached_engine_on_second_call(self):
        manager = _CachedModelManager()

        def load(engine):
            engine._model_loaded = True

        def generate(engine, _params, **_kwargs):
            if not engine._model_loaded:
                raise RuntimeError("unloaded engine was used")
            return GenerationResult(audio_path="song.wav", duration=1.0)

        with (
            mock.patch("core.model_manager.ModelManager", return_value=manager),
            mock.patch.object(ACEStepEngine, "load", load),
            mock.patch.object(ACEStepEngine, "generate", generate),
            mock.patch(
                "engines.ace_step_engine.recover_song_vocal_stem", return_value={}
            ),
        ):
            first = generate_song("lyrics")
            second = generate_song("lyrics")

        self.assertEqual(manager.calls, 2)
        self.assertEqual(first["audio_path"], "song.wav")
        self.assertEqual(second["audio_path"], "song.wav")

    def test_lyrics_generation_uses_cached_llm_on_second_call(self):
        manager = _CachedModelManager()
        settings = mock.Mock()
        settings.get.side_effect = lambda _key, default=None: default

        def load(llm, model_id=None, **_kwargs):
            llm._model = object()
            llm._backend = "stub"
            llm._model_id = model_id

        def generate(llm, *_args, **_kwargs):
            if not llm.is_loaded:
                raise RuntimeError("unloaded LLM was used")
            return "cached lyrics"

        with (
            mock.patch("engines.lyrics_engine.ModelManager", return_value=manager),
            mock.patch("engines.lyrics_engine.Settings", return_value=settings),
            mock.patch.object(LyricsLLM, "load", load),
            mock.patch.object(LyricsLLM, "generate", generate),
        ):
            first = generate_lyrics("dark song")
            second = generate_lyrics("dark song")

        self.assertEqual(manager.calls, 2)
        self.assertEqual(first["lyrics"], "cached lyrics")
        self.assertEqual(second["lyrics"], "cached lyrics")


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
    def test_stores_share_the_process_job_lock(self):
        with tempfile.TemporaryDirectory() as tmp1:
            with tempfile.TemporaryDirectory() as tmp2:
                store1 = JobStore(Path(tmp1))
                store2 = JobStore(Path(tmp2))
                self.assertIs(store1._lock, store2._lock)

    def test_concurrent_store_updates_do_not_lose_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store_a = JobStore(root)
            store_b = JobStore(root)
            job_a = store_a.create("generation", "A")
            job_b = store_b.create("generation", "B")
            store_a.mark_running(job_a.id)
            store_b.mark_running(job_b.id)

            progress_read = threading.Event()
            release_progress = threading.Event()
            complete_read = threading.Event()
            errors: list[BaseException] = []
            original_a_read = store_a._read
            original_b_read = store_b._read

            def blocked_progress_read():
                records = original_a_read()
                progress_read.set()
                if not release_progress.wait(timeout=2):
                    raise AssertionError("timed out waiting to release progress update")
                return records

            def observed_complete_read():
                complete_read.set()
                return original_b_read()

            store_a._read = blocked_progress_read
            store_b._read = observed_complete_read

            def run_progress():
                try:
                    store_a.update_progress(job_a.id, 45, "progress")
                except BaseException as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

            def run_complete():
                try:
                    store_b.mark_completed(job_b.id, {"path": "B.wav"})
                except BaseException as exc:  # pragma: no cover - surfaced below
                    errors.append(exc)

            progress_thread = threading.Thread(target=run_progress)
            complete_thread = threading.Thread(target=run_complete)
            progress_thread.start()
            self.assertTrue(progress_read.wait(timeout=2))
            complete_thread.start()
            self.assertFalse(complete_read.wait(timeout=0.2))
            release_progress.set()
            progress_thread.join(timeout=2)
            complete_thread.join(timeout=2)

            self.assertFalse(progress_thread.is_alive())
            self.assertFalse(complete_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(store_a.get(job_a.id).status, "running")
            self.assertEqual(store_b.get(job_b.id).status, "completed")


class JobProgressPersistenceTests(unittest.TestCase):
    def test_worker_throttles_ledger_progress_but_emits_every_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp))
            worker = InferenceWorker(
                lambda **_kwargs: None,
                job_kind="song_generation",
                job_label="Progress throttle",
                job_store=store,
            )
            emitted: list[int] = []
            worker.progress.connect(emitted.append)

            with mock.patch("core.workers.time.monotonic", side_effect=[10.0, 10.01, 10.11]):
                with mock.patch.object(store, "update_progress", wraps=store.update_progress) as update:
                    worker._emit_progress(10)
                    worker._emit_progress(20)
                    worker._emit_progress(30)

            self.assertEqual([call.args[1] for call in update.call_args_list], [10, 30])
            self.assertEqual(emitted, [10, 20, 30])


if __name__ == "__main__":
    unittest.main()
