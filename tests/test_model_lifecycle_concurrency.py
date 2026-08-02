import os
import threading
import time
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.model_manager import (
    DownloadInFlightError,
    ModelCategory,
    ModelInfo,
    ModelManager,
    ModelStatus,
    StaleModelRequestError,
)


class _FakeEngine:
    """Minimal engine object with the teardown hook ModelManager looks for."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.unloaded = False

    def unload_model(self):
        self.unloaded = True


def _info(model_id: str) -> ModelInfo:
    return ModelInfo(
        model_id=model_id,
        name=model_id.title(),
        description="test",
        category=ModelCategory.EXTRAS,
        vram_gb=1.0,
        disk_gb=0.01,
        license="MIT",
        source=f"example/{model_id}",
        revision="a" * 40,
        loader_module="engines.sfx_engine",
        loader_fn="load_model",
    )


class ModelLifecycleConcurrencyTests(unittest.TestCase):
    """The GUI reads lifecycle state while workers load, unload, and download."""

    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.mgr = ModelManager()
        self._old_registry = self.mgr._registry
        self._old_status = dict(self.mgr._status)
        self.addCleanup(self._restore)

        self.mgr._registry = {mid: _info(mid) for mid in ("alpha", "beta", "gamma")}
        self.mgr._status = {
            mid: ModelStatus.DOWNLOADED for mid in self.mgr._registry
        }
        self.mgr.unload()

        patcher = mock.patch.object(
            ModelManager, "require_verified_model", lambda _self, _mid: {}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        cached = mock.patch.object(
            ModelManager, "_is_model_cached", lambda _self, _mid: True
        )
        cached.start()
        self.addCleanup(cached.stop)

    def _restore(self):
        self.mgr.unload()
        self.mgr._registry = self._old_registry
        self.mgr._status = self._old_status

    def _loader(self, model_id: str, delay: float = 0.0):
        def _load():
            if delay:
                time.sleep(delay)
            return _FakeEngine(model_id)
        return _load

    def test_single_valid_state_after_concurrent_lifecycle_churn(self):
        errors: list[BaseException] = []
        barrier = threading.Barrier(9)

        def loader_thread(model_id: str):
            try:
                barrier.wait(timeout=15)
                for _ in range(6):
                    try:
                        self.mgr.load_model(model_id, self._loader(model_id))
                    except StaleModelRequestError:
                        pass
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        def unloader_thread():
            try:
                barrier.wait(timeout=15)
                for _ in range(6):
                    self.mgr.unload()
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        def observer_thread():
            try:
                barrier.wait(timeout=15)
                for _ in range(60):
                    snapshot = self.mgr.lifecycle_snapshot()
                    current = snapshot["current_model_id"]
                    # A model identity is never observable without a model object.
                    if current is not None and not snapshot["has_model"]:
                        raise AssertionError(f"torn state: {snapshot}")
                    if current is None and snapshot["has_model"]:
                        raise AssertionError(f"orphaned model object: {snapshot}")
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [
            threading.Thread(target=loader_thread, args=(mid,))
            for mid in ("alpha", "beta", "gamma")
        ]
        threads += [threading.Thread(target=unloader_thread) for _ in range(3)]
        threads += [threading.Thread(target=observer_thread) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        self.assertEqual([repr(e) for e in errors], [])
        self.assertFalse(any(t.is_alive() for t in threads))

        snapshot = self.mgr.lifecycle_snapshot()
        current = snapshot["current_model_id"]
        self.assertEqual(current is not None, snapshot["has_model"])
        if current is not None:
            self.assertIn(current, self.mgr._registry)
            self.assertEqual(snapshot["status"][current], ModelStatus.LOADED.value)
            loaded = [
                mid
                for mid, status in snapshot["status"].items()
                if status == ModelStatus.LOADED.value
            ]
            self.assertEqual(loaded, [current])

    def test_stale_loader_does_not_overwrite_a_newer_request(self):
        release = threading.Event()
        built: list[_FakeEngine] = []

        def slow_loader():
            engine = _FakeEngine("alpha")
            built.append(engine)
            release.wait(timeout=10)
            return engine

        result: dict[str, object] = {}

        def worker():
            try:
                self.mgr.load_model("alpha", slow_loader)
            except BaseException as exc:  # noqa: BLE001 - asserted below
                result["error"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        # Wait until the slow loader is actually running.
        for _ in range(200):
            if built:
                break
            time.sleep(0.01)
        self.assertTrue(built, "slow loader never started")

        # A newer request arrives while alpha is still loading.
        self.mgr.unload()
        release.set()
        thread.join(timeout=15)

        self.assertIsInstance(result.get("error"), StaleModelRequestError)
        self.assertIsNone(self.mgr.current_model_id)
        self.assertIsNone(self.mgr.current_model)
        self.assertTrue(built[0].unloaded, "superseded model was not released")

    def test_superseded_activation_reports_cancelled_not_active(self):
        release = threading.Event()
        started = threading.Event()

        def slow_loader():
            started.set()
            release.wait(timeout=10)
            return _FakeEngine("alpha")

        outcome: dict[str, object] = {}

        def worker():
            with mock.patch.object(
                ModelManager, "_dynamic_load", lambda _self, _info: slow_loader()
            ):
                outcome["result"] = self.mgr.activate_model("alpha")

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(started.wait(timeout=10))
        self.mgr.unload()
        release.set()
        thread.join(timeout=15)

        result = outcome["result"]
        self.assertFalse(result.is_success)
        self.assertTrue(result.cancelled)
        self.assertIsNone(self.mgr.current_model_id)

    def test_loading_a_second_model_releases_the_first(self):
        first = self.mgr.load_model("alpha", self._loader("alpha"))
        second = self.mgr.load_model("beta", self._loader("beta"))

        self.assertTrue(first.unloaded)
        self.assertFalse(second.unloaded)
        self.assertEqual(self.mgr.current_model_id, "beta")
        self.assertEqual(self.mgr.get_status("alpha"), ModelStatus.DOWNLOADED)
        self.assertEqual(self.mgr.get_status("beta"), ModelStatus.LOADED)

    def test_unload_if_current_ignores_other_models(self):
        self.mgr.load_model("alpha", self._loader("alpha"))
        self.assertFalse(self.mgr.unload_if_current("beta"))
        self.assertEqual(self.mgr.current_model_id, "alpha")
        self.assertTrue(self.mgr.unload_if_current("alpha"))
        self.assertIsNone(self.mgr.current_model_id)

    def test_concurrent_download_of_same_model_is_rejected(self):
        started = threading.Event()
        release = threading.Event()

        def body(_self, _info, _model_id, **_kwargs):
            started.set()
            release.wait(timeout=10)
            return None

        with mock.patch.object(ModelManager, "_download_model_locked", body), \
                mock.patch.object(
                    ModelManager, "is_offline", property(lambda _self: False)
                ):
            thread = threading.Thread(
                target=lambda: self.mgr.download_model("alpha")
            )
            thread.start()
            self.assertTrue(started.wait(timeout=10))
            with self.assertRaises(DownloadInFlightError):
                self.mgr.download_model("alpha")
            release.set()
            thread.join(timeout=15)

        # The in-flight marker is cleared, so a later download is allowed.
        self.assertNotIn("alpha", self.mgr._downloads_in_flight)


if __name__ == "__main__":
    unittest.main()
