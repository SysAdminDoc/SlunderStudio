import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.engine_contract import (
    ENGINE_CAPABILITIES,
    ActivationOutcome,
    ArtifactKind,
    CapabilityReadiness,
    EngineArtifact,
    EngineBatchResult,
    EngineRunResult,
    ModelReadiness,
    RunMode,
    RunOutcome,
    CAP_MIDI_GENERATE,
    CAP_PRODUCER_RUN,
    CAP_VOCAL_CONVERT,
    adapt_engine_result,
)
from core.model_manager import (
    BUILTIN_MODELS,
    ModelManager,
    ModelStatus,
)
from core.voice_bank import (
    VOICE_OPERATION_CONVERSION,
    VoiceProfile,
)
from engines.rvc_engine import RVCEngine
from ui.model_hub import ModelCard


@dataclass
class _Result:
    error: str = ""
    output_kind: str = "model"
    can_route: bool = True
    is_success: bool = True


class EngineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_capabilities_reference_registered_models_and_declared_outputs(self):
        for capability in ENGINE_CAPABILITIES.values():
            with self.subTest(capability=capability.id):
                self.assertTrue(capability.model_ids)
                self.assertTrue(capability.outputs)
                for model_id in capability.model_ids:
                    self.assertIn(model_id, BUILTIN_MODELS)

    def test_adapter_distinguishes_model_demo_failure_and_cancel(self):
        artifact = EngineArtifact(ArtifactKind.MIDI, payload=object())

        model = adapt_engine_result(
            CAP_MIDI_GENERATE,
            _Result(),
            [artifact],
            model_id="midi-llm-1b",
        )
        demo = adapt_engine_result(
            CAP_MIDI_GENERATE,
            _Result(output_kind="demo"),
            [EngineArtifact(ArtifactKind.MIDI, payload=object())],
        )
        failed = adapt_engine_result(
            CAP_MIDI_GENERATE,
            _Result(error="generation failed", is_success=False),
        )
        cancelled = adapt_engine_result(
            CAP_MIDI_GENERATE,
            type("Cancelled", (), {"cancelled": True})(),
        )

        self.assertEqual(RunOutcome.MODEL, model.outcome)
        self.assertEqual(RunOutcome.DEMO, demo.outcome)
        self.assertEqual(RunOutcome.FAILED, failed.outcome)
        self.assertEqual("generation failed", failed.error)
        self.assertEqual(RunOutcome.CANCELLED, cancelled.outcome)

    def test_adapter_rejects_missing_and_undeclared_artifacts(self):
        missing = adapt_engine_result(CAP_MIDI_GENERATE, _Result(), [])
        unexpected = adapt_engine_result(
            CAP_MIDI_GENERATE,
            _Result(),
            [EngineArtifact(ArtifactKind.AUDIO, payload=object())],
        )

        self.assertFalse(missing.is_success)
        self.assertIn("without a declared output", missing.error)
        self.assertFalse(unexpected.is_success)
        self.assertIn("undeclared artifact", unexpected.error)

    def test_batch_contract_reports_partial_success_and_paths(self):
        good = EngineRunResult(
            capability_id=CAP_MIDI_GENERATE,
            outcome=RunOutcome.MODEL,
            artifacts=[
                EngineArtifact(
                    ArtifactKind.MIDI,
                    path="generated.mid",
                )
            ],
        )
        bad = EngineRunResult.failure(CAP_MIDI_GENERATE, "failed")
        batch = EngineBatchResult(CAP_MIDI_GENERATE, [good, bad])

        self.assertTrue(batch.is_success)
        self.assertEqual([good], batch.successful_runs)
        self.assertEqual(["generated.mid"], batch.output_paths)
        self.assertEqual(1, batch.job_metadata()["engine_batch"]["failure_count"])

    def test_readiness_requires_activation_profiles_and_explicit_demos(self):
        manager = ModelManager()

        def readiness(model_id):
            return ModelReadiness(
                model_id=model_id,
                installed=True,
                verified=True,
                loadable=True,
                active=model_id == "rvc-v2",
                status="loaded" if model_id == "rvc-v2" else "downloaded",
                remedy=f"Activate {model_id} in Model Hub.",
            )

        with mock.patch.object(manager, "get_model_readiness", side_effect=readiness):
            midi_blocked = manager.get_capability_readiness(CAP_MIDI_GENERATE)
            midi_demo = manager.get_capability_readiness(
                CAP_MIDI_GENERATE,
                allow_demo=True,
            )
            producer = manager.get_capability_readiness(CAP_PRODUCER_RUN)
            rvc_missing_profile = manager.get_capability_readiness(
                CAP_VOCAL_CONVERT,
                allow_demo=True,
                profile_ready=False,
            )
            rvc_demo_off = manager.get_capability_readiness(
                CAP_VOCAL_CONVERT,
                profile_ready=True,
            )
            rvc_demo_on = manager.get_capability_readiness(
                CAP_VOCAL_CONVERT,
                allow_demo=True,
                profile_ready=True,
            )

        self.assertFalse(midi_blocked.can_run)
        self.assertEqual(RunMode.DEMO, midi_demo.mode)
        self.assertEqual(RunMode.MODEL, producer.mode)
        self.assertIn("consent-ready RVC", rvc_missing_profile.remedy)
        self.assertFalse(rvc_demo_off.can_run)
        self.assertIn("explicit demo", rvc_demo_off.remedy)
        self.assertEqual(RunMode.DEMO, rvc_demo_on.mode)

    def test_activation_and_deactivation_release_engine_resources(self):
        manager = ModelManager()
        engine = mock.Mock()
        old_registry = manager._registry
        old_status = manager._status
        old_current_id = manager._current_model_id
        old_current = manager._current_model
        try:
            manager._registry = {"diffsinger": BUILTIN_MODELS["diffsinger"]}
            manager._status = {"diffsinger": ModelStatus.DOWNLOADED}
            manager._current_model_id = None
            manager._current_model = None
            with mock.patch.object(manager, "require_verified_model"), \
                    mock.patch.object(manager, "_dynamic_load", return_value=engine), \
                    mock.patch.object(manager, "_is_model_cached", return_value=True), \
                    mock.patch.object(manager, "_emit_gpu_status"):
                activated = manager.activate_model("diffsinger")
                deactivated = manager.deactivate_model("diffsinger")

            self.assertEqual(ActivationOutcome.ACTIVE, activated.outcome)
            self.assertEqual(engine, activated.engine)
            self.assertEqual(ActivationOutcome.INACTIVE, deactivated.outcome)
            engine.unload_model.assert_called_once_with()
            self.assertIsNone(manager.current_model)
            self.assertEqual(ModelStatus.DOWNLOADED, manager.get_status("diffsinger"))
        finally:
            manager._registry = old_registry
            manager._status = old_status
            manager._current_model_id = old_current_id
            manager._current_model = old_current

    def test_model_card_routes_activate_cancel_and_deactivate_actions(self):
        info = BUILTIN_MODELS["midi-llm-1b"]
        manager = ModelManager()
        old_status = manager._status.get(info.model_id)
        readiness = ModelReadiness(
            model_id=info.model_id,
            installed=True,
            verified=True,
            loadable=True,
            active=False,
            status="downloaded",
            remedy="Activate MIDI-LLM in Model Hub.",
        )
        with mock.patch.object(
            ModelManager,
            "get_model_readiness",
            return_value=readiness,
        ):
            card = ModelCard(info)
            activated = []
            cancelled = []
            deactivated = []
            card.activation_requested.connect(activated.append)
            card.activation_cancel_requested.connect(cancelled.append)
            card.deactivation_requested.connect(deactivated.append)
            try:
                manager._status[info.model_id] = ModelStatus.DOWNLOADED
                card.update_status(ModelStatus.DOWNLOADED)
                self.assertEqual("Activate", card._action_btn.text())
                card._action_btn.click()

                manager._status[info.model_id] = ModelStatus.LOADING
                card.update_status(ModelStatus.LOADING)
                self.assertEqual("Cancel Activation", card._action_btn.text())
                card._action_btn.click()

                manager._status[info.model_id] = ModelStatus.LOADED
                card.update_status(ModelStatus.LOADED)
                self.assertEqual("Deactivate", card._action_btn.text())
                card._action_btn.click()
            finally:
                card.deleteLater()
                if old_status is None:
                    manager._status.pop(info.model_id, None)
                else:
                    manager._status[info.model_id] = old_status

        self.assertEqual([info.model_id], activated)
        self.assertEqual([info.model_id], cancelled)
        self.assertEqual([info.model_id], deactivated)

    def test_rvc_demo_profile_uses_verified_base_without_checkpoint_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = VoiceProfile(
                name="Consented demo voice",
                engine="rvc",
                owner_name="Singer",
                consent_status="confirmed",
                consent_source="Self-recorded",
                language="en",
                permitted_uses=[VOICE_OPERATION_CONVERSION],
            )
            engine = RVCEngine()

            engine.activate_base_model(tmp)
            engine.prepare_demo_profile(profile)

            self.assertTrue(engine.is_loaded)
            self.assertIs(profile, engine._profile)
            self.assertIsNone(engine._model)
            engine.unload_model()
            self.assertFalse(engine.is_loaded)


if __name__ == "__main__":
    unittest.main()
