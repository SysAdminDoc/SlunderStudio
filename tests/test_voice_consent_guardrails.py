import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.provenance import read_provenance_sidecar
from core.voice_bank import (
    VOICE_OPERATION_CLONE,
    VOICE_OPERATION_CONVERSION,
    VoiceProfile,
    ensure_voice_profile_allowed,
    validate_voice_profile,
    voice_profile_provenance,
)
from engines.rvc_engine import GPTSoVITSEngine, RVCEngine, VoiceResult


def _consented_profile(**overrides) -> VoiceProfile:
    data = {
        "name": "Consented Voice",
        "engine": "gpt_sovits",
        "owner_name": "Singer",
        "consent_status": "confirmed",
        "consent_source": "Self-recorded / my voice",
        "consent_scope": "Clone + conversion",
        "language": "en",
        "permitted_uses": [VOICE_OPERATION_CLONE, VOICE_OPERATION_CONVERSION],
        "license": "user-confirmed",
    }
    data.update(overrides)
    return VoiceProfile(**data)


class VoiceConsentGuardrailTests(unittest.TestCase):
    def test_missing_consent_blocks_voice_operations(self):
        profile = VoiceProfile(name="Incomplete", engine="gpt_sovits")

        issues = validate_voice_profile(profile, VOICE_OPERATION_CLONE)

        self.assertIn("voice owner is required", issues)
        self.assertIn("consent status must be confirmed", issues)
        self.assertIn("permitted use metadata is required", issues)
        with self.assertRaises(RuntimeError) as ctx:
            ensure_voice_profile_allowed(profile, VOICE_OPERATION_CLONE)
        self.assertIn("consent metadata incomplete", str(ctx.exception))

    def test_confirmed_consent_exports_profile_provenance(self):
        profile = _consented_profile()

        issues = validate_voice_profile(profile, VOICE_OPERATION_CONVERSION)
        provenance = voice_profile_provenance(profile)

        self.assertEqual(issues, [])
        self.assertGreater(provenance["consent_recorded_at"], 0.0)
        self.assertEqual(provenance["owner_name"], "Singer")
        self.assertEqual(provenance["permitted_uses"], ["voice-clone", "voice-conversion"])

    def test_rvc_load_blocks_missing_consent_before_checkpoint_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "voice.safetensors"
            checkpoint.write_bytes(b"not real safetensors")
            profile = VoiceProfile(
                name="Missing Consent",
                engine="rvc",
                model_path=str(checkpoint),
                trusted=True,
            )
            engine = RVCEngine()

            with self.assertRaises(RuntimeError) as ctx:
                engine.load_model(profile, device="cpu")

            self.assertIn("consent metadata incomplete", str(ctx.exception))

    def test_clone_sidecar_embeds_voice_profile_consent(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = _consented_profile(ref_audio_path=os.path.join(tmp, "voice.wav"))
            engine = GPTSoVITSEngine()
            engine._output_dir = tmp
            result = VoiceResult(
                audio=np.zeros(64, dtype=np.float32),
                sample_rate=32000,
                duration=64 / 32000,
                output_kind="demo",
                provenance={
                    "module": "vocal_suite",
                    "operation": "gpt_sovits_clone",
                    "model_id": "gpt-sovits-v2",
                    "parameters": {"language": "en"},
                },
            )

            output = engine.save_output(result, profile=profile)
            data = read_provenance_sidecar(output)

            self.assertEqual(data["source_asset_ids"], [profile.id])
            self.assertEqual(data["extra"]["voice_profile"]["owner_name"], "Singer")
            self.assertEqual(data["extra"]["voice_profile"]["consent_status"], "confirmed")
            self.assertIn("voice-clone", data["extra"]["voice_profile"]["permitted_uses"])


if __name__ == "__main__":
    unittest.main()
