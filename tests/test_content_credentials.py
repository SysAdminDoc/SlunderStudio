import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.audio_export import ExportSettings, export_from_numpy
from core.content_credentials import (
    C2PAConfig,
    ContentCredentialsError,
    c2pa_format_for_export,
    build_c2pa_manifest,
)


# These are the C2PA project's public FOR TESTING_ONLY ES256 credentials. They
# are intentionally confined to tests and must never be used for releases.
TEST_CERTIFICATE_CHAIN = """-----BEGIN CERTIFICATE-----
MIIChzCCAi6gAwIBAgIUcCTmJHYF8dZfG0d1UdT6/LXtkeYwCgYIKoZIzj0EAwIw
gYwxCzAJBgNVBAYTAlVTMQswCQYDVQQIDAJDQTESMBAGA1UEBwwJU29tZXdoZXJl
MScwJQYDVQQKDB5DMlBBIFRlc3QgSW50ZXJtZWRpYXRlIFJvb3QgQ0ExGTAXBgNV
BAsMEEZPUiBURVNUSU5HX09OTFkxGDAWBgNVBAMMD0ludGVybWVkaWF0ZSBDQTAe
Fw0yMjA2MTAxODQ2NDBaFw0zMDA4MjYxODQ2NDBaMIGAMQswCQYDVQQGEwJVUzEL
MAkGA1UECAwCQ0ExEjAQBgNVBAcMCVNvbWV3aGVyZTEfMB0GA1UECgwWQzJQQSBU
ZXN0IFNpZ25pbmcgQ2VydDEZMBcGA1UECwwQRk9SIFRFU1RJTkdfT05MWTEUMBIG
A1UEAwwLQzJQQSBTaWduZXIwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAAQPaL6R
kAkYkKU4+IryBSYxJM3h77sFiMrbvbI8fG7w2Bbl9otNG/cch3DAw5rGAPV7NWky
l3QGuV/wt0MrAPDoo3gwdjAMBgNVHRMBAf8EAjAAMBYGA1UdJQEB/wQMMAoGCCsG
AQUFBwMEMA4GA1UdDwEB/wQEAwIGwDAdBgNVHQ4EFgQUFznP0y83joiNOCedQkxT
tAMyNcowHwYDVR0jBBgwFoAUDnyNcma/osnlAJTvtW6A4rYOL2swCgYIKoZIzj0E
AwIDRwAwRAIgOY/2szXjslg/MyJFZ2y7OH8giPYTsvS7UPRP9GI9NgICIDQPMKrE
LQUJEtipZ0TqvI/4mieoyRCeIiQtyuS0LACz
-----END CERTIFICATE-----
-----BEGIN CERTIFICATE-----
MIICajCCAg+gAwIBAgIUfXDXHH+6GtA2QEBX2IvJ2YnGMnUwCgYIKoZIzj0EAwIw
dzELMAkGA1UEBhMCVVMxCzAJBgNVBAgMAkNBMRIwEAYDVQQHDAlTb21ld2hlcmUx
GjAYBgNVBAoMEUMyUEEgVGVzdCBSb290IENBMRkwFwYDVQQLDBBGT1IgVEVTVElO
R19PTkxZMRAwDgYDVQQDDAdSb290IENBMB4XDTIyMDYxMDE4NDY0MFoXDTMwMDgy
NzE4NDY0MFowgYwxCzAJBgNVBAYTAlVTMQswCQYDVQQIDAJDQTESMBAGA1UEBwwJ
U29tZXdoZXJlMScwJQYDVQQKDB5DMlBBIFRlc3QgSW50ZXJtZWRpYXRlIFJvb3Qg
Q0ExGTAXBgNVBA0MEEZPUiBURVNUSU5HX09OTFkxGDAWBgNVBAMMD0ludGVybWVk
aWF0ZSBDQTBZMBMGByqGSM49AgEGCCqGSM49AwEHA0IABHllI4O7a0EkpTYAWfPM
D6Rnfk9iqhEmCQKMOR6J47Rvh2GGjUw4CS+aLT89ySukPTnzGsMQ4jK9d3V4Aq4Q
LsOjYzBhMA8GA1UdEwEB/wQFMAMBAf8wDgYDVR0PAQH/BAQDAgGGMB0GA1UdDgQW
BBQOfI1yZr+iyeUAlO+1boDitg4vazAfBgNVHSMEGDAWgBRembiG4Xgb2VcVWnUA
UrYpDsuojDAKBggqhkjOPQQDAgNJADBGAiEAtdZ3+05CzFo90fWeZ4woeJcNQC4B
84Ill3YeZVvR8ZECIQDVRdha1xEDKuNTAManY0zthSosfXcvLnZui1A/y/DYeg==
-----END CERTIFICATE-----
"""
TEST_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgfNJBsaRLSeHizv0m
GL+gcn78QmtfLSm+n+qG9veC2W2hRANCAAQPaL6RkAkYkKU4+IryBSYxJM3h77sF
iMrbvbI8fG7w2Bbl9otNG/cch3DAw5rGAPV7NWkyl3QGuV/wt0MrAPDo
-----END PRIVATE KEY-----
"""


class ContentCredentialsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cert_path = self.root / "c2pa-certs.pem"
        self.key_path = self.root / "c2pa-private.key"
        self.cert_path.write_text(TEST_CERTIFICATE_CHAIN, encoding="utf-8")
        self.key_path.write_text(TEST_PRIVATE_KEY, encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_c2pa_is_off_by_default_for_exports(self):
        output = self.root / "plain.wav"
        export_from_numpy(
            np.zeros((4800, 1), dtype=np.float32),
            48000,
            str(output),
            ExportSettings(format="wav", sample_rate=48000, c2pa_enabled=False),
        )
        sidecar = json.loads(
            Path(str(output) + ".provenance.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("c2pa", sidecar["extra"])

    def test_manifest_includes_ai_disclosure_and_stable_digest(self):
        sidecar = {
            "schema_version": 2,
            "app_version": "0.1.31",
            "module": "song_forge",
            "operation": "generate",
            "output_kind": "model",
            "model": {
                "id": "ace-step-v1.5",
                "name": "ACE-Step 1.5",
                "revision": "abc123",
            },
            "parameters": {"seed": 7},
            "source_paths": [],
            "source_hashes": {},
            "export_format": "wav",
            "artifact": {"sha256": "raw"},
            "extra": {"delivery": {"verification": {"sha256": "raw"}}},
        }
        manifest, digest = build_c2pa_manifest(sidecar, export_format="wav")
        labels = [item["label"] for item in manifest["assertions"]]
        self.assertIn("c2pa.ai-disclosure", labels)
        self.assertIn("org.slunderstudio.provenance", labels)
        disclosure = next(
            item for item in manifest["assertions"]
            if item["label"] == "c2pa.ai-disclosure"
        )
        self.assertEqual(
            disclosure["data"]["contentProfile"]["humanOversightLevel"],
            "prompt_guided",
        )
        identity = next(
            item for item in manifest["assertions"]
            if item["label"] == "org.slunderstudio.provenance"
        )
        self.assertEqual(digest, identity["data"]["provenanceDigest"])

        sidecar["artifact"]["sha256"] = "final"
        sidecar["extra"]["delivery"]["verification"]["sha256"] = "final"
        _manifest_after, digest_after = build_c2pa_manifest(sidecar, export_format="wav")
        self.assertEqual(digest, digest_after)

    def test_signed_wav_round_trips_and_updates_sidecar(self):
        output = self.root / "signed.wav"
        export_from_numpy(
            np.zeros((4800, 1), dtype=np.float32),
            48000,
            str(output),
            ExportSettings(format="wav", sample_rate=48000, c2pa_enabled=True),
            c2pa_config=C2PAConfig(
                certificate_path=str(self.cert_path),
                private_key_path=str(self.key_path),
            ),
        )
        sidecar = json.loads(
            Path(str(output) + ".provenance.json").read_text(encoding="utf-8")
        )
        result = sidecar["extra"]["c2pa"]
        self.assertEqual(result["status"], "embedded")
        self.assertEqual(result["validation_state"], "Valid")
        self.assertIn("signingCredential.untrusted", result["validation_codes"])
        self.assertIn("org.slunderstudio.provenance", result["manifest_labels"])
        self.assertEqual(
            result["provenance_digest"],
            sidecar["extra"]["c2pa"]["provenance_digest"],
        )
        self.assertEqual(
            sidecar["artifact"]["sha256"],
            sidecar["extra"]["delivery"]["verification"]["sha256"],
        )
        self.assertGreater(output.stat().st_size, 0)

    def test_missing_credentials_fail_closed_and_remove_output(self):
        output = self.root / "missing.wav"
        with self.assertRaises(ContentCredentialsError):
            export_from_numpy(
                np.zeros((4800, 1), dtype=np.float32),
                48000,
                str(output),
                ExportSettings(format="wav", sample_rate=48000, c2pa_enabled=True),
            )
        self.assertFalse(output.exists())
        self.assertFalse(Path(str(output) + ".provenance.json").exists())

    def test_unsupported_format_is_explicit(self):
        self.assertEqual(c2pa_format_for_export("wav"), "audio/wav")
        with self.assertRaises(ContentCredentialsError):
            c2pa_format_for_export("ogg")


if __name__ == "__main__":
    unittest.main()
