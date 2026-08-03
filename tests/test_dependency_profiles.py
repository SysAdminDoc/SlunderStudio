import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.dependency_profiles import (
    DENYLISTED_PROFILE_PACKAGES,
    DependencyProfile,
    DependencyProfileError,
    LockEntry,
    WheelArtifact,
    create_sbom,
    get_profile,
    load_registry,
    lock_from_pip_report,
    offline_install_command,
    parse_lock,
    registry_diagnostics,
    validate_profile_registry_security,
    validate_profile,
    verify_wheelhouse,
    version_at_least,
    version_less_than,
)


class DependencyProfileTests(unittest.TestCase):
    def test_declared_security_floor_and_optional_gguf_pin_are_current(self):
        registry = load_registry()
        torch_floor = registry["advisory_floors"]["torch"]
        self.assertEqual(torch_floor["minimum"], "2.10.0")
        self.assertIn("CVE-2026-24747", torch_floor["advisory"])
        self.assertIn("GHSA-63cw-57p8-fm3p", torch_floor["advisory"])

        requirements = (
            Path(__file__).resolve().parents[1] / "requirements.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("# llama-cpp-python==0.3.34", requirements)
        self.assertNotIn("llama-cpp-python>=", requirements)
        self.assertNotIn("pyloudnorm", requirements)
        self.assertEqual(requirements.count("psutil"), 1)

    def test_security_controls_keep_denylisted_packages_out_of_every_profile(self):
        registry = load_registry()
        controls = registry["security_controls"]
        self.assertEqual(set(controls["denylisted_packages"]), DENYLISTED_PROFILE_PACKAGES)
        self.assertIn("config.json", controls["transformers_config_scan"])
        self.assertIn("transformers<4.58.0", controls["compatibility_note"])
        validate_profile_registry_security()

    def test_all_enabled_profiles_are_complete_hashed_and_above_floors(self):
        registry = load_registry()
        expected = {
            "windows-cpu",
            "windows-cuda",
            "linux-cpu",
            "linux-cuda",
            "macos-mps",
        }
        enabled = {
            name for name, data in registry["profiles"].items() if data.get("enabled")
        }
        self.assertEqual(enabled, expected)
        for name in sorted(enabled):
            profile, entries = validate_profile(name)
            self.assertGreaterEqual(len(entries), 35)
            self.assertTrue(profile.lock_path.is_file())
            self.assertTrue(all(entry.hashes for entry in entries.values()))
            self.assertTrue(
                version_at_least(
                    entries["torch"].version,
                    registry["advisory_floors"]["torch"]["minimum"],
                )
            )
            self.assertTrue(version_at_least(entries["transformers"].version, "4.53.0"))
            self.assertTrue(version_less_than(entries["transformers"].version, "4.58.0"))

    def test_directml_fails_closed_below_safe_torch_floor(self):
        registry = load_registry()
        profile = get_profile(
            "windows-directml",
            registry=registry,
            require_enabled=False,
        )
        self.assertFalse(profile.enabled)
        self.assertIn("2.4.1", profile.reason)
        self.assertIn("2.10.0", profile.reason)
        with self.assertRaisesRegex(DependencyProfileError, "disabled"):
            get_profile("windows-directml", registry=registry)

    def test_lock_rejects_unpinned_unhashed_and_vulnerable_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unhashed = root / "unhashed.txt"
            unhashed.write_text("torch==2.11.0\n", encoding="utf-8")
            with self.assertRaisesRegex(DependencyProfileError, "no SHA-256"):
                parse_lock(unhashed)

            ranged = root / "ranged.txt"
            ranged.write_text(
                "torch>=2.11.0 --hash=sha256:" + "a" * 64 + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DependencyProfileError, "not exactly pinned"):
                parse_lock(ranged)

            registry = json.loads(
                (Path(__file__).resolve().parents[1] / "requirements" / "profiles.json")
                .read_text(encoding="utf-8")
            )
            registry["profiles"] = {
                "bad": {
                    "enabled": True,
                    "backend": "cpu",
                    "system": "Windows",
                    "machines": ["AMD64"],
                    "python_tag": "cp312",
                    "lock": "bad.txt",
                    "roots": {"torch": "2.5.1", "transformers": "4.57.6"},
                    "smoke_imports": ["torch"],
                }
            }
            registry_path = root / "profiles.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            (root / "bad.txt").write_text(
                "torch==2.5.1 --hash=sha256:" + "a" * 64 + "\n"
                "transformers==4.57.6 --hash=sha256:" + "b" * 64 + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DependencyProfileError, "below security floor"):
                validate_profile("bad", registry_path=registry_path)

            with self.assertRaisesRegex(DependencyProfileError, "below security floor"):
                validate_profile_registry_security(registry_path)

    def test_denylisted_package_cannot_enter_a_profile_or_install_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = root / "unsafe.txt"
            lock.write_text(
                "kernels==0.1.0 --hash=sha256:" + "a" * 64 + "\n",
                encoding="utf-8",
            )
            registry = json.loads(
                (Path(__file__).resolve().parents[1] / "requirements" / "profiles.json")
                .read_text(encoding="utf-8")
            )
            registry["profiles"] = {
                "unsafe": {
                    "enabled": True,
                    "backend": "cpu",
                    "system": "Windows",
                    "machines": ["AMD64"],
                    "python_tag": "cp312",
                    "lock": "unsafe.txt",
                    "roots": {"kernels": "0.1.0"},
                    "smoke_imports": [],
                }
            }
            registry_path = root / "profiles.json"
            registry_path.write_text(json.dumps(registry), encoding="utf-8")

            with self.assertRaisesRegex(DependencyProfileError, "deny-listed"):
                validate_profile("unsafe", registry_path=registry_path)
            with self.assertRaisesRegex(DependencyProfileError, "deny-listed"):
                validate_profile_registry_security(registry_path)

            profile = DependencyProfile(
                name="unsafe",
                enabled=True,
                backend="cpu",
                system="Windows",
                machines=("AMD64",),
                python_tag="cp312",
                lock_path=lock,
                index_url="",
                extra_index_url="",
                roots={"kernels": "0.1.0"},
                smoke_imports=(),
            )
            with self.assertRaisesRegex(DependencyProfileError, "deny-listed"):
                offline_install_command(profile, root / "wheelhouse")

    def test_wheelhouse_rejects_tampering_and_sbom_covers_every_lock_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch_wheel = root / "torch-2.11.0+cpu-py3-none-any.whl"
            transformer_wheel = root / "transformers-4.57.6-py3-none-any.whl"
            torch_wheel.write_bytes(b"torch wheel")
            transformer_wheel.write_bytes(b"transformers wheel")
            entries = {
                "torch": LockEntry(
                    "torch",
                    "2.11.0+cpu",
                    (hashlib.sha256(torch_wheel.read_bytes()).hexdigest(),),
                ),
                "transformers": LockEntry(
                    "transformers",
                    "4.57.6",
                    (hashlib.sha256(transformer_wheel.read_bytes()).hexdigest(),),
                ),
            }
            artifacts = verify_wheelhouse(entries, root)
            profile = DependencyProfile(
                name="test-cpu",
                enabled=True,
                backend="cpu",
                system="Windows",
                machines=("AMD64",),
                python_tag="cp312",
                lock_path=root / "test.txt",
                index_url="https://example.invalid/simple",
                extra_index_url="",
                roots={"torch": "2.11.0+cpu", "transformers": "4.57.6"},
                smoke_imports=("torch",),
            )
            profile.lock_path.write_text(
                "torch==2.11.0+cpu --hash=sha256:" + entries["torch"].hashes[0] + "\n"
                "transformers==4.57.6 --hash=sha256:"
                + entries["transformers"].hashes[0]
                + "\n",
                encoding="utf-8",
            )
            sbom = create_sbom(profile, artifacts)
            self.assertEqual(sbom["bomFormat"], "CycloneDX")
            self.assertEqual(len(sbom["components"]), len(entries))
            self.assertEqual(
                {component["name"] for component in sbom["components"]},
                {"torch", "transformers"},
            )
            with self.assertRaisesRegex(DependencyProfileError, "does not match the lock"):
                create_sbom(profile, {"torch": artifacts["torch"]})

            torch_wheel.write_bytes(b"tampered")
            with self.assertRaisesRegex(DependencyProfileError, "hash mismatch"):
                verify_wheelhouse(entries, root)

    def test_offline_install_command_cannot_reach_an_index(self):
        profile, _ = validate_profile("windows-cpu")
        command = offline_install_command(profile, Path("wheelhouse"))
        self.assertIn("--no-index", command)
        self.assertIn("--require-hashes", command)
        self.assertIn("--only-binary=:all:", command)
        self.assertNotIn("--index-url", command)

    def test_pip_report_conversion_requires_every_root_and_archive_hash(self):
        registry = load_registry()
        profile = get_profile("windows-cpu", registry=registry)
        install = []
        for package, version in profile.roots.items():
            install.append({
                "metadata": {"name": package, "version": version},
                "download_info": {
                    "archive_info": {"hashes": {"sha256": "a" * 64}}
                },
            })
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                json.dumps({"pip_version": "test", "install": install}),
                encoding="utf-8",
            )
            lock = lock_from_pip_report("windows-cpu", report)
        self.assertIn("torch==2.11.0+cpu", lock)
        self.assertEqual(lock.count("--hash=sha256:"), len(profile.roots))

        install[0]["download_info"]["archive_info"]["hashes"] = {}
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(json.dumps({"install": install}), encoding="utf-8")
            with self.assertRaisesRegex(DependencyProfileError, "unhashed"):
                lock_from_pip_report("windows-cpu", report)

    def test_diagnostics_report_profile_validity_and_advisory_state(self):
        with mock.patch(
            "core.dependency_profiles.importlib.metadata.version",
            side_effect=lambda name: {"torch": "2.5.1", "transformers": "4.57.6"}[name],
        ):
            report = registry_diagnostics()
        self.assertFalse(report["advisory_status"]["torch"]["compliant"])
        self.assertTrue(report["advisory_status"]["transformers"]["compliant"])
        self.assertTrue(report["profiles"]["windows-cpu"]["package_count"] >= 35)
        self.assertFalse(report["profiles"]["windows-directml"]["enabled"])


if __name__ == "__main__":
    unittest.main()
