import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


def load_build_script():
    build_script_path = Path(__file__).resolve().parents[1] / "build" / "build.py"
    spec = importlib.util.spec_from_file_location(
        "slunder_build_artifacts_for_test",
        build_script_path,
    )
    build_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_script)
    return build_script


class BuildArtifactTests(unittest.TestCase):
    def setUp(self):
        self.build_script = load_build_script()

    def test_numpy_is_bundled(self):
        command = self.build_script.build_command(onefile=False)
        hidden_imports = {
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--hidden-import"
        }
        self.assertIn("numpy", hidden_imports)

    def test_dynamic_engine_imports_reach_pyinstaller_command(self):
        command = self.build_script.build_command(onefile=False)
        hidden_imports = {
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--hidden-import"
        }
        self.assertTrue(
            {
                "engines.ace_step_engine",
                "engines.lyrics_templates",
                "core.audio_export",
                "core.content_credentials",
                "c2pa",
            }.issubset(hidden_imports)
        )

    def test_command_excludes_polluted_and_unused_modules(self):
        command = self.build_script.build_command(onefile=False)
        excluded = {
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--exclude-module"
        }
        self.assertTrue({"pytest", "aiohttp", "PySide6.QtWebEngineCore"}.issubset(excluded))

    def test_command_disables_upx_and_build_tools_are_pinned(self):
        command = self.build_script.build_command(onefile=False)
        self.assertIn("--noupx", command)
        requirements = (
            self.build_script.PROJECT_ROOT / "build" / "requirements-build.txt"
        ).read_text(encoding="utf-8")
        lock = (
            self.build_script.PROJECT_ROOT / "build" / "requirements-build-lock.txt"
        ).read_text(encoding="utf-8")
        self.assertRegex(requirements, r"(?m)^pyinstaller==[0-9.]+$")
        self.assertRegex(lock, r"(?m)^pyinstaller==[0-9.]+ \\")

    def test_locked_packages_normalizes_requirement_extras(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "requirements-lock.txt"
            lock.write_text(
                "pydantic[email]==2.13.4 \\\n+    --hash=sha256:example\n"
                "model-signing==1.1.1 \\\n+    --hash=sha256:example\n",
                encoding="utf-8",
            )

            self.assertEqual(
                self.build_script._locked_packages(lock),
                {"pydantic": "2.13.4", "model-signing": "1.1.1"},
            )

    def test_reproducible_environment_sets_hash_and_source_controls(self):
        with mock.patch.object(self.build_script, "source_date_epoch", return_value="1700000000"):
            environment = self.build_script.reproducible_build_environment(
                {"PYTHONHASHSEED": "random", "SOURCE_DATE_EPOCH": "old"}
            )
        self.assertEqual(environment["PYTHONHASHSEED"], "0")
        self.assertEqual(environment["SOURCE_DATE_EPOCH"], "1700000000")

    def test_frozen_module_audit_rejects_unlocked_top_level_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            internal = Path(tmp) / "_internal"
            (internal / "numpy").mkdir(parents=True)
            (internal / "assets").mkdir()
            (internal / "unexpected_cloud_package").mkdir()
            with self.assertRaisesRegex(RuntimeError, "unexpected_cloud_package"):
                self.build_script.audit_frozen_modules(Path(tmp))

    def test_frozen_metadata_removes_installer_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "SlunderStudio"
            record = root / "_internal" / "sample-1.0.dist-info" / "RECORD"
            record.parent.mkdir(parents=True)
            record.write_text("temporary build state", encoding="utf-8")

            self.build_script.remove_nondeterministic_frozen_metadata(root)

            self.assertFalse(record.exists())

    def test_clean_artifacts_removes_stale_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_script.PROJECT_ROOT = root
            stale_paths = [
                self.build_script.onefolder_dir(),
                self.build_script.build_dir(),
            ]
            for path in stale_paths:
                path.mkdir(parents=True)
                (path / "stale.txt").write_text("old", encoding="utf-8")
            for path in [
                self.build_script.onefile_path(),
                self.build_script.onedir_zip_path(),
                self.build_script.frozen_sbom_path(onefile=False),
                self.build_script.frozen_sbom_path(onefile=True),
                root / "dist" / "SlunderStudio-v0.0.1-win64.zip",
                self.build_script.checksum_path(),
                self.build_script.spec_path(),
            ]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("old", encoding="utf-8")

            self.build_script.clean_artifacts()

            for path in stale_paths:
                self.assertFalse(path.exists())
            self.assertFalse(self.build_script.onefile_path().exists())
            self.assertFalse(self.build_script.onedir_zip_path().exists())
            self.assertFalse((root / "dist" / "SlunderStudio-v0.0.1-win64.zip").exists())
            self.assertFalse(self.build_script.checksum_path().exists())
            self.assertFalse(self.build_script.spec_path().exists())

    def test_zip_and_checksums_cover_release_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_script.PROJECT_ROOT = root
            exe = self.build_script.executable_path(onefile=False)
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"binary")
            data = exe.parent / "_internal" / "helper.dll"
            data.parent.mkdir()
            data.write_bytes(b"helper")

            zip_path = self.build_script.create_onedir_zip()
            checksums = self.build_script.write_checksums([exe, zip_path])

            with zipfile.ZipFile(zip_path) as bundle:
                self.assertIn("SlunderStudio/SlunderStudio.exe", bundle.namelist())
                self.assertIn("SlunderStudio/_internal/helper.dll", bundle.namelist())
                timestamps = {info.date_time for info in bundle.infolist()}
                self.assertEqual(len(timestamps), 1)
            first_hash = self.build_script.sha256_file(zip_path)
            self.build_script.create_onedir_zip()
            self.assertEqual(first_hash, self.build_script.sha256_file(zip_path))
            checksum_text = checksums.read_text(encoding="utf-8")
            self.assertIn("SlunderStudio/SlunderStudio.exe", checksum_text)
            self.assertIn(zip_path.name, checksum_text)
            self.assertRegex(checksum_text, r"^[0-9a-f]{64}  ", msg=checksum_text)

    def test_frozen_sbom_is_deterministic_and_covers_bundle_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "SlunderStudio"
            executable = root / "SlunderStudio.exe"
            helper = root / "_internal" / "helper.dll"
            executable.parent.mkdir(parents=True)
            helper.parent.mkdir(parents=True)
            executable.write_bytes(b"binary")
            helper.write_bytes(b"helper")

            sbom = self.build_script.create_frozen_sbom(
                root,
                source_date_epoch="1700000000",
            )
            payload = json.loads(sbom.read_text(encoding="utf-8"))
            self.assertEqual(payload["specVersion"], "1.7")
            self.assertEqual(
                {component["name"] for component in payload["components"]},
                {"SlunderStudio.exe", "_internal/helper.dll"},
            )
            hashes = {
                component["name"]: component["hashes"][0]["content"]
                for component in payload["components"]
            }
            self.assertEqual(hashes["SlunderStudio.exe"], self.build_script.sha256_file(executable))
            self.assertEqual(hashes["_internal/helper.dll"], self.build_script.sha256_file(helper))
            self.assertEqual(payload["metadata"]["timestamp"], "2023-11-14T22:13:20Z")
            first_bytes = sbom.read_bytes()
            self.build_script.create_frozen_sbom(root, source_date_epoch="1700000000")
            self.assertEqual(first_bytes, sbom.read_bytes())

    def test_release_artifacts_are_unsigned(self):
        """Signing is policy-excluded; no code signing path may exist."""
        source = Path(self.build_script.__file__).read_text(encoding="utf-8")
        for banned in ("signtool", "codesign", "SLUNDER_SIGN", "Authenticode", "notariz"):
            self.assertNotIn(banned.lower(), source.lower(), banned)
        self.assertFalse(hasattr(self.build_script, "sign_executables"))

    def test_version_comes_from_the_single_source(self):
        from core.version import APP_VERSION, APP_NAME

        self.assertEqual(self.build_script.APP_VERSION, APP_VERSION)
        self.assertEqual(self.build_script.APP_NAME, APP_NAME)

    def test_artifact_names_carry_version_and_platform(self):
        zip_name = self.build_script.onedir_zip_path().name
        self.assertIn(self.build_script.APP_VERSION, zip_name)
        self.assertIn(self.build_script.platform_tag(), zip_name)

    def test_executable_name_matches_the_platform(self):
        with mock.patch.object(self.build_script.sys, "platform", "win32"):
            self.assertEqual(self.build_script.executable_name(), "SlunderStudio.exe")
        with mock.patch.object(self.build_script.sys, "platform", "linux"):
            self.assertEqual(self.build_script.executable_name(), "SlunderStudio")

    def test_windows_smoke_uses_private_desktop_and_cleans_up(self):
        exe = Path("dist/SlunderStudio/SlunderStudio.exe")
        device = r"\\.\DISPLAY6"
        desktop = "CodexVisualIsolation-test"
        isolation_outputs = [
            json.dumps({"deviceName": device, "primary": False}),
            json.dumps({"processId": 42, "desktop": desktop, "display": device}),
            "placement proof passed",
            "swept 0 window(s)",
        ]
        with mock.patch.object(self.build_script.sys, "platform", "win32"), \
                mock.patch.object(self.build_script, "_run_visual_isolation", side_effect=isolation_outputs) as isolate, \
                mock.patch.object(self.build_script.time, "sleep"), \
                mock.patch.object(self.build_script, "process_ids_for_exe", side_effect=[[], [42]]), \
                mock.patch.object(self.build_script, "_stop_isolated_process") as stop:
            self.build_script.smoke_launch(exe, seconds=0)

        self.assertEqual(isolate.call_args_list[0], mock.call("ensure"))
        self.assertEqual(isolate.call_args_list[1], mock.call("launch", "-FilePath", str(exe)))
        stop.assert_called_once_with(42)

    def test_smoke_launch_rejects_recursive_processes(self):
        exe = Path("dist/SlunderStudio.exe")
        device = r"\\.\DISPLAY6"
        desktop = "CodexVisualIsolation-test"
        with mock.patch.object(self.build_script.sys, "platform", "win32"), \
                mock.patch.object(
                    self.build_script,
                    "_run_visual_isolation",
                    side_effect=[
                        json.dumps({"deviceName": device, "primary": False}),
                        json.dumps({"processId": 42, "desktop": desktop, "display": device}),
                        "placement proof passed",
                        "swept 0 window(s)",
                    ],
                ), \
                mock.patch.object(self.build_script.time, "sleep"), \
                mock.patch.object(self.build_script, "process_ids_for_exe", return_value=[]), \
                mock.patch.object(
                    self.build_script,
                    "process_tree_for_exe",
                    return_value={42: 1, 43: 42, 44: 43},
                ), \
                mock.patch.object(self.build_script, "_stop_isolated_process") as stop:
            with self.assertRaises(RuntimeError):
                self.build_script.smoke_launch(exe, seconds=0, onefile=True)

        stop.assert_called_once_with(42)

    def test_onefile_smoke_accepts_bootloader_parent_and_child(self):
        exe = Path("dist/SlunderStudio.exe")
        device = r"\\.\DISPLAY6"
        desktop = "CodexVisualIsolation-test"
        with mock.patch.object(self.build_script.sys, "platform", "win32"), \
                mock.patch.object(
                    self.build_script,
                    "_run_visual_isolation",
                    side_effect=[
                        json.dumps({"deviceName": device, "primary": False}),
                        json.dumps({"processId": 42, "desktop": desktop, "display": device}),
                        "placement proof passed",
                        "swept 0 window(s)",
                    ],
                ), \
                mock.patch.object(self.build_script.time, "sleep"), \
                mock.patch.object(self.build_script, "process_ids_for_exe", return_value=[]), \
                mock.patch.object(
                    self.build_script,
                    "process_tree_for_exe",
                    return_value={42: 1, 43: 42},
                ), \
                mock.patch.object(self.build_script, "_stop_isolated_process") as stop:
            self.build_script.smoke_launch(exe, seconds=0, onefile=True)

        stop.assert_called_once_with(42)

    def test_onefile_smoke_rejects_recursive_processes(self):
        exe = Path("dist/SlunderStudio.exe")
        device = r"\\.\DISPLAY6"
        desktop = "CodexVisualIsolation-test"
        with mock.patch.object(self.build_script.sys, "platform", "win32"), \
                mock.patch.object(
                    self.build_script,
                    "_run_visual_isolation",
                    side_effect=[
                        json.dumps({"deviceName": device, "primary": False}),
                        json.dumps({"processId": 42, "desktop": desktop, "display": device}),
                        "placement proof passed",
                        "swept 0 window(s)",
                    ],
                ), \
                mock.patch.object(self.build_script.time, "sleep"), \
                mock.patch.object(self.build_script, "process_ids_for_exe", return_value=[]), \
                mock.patch.object(
                    self.build_script,
                    "process_tree_for_exe",
                    return_value={42: 1, 43: 42, 44: 43},
                ), \
                mock.patch.object(self.build_script, "_stop_isolated_process") as stop:
            with self.assertRaises(RuntimeError):
                self.build_script.smoke_launch(exe, seconds=0, onefile=True)

        stop.assert_called_once_with(42)

    def test_process_ids_for_exe_embeds_escaped_powershell_path(self):
        exe = Path("dist/SlunderStudio/SlunderStudio.exe")
        run_result = mock.Mock(returncode=0, stdout="42\n43\n", stderr="")
        with mock.patch.object(self.build_script.sys, "platform", "win32"), \
                mock.patch.object(self.build_script.subprocess, "run", return_value=run_result) as run:
            ids = self.build_script.process_ids_for_exe(exe)

        self.assertEqual(ids, [42, 43])
        command = run.call_args.args[0]
        self.assertIn(str(exe), command[3])
        self.assertEqual(len(command), 4)

    def test_posix_smoke_fails_when_the_app_exits_immediately(self):
        exe = Path("dist/SlunderStudio/SlunderStudio")
        process = mock.Mock(pid=42)
        process.poll.return_value = 1
        process.returncode = 1
        with mock.patch.object(self.build_script.sys, "platform", "linux"), \
                mock.patch.object(self.build_script.subprocess, "Popen", return_value=process), \
                mock.patch.object(self.build_script.time, "sleep"), \
                mock.patch.object(self.build_script, "process_ids_for_exe", return_value=[42]):
            with self.assertRaises(RuntimeError):
                self.build_script.smoke_launch(exe, seconds=0)
        process.terminate.assert_called_once()

    def test_posix_smoke_passes_for_a_single_live_process(self):
        exe = Path("dist/SlunderStudio/SlunderStudio")
        process = mock.Mock(pid=42)
        process.poll.return_value = None
        with mock.patch.object(self.build_script.sys, "platform", "linux"), \
                mock.patch.object(self.build_script.subprocess, "Popen", return_value=process), \
                mock.patch.object(self.build_script.time, "sleep"), \
                mock.patch.object(self.build_script, "process_ids_for_exe", return_value=[42]):
            self.build_script.smoke_launch(exe, seconds=0)
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
