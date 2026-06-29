import importlib.util
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
            checksum_text = checksums.read_text(encoding="utf-8")
            self.assertIn("SlunderStudio/SlunderStudio.exe", checksum_text)
            self.assertIn(zip_path.name, checksum_text)
            self.assertRegex(checksum_text, r"^[0-9a-f]{64}  ", msg=checksum_text)

    def test_signing_skips_without_certificate_configuration(self):
        exe = Path("dist/SlunderStudio/SlunderStudio.exe")
        with mock.patch.object(self.build_script.sys, "platform", "win32"), \
                mock.patch.object(self.build_script.shutil, "which", return_value="signtool"):
            with mock.patch.dict(self.build_script.os.environ, {}, clear=True):
                signed = self.build_script.sign_executables([exe])
        self.assertEqual(signed, [])

    def test_signing_uses_thumbprint_when_configured(self):
        exe = Path("dist/SlunderStudio/SlunderStudio.exe")
        run_result = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(self.build_script.sys, "platform", "win32"), \
                mock.patch.object(self.build_script.shutil, "which", return_value="signtool"), \
                mock.patch.object(self.build_script.subprocess, "run", return_value=run_result) as run, \
                mock.patch.dict(
                    self.build_script.os.environ,
                    {"SLUNDER_SIGN_CERT_SHA1": "ABC123"},
                    clear=True,
                ):
            signed = self.build_script.sign_executables([exe])

        self.assertEqual(signed, [exe])
        args = run.call_args.args[0]
        self.assertIn("/sha1", args)
        self.assertIn("ABC123", args)
        self.assertEqual(args[-1], str(exe))

    def test_smoke_launch_requires_single_process_and_cleans_up(self):
        exe = Path("dist/SlunderStudio/SlunderStudio.exe")
        process = mock.Mock(pid=42)
        with mock.patch.object(self.build_script.sys, "platform", "win32"), \
                mock.patch.object(self.build_script.subprocess, "Popen", return_value=process), \
                mock.patch.object(self.build_script.time, "sleep"), \
                mock.patch.object(self.build_script, "process_ids_for_exe", side_effect=[[], [42]]), \
                mock.patch.object(self.build_script, "terminate_process_tree") as terminate:
            self.build_script.smoke_launch(exe, seconds=0)

        terminate.assert_called_once_with([42])

    def test_smoke_launch_rejects_recursive_processes(self):
        exe = Path("dist/SlunderStudio/SlunderStudio.exe")
        process = mock.Mock(pid=42)
        with mock.patch.object(self.build_script.sys, "platform", "win32"), \
                mock.patch.object(self.build_script.subprocess, "Popen", return_value=process), \
                mock.patch.object(self.build_script.time, "sleep"), \
                mock.patch.object(self.build_script, "process_ids_for_exe", side_effect=[[], [42, 43]]), \
                mock.patch.object(self.build_script, "terminate_process_tree") as terminate:
            with self.assertRaises(RuntimeError):
                self.build_script.smoke_launch(exe, seconds=0)

        terminate.assert_called_once_with([42, 43])

    def test_process_ids_for_exe_embeds_escaped_powershell_path(self):
        exe = Path("dist/SlunderStudio/SlunderStudio.exe")
        run_result = mock.Mock(returncode=0, stdout="42\n43\n", stderr="")
        with mock.patch.object(self.build_script.subprocess, "run", return_value=run_result) as run:
            ids = self.build_script.process_ids_for_exe(exe)

        self.assertEqual(ids, [42, 43])
        command = run.call_args.args[0]
        self.assertIn(str(exe), command[3])
        self.assertEqual(len(command), 4)


if __name__ == "__main__":
    unittest.main()
