import builtins
import contextlib
import io
import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from core import deps


class DependencyDiagnosticsTests(unittest.TestCase):
    def test_missing_dependency_raises_without_installing(self):
        with mock.patch("core.deps._install") as install:
            with self.assertRaises(deps.MissingDependencyError) as ctx:
                deps.ensure(
                    "definitely_missing_slunder_dependency",
                    pip_name="slunder-missing-package",
                )

        install.assert_not_called()
        message = str(ctx.exception)
        self.assertIn("requirements.txt", message)
        self.assertIn("slunder-missing-package", message)
        self.assertIn("-m pip install", message)

    def test_install_compatibility_shim_refuses_mutation(self):
        with self.assertRaises(deps.MissingDependencyError) as ctx:
            deps._install("slunder-missing-package", "missing_import")

        self.assertIn("slunder-missing-package", str(ctx.exception))

    def test_build_preflight_does_not_install_pyinstaller(self):
        build_script_path = Path(__file__).resolve().parents[1] / "build" / "build.py"
        spec = importlib.util.spec_from_file_location(
            "slunder_build_for_test",
            build_script_path,
        )
        build_script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_script)

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "PyInstaller":
                raise ImportError("simulated missing PyInstaller")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with mock.patch.object(subprocess, "check_call") as check_call:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as ctx:
                        build_script.require_pyinstaller()

        self.assertEqual(ctx.exception.code, 2)
        check_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
