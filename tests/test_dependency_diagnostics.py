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


    def test_requirements_lock_exists_and_is_parseable(self):
        lock_path = Path(__file__).resolve().parents[1] / "requirements-lock.txt"
        self.assertTrue(lock_path.is_file(), "requirements-lock.txt missing")

        lines = lock_path.read_text(encoding="utf-8").strip().splitlines()
        pinned = [
            line.strip() for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertTrue(len(pinned) > 0, "requirements-lock.txt has no pinned packages")

        for line in pinned:
            self.assertIn("==", line, f"Lock entry not pinned: {line}")

    def test_requirements_lock_covers_core_requirements(self):
        root = Path(__file__).resolve().parents[1]
        req_path = root / "requirements.txt"
        lock_path = root / "requirements-lock.txt"

        def parse_names(path: Path) -> set[str]:
            names = set()
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name = line.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip()
                names.add(name.lower().replace("-", "_"))
            return names

        req_names = parse_names(req_path)
        lock_names = parse_names(lock_path)

        for name in req_names:
            self.assertIn(
                name, lock_names,
                f"Core dependency '{name}' from requirements.txt missing from lock file",
            )

    def test_locked_core_packages_are_importable(self):
        import_map = {
            "pyside6": "PySide6",
            "numpy": "numpy",
            "sounddevice": "sounddevice",
            "soundfile": "soundfile",
            "huggingface_hub": "huggingface_hub",
            "pyqtgraph": "pyqtgraph",
            "librosa": "librosa",
            "psutil": "psutil",
        }
        for pkg, module in import_map.items():
            spec = importlib.util.find_spec(module)
            self.assertIsNotNone(spec, f"Locked package {pkg} ({module}) is not importable")


if __name__ == "__main__":
    unittest.main()
