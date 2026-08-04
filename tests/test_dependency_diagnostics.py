import ast
import builtins
import contextlib
import io
import importlib.util
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from core import deps


class DependencyDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def _requirement_names(path: Path) -> set[str]:
        names = set()
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", "-")):
                continue
            name = re.split(r"[<>=!~;\\[]", line, maxsplit=1)[0].strip()
            if name:
                names.add(name.lower().replace("-", "_"))
        return names

    @staticmethod
    def _third_party_imports(root: Path) -> set[str]:
        imports = set()
        local_modules = {"core", "engines"}
        stdlib = set(getattr(sys, "stdlib_module_names", ()))
        for folder in (root / "core", root / "engines"):
            for path in folder.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.update(alias.name.split(".", 1)[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        imports.add(node.module.split(".", 1)[0])
        return imports - stdlib - local_modules

    def test_missing_dependency_raises_without_installing(self):
        with mock.patch.object(deps.importlib, "import_module", side_effect=ImportError):
            with self.assertRaises(deps.MissingDependencyError) as ctx:
                deps.ensure(
                    "definitely_missing_slunder_dependency",
                    pip_name="slunder-missing-package",
                )

        message = str(ctx.exception)
        self.assertIn("requirements.txt", message)
        self.assertIn("slunder-missing-package", message)
        self.assertIn("-m pip install", message)

    def test_install_compatibility_shim_refuses_mutation(self):
        self.assertFalse(hasattr(deps, "_install"))

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
            if re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*==\S+", line.strip())
        ]
        self.assertTrue(len(pinned) > 0, "requirements-lock.txt has no pinned packages")
        hashes = [line for line in lines if "--hash=sha256:" in line]
        self.assertGreaterEqual(
            len(hashes), len(pinned), "Every locked package must have a hash",
        )

        for line in pinned:
            self.assertIn("==", line, f"Lock entry not pinned: {line}")

    def test_requirements_lock_covers_core_requirements(self):
        root = Path(__file__).resolve().parents[1]
        req_path = root / "requirements.txt"
        lock_path = root / "requirements-lock.txt"

        req_names = self._requirement_names(req_path)
        lock_names = self._requirement_names(lock_path)

        for name in req_names:
            self.assertIn(
                name, lock_names,
                f"Core dependency '{name}' from requirements.txt missing from lock file",
            )

    def test_numeric_runtime_lock_uses_supported_numpy_scipy_range(self):
        root = Path(__file__).resolve().parents[1]
        requirements = (root / "requirements.txt").read_text(encoding="utf-8")
        lock = (root / "requirements-lock.txt").read_text(encoding="utf-8")

        self.assertIn("numpy>=2.0,<2.5", requirements)
        self.assertIn("scipy>=1.18", requirements)

        def locked_version(name):
            match = re.search(
                rf"(?m)^{re.escape(name)}==([^\s\\]+)",
                lock,
            )
            self.assertIsNotNone(match, f"{name} missing from runtime lock")
            return tuple(int(part) for part in match.group(1).split("."))

        numpy_version = locked_version("numpy")
        self.assertGreaterEqual(numpy_version, (2, 0))
        self.assertLess(numpy_version, (2, 5))
        self.assertEqual(locked_version("scipy"), (1, 18, 0))
        self.assertEqual(locked_version("numba"), (0, 66, 0))

    def test_direct_runtime_imports_have_declared_or_optional_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        req_names = self._requirement_names(root / "requirements.txt")
        lock_names = self._requirement_names(root / "requirements-lock.txt")

        # These imports are deliberately lazy and are supplied by optional
        # engine profiles rather than the small core runtime lock.
        optional_imports = {
            "demucs",
            "audio_separator",
            "diffusers",
            "faiss",
            "fluidsynth",
            "g2p_en",
            "keyring",
            "llama_cpp",
            "onnxruntime",
            "pretty_midi",
            "pypinyin",
            "safetensors",
            "stable_audio_tools",
            "torch",
            "torchaudio",
            "transformers",
            "yaml",
        }
        import_to_package = {
            "PySide6": "PySide6",
            "huggingface_hub": "huggingface-hub",
            "sounddevice": "sounddevice",
            "soundfile": "soundfile",
            "tqdm": "tqdm",
        }
        for import_name in sorted(
            self._third_party_imports(root) - optional_imports
        ):
            package_name = import_to_package.get(import_name, import_name)
            normalized = package_name.lower().replace("-", "_")
            self.assertIn(
                normalized,
                req_names,
                f"Direct import '{import_name}' lacks a requirements.txt entry",
            )
            self.assertIn(
                normalized,
                lock_names,
                f"Direct import '{import_name}' lacks a locked package",
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
