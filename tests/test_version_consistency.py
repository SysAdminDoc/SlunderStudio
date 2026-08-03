import re
import unittest
from pathlib import Path

from core.version import APP_NAME, APP_VERSION, version_tuple

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"dist", "build", "__pycache__", ".git", ".venv", ".pytest_cache", "assets"}
VERSION_LITERAL = re.compile(r"\bv?\d+\.\d+\.\d+\b")


def source_files():
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


class VersionConsistencyTests(unittest.TestCase):
    """One version source drives the UI, docs, and artifact metadata."""

    def test_settings_and_main_reexport_the_single_source(self):
        from core.settings import APP_VERSION as settings_version
        import main

        self.assertEqual(settings_version, APP_VERSION)
        self.assertEqual(main.APP_VERSION, APP_VERSION)

    def test_no_module_hardcodes_its_own_app_version(self):
        offenders = []
        for path in source_files():
            if path.name == "version.py" and path.parent.name == "core":
                continue
            if path.parent.name == "tests":
                # Tests may assert against deliberately fake version strings.
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                # "Slunder Studio v{APP_VERSION}" is fine; a literal is not.
                if re.search(r"Slunder Studio v\d", line):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
                if re.search(r"^\s*APP_VERSION\s*=\s*[\"']", line):
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
        self.assertEqual(offenders, [], "hardcoded version strings: " + ", ".join(offenders))

    def test_readme_badge_matches(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"badge/version-{APP_VERSION}-blue", readme)

    def test_lock_version_literals_match_the_single_source(self):
        lock = (ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
        versions = re.findall(r"Slunder Studio v(\d+\.\d+\.\d+)", lock)
        self.assertTrue(
            all(version == APP_VERSION for version in versions),
            f"requirements-lock version literals drifted: {versions}",
        )

    def test_changelog_has_an_entry_for_this_version_or_unreleased(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertTrue(
            f"## {APP_VERSION}" in changelog
            or f"## v{APP_VERSION}" in changelog
            or "## Unreleased" in changelog,
            "CHANGELOG has neither an Unreleased section nor this version",
        )

    def test_version_tuple_is_numeric_and_padded(self):
        parts = version_tuple(4)
        self.assertEqual(len(parts), 4)
        self.assertTrue(all(isinstance(p, int) for p in parts))
        self.assertEqual(parts[:3], tuple(int(x) for x in APP_VERSION.split(".")[:3]))

    def test_build_script_artifact_names_use_the_single_source(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "slunder_build_for_version_test", ROOT / "build" / "build.py"
        )
        build_script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_script)
        self.assertEqual(build_script.APP_VERSION, APP_VERSION)
        self.assertEqual(build_script.APP_NAME, APP_NAME)
        self.assertIn(APP_VERSION, build_script.onedir_zip_path().name)

    def test_build_command_contains_the_runtime_hidden_imports(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "slunder_build_for_command_test", ROOT / "build" / "build.py"
        )
        build_script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_script)
        command = build_script.build_command(onefile=False)
        hidden_imports = {
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--hidden-import"
        }
        self.assertIn("numpy", hidden_imports)


if __name__ == "__main__":
    unittest.main()
