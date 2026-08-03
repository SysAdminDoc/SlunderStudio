#!/usr/bin/env python3
"""
Slunder Studio — Build Script
Creates a standalone executable using PyInstaller.

Usage:
    python build/build.py

Outputs:
    dist/SlunderStudio/                          (one-folder distribution)
    dist/SlunderStudio[.exe]                     (one-file, if --onefile)
    dist/SlunderStudio-vX.Y.Z-<platform>.zip     (one-folder archive)
    dist/SHA256SUMS.txt                          (recorded hashes)

All artifacts are unsigned. There is no signing step and none will be added.
"""
import os
import sys
import subprocess
import shutil
import hashlib
import importlib.metadata
import json
import platform
import re
import tempfile
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.version import APP_NAME, APP_VERSION  # noqa: E402 - needs sys.path above

ENTRY_POINT = "main.py"
RUNTIME_LOCK_PATH = PROJECT_ROOT / "requirements-lock.txt"
BUILD_LOCK_PATH = PROJECT_ROOT / "build" / "requirements-build-lock.txt"
VISUAL_ISOLATION_SCRIPT = Path.home() / ".claude" / "scripts" / "visual-isolation.ps1"

# These are build-only packages. They are allowed in the isolated builder but
# are excluded from the application graph and must never leak into the bundle.
BUILD_TOOL_PACKAGES = {"pip", "setuptools", "wheel"}

# PyInstaller hooks can discover optional packages that are present in a
# polluted developer interpreter. Keep the application graph limited to the
# runtime lock and the optional engines' lazy-import boundary.
EXCLUDED_MODULES = (
    "pytest",
    "pytest_asyncio",
    "pytest_cov",
    "hypothesis",
    "coverage",
    "mypy",
    "mcp",
    "aiohttp",
    "asyncpg",
    "boto3",
    "botocore",
    "duckdb",
    "email_validator",
    "keyring",
    "redis",
    "sqlalchemy",
    "psycopg2",
    "watchfiles",
    "websockets",
    "tornado",
    "httptools",
    "lxml",
    "pythonnet",
    "clr_loader",
    "win32com",
    "tkinter",
    "_tkinter",
    "setuptools",
    "pip",
    "wheel",
    # The shell uses Core, Gui and Widgets only. These Qt modules are not
    # imported by the app and can pull hundreds of megabytes of unused DLLs.
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtHttpServer",
    "PySide6.QtLocation",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNetworkAuth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtPrintSupport",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialBus",
    "PySide6.QtSerialPort",
    "PySide6.QtShaderTools",
    "PySide6.QtSpatialAudio",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtTextToSpeech",
    "PySide6.QtVirtualKeyboard",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets",
    "PySide6.QtXml",
    "PySide6.QtXmlPatterns",
)

# Non-package directories intentionally copied by PyInstaller. Every other
# top-level directory in _internal must resolve to a package in a lock file.
FROZEN_SUPPORT_DIRECTORIES = {
    "_sounddevice_data",
    "_soundfile_data",
    "_tcl_data",
    "_tk_data",
    "assets",
    "requirements",
    "tcl8",
    "tk8",
}


def require_pyinstaller():
    """Fail with setup instructions if the locked build tool is not present."""
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed.")
        print("Run this setup command before building:")
        print(
            f'  "{sys.executable}" -m pip install --require-hashes '
            f'-r "{BUILD_LOCK_PATH}"'
        )
        print("Then rerun:")
        print(f'  "{sys.executable}" build/build.py --no-smoke')
        sys.exit(2)


def _normalize_distribution_name(name: str) -> str:
    """Normalize a distribution name using the PEP 503 comparison form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_packages(path: Path) -> dict[str, str]:
    """Read pinned package versions from a pip-compile lock file."""
    if not path.is_file():
        raise RuntimeError(f"Required lock file is missing: {path}")
    packages: dict[str, str] = {}
    pattern = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            packages[_normalize_distribution_name(match.group(1))] = match.group(2)
    if not packages:
        raise RuntimeError(f"Lock file has no pinned packages: {path}")
    return packages


def _installed_packages() -> dict[str, str]:
    """Return installed distribution versions without consulting user metadata."""
    return {
        _normalize_distribution_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }


def validate_build_environment():
    """Require a venv containing exactly the two hash-locked package sets."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            "Refusing to build from the base Python installation. "
            "Run `py -3.12 build/build.py --clean-env` to create a temporary "
            "hash-locked builder."
        )

    runtime = _locked_packages(RUNTIME_LOCK_PATH)
    build_tools = _locked_packages(BUILD_LOCK_PATH)
    expected = {**runtime, **build_tools}
    installed = _installed_packages()

    version_mismatches = {
        name: f"expected {version}, found {installed.get(name, 'missing')}"
        for name, version in expected.items()
        if installed.get(name) != version
    }
    unexpected = sorted(
        name
        for name in installed
        if name not in expected and name not in {_normalize_distribution_name(p) for p in BUILD_TOOL_PACKAGES}
    )
    if version_mismatches or unexpected:
        details = []
        if version_mismatches:
            details.append(
                "version mismatches: "
                + ", ".join(f"{name} ({detail})" for name, detail in sorted(version_mismatches.items()))
            )
        if unexpected:
            details.append("unexpected packages: " + ", ".join(unexpected))
        raise RuntimeError(
            "The build environment is not the locked runtime/build environment; "
            + "; ".join(details)
            + ". Run `py -3.12 build/build.py --clean-env`."
        )
    require_pyinstaller()


def _python_in_venv(venv_path: Path) -> Path:
    """Return the platform-specific Python executable in a venv."""
    executable = "python.exe" if sys.platform == "win32" else "python"
    subdir = "Scripts" if sys.platform == "win32" else "bin"
    return venv_path / subdir / executable


def _run_checked(command: list[str], *, env: dict[str, str] | None = None):
    """Run a setup/build child process and preserve its useful output."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run_in_clean_environment(arguments: list[str]):
    """Create, populate, use, and remove a temporary locked build venv."""
    temp_root = Path(tempfile.mkdtemp(prefix="slunder-build-")).resolve()
    temp_parent = Path(tempfile.gettempdir()).resolve()
    if temp_root.parent != temp_parent:
        raise RuntimeError(f"Unexpected temporary build location: {temp_root}")
    venv_path = temp_root / "venv"
    try:
        _run_checked([sys.executable, "-m", "venv", str(venv_path)])
        python = _python_in_venv(venv_path)
        _run_checked(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "-r",
                str(RUNTIME_LOCK_PATH),
            ]
        )
        _run_checked(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "-r",
                str(BUILD_LOCK_PATH),
            ]
        )
        child_env = os.environ.copy()
        child_env["PYTHONNOUSERSITE"] = "1"
        child_env["SLUNDER_CLEAN_BUILD"] = "1"
        forwarded = [argument for argument in arguments if argument != "--clean-env"]
        _run_checked([str(python), str(Path(__file__).resolve()), *forwarded], env=child_env)
    finally:
        shutil.rmtree(temp_root)


def build(onefile: bool = False, smoke: bool = True):
    """Run the PyInstaller build."""
    validate_build_environment()

    os.chdir(PROJECT_ROOT)
    clean_artifacts()

    cmd = build_command(onefile=onefile)

    print(f"Building {APP_NAME} v{APP_VERSION}...")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nBuild failed with exit code {result.returncode}")
        sys.exit(1)

    exe_path = executable_path(onefile)
    if not exe_path.is_file():
        print(f"\nBuild failed: expected executable was not created: {exe_path}")
        sys.exit(1)

    if not onefile:
        audit_frozen_modules(onefolder_dir())

    if smoke:
        smoke_launch(exe_path, onefile=onefile)
    else:
        print("Smoke launch skipped by --no-smoke.")

    artifacts = [exe_path]
    if not onefile:
        artifacts.append(create_onedir_zip())

    checksum_path = write_checksums(artifacts)
    if onefile:
        print(f"\nBuild successful: {exe_path}")
    else:
        print(f"\nBuild successful: {onefolder_dir()}/")
        print(f"Run: {exe_path}")
    print(f"Checksums: {checksum_path}")


def build_command(onefile: bool = False) -> list[str]:
    """Construct the exact PyInstaller command used by :func:`build`."""

    # Collect data files
    datas = [
        ("assets/locales", "assets/locales"),
        ("assets/templates", "assets/templates"),
        ("requirements/profiles.json", "requirements"),
        ("requirements/profiles", "requirements/profiles"),
    ]

    # Hidden imports (engines that are dynamically loaded)
    hidden = [
        "engines.ace_step_engine",
        "engines.lyrics_engine",
        "engines.midi_llm_engine",
        "engines.fluidsynth_engine",
        "engines.diffsinger_engine",
        "engines.rvc_engine",
        "engines.demucs_engine",
        "engines.sfx_engine",
        "engines.ai_producer",
        "engines.audio_analyzer",
        "engines.lyrics_templates",
        "engines.melody_extractor",
        "engines.style_tags",
        "engines.vocal_tuning",
        "core.chord_chart",
        "core.audio_engine",
        "core.audio_export",
        "core.lyrics_db",
        "core.mastering",
        "core.midi_utils",
        "core.model_manager",
        "core.project",
        "core.settings",
        "core.diagnostics",
        "core.dependency_profiles",
        "core.voice_bank",
        "core.workers",
        "numpy",
        "sounddevice",
        "soundfile",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--windowed",  # no console window
        "--noconfirm",
        "--clean",
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # Data files
    for src, dest in datas:
        if os.path.exists(src):
            cmd.extend(["--add-data", f"{src}{os.pathsep}{dest}"])

    # Hidden imports
    for imp in hidden:
        cmd.extend(["--hidden-import", imp])

    # Keep optional developer/cloud packages and unused Qt modules out of the
    # application graph even if a caller bypasses the clean-env bootstrap.
    for module in EXCLUDED_MODULES:
        cmd.extend(["--exclude-module", module])

    runtime_hook = os.path.join("assets", "runtime_hook_mp.py")
    if os.path.isfile(runtime_hook):
        cmd.extend(["--runtime-hook", runtime_hook])

    icon_path = find_icon()
    if icon_path:
        cmd.extend(["--icon", str(icon_path)])
    else:
        print("No platform icon found; building without one.")

    # Version info (Windows)
    if sys.platform == "win32":
        _create_version_file()
        version_file = os.path.join("build", "version_info.txt")
        if os.path.isfile(version_file):
            cmd.extend(["--version-file", version_file])

    # Entry point
    cmd.append(ENTRY_POINT)

    return cmd


def _frozen_package_candidates(name: str, locked: set[str] | None = None) -> set[str]:
    """Map a frozen top-level directory to possible locked distributions."""
    package_name = name
    if package_name.endswith(".dist-info"):
        package_name = package_name[: -len(".dist-info")]
    candidates = {_normalize_distribution_name(package_name)}
    if name.endswith(".dist-info") and locked:
        normalized = _normalize_distribution_name(package_name)
        candidates.update(
            lock_name
            for lock_name in locked
            if normalized == lock_name or normalized.startswith(lock_name + "-")
        )
    if package_name.endswith(".libs"):
        candidates.add(_normalize_distribution_name(package_name[: -len(".libs")]))

    package_map = importlib.metadata.packages_distributions()
    for import_name in (name, package_name):
        for distribution in package_map.get(import_name, []):
            candidates.add(_normalize_distribution_name(distribution))
    return candidates


def audit_frozen_modules(distribution_dir: Path):
    """Reject bundled package directories that are absent from a lock file."""
    internal_dir = distribution_dir / "_internal"
    if not internal_dir.is_dir():
        raise RuntimeError(f"PyInstaller internal directory is missing: {internal_dir}")

    locked = set(_locked_packages(RUNTIME_LOCK_PATH)) | set(_locked_packages(BUILD_LOCK_PATH))
    unknown: list[str] = []
    for entry in sorted(internal_dir.iterdir(), key=lambda path: path.name.lower()):
        if not entry.is_dir() or entry.name in FROZEN_SUPPORT_DIRECTORIES:
            continue
        if not _frozen_package_candidates(entry.name, locked) & locked:
            unknown.append(entry.name)
    if unknown:
        raise RuntimeError(
            "Distributable contains top-level packages absent from the locked "
            "runtime/build inputs: " + ", ".join(unknown)
        )


def clean_artifacts():
    """Remove stale distributables before building."""
    paths = [
        onefolder_dir(),
        dist_dir() / f"{APP_NAME}.app",
        build_dir(),
        onefile_path(),
        onedir_zip_path(),
        checksum_path(),
        spec_path(),
    ]
    if dist_dir().exists():
        paths.extend(dist_dir().glob(f"{APP_NAME}-v*-*.zip"))

    for path in sorted(set(paths), key=lambda item: str(item), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def dist_dir() -> Path:
    return PROJECT_ROOT / "dist"


def build_dir() -> Path:
    return PROJECT_ROOT / "build" / APP_NAME


def onefolder_dir() -> Path:
    return dist_dir() / APP_NAME


def find_icon() -> Path | None:
    """First existing icon for this platform, checked in assets/ then repo root."""
    if sys.platform == "win32":
        names = ("icon.ico", "icon.png")
    elif sys.platform == "darwin":
        names = ("icon.icns", "icon.png")
    else:
        names = ("icon.png",)
    for directory in (PROJECT_ROOT / "assets", PROJECT_ROOT):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def executable_name() -> str:
    return f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME


def onefile_path() -> Path:
    return dist_dir() / executable_name()


def executable_path(onefile: bool) -> Path:
    if onefile:
        return onefile_path()
    if sys.platform == "darwin":
        # --windowed produces an .app bundle; the binary lives inside it.
        bundle = dist_dir() / f"{APP_NAME}.app"
        if bundle.is_dir():
            return bundle / "Contents" / "MacOS" / APP_NAME
    return onefolder_dir() / executable_name()


def platform_tag() -> str:
    """Stable, human-readable tag for artifact names."""
    machine = (platform.machine() or "unknown").lower()
    machine = {"amd64": "x64", "x86_64": "x64", "aarch64": "arm64"}.get(machine, machine)
    if sys.platform == "win32":
        return f"win-{machine}"
    if sys.platform == "darwin":
        return f"macos-{machine}"
    return f"{sys.platform}-{machine}"


def onedir_zip_path() -> Path:
    return dist_dir() / f"{APP_NAME}-v{APP_VERSION}-{platform_tag()}.zip"


def checksum_path() -> Path:
    return dist_dir() / "SHA256SUMS.txt"


def spec_path() -> Path:
    return PROJECT_ROOT / f"{APP_NAME}.spec"


def create_onedir_zip() -> Path:
    """Zip the one-folder distribution for release upload."""
    source_dir = onefolder_dir()
    if sys.platform == "darwin" and (dist_dir() / f"{APP_NAME}.app").is_dir():
        source_dir = dist_dir() / f"{APP_NAME}.app"
    target = onedir_zip_path()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"One-folder distribution missing: {source_dir}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(dist_dir()))
    print(f"Packaged ZIP: {target}")
    return target


def write_checksums(artifacts: list[Path], target: Path | None = None) -> Path:
    """Write SHA256 checksums for release artifacts."""
    target = target or checksum_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for artifact in artifacts:
        digest = sha256_file(artifact)
        rel = artifact.relative_to(dist_dir()) if artifact.is_relative_to(dist_dir()) else artifact
        lines.append(f"{digest}  {str(rel).replace(os.sep, '/')}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def smoke_launch(
    exe_path: Path,
    seconds: float | None = None,
    *,
    onefile: bool = False,
):
    """Launch the packaged app and verify it starts and does not fork-bomb."""
    seconds = seconds if seconds is not None else float(
        os.environ.get("SLUNDER_BUILD_SMOKE_SECONDS", "8")
    )
    if sys.platform == "win32":
        _smoke_launch_windows(exe_path, seconds, onefile=onefile)
    else:
        _smoke_launch_posix(exe_path, seconds, onefile=onefile)


def _smoke_launch_windows(exe_path: Path, seconds: float, *, onefile: bool = False):
    """Smoke-test Windows builds on the isolated non-input virtual display."""
    before = set(process_ids_for_exe(exe_path))
    if before:
        raise RuntimeError(f"Smoke launch blocked: {exe_path} is already running ({sorted(before)})")

    ensure_info = _last_json_object(_run_visual_isolation("ensure"))
    if not ensure_info or not ensure_info.get("deviceName") or ensure_info.get("primary"):
        raise RuntimeError(
            "Visual isolation did not prove a non-primary virtual display; refusing GUI smoke test"
        )

    launch_info = _last_json_object(
        _run_visual_isolation("launch", "-FilePath", str(exe_path))
    )
    if not launch_info:
        raise RuntimeError("Visual isolation did not return launch proof")
    if launch_info.get("display") != ensure_info["deviceName"]:
        raise RuntimeError(
            "Visual isolation launch display does not match the ensured virtual display"
        )
    process_id = launch_info.get("processId")
    desktop = launch_info.get("desktop")
    if not isinstance(process_id, int) or not desktop:
        raise RuntimeError("Visual isolation launch proof is missing process or desktop identity")

    try:
        time.sleep(seconds)
        _run_visual_isolation(
            "verify",
            "-ProcessId",
            str(process_id),
            "-DesktopName",
            desktop,
        )
        if onefile:
            tree = process_tree_for_exe(exe_path)
            parent_pid, child_pid = validate_onefile_process_tree(tree, process_id)
            print(
                "Packaged smoke ok: process_count=2 "
                f"parent_pid={parent_pid} child_pid={child_pid}"
            )
        else:
            ids = process_ids_for_exe(exe_path)
            if len(ids) != 1:
                raise RuntimeError(
                    f"Packaged smoke expected one {APP_NAME}.exe process, "
                    f"saw {len(ids)}: {ids}"
                )
            print(f"Packaged smoke ok: process_count=1 pid={ids[0]}")
    finally:
        try:
            _run_visual_isolation(
                "sweep",
                "-ProcessId",
                str(process_id),
                "-DesktopName",
                desktop,
            )
        finally:
            _stop_isolated_process(process_id)


def _run_visual_isolation(*arguments: str) -> str:
    """Run the repository-approved private-desktop isolation helper."""
    if not VISUAL_ISOLATION_SCRIPT.is_file():
        raise RuntimeError(
            f"Visual isolation helper is missing: {VISUAL_ISOLATION_SCRIPT}; refusing GUI smoke test"
        )
    result = subprocess.run(
        ["pwsh", "-File", str(VISUAL_ISOLATION_SCRIPT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Visual isolation command failed ({result.returncode}): {' '.join(arguments)}"
        )
    return result.stdout


def _last_json_object(output: str) -> dict | None:
    """Read the last JSON object emitted by the isolation helper."""
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _stop_isolated_process(process_id: int):
    """Stop only the process tree returned by the isolation launch proof."""
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(process_id)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    verify_script = (
        "for ($attempt = 0; $attempt -lt 10; $attempt++) { "
        f"$targetProcess = Get-Process -Id {process_id} -ErrorAction SilentlyContinue; "
        "if (-not $targetProcess) { exit 0 }; "
        "Start-Sleep -Milliseconds 500 }; "
        f"throw 'isolated process {process_id} still running'"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", verify_script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"isolated process {process_id} was not cleaned up")


def _smoke_launch_posix(exe_path: Path, seconds: float, *, onefile: bool = False):
    """Start the packaged app offscreen and confirm it stays up without forking."""
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    process = subprocess.Popen(
        [str(exe_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    time.sleep(seconds)
    try:
        if process.poll() is not None:
            raise RuntimeError(
                f"Packaged smoke failed: process exited with {process.returncode}"
            )
        if onefile:
            tree = process_tree_for_exe(exe_path)
            parent_pid, child_pid = validate_onefile_process_tree(tree, process.pid)
            print(
                "Packaged smoke ok: process_count=2 "
                f"parent_pid={parent_pid} child_pid={child_pid}"
            )
        else:
            count = len(process_ids_for_exe(exe_path))
            if count > 1:
                raise RuntimeError(
                    f"Packaged smoke expected one {APP_NAME} process, saw {count}"
                )
            print(f"Packaged smoke ok: process_count={max(count, 1)} pid={process.pid}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def process_ids_for_exe(exe_path: Path) -> list[int]:
    if sys.platform != "win32":
        return _process_ids_posix(exe_path)
    return _process_ids_windows(exe_path)


def process_tree_for_exe(exe_path: Path) -> dict[int, int]:
    """Return matching process IDs and their parent IDs."""
    if sys.platform != "win32":
        return _process_tree_posix(exe_path)
    return _process_tree_windows(exe_path)


def validate_onefile_process_tree(tree: dict[int, int], root_pid: int) -> tuple[int, int]:
    """Require exactly one PyInstaller bootloader child under the launched process."""
    if root_pid not in tree or len(tree) != 2:
        raise RuntimeError(
            f"Packaged onefile smoke expected one parent and child rooted at {root_pid}, "
            f"saw {tree}"
        )
    children = [pid for pid, parent_pid in tree.items() if parent_pid == root_pid]
    if len(children) != 1:
        raise RuntimeError(
            f"Packaged onefile smoke expected one child of {root_pid}, saw {tree}"
        )
    return root_pid, children[0]


def _process_ids_posix(exe_path: Path) -> list[int]:
    pgrep = shutil.which("pgrep")
    if not pgrep:
        return []
    result = subprocess.run(
        [pgrep, "-f", str(exe_path)], capture_output=True, text=True, check=False
    )
    return [int(line) for line in result.stdout.split() if line.strip().isdigit()]


def _process_ids_windows(exe_path: Path) -> list[int]:
    escaped_path = str(exe_path).replace("'", "''")
    script = (
        f"$exe = [System.IO.Path]::GetFullPath('{escaped_path}'); "
        f"Get-CimInstance Win32_Process -Filter \"name = '{APP_NAME}.exe'\" | "
        "Where-Object { $_.ExecutablePath -eq $exe } | "
        "ForEach-Object { $_.ProcessId }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to inspect running {APP_NAME} processes: {result.stderr.strip()}")
    ids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            ids.append(int(line))
    return ids


def _process_tree_windows(exe_path: Path) -> dict[int, int]:
    escaped_path = str(exe_path).replace("'", "''")
    script = (
        f"$exe = [System.IO.Path]::GetFullPath('{escaped_path}'); "
        f"Get-CimInstance Win32_Process -Filter \"name = '{APP_NAME}.exe'\" | "
        "Where-Object { $_.ExecutablePath -eq $exe } | "
        "ForEach-Object { \"$($_.ProcessId)|$($_.ParentProcessId)\" }"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to inspect running {APP_NAME} processes: {result.stderr.strip()}")
    tree: dict[int, int] = {}
    for line in result.stdout.splitlines():
        parts = line.strip().split("|", 1)
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            tree[int(parts[0])] = int(parts[1])
    return tree


def _process_tree_posix(exe_path: Path) -> dict[int, int]:
    ids = process_ids_for_exe(exe_path)
    if not ids:
        return {}
    ps = shutil.which("ps")
    if not ps:
        raise RuntimeError("Unable to inspect packaged process parents: ps is unavailable")
    result = subprocess.run(
        [ps, "-o", "pid=,ppid=", "-p", ",".join(str(pid) for pid in ids)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Unable to inspect packaged process parents: {result.stderr.strip()}")
    tree: dict[int, int] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            tree[int(parts[0])] = int(parts[1])
    return tree


def terminate_process_tree(process_ids: list[int]):
    for pid in sorted(set(process_ids)):
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid), "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _create_version_file():
    """Create the Windows version-info file from the single version source."""
    from core.version import version_tuple

    parts = [str(n) for n in version_tuple(4)]

    content = f"""
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({','.join(parts)}),
    prodvers=({','.join(parts)}),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(u'040904B0', [
        StringStruct(u'CompanyName', u'SysAdminDoc'),
        StringStruct(u'FileDescription', u'Slunder Studio - AI Music Suite'),
        StringStruct(u'FileVersion', u'{APP_VERSION}'),
        StringStruct(u'InternalName', u'{APP_NAME}'),
        StringStruct(u'OriginalFilename', u'{APP_NAME}.exe'),
        StringStruct(u'ProductName', u'Slunder Studio'),
        StringStruct(u'ProductVersion', u'{APP_VERSION}'),
      ])
    ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    os.makedirs("build", exist_ok=True)
    with open(os.path.join("build", "version_info.txt"), "w") as f:
        f.write(content.strip())


def main(arguments: list[str] | None = None):
    """Run a build, bootstrapping the locked venv when requested."""
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if "--clean-env" in arguments:
        run_in_clean_environment(arguments)
        return
    onefile = "--onefile" in arguments
    smoke = "--no-smoke" not in arguments
    build(onefile=onefile, smoke=smoke)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Build blocked: {exc}", file=sys.stderr)
        sys.exit(2)
