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
import platform
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.version import APP_NAME, APP_VERSION  # noqa: E402 - needs sys.path above

ENTRY_POINT = "main.py"


def require_pyinstaller():
    """Fail with setup instructions if PyInstaller is not present."""
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller is not installed.")
        print("Run this setup command before building:")
        print(f'  "{sys.executable}" -m pip install pyinstaller')
        print("Then rerun:")
        print(f'  "{sys.executable}" build/build.py')
        sys.exit(2)


def build(onefile: bool = False, smoke: bool = True):
    """Run the PyInstaller build."""
    require_pyinstaller()

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
        "numpy._core._exceptions",
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
    before = set(process_ids_for_exe(exe_path))
    if before:
        raise RuntimeError(f"Smoke launch blocked: {exe_path} is already running ({sorted(before)})")

    process = subprocess.Popen(
        [str(exe_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    time.sleep(seconds)
    ids: list[int] = []
    try:
        if onefile:
            tree = process_tree_for_exe(exe_path)
            ids = sorted(tree)
            parent_pid, child_pid = validate_onefile_process_tree(tree, process.pid)
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
        terminate_process_tree(ids or [process.pid])


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


if __name__ == "__main__":
    onefile = "--onefile" in sys.argv
    smoke = "--no-smoke" not in sys.argv
    build(onefile=onefile, smoke=smoke)
