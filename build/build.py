#!/usr/bin/env python3
"""
Slunder Studio — Build Script
Creates a standalone executable using PyInstaller.

Usage:
    python build/build.py

Outputs:
    dist/SlunderStudio/          (one-folder distribution)
    dist/SlunderStudio.exe       (Windows one-file, if --onefile)
"""
import os
import sys
import subprocess
import shutil
import hashlib
import time
import zipfile
from pathlib import Path

APP_NAME = "SlunderStudio"
APP_VERSION = "0.1.17"
ENTRY_POINT = "main.py"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

    # Collect data files
    datas = [
        ("assets/templates", "assets/templates"),
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
        "engines.style_tags",
        "core.audio_engine",
        "core.audio_export",
        "core.lyrics_db",
        "core.mastering",
        "core.midi_utils",
        "core.model_manager",
        "core.project",
        "core.settings",
        "core.diagnostics",
        "core.voice_bank",
        "core.workers",
        "numpy",
        "sounddevice",
        "soundfile",
    ]

    # Build command
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

    # Icon (if exists)
    icon_path = os.path.join("assets", "icon.ico")
    if os.path.isfile(icon_path):
        cmd.extend(["--icon", icon_path])

    # Version info (Windows)
    if sys.platform == "win32":
        _create_version_file()
        version_file = os.path.join("build", "version_info.txt")
        if os.path.isfile(version_file):
            cmd.extend(["--version-file", version_file])

    # Entry point
    cmd.append(ENTRY_POINT)

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

    sign_executables([exe_path])

    if smoke:
        smoke_launch(exe_path)
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


def clean_artifacts():
    """Remove stale distributables before building."""
    for path in [
        onefolder_dir(),
        build_dir(),
        onefile_path(),
        onedir_zip_path(),
        checksum_path(),
        spec_path(),
    ]:
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


def onefile_path() -> Path:
    return dist_dir() / f"{APP_NAME}.exe"


def executable_path(onefile: bool) -> Path:
    return onefile_path() if onefile else onefolder_dir() / f"{APP_NAME}.exe"


def onedir_zip_path() -> Path:
    platform_tag = "win64" if sys.platform == "win32" else sys.platform
    return dist_dir() / f"{APP_NAME}-v{APP_VERSION}-{platform_tag}.zip"


def checksum_path() -> Path:
    return dist_dir() / "SHA256SUMS.txt"


def spec_path() -> Path:
    return PROJECT_ROOT / f"{APP_NAME}.spec"


def create_onedir_zip() -> Path:
    """Zip the one-folder distribution for release upload."""
    source_dir = onefolder_dir()
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


def sign_executables(exe_paths: list[Path]) -> list[Path]:
    """Authenticode-sign executables when signing configuration is present."""
    if sys.platform != "win32":
        print("Signing skipped: Authenticode signing is Windows-only.")
        return []

    signtool = os.environ.get("SLUNDER_SIGNTOOL") or shutil.which("signtool")
    cert_sha1 = os.environ.get("SLUNDER_SIGN_CERT_SHA1", "").strip()
    cert_file = os.environ.get("SLUNDER_SIGN_CERT_FILE", "").strip()
    cert_password = os.environ.get("SLUNDER_SIGN_CERT_PASSWORD", "")
    timestamp_url = os.environ.get("SLUNDER_SIGN_TIMESTAMP_URL", "http://timestamp.digicert.com")

    if not signtool:
        print("Signing skipped: signtool was not found.")
        return []
    if not cert_sha1 and not cert_file:
        print("Signing skipped: set SLUNDER_SIGN_CERT_SHA1 or SLUNDER_SIGN_CERT_FILE to enable signing.")
        return []

    signed: list[Path] = []
    for exe_path in exe_paths:
        cmd = [
            signtool,
            "sign",
            "/fd",
            "SHA256",
            "/tr",
            timestamp_url,
            "/td",
            "SHA256",
        ]
        if cert_sha1:
            cmd.extend(["/sha1", cert_sha1])
        else:
            cmd.extend(["/f", cert_file])
            if cert_password:
                cmd.extend(["/p", cert_password])
        cmd.append(str(exe_path))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Signing failed for {exe_path}")
        signed.append(exe_path)
        print(f"Signed: {exe_path}")
    return signed


def smoke_launch(exe_path: Path, seconds: float | None = None):
    """Launch the packaged app and verify it does not recursively spawn."""
    if sys.platform != "win32":
        print("Smoke launch skipped: process-count smoke is Windows-only.")
        return

    seconds = seconds if seconds is not None else float(os.environ.get("SLUNDER_BUILD_SMOKE_SECONDS", "8"))
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
    ids = process_ids_for_exe(exe_path)
    try:
        if len(ids) != 1:
            raise RuntimeError(f"Packaged smoke expected one {APP_NAME}.exe process, saw {len(ids)}: {ids}")
        print(f"Packaged smoke ok: process_count=1 pid={ids[0]}")
    finally:
        terminate_process_tree(ids or [process.pid])


def process_ids_for_exe(exe_path: Path) -> list[int]:
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


def terminate_process_tree(process_ids: list[int]):
    for pid in sorted(set(process_ids)):
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid), "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _create_version_file():
    """Create Windows version info file."""
    parts = APP_VERSION.split(".")
    while len(parts) < 4:
        parts.append("0")

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
