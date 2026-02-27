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

APP_NAME = "SlunderStudio"
APP_VERSION = "0.1.0"
ENTRY_POINT = "main.py"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_pyinstaller():
    """Install PyInstaller if not present."""
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "pyinstaller", "-q"
        ])


def build(onefile: bool = False):
    """Run the PyInstaller build."""
    ensure_pyinstaller()

    os.chdir(PROJECT_ROOT)

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

    if result.returncode == 0:
        dist_path = os.path.join("dist", APP_NAME)
        if onefile:
            exe = os.path.join("dist", f"{APP_NAME}.exe")
            print(f"\nBuild successful: {exe}")
        else:
            print(f"\nBuild successful: {dist_path}/")
            print(f"Run: {os.path.join(dist_path, APP_NAME)}")
    else:
        print(f"\nBuild failed with exit code {result.returncode}")
        sys.exit(1)


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
    build(onefile=onefile)
