"""
Slunder Studio v0.1.16 - Dependency diagnostics.

This module never installs packages. Runtime code may import through these
helpers, but missing packages are reported with explicit setup commands.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

# Map of import name -> pip package name where they differ.
_PIP_NAMES = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "soundfile": "soundfile",
    "sounddevice": "sounddevice",
    "sklearn": "scikit-learn",
    "faiss": "faiss-cpu",
    "pyqtgraph": "pyqtgraph",
    "librosa": "librosa",
    "psutil": "psutil",
    "pypinyin": "pypinyin",
    "g2p_en": "g2p-en",
    "fluidsynth": "pyfluidsynth",
    "onnxruntime": "onnxruntime",
    "torchaudio": "torchaudio",
    "stable_audio_tools": "stable-audio-tools",
    "acestep": "ace-step",
    "whisper": "openai-whisper",
    "demucs": "demucs",
    "llama_cpp": "llama-cpp-python",
    "pretty_midi": "pretty-midi",
    "torch": "torch",
    "transformers": "transformers",
}

CORE_RUNTIME_PACKAGES: tuple[tuple[str, str], ...] = (
    ("PySide6", "PySide6"),
    ("numpy", "numpy"),
    ("sounddevice", "sounddevice"),
    ("soundfile", "soundfile"),
    ("huggingface_hub", "huggingface-hub"),
    ("pyqtgraph", "pyqtgraph"),
    ("librosa", "librosa"),
    ("psutil", "psutil"),
)

_checked_this_session: set[str] = set()


class MissingDependencyError(ImportError):
    """Raised when a dependency is missing and must be installed explicitly."""

    def __init__(self, missing: Sequence[tuple[str, str]]):
        self.missing = tuple(missing)
        super().__init__(format_missing_dependency_message(self.missing))


def project_root() -> Path:
    """Return the repository root used for setup commands."""
    return Path(__file__).resolve().parents[1]


def requirements_file() -> Path:
    return project_root() / "requirements.txt"


def pip_name_for(import_name: str, pip_name: Optional[str] = None) -> str:
    return pip_name or _PIP_NAMES.get(import_name, import_name)


def setup_commands() -> list[str]:
    """Commands that recreate the supported local dependency environment."""
    root = project_root()
    req = requirements_file()
    return [
        f'cd "{root}"',
        f'"{sys.executable}" -m pip install -r "{req}"',
        f'"{sys.executable}" "{root / "main.py"}"',
    ]


def package_install_command(packages: Iterable[str]) -> str:
    unique = []
    for package in packages:
        if package not in unique:
            unique.append(package)
    quoted = " ".join(unique)
    return f'"{sys.executable}" -m pip install {quoted}'


def dependency_status(
    packages: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Return missing import/package pairs without installing anything."""
    missing: list[tuple[str, str]] = []
    for import_name, pip_name in packages:
        try:
            importlib.import_module(import_name)
            _checked_this_session.add(import_name)
        except ImportError:
            missing.append((import_name, pip_name))
    return missing


def ensure(*packages: str, pip_name: Optional[str] = None) -> None:
    """Verify one or more packages are importable; raise with setup commands."""
    missing: list[tuple[str, str]] = []
    for pkg in packages:
        if pkg in _checked_this_session:
            continue
        try:
            importlib.import_module(pkg)
            _checked_this_session.add(pkg)
        except ImportError:
            missing.append((
                pkg,
                pip_name_for(pkg, pip_name if len(packages) == 1 else None),
            ))

    if missing:
        raise MissingDependencyError(missing)


def require(import_name: str, pip_name: Optional[str] = None):
    """Import and return a module, or raise MissingDependencyError."""
    try:
        module = importlib.import_module(import_name)
        _checked_this_session.add(import_name)
        return module
    except ImportError as exc:
        raise MissingDependencyError(
            [(import_name, pip_name_for(import_name, pip_name))]
        ) from exc


def _install(pip_name: str, import_name: str) -> None:
    """Compatibility shim for old callers; intentionally refuses mutation."""
    raise MissingDependencyError([(import_name, pip_name)])


def check_available(import_name: str) -> bool:
    """Quick check if a package is importable without installing."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def format_missing_dependency_message(
    missing: Sequence[tuple[str, str]],
) -> str:
    missing_lines = "\n".join(
        f"  - {pip_name} (import: {import_name})"
        for import_name, pip_name in missing
    )
    setup = "\n".join(f"  {cmd}" for cmd in setup_commands())
    direct = package_install_command(pip_name for _, pip_name in missing)
    return (
        "Slunder Studio cannot continue because Python dependencies "
        "are missing.\n\n"
        f"Missing:\n{missing_lines}\n\n"
        "Install the supported runtime environment:\n"
        f"{setup}\n\n"
        "For optional engine packages, this direct install command may also "
        "be used:\n"
        f"  {direct}"
    )
