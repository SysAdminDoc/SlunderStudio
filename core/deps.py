"""
Slunder Studio v0.1.0 — Dependency Manager
Auto-installs missing packages at runtime. No user intervention required.
"""
import sys
import os
import subprocess
import importlib
from typing import Optional

# Map of import name -> pip package name (only where they differ)
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

# Track packages we've already installed this session (skip re-checks)
_installed_this_session: set = set()


def ensure(*packages: str, pip_name: Optional[str] = None) -> None:
    """
    Import one or more packages, auto-installing any that are missing.
    Guaranteed silent to the user — all install output goes to console only.
    """
    for pkg in packages:
        if pkg in _installed_this_session:
            continue
        try:
            importlib.import_module(pkg)
            _installed_this_session.add(pkg)
        except ImportError:
            pip = pip_name if (pip_name and len(packages) == 1) else _PIP_NAMES.get(pkg, pkg)
            _install(pip, pkg)
            _installed_this_session.add(pkg)


def _install(pip_name: str, import_name: str) -> None:
    """Install a package via pip with multiple fallback strategies."""
    print(f"[Slunder Studio] Auto-installing: {pip_name} ...")

    strategies = [
        [sys.executable, "-m", "pip", "install", pip_name],
        [sys.executable, "-m", "pip", "install", pip_name, "--user"],
        [sys.executable, "-m", "pip", "install", pip_name, "--break-system-packages"],
        [sys.executable, "-m", "pip", "install", pip_name, "--force-reinstall"],
    ]

    last_error = ""
    for cmd in strategies:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                # Force Python to see the newly installed package
                importlib.invalidate_caches()
                # Refresh sys.path for --user installs
                if "--user" in cmd:
                    import site
                    if hasattr(site, "getusersitepackages"):
                        user_site = site.getusersitepackages()
                        if user_site not in sys.path:
                            sys.path.insert(0, user_site)
                importlib.import_module(import_name)
                print(f"[Slunder Studio] Installed: {pip_name}")
                return
            else:
                last_error = result.stderr[-300:] if result.stderr else "unknown error"
        except subprocess.TimeoutExpired:
            last_error = "Install timed out (600s)"
        except ImportError:
            last_error = f"pip succeeded but import '{import_name}' still fails"
        except Exception as e:
            last_error = str(e)

    raise ImportError(
        f"Auto-install failed for '{pip_name}': {last_error}"
    )


def require(import_name: str, pip_name: Optional[str] = None):
    """
    Import and return a module, auto-installing if missing.
    """
    if import_name in _installed_this_session:
        return importlib.import_module(import_name)
    try:
        mod = importlib.import_module(import_name)
        _installed_this_session.add(import_name)
        return mod
    except ImportError:
        pip = pip_name or _PIP_NAMES.get(import_name, import_name)
        _install(pip, import_name)
        _installed_this_session.add(import_name)
        return importlib.import_module(import_name)


def check_available(import_name: str) -> bool:
    """Quick check if a package is importable without installing."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False
