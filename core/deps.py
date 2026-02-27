"""
Slunder Studio v0.1.0 — Dependency Manager
Auto-installs missing packages at runtime. No user intervention required.
"""
import sys
import os
import subprocess
import importlib
import re
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


# -- CUDA / platform detection ------------------------------------------------

def _detect_cuda_version() -> Optional[str]:
    """
    Detect CUDA version. Returns version string like '12.4' or None.
    Checks torch first, then nvidia-smi output.
    """
    try:
        import torch
        if torch.cuda.is_available():
            # torch.version.cuda returns e.g. '12.4'
            return torch.version.cuda
    except (ImportError, AttributeError):
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            # Parse "CUDA Version: 12.4" from nvidia-smi output
            m = re.search(r"CUDA Version:\s*([\d.]+)", r.stdout)
            if m:
                return m.group(1)
            return "12.4"  # nvidia-smi works but can't parse version
    except Exception:
        pass
    return None


def _cuda_wheel_tags(cuda_ver: str) -> list[str]:
    """
    Given a CUDA version like '12.4', return wheel index tags to try
    in priority order (exact match first, then nearby).
    """
    try:
        major, minor = cuda_ver.split(".")[:2]
        major, minor = int(major), int(minor)
    except (ValueError, IndexError):
        return ["cu124", "cu121"]

    # Build tag list: exact, then down, then up
    tags = [f"cu{major}{minor}"]
    for offset in range(1, 4):
        if minor - offset >= 0:
            tags.append(f"cu{major}{minor - offset}")
        tags.append(f"cu{major}{minor + offset}")
    return tags


# -- Install logic -------------------------------------------------------------

_LLAMA_WHEEL_BASE = "https://abetlen.github.io/llama-cpp-python/whl"


def _install_llama_cpp(import_name: str) -> None:
    """
    Special installer for llama-cpp-python.
    Tries pre-built binary wheels (no compiler needed), multiple CUDA
    versions, CPU fallback, then source build as last resort.
    """
    import platform
    pip_name = "llama-cpp-python"
    print(f"[Slunder Studio] Installing {pip_name} (pre-built binary)...")

    is_mac = platform.system() == "Darwin"
    is_win = platform.system() == "Windows"
    cuda_ver = _detect_cuda_version()

    # Build ordered list of wheel index URLs to try
    wheel_urls = []

    # CUDA wheels (if GPU present)
    if cuda_ver:
        for tag in _cuda_wheel_tags(cuda_ver):
            wheel_urls.append(f"{_LLAMA_WHEEL_BASE}/{tag}")

    # Metal wheels (macOS only)
    if is_mac:
        wheel_urls.append(f"{_LLAMA_WHEEL_BASE}/metal")

    # CPU wheel (always)
    wheel_urls.append(f"{_LLAMA_WHEEL_BASE}/cpu")

    # Phase 1: Try each wheel index with --only-binary (no source build)
    for url in wheel_urls:
        cmd = [
            sys.executable, "-m", "pip", "install",
            pip_name,
            "--only-binary=:all:",
            "--extra-index-url", url,
        ]
        try:
            print(f"[Slunder Studio]   Trying {url.split('/')[-1]} wheel...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                importlib.invalidate_caches()
                importlib.import_module(import_name)
                print(f"[Slunder Studio] Installed: {pip_name} (binary)")
                return
        except (subprocess.TimeoutExpired, ImportError, Exception):
            continue

    # Phase 2: Try plain pip with --only-binary (PyPI may have platform wheels)
    try:
        print("[Slunder Studio]   Trying PyPI binary wheel...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            pip_name, "--only-binary=:all:",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            importlib.invalidate_caches()
            importlib.import_module(import_name)
            print(f"[Slunder Studio] Installed: {pip_name} (PyPI binary)")
            return
    except (subprocess.TimeoutExpired, ImportError, Exception):
        pass

    # Phase 3: Source build (requires C++ compiler)
    print("[Slunder Studio]   No binary wheel found. Attempting source build...")
    env = os.environ.copy()
    if cuda_ver:
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
    try:
        cmd = [sys.executable, "-m", "pip", "install", pip_name]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if result.returncode == 0:
            importlib.invalidate_caches()
            importlib.import_module(import_name)
            print(f"[Slunder Studio] Installed: {pip_name} (source build)")
            return
    except (subprocess.TimeoutExpired, ImportError, Exception):
        pass

    # All strategies failed — give clear instructions
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    plat = platform.system()
    cuda_info = f"CUDA {cuda_ver}" if cuda_ver else "no CUDA"

    instructions = (
        f"\n{'='*60}\n"
        f"  LLAMA-CPP-PYTHON INSTALL FAILED\n"
        f"  Python {py_ver} | {plat} | {cuda_info}\n"
        f"{'='*60}\n"
        f"\n"
        f"  No pre-built wheel found for your platform.\n"
        f"  Fix options:\n"
        f"\n"
    )
    if is_win:
        instructions += (
            f"  1. Install Visual Studio Build Tools (recommended):\n"
            f"     https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
            f"     Select 'Desktop development with C++'\n"
            f"     Then restart Slunder Studio.\n"
            f"\n"
            f"  2. Manual wheel install:\n"
            f"     pip install llama-cpp-python \\\n"
            f"       --extra-index-url {_LLAMA_WHEEL_BASE}/cu121\n"
        )
    elif is_mac:
        instructions += (
            f"  1. Install Xcode Command Line Tools:\n"
            f"     xcode-select --install\n"
            f"     Then restart Slunder Studio.\n"
            f"\n"
            f"  2. Manual wheel install:\n"
            f"     pip install llama-cpp-python \\\n"
            f"       --extra-index-url {_LLAMA_WHEEL_BASE}/metal\n"
        )
    else:
        instructions += (
            f"  1. Install build tools:\n"
            f"     sudo apt install build-essential cmake\n"
            f"     Then restart Slunder Studio.\n"
            f"\n"
            f"  2. Manual wheel install:\n"
            f"     pip install llama-cpp-python \\\n"
            f"       --extra-index-url {_LLAMA_WHEEL_BASE}/cpu\n"
        )

    raise ImportError(instructions)


# -- Generic installer ---------------------------------------------------------

def _install(pip_name: str, import_name: str) -> None:
    """Install a package via pip with multiple fallback strategies."""
    # Special handling for packages that need pre-built wheels
    if pip_name == "llama-cpp-python":
        return _install_llama_cpp(import_name)

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
