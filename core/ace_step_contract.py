"""Canonical ACE-Step 1.5 model, adapter, and capability contract."""
from __future__ import annotations


ACE_STEP_MODEL_ID = "ace-step-v1.5"
ACE_STEP_VERSION = "1.5"
ACE_STEP_DISPLAY_NAME = "ACE-Step 1.5 XL Turbo (DiT)"
ACE_STEP_ADAPTER = "diffusers.AceStepPipeline"
ACE_STEP_SOURCE = "ACE-Step/acestep-v15-xl-turbo-diffusers"
ACE_STEP_REVISION = "200ba991ae448051e14b0183157e35c2d27c9fb0"
ACE_STEP_LICENSE = "MIT"
ACE_STEP_LICENSE_URL = (
    "https://huggingface.co/ACE-Step/acestep-v15-xl-turbo-diffusers"
)

ACE_STEP_SAMPLE_RATE = 48_000
ACE_STEP_MIN_DURATION = 10.0
ACE_STEP_MAX_DURATION = 600.0
ACE_STEP_DEFAULT_STEPS = 8
ACE_STEP_DEFAULT_SHIFT = 3.0
ACE_STEP_PYTHON_VERSIONS = ((3, 11), (3, 12))
ACE_STEP_DEPENDENCY_BOUNDS = {
    "torch": ("2.6.0", None),
    "transformers": ("4.53.0", "4.58.0"),
    "diffusers": ("0.37.0", None),
    "accelerate": ("1.12.0", None),
}

# "extend" is an application-level workflow implemented with the native
# repaint task over a zero-padded continuation region.
ACE_STEP_NATIVE_TASKS = (
    "text2music",
    "cover",
    "repaint",
    "extract",
    "lego",
    "complete",
)
ACE_STEP_APP_TASKS = ("text2music", "cover", "repaint", "extend")
ACE_STEP_SOURCE_TASKS = ("cover", "repaint", "extend")
ACE_STEP_CAPABILITIES = (
    "48 kHz stereo",
    "10 s to 10 min",
    "text-to-music",
    "reference-audio cover",
    "source-audio repaint",
    "source-audio extend",
    "50+ lyric languages",
    "CUDA CPU offload",
    "Apple Silicon MPS",
    "CPU fallback",
)

# The top-level pickle-backed silence latent is a converter artifact. The
# Diffusers model index loads the safetensors condition encoder instead.
ACE_STEP_IGNORE_PATTERNS = (
    ".gitattributes",
    "README.md",
    "silence_latent.pt",
)
