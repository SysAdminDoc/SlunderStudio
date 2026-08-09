"""Backend and checkpoint metadata for source-separation engines."""
from __future__ import annotations

from dataclasses import dataclass


COMMERCIAL_USE_ALLOWED = "allowed"
COMMERCIAL_USE_UNKNOWN = "unknown"
LONG_FILE_THRESHOLD_SECONDS = 180.0
LONG_FILE_POLICY_WARNING = "warn_no_crossfade"


@dataclass(frozen=True)
class SeparatorBackend:
    id: str
    name: str
    package: str
    import_name: str
    license: str
    license_url: str
    device_support: tuple[str, ...]
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeparatorCheckpoint:
    id: str
    backend_id: str
    model_filename: str
    name: str
    stems: tuple[str, ...]
    checkpoint_license: str
    checkpoint_license_url: str
    commercial_use: str
    vram_gb: float
    ram_gb: float
    chunking: str
    quality: str
    speed: str
    limitations: tuple[str, ...] = ()
    credit_required: str = ""
    long_file_threshold_seconds: float = LONG_FILE_THRESHOLD_SECONDS
    long_file_policy: str = LONG_FILE_POLICY_WARNING

    def metadata(self) -> dict:
        """Return the run-stable metadata stamped into provenance."""
        return {
            "checkpoint_id": self.id,
            "backend_id": self.backend_id,
            "model_filename": self.model_filename,
            "name": self.name,
            "stems": list(self.stems),
            "checkpoint_license": self.checkpoint_license,
            "checkpoint_license_url": self.checkpoint_license_url,
            "commercial_use": self.commercial_use,
            "vram_gb": self.vram_gb,
            "ram_gb": self.ram_gb,
            "chunking": self.chunking,
            "quality": self.quality,
            "speed": self.speed,
            "limitations": list(self.limitations),
            "credit_required": self.credit_required,
            "long_file_threshold_seconds": self.long_file_threshold_seconds,
            "long_file_policy": self.long_file_policy,
        }


SEPARATOR_BACKENDS: dict[str, SeparatorBackend] = {
    "demucs": SeparatorBackend(
        id="demucs",
        name="Demucs",
        package="demucs",
        import_name="demucs",
        license="MIT",
        license_url="https://github.com/facebookresearch/demucs",
        device_support=("cpu", "cuda"),
        limitations=("Upstream Demucs is archived; no new model features are expected.",),
    ),
    "audio-separator": SeparatorBackend(
        id="audio-separator",
        name="Audio Separator",
        package="audio-separator",
        import_name="audio_separator",
        license="MIT",
        license_url="https://github.com/nomadkaraoke/python-audio-separator",
        device_support=("cpu", "cuda"),
        limitations=(
            "Checkpoint licenses vary; verify the selected checkpoint before release.",
            "Large files should use bounded chunking to control memory use.",
        ),
    ),
}


SEPARATOR_CHECKPOINTS: dict[str, SeparatorCheckpoint] = {
    "demucs-htdemucs": SeparatorCheckpoint(
        id="demucs-htdemucs",
        backend_id="demucs",
        model_filename="htdemucs",
        name="Demucs v4 — htdemucs (4 stems)",
        stems=("drums", "bass", "other", "vocals"),
        checkpoint_license="MIT",
        checkpoint_license_url="https://github.com/facebookresearch/demucs",
        commercial_use=COMMERCIAL_USE_ALLOWED,
        vram_gb=4.0,
        ram_gb=8.0,
        chunking="Demucs segment processing",
        quality="Reference 4-stem baseline",
        speed="Moderate",
        limitations=("Upstream project is archived.",),
    ),
    "demucs-htdemucs-6s": SeparatorCheckpoint(
        id="demucs-htdemucs-6s",
        backend_id="demucs",
        model_filename="htdemucs_6s",
        name="Demucs v4 — htdemucs_6s (6 stems)",
        stems=("drums", "bass", "other", "vocals", "guitar", "piano"),
        checkpoint_license="MIT",
        checkpoint_license_url="https://github.com/facebookresearch/demucs",
        commercial_use=COMMERCIAL_USE_ALLOWED,
        vram_gb=4.5,
        ram_gb=10.0,
        chunking="Demucs segment processing",
        quality="Six-stem instrument split",
        speed="Moderate",
        limitations=("Upstream project is archived.",),
    ),
    "audio-separator-bs-roformer": SeparatorCheckpoint(
        id="audio-separator-bs-roformer",
        backend_id="audio-separator",
        model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        name="BS-Roformer Viperx (vocals + instrumental)",
        stems=("vocals", "instrumental"),
        checkpoint_license="Unknown — verify checkpoint terms",
        checkpoint_license_url="",
        commercial_use=COMMERCIAL_USE_UNKNOWN,
        vram_gb=4.0,
        ram_gb=8.0,
        chunking="Optional fixed-duration chunks; concatenation has no crossfade",
        quality="Published SDR-ranked vocal separation checkpoint",
        speed="Moderate",
        limitations=(
            "Only vocals and instrumental outputs are declared.",
            "Checkpoint provenance and commercial terms must be verified separately from the MIT wrapper.",
        ),
        credit_required="Credit the Audio Separator/UVR ecosystem as required by the selected checkpoint terms.",
    ),
}


def get_separator_backend(backend_id: str) -> SeparatorBackend:
    try:
        return SEPARATOR_BACKENDS[backend_id]
    except KeyError as exc:
        raise KeyError(f"Unknown separator backend: {backend_id}") from exc


def get_separator_checkpoint(checkpoint_id: str) -> SeparatorCheckpoint:
    try:
        return SEPARATOR_CHECKPOINTS[checkpoint_id]
    except KeyError as exc:
        raise KeyError(f"Unknown separator checkpoint: {checkpoint_id}") from exc


def separator_checkpoints(*, backend_id: str | None = None) -> tuple[SeparatorCheckpoint, ...]:
    values = tuple(SEPARATOR_CHECKPOINTS.values())
    if backend_id is None:
        return values
    return tuple(item for item in values if item.backend_id == backend_id)


def separator_artifact_policy(
    checkpoint: SeparatorCheckpoint,
    duration_seconds: float,
) -> dict:
    """Describe the output contract for one source/checkpoint combination.

    The adapters normalize every returned stem back to the source rate and
    frame count. Long inputs still receive an explicit warning because the
    configured backends may concatenate internal chunks without an
    application-level crossfade.
    """
    duration = max(0.0, float(duration_seconds or 0.0))
    is_long = duration > checkpoint.long_file_threshold_seconds
    return {
        "mode": "preserve_input_rate_duration",
        "source_duration_seconds": duration,
        "long_file": is_long,
        "threshold_seconds": checkpoint.long_file_threshold_seconds,
        "crossfade": False,
        "policy": checkpoint.long_file_policy if is_long else "native_rate_duration",
        "warning": (
            "long_file_no_crossfade"
            if is_long and checkpoint.long_file_policy == LONG_FILE_POLICY_WARNING
            else ""
        ),
    }


def checkpoint_id_for_demucs_model(model_name: str) -> str:
    return {
        "htdemucs_6s": "demucs-htdemucs-6s",
        "htdemucs": "demucs-htdemucs",
        "htdemucs_ft": "demucs-htdemucs",
        "mdx_extra": "demucs-htdemucs",
    }.get(model_name, model_name if model_name in SEPARATOR_CHECKPOINTS else "demucs-htdemucs")
