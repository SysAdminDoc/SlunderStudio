"""Configuration contract for interchangeable song-generation backends.

The desktop surface consumes a model id, not a concrete generator class.  A
new generator therefore only needs a pinned registry entry and an adapter that
implements the shared loader/generation entry points.  Runtime profiles remain
separate from this small core registry so an incompatible optional stack cannot
silently replace the supported ACE-Step path.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from core.ace_step_contract import (
    ACE_STEP_ADAPTER,
    ACE_STEP_CAPABILITIES,
    ACE_STEP_DISPLAY_NAME,
    ACE_STEP_LICENSE,
    ACE_STEP_LICENSE_URL,
    ACE_STEP_MODEL_ID,
    ACE_STEP_REVISION,
    ACE_STEP_SOURCE,
)


_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
COMMERCIAL_USE_ALLOWED = "allowed"


class SongGeneratorRegistryError(ValueError):
    """Raised when a generator entry would weaken the model policy."""


def default_local_mirror(model_id: str, revision: str) -> str:
    """Return the portable cache identity for one immutable model revision."""
    safe_id = str(model_id).replace("/", "--")
    return f"models/{safe_id}/{revision}"


@dataclass(frozen=True)
class SongGeneratorConfig:
    """Declarative identity and adapter contract for one song generator."""

    generator_id: str
    model_id: str
    display_name: str
    adapter_module: str
    loader_fn: str
    generation_fn: str
    source: str
    revision: str
    license: str
    license_url: str
    commercial_use: str
    local_mirror: str
    capabilities: tuple[str, ...] = ()
    operations: tuple[str, ...] = ("generate_song",)
    runtime_profile: str = ""
    enabled: bool = True
    availability_reason: str = ""
    description: str = ""
    vram_gb: float = 0.0
    disk_gb: float = 0.0
    is_core: bool = False

    @property
    def requires_license_acceptance(self) -> bool:
        """Return whether a user must explicitly accept the model terms."""
        return not self.license.strip() or self.commercial_use != COMMERCIAL_USE_ALLOWED

    @property
    def is_pinned(self) -> bool:
        """Return whether source, revision, and local mirror are immutable."""
        return bool(
            self.source.strip()
            and _COMMIT_SHA_RE.fullmatch(self.revision.strip())
            and self.revision in self.local_mirror
        )


def validate_song_generator_config(
    config: SongGeneratorConfig,
    *,
    license_accepted: bool = False,
) -> SongGeneratorConfig:
    """Validate a generator before it can enter an active registry."""
    required = {
        "generator_id": config.generator_id,
        "model_id": config.model_id,
        "display_name": config.display_name,
        "adapter_module": config.adapter_module,
        "loader_fn": config.loader_fn,
        "generation_fn": config.generation_fn,
        "source": config.source,
        "revision": config.revision,
        "license_url": config.license_url,
        "local_mirror": config.local_mirror,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise SongGeneratorRegistryError(
            "Song generator configuration is missing: " + ", ".join(missing)
        )
    if not config.is_pinned:
        raise SongGeneratorRegistryError(
            f"{config.display_name} must use a 40-character revision and a "
            "local mirror containing that revision."
        )
    if "generate_song" not in config.operations:
        raise SongGeneratorRegistryError(
            f"{config.display_name} must declare a generate_song operation."
        )
    if any(
        not str(operation).strip()
        or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(operation))
        for operation in config.operations
    ):
        raise SongGeneratorRegistryError(
            f"{config.display_name} declares an invalid adapter operation."
        )
    if config.requires_license_acceptance and not license_accepted:
        raise SongGeneratorRegistryError(
            f"{config.display_name} is not commercially cleared by default; "
            "record explicit license acceptance before enabling it."
        )
    if config.enabled and config.availability_reason:
        raise SongGeneratorRegistryError(
            f"{config.display_name} is enabled but declares unavailable: "
            f"{config.availability_reason}"
        )
    return config


class SongGeneratorRegistry:
    """Stable, policy-checked collection of generator configurations."""

    def __init__(self, configs: Iterable[SongGeneratorConfig] = ()):
        self._configs: dict[str, SongGeneratorConfig] = {}
        for config in configs:
            self.register(config)

    def register(
        self,
        config: SongGeneratorConfig,
        *,
        license_accepted: bool = False,
        replace: bool = False,
    ) -> SongGeneratorConfig:
        validate_song_generator_config(config, license_accepted=license_accepted)
        if not replace and config.model_id in self._configs:
            raise SongGeneratorRegistryError(
                f"A song generator is already registered for {config.model_id}."
            )
        self._configs[config.model_id] = config
        return config

    def get(self, model_id: str) -> SongGeneratorConfig | None:
        return self._configs.get(model_id)

    def values(self) -> tuple[SongGeneratorConfig, ...]:
        return tuple(self._configs.values())

    def active_model_ids(self) -> tuple[str, ...]:
        return tuple(config.model_id for config in self.values() if config.enabled)

    def validate(self) -> tuple[SongGeneratorConfig, ...]:
        for config in self.values():
            validate_song_generator_config(config)
        return self.values()


ACE_STEP_GENERATOR = SongGeneratorConfig(
    generator_id="ace-step",
    model_id=ACE_STEP_MODEL_ID,
    display_name=ACE_STEP_DISPLAY_NAME,
    adapter_module="engines.ace_step_engine",
    loader_fn="load_model",
    generation_fn="generate_song",
    source=ACE_STEP_SOURCE,
    revision=ACE_STEP_REVISION,
    license=ACE_STEP_LICENSE,
    license_url=ACE_STEP_LICENSE_URL,
    commercial_use=COMMERCIAL_USE_ALLOWED,
    local_mirror=default_local_mirror(ACE_STEP_MODEL_ID, ACE_STEP_REVISION),
    capabilities=tuple(ACE_STEP_CAPABILITIES),
    operations=(
        "generate_song",
        "generate_song_batch",
        "generate_cover",
        "generate_extend",
        "generate_repaint",
        "generate_seed_grid",
    ),
    runtime_profile="ace-step-1.5",
    description=(
        "Official ACE-Step 1.5 XL Turbo song generator with source-conditioned "
        "editing and long-form rendering."
    ),
    vram_gb=16.0,
    disk_gb=10.4,
    is_core=True,
)

# HeartMuLa is intentionally staged as configuration-only.  Its official
# package currently pins a separate torch/torchaudio stack, so enabling it in
# the desktop profile before that isolation exists would make the fallback less
# reliable than the upstream concentration it is meant to address.
HEARTMULA_GENERATOR = SongGeneratorConfig(
    generator_id="heartmula",
    model_id="heartmula-oss-3b",
    display_name="HeartMuLa OSS 3B",
    adapter_module="engines.heartmula_engine",
    loader_fn="load_model",
    generation_fn="generate_song",
    source="HeartMuLa/HeartMuLa-oss-3B",
    revision="d12ac79c6b3387d5c9eee456323495d8f08bb09d",
    license="Apache-2.0",
    license_url="https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B",
    commercial_use=COMMERCIAL_USE_ALLOWED,
    local_mirror=default_local_mirror(
        "heartmula-oss-3b",
        "d12ac79c6b3387d5c9eee456323495d8f08bb09d",
    ),
    capabilities=("lyrics-conditioned music", "multilingual tags", "local generation"),
    operations=("generate_song",),
    runtime_profile="heartmula-python310",
    enabled=False,
    availability_reason=(
        "requires an isolated HeartMuLa runtime profile for its separate torch "
        "stack before it can be enabled"
    ),
)


SONG_GENERATOR_REGISTRY = SongGeneratorRegistry(
    (ACE_STEP_GENERATOR, HEARTMULA_GENERATOR)
)


def get_song_generator(model_id: str) -> SongGeneratorConfig | None:
    """Return the declarative generator config for a model id."""
    return SONG_GENERATOR_REGISTRY.get(model_id)


def active_song_generator_model_ids() -> tuple[str, ...]:
    """Return model ids that are safe to advertise for song generation."""
    return SONG_GENERATOR_REGISTRY.active_model_ids()
