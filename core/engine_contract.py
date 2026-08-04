"""Shared capability, readiness, and result contracts for AI engines."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

from core.song_generator_registry import active_song_generator_model_ids


class ArtifactKind(str, Enum):
    AUDIO = "audio"
    MIDI = "midi"
    STEMS = "stems"
    LYRICS = "lyrics"
    DATA = "data"
    PROVENANCE = "provenance"


class RunMode(str, Enum):
    MODEL = "model"
    DEMO = "demo"
    UNAVAILABLE = "unavailable"


class RunOutcome(str, Enum):
    MODEL = "model"
    DEMO = "demo"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActivationOutcome(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class EngineCapability:
    """Static declaration for one user-visible engine action."""

    id: str
    label: str
    model_ids: tuple[str, ...]
    outputs: tuple[ArtifactKind, ...]
    requires_activation: bool = True
    auto_activates: bool = False
    profile_requirement: str = ""
    supports_demo: bool = False
    demo_requires_activation: bool = False
    model_output_available: bool = True
    unavailable_reason: str = ""

    @property
    def output_summary(self) -> str:
        return ", ".join(kind.value for kind in self.outputs)


@dataclass(frozen=True)
class ModelReadiness:
    model_id: str
    installed: bool
    verified: bool
    loadable: bool
    active: bool
    status: str
    missing_packages: tuple[str, ...] = ()
    remedy: str = ""


@dataclass(frozen=True)
class CapabilityReadiness:
    capability: EngineCapability
    mode: RunMode
    can_run: bool
    model_id: str = ""
    active_model_id: str = ""
    profile_ready: bool = False
    remedy: str = ""

    @property
    def output_summary(self) -> str:
        return self.capability.output_summary


@dataclass
class EngineActivationResult:
    model_id: str
    outcome: ActivationOutcome
    message: str = ""
    error: str = ""
    engine: Any = field(default=None, repr=False, compare=False)

    @property
    def is_success(self) -> bool:
        return self.outcome in {
            ActivationOutcome.ACTIVE,
            ActivationOutcome.INACTIVE,
        }

    @property
    def cancelled(self) -> bool:
        return self.outcome == ActivationOutcome.CANCELLED

    def job_metadata(self) -> dict[str, Any]:
        return {
            "engine_activation": {
                "model_id": self.model_id,
                "outcome": self.outcome.value,
                "message": self.message,
                "error": self.error,
            }
        }


@dataclass
class EngineArtifact:
    """A typed result artifact; payload is intentionally process-local."""

    kind: ArtifactKind
    path: str = ""
    payload: Any = field(default=None, repr=False, compare=False)
    provenance_path: str = ""
    routable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return bool(self.path) and Path(self.path).is_file()

    def job_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "path": self.path,
            "provenance_path": self.provenance_path,
            "routable": self.routable,
            "metadata": dict(self.metadata),
        }


@dataclass
class EngineRunResult:
    """Normalized result consumed by views, workers, routing, and jobs."""

    capability_id: str
    outcome: RunOutcome
    artifacts: list[EngineArtifact] = field(default_factory=list)
    model_id: str = ""
    message: str = ""
    error: str = ""
    source_result: Any = field(default=None, repr=False, compare=False)

    @property
    def is_success(self) -> bool:
        return self.outcome in {
            RunOutcome.MODEL,
            RunOutcome.DEMO,
            RunOutcome.DEGRADED,
        }

    @property
    def is_demo(self) -> bool:
        return self.outcome == RunOutcome.DEMO

    @property
    def is_cancelled(self) -> bool:
        return self.outcome == RunOutcome.CANCELLED

    @property
    def can_route(self) -> bool:
        return self.is_success and any(
            artifact.routable for artifact in self.artifacts
        )

    @property
    def output_paths(self) -> list[str]:
        return list(dict.fromkeys(
            artifact.path for artifact in self.artifacts if artifact.path
        ))

    def first_artifact(
        self,
        kind: Optional[ArtifactKind] = None,
    ) -> Optional[EngineArtifact]:
        for artifact in self.artifacts:
            if kind is None or artifact.kind == kind:
                return artifact
        return None

    def job_metadata(self) -> dict[str, Any]:
        return {
            "engine_result": {
                "capability_id": self.capability_id,
                "outcome": self.outcome.value,
                "model_id": self.model_id,
                "message": self.message,
                "error": self.error,
                "artifacts": [
                    artifact.job_metadata() for artifact in self.artifacts
                ],
            }
        }

    @classmethod
    def failure(
        cls,
        capability_id: str,
        error: str,
        *,
        model_id: str = "",
        source_result: Any = None,
    ) -> "EngineRunResult":
        return cls(
            capability_id=capability_id,
            outcome=RunOutcome.FAILED,
            model_id=model_id,
            error=str(error or "Engine action failed"),
            source_result=source_result,
        )

    @classmethod
    def cancelled(
        cls,
        capability_id: str,
        message: str = "Cancelled",
        *,
        model_id: str = "",
    ) -> "EngineRunResult":
        return cls(
            capability_id=capability_id,
            outcome=RunOutcome.CANCELLED,
            model_id=model_id,
            message=message,
        )


@dataclass
class EngineBatchResult:
    """Normalized collection for a single user action that yields many runs."""

    capability_id: str
    runs: list[EngineRunResult] = field(default_factory=list)
    error: str = ""

    @property
    def successful_runs(self) -> list[EngineRunResult]:
        return [run for run in self.runs if run.is_success]

    @property
    def is_success(self) -> bool:
        return bool(self.successful_runs)

    @property
    def output_paths(self) -> list[str]:
        return list(dict.fromkeys(
            path
            for run in self.runs
            for path in run.output_paths
        ))

    def job_metadata(self) -> dict[str, Any]:
        return {
            "engine_batch": {
                "capability_id": self.capability_id,
                "success_count": len(self.successful_runs),
                "failure_count": len(self.runs) - len(self.successful_runs),
                "runs": [run.job_metadata()["engine_result"] for run in self.runs],
            }
        }


CAP_SONG_GENERATE = "song.generate"
CAP_LYRICS_GENERATE = "lyrics.generate"
CAP_PRODUCER_RUN = "producer.run"
CAP_MIDI_GENERATE = "midi.generate"
CAP_MIDI_RENDER = "midi.render"
CAP_SFX_GENERATE = "sfx.generate"
CAP_VOCAL_SYNTHESIZE = "vocal.synthesize"
CAP_VOCAL_CONVERT = "vocal.convert"
CAP_VOCAL_CLONE = "vocal.clone"
CAP_STEM_SEPARATE = "stem.separate"


ENGINE_CAPABILITIES: dict[str, EngineCapability] = {
    CAP_SONG_GENERATE: EngineCapability(
        id=CAP_SONG_GENERATE,
        label="Generate song",
        model_ids=active_song_generator_model_ids(),
        outputs=(ArtifactKind.AUDIO, ArtifactKind.PROVENANCE),
        auto_activates=True,
        supports_demo=True,
    ),
    CAP_LYRICS_GENERATE: EngineCapability(
        id=CAP_LYRICS_GENERATE,
        label="Generate lyrics",
        model_ids=("llama-3.1-8b-q4", "llama-3.2-3b-q4", "qwen-2.5-14b-q4"),
        outputs=(ArtifactKind.LYRICS,),
        auto_activates=True,
    ),
    CAP_PRODUCER_RUN: EngineCapability(
        id=CAP_PRODUCER_RUN,
        label="Produce song",
        model_ids=active_song_generator_model_ids(),
        outputs=(ArtifactKind.AUDIO, ArtifactKind.PROVENANCE),
        auto_activates=True,
        supports_demo=True,
    ),
    CAP_MIDI_GENERATE: EngineCapability(
        id=CAP_MIDI_GENERATE,
        label="Generate MIDI",
        model_ids=("midi-llm-1b",),
        outputs=(ArtifactKind.MIDI, ArtifactKind.PROVENANCE),
        supports_demo=True,
    ),
    CAP_MIDI_RENDER: EngineCapability(
        id=CAP_MIDI_RENDER,
        label="Render MIDI audio",
        model_ids=("midi-llm-1b",),
        outputs=(ArtifactKind.AUDIO, ArtifactKind.PROVENANCE),
        requires_activation=False,
        supports_demo=True,
    ),
    CAP_SFX_GENERATE: EngineCapability(
        id=CAP_SFX_GENERATE,
        label="Generate sound effect",
        model_ids=("stable-audio-open",),
        outputs=(ArtifactKind.AUDIO, ArtifactKind.PROVENANCE),
        supports_demo=True,
    ),
    CAP_VOCAL_SYNTHESIZE: EngineCapability(
        id=CAP_VOCAL_SYNTHESIZE,
        label="Synthesize singing",
        model_ids=("diffsinger",),
        outputs=(ArtifactKind.AUDIO, ArtifactKind.PROVENANCE),
        profile_requirement="a local DiffSinger voice profile",
    ),
    CAP_VOCAL_CONVERT: EngineCapability(
        id=CAP_VOCAL_CONVERT,
        label="Convert voice",
        model_ids=("rvc-v2",),
        outputs=(ArtifactKind.AUDIO, ArtifactKind.PROVENANCE),
        profile_requirement="a consent-ready RVC voice profile",
        supports_demo=False,
        model_output_available=False,
        unavailable_reason=(
            "RVC conversion is unavailable until a verified local RVC inference "
            "adapter is bundled; no placeholder audio will be generated."
        ),
    ),
    CAP_VOCAL_CLONE: EngineCapability(
        id=CAP_VOCAL_CLONE,
        label="Clone voice",
        model_ids=("gpt-sovits-v2",),
        outputs=(ArtifactKind.AUDIO, ArtifactKind.PROVENANCE),
        profile_requirement="a consent-ready GPT-SoVITS voice profile",
        supports_demo=False,
        model_output_available=False,
        unavailable_reason=(
            "GPT-SoVITS cloning is unavailable until a verified local GPT-SoVITS "
            "inference adapter is bundled; no placeholder audio will be generated."
        ),
    ),
    CAP_STEM_SEPARATE: EngineCapability(
        id=CAP_STEM_SEPARATE,
        label="Separate stems",
        model_ids=("demucs-v4", "audio-separator"),
        outputs=(ArtifactKind.STEMS, ArtifactKind.AUDIO, ArtifactKind.PROVENANCE),
    ),
}


def get_capability(capability_id: str) -> EngineCapability:
    try:
        return ENGINE_CAPABILITIES[capability_id]
    except KeyError as exc:
        raise ValueError(f"Unknown engine capability: {capability_id}") from exc


def adapt_engine_result(
    capability_id: str,
    result: Any,
    artifacts: Iterable[EngineArtifact] = (),
    *,
    model_id: str = "",
    message: str = "",
) -> EngineRunResult:
    """Normalize an existing engine result without discarding its typed payload."""
    capability = get_capability(capability_id)
    artifact_list = list(artifacts)
    if result is None:
        return EngineRunResult.failure(
            capability_id,
            "Engine returned no result",
            model_id=model_id,
        )

    cancelled = bool(
        getattr(result, "cancelled", False)
        or getattr(result, "is_cancelled", False)
    )
    output_kind = str(getattr(result, "output_kind", "") or "").lower()
    if cancelled or output_kind in {"cancelled", "canceled"}:
        return EngineRunResult.cancelled(
            capability_id,
            model_id=model_id,
        )

    error = getattr(result, "error", "")
    if error:
        return EngineRunResult.failure(
            capability_id,
            str(error),
            model_id=model_id,
            source_result=result,
        )

    declared = set(capability.outputs)
    unexpected = [
        artifact.kind.value
        for artifact in artifact_list
        if artifact.kind not in declared
    ]
    if unexpected:
        return EngineRunResult.failure(
            capability_id,
            "Engine returned undeclared artifact kinds: "
            + ", ".join(sorted(set(unexpected))),
            model_id=model_id,
            source_result=result,
        )
    if capability.outputs and not artifact_list:
        return EngineRunResult.failure(
            capability_id,
            "Engine finished without a declared output artifact",
            model_id=model_id,
            source_result=result,
        )

    semantic_success = getattr(result, "is_success", None)
    if callable(semantic_success):
        semantic_success = semantic_success()
    if semantic_success is False:
        return EngineRunResult.failure(
            capability_id,
            "Engine returned an unsuccessful result",
            model_id=model_id,
            source_result=result,
        )

    is_demo = bool(getattr(result, "is_demo", False)) or output_kind == "demo"
    is_degraded = bool(getattr(result, "is_degraded", False))
    outcome = (
        RunOutcome.DEMO if is_demo
        else RunOutcome.DEGRADED if is_degraded
        else RunOutcome.MODEL
    )
    legacy_can_route = getattr(result, "can_route", True)
    if not legacy_can_route:
        for artifact in artifact_list:
            artifact.routable = False

    return EngineRunResult(
        capability_id=capability_id,
        outcome=outcome,
        artifacts=artifact_list,
        model_id=model_id,
        message=message,
        source_result=result,
    )
