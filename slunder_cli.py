"""Headless command line access to Slunder Studio engine contracts.

The CLI deliberately stays outside the Qt GUI shell.  Generation and export
commands still run through ``InferenceWorker`` so progress, cancellation,
failure state, output cleanup, and durable ``JobStore`` records remain the
same as the desktop workflows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass, replace
import json
import os
from pathlib import Path
import queue
import signal
import sys
import time
from typing import Any, Callable, Sequence

from core.audio_export import ExportSettings, configured_export_settings, export_audio
from core.engine_contract import (
    ArtifactKind,
    CAP_LYRICS_GENERATE,
    CAP_MIDI_GENERATE,
    CAP_SFX_GENERATE,
    EngineArtifact,
    EngineBatchResult,
    EngineRunResult,
    adapt_engine_result,
)
from core.job_state import JobStatus, JobStore
from core.midi_utils import save_midi
from core.provenance import (
    provenance_replayability,
    read_provenance_sidecar,
    sidecar_path_for,
    write_provenance_sidecar,
)
from core.settings import APP_VERSION
from core.version import APP_NAME
from core.workers import CancelledJobError, InferenceWorker


CLI_SCHEMA_VERSION = 1
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_CANCELLED = 130


class CLIError(ValueError):
    """An actionable command-line validation or contract error."""


@dataclass
class CLIExportResult:
    """Typed export result retained in the shared worker/job contract."""

    output_path: str
    provenance_path: str
    format: str

    @property
    def is_success(self) -> bool:
        return bool(self.output_path and Path(self.output_path).is_file())

    @property
    def output_kind(self) -> str:
        return "export"

    def job_metadata(self) -> dict[str, Any]:
        return {
            "cli_export": {
                "output_path": self.output_path,
                "provenance_path": self.provenance_path,
                "format": self.format,
            }
        }


@dataclass
class HeadlessExecution:
    """Terminal state from one worker-backed headless operation."""

    job_id: str
    status: str
    result: Any = None
    error: str = ""
    interrupted: bool = False


def _json_safe(value: Any) -> Any:
    """Convert public result values without serializing audio/model payloads."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return str(value)


def _artifact_payload(artifact: EngineArtifact) -> dict[str, Any]:
    return {
        "kind": artifact.kind.value,
        "path": artifact.path,
        "provenance_path": artifact.provenance_path,
        "routable": artifact.routable,
        "metadata": _json_safe(artifact.metadata),
    }


def result_payload(result: Any) -> Any:
    """Return a bounded JSON representation of a shared engine result."""
    if isinstance(result, EngineRunResult):
        payload = {
            "capability_id": result.capability_id,
            "outcome": result.outcome.value,
            "model_id": result.model_id,
            "message": result.message,
            "error": result.error,
            "artifacts": [_artifact_payload(item) for item in result.artifacts],
        }
        if result.source_result is not None:
            payload["source_result"] = result_payload(result.source_result)
        return payload
    if isinstance(result, EngineBatchResult):
        return {
            "capability_id": result.capability_id,
            "error": result.error,
            "runs": [result_payload(item) for item in result.runs],
        }
    if isinstance(result, dict):
        return _json_safe(result)
    if isinstance(result, CLIExportResult):
        return asdict(result)
    if hasattr(result, "midi_data"):
        midi_data = getattr(result, "midi_data", None)
        return {
            "error": str(getattr(result, "error", "") or ""),
            "output_kind": str(getattr(result, "output_kind", "") or ""),
            "can_route": bool(getattr(result, "can_route", False)),
            "generation_time": float(getattr(result, "generation_time", 0.0) or 0.0),
            "token_count": int(getattr(result, "token_count", 0) or 0),
            "provenance_path": str(getattr(result, "provenance_path", "") or ""),
            "midi": {
                "tracks": int(getattr(midi_data, "track_count", 0) or 0),
                "notes": int(getattr(midi_data, "total_notes", 0) or 0),
                "duration": float(getattr(midi_data, "duration", 0.0) or 0.0),
                "tempo": float(getattr(midi_data, "tempo", 0.0) or 0.0),
            }
            if midi_data is not None
            else None,
        }
    if hasattr(result, "file_path"):
        return {
            "error": str(getattr(result, "error", "") or ""),
            "output_kind": str(getattr(result, "output_kind", "") or ""),
            "is_demo": bool(getattr(result, "is_demo", False)),
            "can_route": bool(getattr(result, "can_route", False)),
            "file_path": str(getattr(result, "file_path", "") or ""),
            "provenance_path": str(getattr(result, "provenance_path", "") or ""),
            "sample_rate": int(getattr(result, "sample_rate", 0) or 0),
            "duration": float(getattr(result, "duration", 0.0) or 0.0),
            "seed": getattr(result, "seed", None),
        }
    return _json_safe(result)


class HeadlessRunner:
    """Run one existing worker contract without a GUI event loop."""

    def __init__(
        self,
        *,
        json_output: bool = False,
        quiet: bool = False,
        job_store: JobStore | None = None,
        event_writer: Callable[[str], None] | None = None,
    ):
        from PySide6.QtCore import QCoreApplication

        self._app = QCoreApplication.instance() or QCoreApplication(["slunder-cli"])
        self.json_output = json_output
        self.quiet = quiet
        self.job_store = job_store or JobStore()
        self._event_writer = event_writer or (lambda message: print(message, file=sys.stderr))

    def _write_event(self, job_id: str, event: tuple[str, Any]) -> None:
        if self.quiet or self.json_output:
            return
        kind, value = event
        if kind == "progress":
            self._event_writer(f"[{job_id}] {int(value)}%")
        elif value:
            self._event_writer(f"[{job_id}] {value}")

    def run(
        self,
        task_fn: Callable,
        *args,
        job_kind: str,
        job_label: str,
        job_inputs: dict[str, Any] | None = None,
        job_metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> HeadlessExecution:
        """Execute one task through InferenceWorker and wait cooperatively."""
        worker = InferenceWorker(
            task_fn,
            *args,
            job_kind=job_kind,
            job_label=job_label,
            job_inputs=job_inputs or {},
            job_metadata=job_metadata or {},
            job_store=self.job_store,
            **kwargs,
        )
        events: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()
        errors: list[str] = []
        cancelled = [False]
        interrupted = [False]
        worker.progress.connect(lambda value: events.put(("progress", value)))
        worker.step_info.connect(lambda message: events.put(("step", message)))
        worker.log.connect(lambda message: events.put(("log", message)))
        worker.error.connect(lambda message: errors.append(str(message)))
        worker.cancelled.connect(lambda: cancelled.__setitem__(0, True))

        previous_handler = signal.getsignal(signal.SIGINT)

        def request_cancel(_signum, _frame):
            if not interrupted[0]:
                interrupted[0] = True
                worker.cancel()
                self._event_writer(f"[{worker.job_id}] cancellation requested")

        try:
            signal.signal(signal.SIGINT, request_cancel)
            worker.start()
            while worker.isRunning():
                self._app.processEvents()
                while True:
                    try:
                        self._write_event(worker.job_id, events.get_nowait())
                    except queue.Empty:
                        break
                time.sleep(0.02)
            worker.wait()
            self._app.processEvents()
            while True:
                try:
                    self._write_event(worker.job_id, events.get_nowait())
                except queue.Empty:
                    break
        finally:
            signal.signal(signal.SIGINT, previous_handler)

        result = worker.result
        result_cancelled = bool(
            getattr(result, "is_cancelled", False)
            or getattr(result, "cancelled", False)
            or (isinstance(result, dict) and result.get("cancelled"))
        )
        semantic_success = getattr(result, "is_success", None)
        if callable(semantic_success):
            semantic_success = semantic_success()
        if interrupted[0] or cancelled[0] or result_cancelled:
            status = JobStatus.CANCELLED
            error = str(getattr(result, "error", "") or "Cancelled")
        elif errors:
            status = JobStatus.FAILED
            error = errors[-1]
        elif semantic_success is False:
            status = JobStatus.FAILED
            error = str(
                getattr(result, "error", "")
                or getattr(result, "message", "")
                or "Task returned an unsuccessful result"
            )
        else:
            status = JobStatus.COMPLETED
            error = ""
        return HeadlessExecution(
            job_id=worker.job_id,
            status=status,
            result=result,
            error=error,
            interrupted=interrupted[0],
        )


def _task_progress(progress_cb, step_cb, value: float, message: str = "") -> None:
    if progress_cb:
        progress_cb(max(0, min(100, int(round(float(value) * 100)))))
    if step_cb and message:
        step_cb(str(message))


def _write_text_artifact(path: Path, text: str) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _build_lyrics_task(args):
    from engines.lyrics_engine import generate_lyrics, generate_lyrics_quick

    def task(progress_cb=None, step_cb=None, log_cb=None, cancel_event=None, **_kwargs):
        if cancel_event is not None and cancel_event.is_set():
            return EngineRunResult.cancelled(CAP_LYRICS_GENERATE, "Lyrics generation cancelled")
        if args.mode == "quick":
            raw = generate_lyrics_quick(
                args.prompt,
                progress_cb=progress_cb,
                step_cb=step_cb,
                log_cb=log_cb,
                cancel_event=cancel_event,
                model_id=args.model_id,
                language=args.language,
            )
        else:
            raw = generate_lyrics(
                args.prompt,
                genre_id=args.genre,
                mood=args.mood,
                language=args.language,
                structure_override=args.structure,
                progress_cb=progress_cb,
                step_cb=step_cb,
                log_cb=log_cb,
                cancel_event=cancel_event,
                model_id=args.model_id,
            )
        if isinstance(raw, dict) and raw.get("cancelled"):
            return EngineRunResult.cancelled(CAP_LYRICS_GENERATE, "Lyrics generation cancelled")
        lyrics = str(raw.get("lyrics", "") if isinstance(raw, dict) else "").strip()
        if not lyrics:
            return EngineRunResult.failure(
                CAP_LYRICS_GENERATE,
                "Lyrics engine returned no text",
                model_id=args.model_id or "",
                source_result=raw,
            )

        artifacts: list[EngineArtifact] = []
        output_path = Path(args.output).expanduser() if args.output else None
        provenance_path = ""
        if output_path is not None:
            _write_text_artifact(output_path, lyrics + "\n")
            sidecar = write_provenance_sidecar(
                output_path,
                module="lyrics",
                operation="generate",
                model_id=str(raw.get("model_id", args.model_id or "")),
                prompt=args.prompt,
                parameters={
                    "mode": args.mode,
                    "genre": args.genre,
                    "mood": args.mood,
                    "language": args.language,
                    "structure": args.structure,
                },
                export_format="txt",
                output_kind="model",
            )
            provenance_path = str(sidecar)
        artifacts.append(
            EngineArtifact(
                kind=ArtifactKind.LYRICS,
                path=str(output_path or ""),
                payload=lyrics,
                provenance_path=provenance_path,
                routable=False,
                metadata={
                    "characters": len(lyrics),
                    "mode": args.mode,
                },
            )
        )
        return adapt_engine_result(
            CAP_LYRICS_GENERATE,
            raw,
            artifacts,
            model_id=str(raw.get("model_id", args.model_id or "")),
        )

    return task


def _build_midi_task(args):
    from engines.midi_llm_engine import MidiGenParams, generate_midi

    params = MidiGenParams(
        prompt=args.prompt,
        style=args.style,
        key=args.key,
        tempo=args.tempo,
        duration_bars=args.duration_bars,
        chord_progression=args.chord_progression,
        drum_groove=args.drum_groove,
        seed=args.seed,
        allow_demo_output=args.demo,
    )

    def task(progress_cb=None, step_cb=None, log_cb=None, cancel_event=None, **_kwargs):
        def progress(value: float, message: str = ""):
            _task_progress(progress_cb, step_cb, value, message)

        result = generate_midi(
            params,
            progress_callback=progress,
            model_id=args.model_id,
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledJobError(
                "MIDI generation cancelled",
                outputs=[str(args.output), str(sidecar_path_for(args.output))],
            )
        if not result.is_success:
            return adapt_engine_result(
                CAP_MIDI_GENERATE,
                result,
                [],
                model_id=args.model_id or "midi-llm-1b",
            )

        provenance = dict(result.provenance or {})
        provenance.setdefault("module", "midi_studio")
        provenance.setdefault("operation", "generate_midi")
        provenance.setdefault("model_id", args.model_id or "midi-llm-1b")
        provenance.setdefault("prompt", args.prompt)
        provenance.setdefault("parameters", asdict(params))
        provenance.setdefault("output_kind", result.output_kind)
        save_midi(result.midi_data, str(args.output), provenance=provenance)
        result.provenance_path = str(sidecar_path_for(args.output))
        artifact = EngineArtifact(
            kind=ArtifactKind.MIDI,
            path=str(args.output),
            payload=result.midi_data,
            provenance_path=result.provenance_path,
            routable=result.can_route,
            metadata={
                "tracks": result.midi_data.track_count,
                "notes": result.midi_data.total_notes,
            },
        )
        return adapt_engine_result(
            CAP_MIDI_GENERATE,
            result,
            [artifact],
            model_id=args.model_id or "midi-llm-1b",
        )

    return task, params


def _build_sfx_task(args):
    from engines.sfx_engine import SFXParams, generate_sfx

    def task(progress_cb=None, step_cb=None, log_cb=None, cancel_event=None, **_kwargs):
        runs: list[EngineRunResult] = []
        for index in range(args.batch):
            if cancel_event is not None and cancel_event.is_set():
                batch = EngineBatchResult(CAP_SFX_GENERATE, runs)
                raise CancelledJobError(
                    f"SFX generation cancelled after {len(runs)} variation(s)",
                    outputs=batch.output_paths,
                    preserved=[path for path in batch.output_paths if Path(path).is_file()],
                    result=batch,
                )
            seed = args.seed + index if args.seed is not None else None
            params = SFXParams(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                duration=args.duration,
                cfg_scale=args.cfg_scale,
                steps=args.steps,
                seed=seed,
                allow_demo_output=args.demo,
            )

            def progress(value: float, message: str = ""):
                overall = (index + max(0.0, min(1.0, float(value)))) / args.batch
                _task_progress(progress_cb, step_cb, overall, f"{index + 1}/{args.batch}: {message}")

            result = generate_sfx(params, progress_callback=progress)
            artifacts: list[EngineArtifact] = []
            if result.file_path:
                artifacts.append(
                    EngineArtifact(
                        kind=ArtifactKind.AUDIO,
                        path=result.file_path,
                        provenance_path=result.provenance_path,
                        routable=result.can_route,
                        metadata={"sample_rate": result.sample_rate},
                    )
                )
            run = adapt_engine_result(
                CAP_SFX_GENERATE,
                result,
                artifacts,
                model_id="stable-audio-open",
            )
            if args.output_dir and run.is_success and result.audio is not None:
                from core.audio_export import export_from_numpy
                from core.audio_export import configured_export_settings
                from core.audio_export import deterministic_filename

                output_dir = Path(args.output_dir).expanduser()
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / deterministic_filename(
                    f"sfx-{args.prompt[:32]}",
                    fmt="wav",
                    variant=f"{index + 1:02d}",
                )
                settings = replace(
                    configured_export_settings(),
                    format="wav",
                    sample_rate=int(result.sample_rate),
                )
                written = export_from_numpy(
                    result.audio,
                    result.sample_rate,
                    str(output_path),
                    settings,
                    module="sfx",
                    operation="cli_export",
                    provenance_extra={
                        "generation_provenance_path": result.provenance_path,
                        "variation": index + 1,
                    },
                    progress_cb=(
                        lambda value: progress_cb(
                            50 + int(value * 0.5)
                        ) if progress_cb else None
                    ),
                    cancel_event=cancel_event,
                )
                result.file_path = written
                result.provenance_path = str(sidecar_path_for(written))
                run.artifacts[0].path = written
                run.artifacts[0].provenance_path = result.provenance_path
            runs.append(run)
        errors = [run.error for run in runs if not run.is_success and run.error]
        return EngineBatchResult(CAP_SFX_GENERATE, runs, error="; ".join(errors))

    return task


def _build_export_task(args):
    base = configured_export_settings()
    fmt = (args.format or Path(args.output).suffix.lstrip(".") or base.format).lower()
    settings = replace(
        base,
        format=fmt,
        sample_rate=args.sample_rate or base.sample_rate,
        bit_depth=args.bit_depth or base.bit_depth,
        normalize=args.normalize or base.normalize,
    )

    def task(progress_cb=None, step_cb=None, log_cb=None, cancel_event=None, **_kwargs):
        written = export_audio(
            str(args.source),
            str(args.output),
            settings,
            module="export",
            operation="cli_export",
            source_paths=[str(args.source)],
            progress_cb=progress_cb,
            step_cb=step_cb,
            log_cb=log_cb,
            cancel_event=cancel_event,
        )
        return CLIExportResult(
            output_path=written,
            provenance_path=str(sidecar_path_for(written)),
            format=settings.format,
        )

    return task


def _execution_payload(command: str, execution: HeadlessExecution) -> dict[str, Any]:
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "command": command,
        "status": execution.status,
        "job_id": execution.job_id,
        "result": result_payload(execution.result),
    }
    if execution.error:
        payload["error"] = execution.error
    return payload


def _emit_payload(payload: dict[str, Any], *, json_output: bool, human_text: str = "") -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif human_text:
        print(human_text)


def _add_output_flags(parser: argparse.ArgumentParser) -> None:
    # Do not let a subparser's default overwrite a global flag supplied before
    # the command (argparse otherwise turns ``--json lyrics ...`` back off).
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit one machine-readable JSON result",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Suppress progress messages",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slunder-cli",
        description="Headless Slunder Studio generation and export commands.",
    )
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable JSON result")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    commands = parser.add_subparsers(dest="command")

    lyrics = commands.add_parser("lyrics", help="Generate lyrics through the lyrics engine")
    _add_output_flags(lyrics)
    lyrics.add_argument("prompt")
    lyrics.add_argument("--mode", choices=("quick", "structured"), default="quick")
    lyrics.add_argument("--genre", default="pop")
    lyrics.add_argument("--mood", default="")
    lyrics.add_argument("--language", default="en")
    lyrics.add_argument("--structure", default="")
    lyrics.add_argument("--model-id", default=None)
    lyrics.add_argument("--output", type=Path, default=None)

    midi = commands.add_parser("midi", help="Generate a MIDI file through the MIDI engine")
    _add_output_flags(midi)
    midi.add_argument("prompt")
    midi.add_argument("--output", type=Path, required=True)
    midi.add_argument("--style", default="")
    midi.add_argument("--key", default="C major")
    midi.add_argument("--tempo", type=float, default=120.0)
    midi.add_argument("--duration-bars", type=int, default=16)
    midi.add_argument("--chord-progression", default="Auto")
    midi.add_argument("--drum-groove", default="Auto")
    midi.add_argument("--seed", type=int, default=None)
    midi.add_argument("--model-id", default=None)
    midi.add_argument("--demo", action="store_true", help="Explicitly allow algorithmic demo output")

    sfx = commands.add_parser("sfx", help="Generate one or more sound effects")
    _add_output_flags(sfx)
    sfx.add_argument("prompt")
    sfx.add_argument("--negative-prompt", default="")
    sfx.add_argument("--duration", type=float, default=5.0)
    sfx.add_argument("--cfg-scale", type=float, default=7.0)
    sfx.add_argument("--steps", type=int, default=100)
    sfx.add_argument("--batch", type=int, choices=range(1, 65), default=1)
    sfx.add_argument("--seed", type=int, default=None)
    sfx.add_argument("--output-dir", type=Path, default=None)
    sfx.add_argument("--demo", action="store_true", help="Explicitly allow algorithmic demo output")

    export = commands.add_parser("export", help="Export an audio file through the shared delivery path")
    _add_output_flags(export)
    export.add_argument("source", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--format", choices=("wav", "flac", "mp3", "ogg", "opus"), default=None)
    export.add_argument("--sample-rate", type=int, default=None)
    export.add_argument("--bit-depth", type=int, choices=(16, 24, 32), default=None)
    export.add_argument("--normalize", action="store_true")

    jobs = commands.add_parser("jobs", help="Inspect durable job records")
    _add_output_flags(jobs)
    jobs.add_argument("--status", choices=tuple(sorted({
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCEL_REQUESTED,
        JobStatus.CANCELLED,
        JobStatus.RECOVERABLE,
    })), default=None)
    jobs.add_argument("--kind", default=None)
    jobs.add_argument("--job-id", default=None)

    provenance = commands.add_parser(
        "provenance",
        help="Inspect an artifact's replayability capability",
    )
    _add_output_flags(provenance)
    provenance.add_argument("path", type=Path)

    return parser


def _run_jobs(args) -> tuple[dict[str, Any], int]:
    store = JobStore()
    if args.job_id:
        record = store.get(args.job_id)
        if record is None:
            raise CLIError(f"Job not found: {args.job_id}")
        records = [record]
    else:
        records = store.list_records(status=args.status, kind=args.kind)
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "command": "jobs",
        "status": JobStatus.COMPLETED,
        "jobs": [record.to_dict() for record in records],
    }
    return payload, EXIT_OK


def _run_provenance(args) -> tuple[dict[str, Any], int]:
    path = args.path.expanduser()
    record = read_provenance_sidecar(path)
    if not record:
        raise CLIError(f"No readable provenance sidecar was found for: {path}")
    capability = provenance_replayability(record)
    return {
        "schema_version": CLI_SCHEMA_VERSION,
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "command": "provenance",
        "status": JobStatus.COMPLETED,
        "artifact_path": str(path),
        "operation": {
            "module": str(record.get("module", "")),
            "name": str(record.get("operation", "")),
            "rerender_key": capability.key,
        },
        "replayability": capability.as_dict(),
    }, EXIT_OK


def _validate_args(args) -> None:
    if args.command == "lyrics" and not args.prompt.strip():
        raise CLIError("Lyrics prompt must not be empty")
    if args.command == "midi":
        if not args.prompt.strip():
            raise CLIError("MIDI prompt must not be empty")
        if args.duration_bars < 1 or args.tempo <= 0:
            raise CLIError("MIDI duration-bars must be positive and tempo must be greater than zero")
        args.output = args.output.expanduser()
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "sfx":
        if not args.prompt.strip():
            raise CLIError("SFX prompt must not be empty")
        if args.duration <= 0 or args.steps < 1:
            raise CLIError("SFX duration and steps must be positive")
        if args.output_dir:
            args.output_dir = args.output_dir.expanduser()
    if args.command == "export":
        args.source = args.source.expanduser()
        args.output = args.output.expanduser()
        if not args.source.is_file():
            raise CLIError(f"Source audio file does not exist: {args.source}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "provenance":
        args.path = args.path.expanduser()
        if not args.path.is_file():
            raise CLIError(f"Artifact or provenance sidecar does not exist: {args.path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    requested_json = "--json" in (list(argv) if argv is not None else sys.argv[1:])
    try:
        args = parser.parse_args(argv)
        if not args.command:
            parser.print_help(sys.stderr)
            return EXIT_USAGE
        _validate_args(args)
        json_output = bool(getattr(args, "json", requested_json))
        quiet = bool(args.quiet)

        if args.command == "jobs":
            payload, code = _run_jobs(args)
            _emit_payload(payload, json_output=json_output, human_text=f"{len(payload['jobs'])} job(s)")
            return code
        if args.command == "provenance":
            payload, code = _run_provenance(args)
            capability = payload["replayability"]
            _emit_payload(
                payload,
                json_output=json_output,
                human_text=f"{capability['state']}: {capability['reason']}",
            )
            return code

        runner = HeadlessRunner(json_output=json_output, quiet=quiet)
        if args.command == "lyrics":
            execution = runner.run(
                _build_lyrics_task(args),
                job_kind="lyrics_generation",
                job_label="Headless lyrics generation",
                job_inputs={"prompt_chars": len(args.prompt), "mode": args.mode},
                job_metadata={"module": "lyrics", "capability_id": CAP_LYRICS_GENERATE},
            )
            human = ""
            if execution.status == JobStatus.COMPLETED:
                source = execution.result.source_result if isinstance(execution.result, EngineRunResult) else {}
                text = str(source.get("lyrics", "") if isinstance(source, dict) else "")
                human = text if not args.output else f"Wrote lyrics: {args.output}"
            payload = _execution_payload("lyrics", execution)
        elif args.command == "midi":
            task, params = _build_midi_task(args)
            execution = runner.run(
                task,
                job_kind="midi_generation",
                job_label="Headless MIDI generation",
                job_inputs={"prompt_chars": len(args.prompt), "output": str(args.output)},
                job_metadata={"module": "midi_studio", "capability_id": CAP_MIDI_GENERATE},
            )
            human = f"Wrote MIDI: {args.output}" if execution.status == JobStatus.COMPLETED else ""
            payload = _execution_payload("midi", execution)
        elif args.command == "sfx":
            execution = runner.run(
                _build_sfx_task(args),
                job_kind="sfx_generation",
                job_label="Headless SFX generation",
                job_inputs={
                    "prompt_chars": len(args.prompt),
                    "duration": args.duration,
                    "batch": args.batch,
                    "demo": args.demo,
                },
                job_metadata={"module": "sfx", "capability_id": CAP_SFX_GENERATE},
            )
            human = f"Completed SFX job: {execution.job_id}" if execution.status == JobStatus.COMPLETED else ""
            payload = _execution_payload("sfx", execution)
        else:
            execution = runner.run(
                _build_export_task(args),
                job_kind="audio_export",
                job_label="Headless audio export",
                job_inputs={"source": str(args.source), "output": str(args.output)},
                job_metadata={"module": "audio_export", "operation": "cli_export"},
            )
            human = f"Wrote export: {args.output}" if execution.status == JobStatus.COMPLETED else ""
            payload = _execution_payload("export", execution)

        _emit_payload(payload, json_output=json_output, human_text=human)
        if payload["status"] == JobStatus.CANCELLED:
            return EXIT_CANCELLED
        return EXIT_OK if payload["status"] == JobStatus.COMPLETED else EXIT_FAILED
    except CLIError as exc:
        payload = {
            "schema_version": CLI_SCHEMA_VERSION,
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "status": "failed",
            "error": str(exc),
        }
        if requested_json:
            _emit_payload(payload, json_output=True)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("cancelled", file=sys.stderr)
        return EXIT_CANCELLED
    except Exception as exc:  # noqa: BLE001 - CLI boundary must return structured failure
        payload = {
            "schema_version": CLI_SCHEMA_VERSION,
            "app_name": APP_NAME,
            "app_version": APP_VERSION,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if requested_json:
            _emit_payload(payload, json_output=True)
        else:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
