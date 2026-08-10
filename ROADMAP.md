# Slunder Studio Roadmap

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Actionable Items

### Export and integration

- [ ] P2 — Enforce inference admission across GUI and CLI processes
  Why: The current controller is process-local, while independent CLI runs can each acquire an inference slot and exceed the intended one-large-model GPU policy. The GUI single-instance lock does not cover headless automation.
  Evidence: `core/admission.py:68-205` uses in-process semaphores and a process-wide singleton; `slunder_cli.py:171-225` creates worker-backed jobs without a machine-wide lease; `main.py` protects only the desktop shell.
  Touches: admission controller, CLI startup/exit handling, diagnostics snapshots, and concurrency/crash-recovery tests.
  Acceptance: GUI and CLI jobs share a bounded machine-wide inference/download lease, stale leases recover safely, cancellation releases capacity, and two independent CLI processes cannot exceed configured limits or corrupt the durable job ledger.
  Complexity: M

- [ ] P2 — Extend the headless CLI through a capability matrix
  Why: The CLI already has a stable JSON/job/provenance contract, but only lyrics, MIDI, SFX, audio export, and job inspection are exposed. Song generation, stem separation, vocals, mixer operations, and project export remain inaccessible to automation even where the desktop engine is already stable.
  Evidence: `slunder_cli.py:626-696` defines the current command set; `core/engine_contract.py` and existing desktop workers already model song, vocal, stem, mixer, and artifact operations; `README.md:24-40` describes the CLI as sharing engine contracts.
  Touches: CLI parser/schema, shared engine adapters, job metadata, provenance payloads, README command matrix, and contract tests.
  Acceptance: A generated capability matrix marks every engine operation as supported or intentionally unavailable; at least the stable song-generation, stem-separation, and project/audio-delivery paths expose JSON job IDs, cancellation/recovery, typed artifacts, and provenance; unsupported operations fail with structured guidance rather than silently falling back.
  Complexity: L

- [ ] P3 — Make release tags and artifact metadata authoritative
  Why: The current source and README identify version 0.1.32, but the repository has tags only through v0.1.31. Reproducible unsigned artifacts are useful only if tag, manifest, changelog, executable name, SBOM, and checksums describe the same release.
  Evidence: `core/version.py`, `README.md:3`, `CHANGELOG.md`, `build/build.py:323-387`, and `git tag --list` in the 2026-08-08 inventory; `build/build.py` already emits SBOM/checksum artifacts and forbids signing.
  Touches: release validation/build metadata, README badge/changelog checks, tag/artifact smoke tests, and release documentation.
  Acceptance: A release check fails on any version/tag/badge/changelog/artifact mismatch; the chosen source of truth is documented; unsigned per-platform artifacts, SBOMs, and SHA-256 sums are generated from that source and the check runs before publication.
  Complexity: S
