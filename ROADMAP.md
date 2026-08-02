# Slunder Studio Roadmap

Incomplete work only for Slunder Studio, an offline local-first AI music creation suite. Priority mapping: P0 = Now, P1 = Next, P2 = Later, P3 = Under Consideration.

## Planned Features

### Export and integration

- [ ] Wire the existing `.dawproject` exporter/validator into Project Manager and Mixer, make archived media names collision-safe, validate against the official schemas, then validate golden packages in Bitwig, Studio One, and Cubase. The core ZIP/XML export already exists; do not reimplement it.
- [ ] Add stem export naming templates for user-selected DAWs.
- [ ] Add project-level MP3/FLAC/Opus delivery with standards-mapped metadata and provenance.
- [ ] Add OSC control with a versioned namespace, loopback default, explicit LAN opt-in, allowlists, and rate/size limits.
- [ ] Add a headless CLI that uses the same engine, job, export, and error contracts as the desktop UI.

### Model hub and operations

- [ ] Add model update checks with explicit release notes, immutable target revisions, health validation, and rollback to the last good revision.
- [ ] Add tested 4-bit/8-bit model variants with measured quality, latency, disk, and VRAM tradeoffs.
- [ ] Add DirectML backend support for AMD GPUs on Windows.
- [ ] Add MPS backend support for Apple Silicon.
- [ ] Add central admission control for concurrent model downloads and inference. Resume and SHA-256 verification already exist; retain them instead of duplicating that work.

### Creative workflows

- [ ] Add Ableton Link sync for tempo-matched sessions after transport timing is testable.
- [ ] Add MIDI controller mapping for Mixer and Piano Roll after their keyboard/action contracts are complete.
- [ ] Expand persistent jobs into a batch-queue panel with retry, resume-on-restart, per-job resource estimates, and export selection.
- [ ] Add lyric rhyme/rhythm scoring as advisory feedback, never as an automatic rewrite.
- [ ] Add ACE-Step LoRA dataset validation/training only after the versioned ACE-Step 1.5 inference adapter is stable.
- [ ] Add ACE-Step Vocal-to-BGM, extract/complete, and non-destructive layer workflows through the shared source-conditioned task contract.
- [ ] Add LRC/synchronized-lyrics export from verified alignment data.
- [ ] Add a process-isolated engine plug-in SDK only after manifest signing is explicitly excluded, permissions are defined, and a bad extension cannot crash the shell.

## Research-Driven Additions

### P1

- [ ] P1 — Make mastering standards-conformant and non-destructive
  Why: RMS-based loudness and sample-peak limiting are labeled as delivery presets, while Dynamic EQ mutates source arrays without preview or undo.
  Evidence: `core/mastering.py`, `ui/mixer_view.py:576-631`; ITU-R BS.1770-5, EBU R128, LANDR, and Ozone.
  Touches: `core/mastering.py`, Mixer mastering state, shared export, reference fixtures, undo/version integration.
  Acceptance: Integrated loudness, LRA, and oversampled dBTP pass published conformance vectors; Analyze/Suggest, gain-matched Preview, Apply, and Revert are separate; originals remain recoverable and exported reports state measured values.
  Complexity: L

- [ ] P1 — Vectorize and benchmark mastering DSP
  Why: Biquad, compressor, and limiter loops scale per sample in Python and block long renders even though current short benchmarks do not justify the prior “unusable for minutes” claim.
  Evidence: `core/mastering.py:286-352`; 2026-07-29 local 48 kHz stereo benchmark.
  Touches: `core/mastering.py`, background mastering worker, numerical-equivalence and performance tests.
  Acceptance: Three-minute stereo fixtures complete within an explicit benchmark budget on the reference CPU, remain off the GUI thread, preserve channel independence/state, and match approved numerical tolerances.
  Complexity: M

- [ ] P1 — Centralize delivery formats, metadata, and provenance
  Why: Mixer writes PCM16 directly while other surfaces use `core/audio_export.py`, creating inconsistent formats, validation, rights warnings, and sidecars.
  Evidence: `ui/mixer_view.py:676-748`, `ui/stem_mixer.py`, `core/audio_export.py`; ID3v2.4, Vorbis comments, RFC 7845, BWF, Suno and Logic exports.
  Touches: `core/audio_export.py`, Mixer/Stem Mixer/project export, metadata schema, codec probes, provenance, round-trip tests.
  Acceptance: Full mix, selected range, clips, stems, and MIDI use one export service; codec availability is probed; filenames are deterministic; BPM/key/language/lyrics/rights/revision map correctly; every successful write is reopened, hashed, and paired with provenance.
  Complexity: L

- [ ] P1 — Bound recovery artifacts and expose one recovery center
  Why: Jobs, crash logs, settings backups, and project versions can grow indefinitely and recovery actions are fragmented.
  Evidence: `core/job_state.py`, `core/settings.py`, `core/project.py`, log/crash paths; Ableton recovery and YuE-UI saved sessions.
  Touches: retention settings, job/log/version stores, recovery UI, diagnostics, cleanup tests.
  Acceptance: Age/count/size policies have safe defaults and dry-run previews; active/recoverable records are never pruned; users can inspect, retry, resume, discard, restore, or reveal artifacts from one screen with redacted failure details.
  Complexity: M

- [ ] P1 — Make cross-module routing transfer real artifacts and context
  Why: Some routes only switch pages and show a toast while discarding the selected file or MIDI context.
  Evidence: `ui/main_window.py:486-501`; current SFX/Vocal-to-Mixer routes; Hacker News demand for granular DAW-style handoff.
  Touches: `ui/main_window.py`, route payload types, Song Forge, MIDI Studio, Vocal Suite, Mixer, project asset registration.
  Acceptance: Each advertised route transfers a typed artifact plus tempo/key/lyrics/provenance, selects it in the destination, registers it to the active project when requested, and has an end-to-end test.
  Complexity: M

### P2

- [ ] P2 — Run SFX batches as cancellable background jobs
  Why: Stable Audio inference runs in a synchronous UI-thread loop for the full batch.
  Evidence: `ui/sfx_view.py:353-415`; commercial and OSS persistent-job patterns.
  Touches: `ui/sfx_view.py`, SFX engine cancellation, shared workers/jobs, progress and partial-result UI.
  Acceptance: The window remains responsive; progress identifies the active variation; cancellation waits for worker termination, preserves verified completed results, removes only owned partials, and supports retry.
  Complexity: M

- [ ] P2 — Run real MIDI generation as a cancellable background job
  Why: MIDI Studio synchronously calls only the demo generator and does not use the installed MIDI model.
  Evidence: `ui/midi_studio_view.py:421-439`; RVC/ACE-Step resource-aware inference patterns.
  Touches: MIDI Studio, `engines/midi_llm_engine.py`, shared engine/job contract, mute/solo render path, tests.
  Acceptance: A loaded model is used when selected; demo mode is explicit; generation is responsive/cancellable; mute/solo affects preview/export; fixed-seed fixtures are deterministic for the pinned runtime.
  Complexity: M

- [ ] P2 — Run reference analysis as a cancellable background job
  Why: Librosa analysis runs synchronously and freezes the GUI on large files.
  Evidence: `ui/reference_panel.py:191-210`; PySide thread-affinity guidance.
  Touches: `ui/reference_panel.py`, `engines/audio_analyzer.py`, shared workers, cache and cancellation tests.
  Acceptance: Large-file analysis keeps the UI responsive, reports stage progress, cancels cleanly, caches by content hash plus analyzer version, and never applies a stale result to a newer selection.
  Complexity: S

- [ ] P2 — Use the DiffSinger model's actual frame timing
  Why: F0 frame-to-time mapping uses an approximation unrelated to the active model hop size.
  Evidence: `engines/diffsinger_engine.py:255-280`; DiffSinger model/config contract.
  Touches: `engines/diffsinger_engine.py`, model profile metadata, alignment fixtures.
  Acceptance: Timing derives from the loaded model sample rate/hop configuration; known pitch events align within one frame across supported profiles and invalid metadata fails explicitly.
  Complexity: S

- [ ] P2 — Complete UI localization with pseudolocale and RTL gates
  Why: The catalog/helper exist, but most views remain hard-coded English and there is no runtime locale control.
  Evidence: `core/i18n.py`, `assets/locales/en.json`, limited `tr()` call sites; ACE-Step/RVC multilingual UIs and Qt translation support.
  Touches: all `ui/*`, i18n extraction/completeness tooling, Settings locale control, layouts, tests and translator docs.
  Acceptance: All user-visible strings are keys; locale changes persist and apply on restart; missing keys fail tests; pseudolocale finds clipping; one RTL locale passes mirrored-layout and keyboard smoke tests.
  Complexity: L

- [ ] P2 — Add a maintained separator adapter with honest model capabilities
  Why: Archived Demucs is the only real backend while current tools expose maintained MDX/MDXC/Roformer/ensemble options and model-specific limitations.
  Evidence: `engines/demucs_engine.py`; UVR, python-audio-separator, Ableton stem separation.
  Touches: separator interface, Vocal Suite, Model Hub, job/export/provenance paths, quality/resource presets.
  Acceptance: Demucs remains one adapter; at least one maintained backend is selectable; each model declares stems, license, device, RAM/VRAM, chunking, quality/speed, and known limitations; originals and per-run settings are preserved.
  Complexity: L

- [ ] P2 — Build a reproducible engine evaluation harness
  Why: Engine claims lack fixed prompts, model revisions, hardware measurements, failure rates, and human-review baselines.
  Evidence: ACE-Step/DiffRhythm papers; VERSA, MAD/MusicPrefs, and MusicEval; current absence of real-inference tests.
  Touches: test fixtures, benchmark runner, provenance/report schema, release checklist, optional metrics dependencies.
  Acceptance: Fixed prompts/seeds/durations/languages record latency, peak RAM/VRAM, failure, adherence, lyric timing, structure, loudness/true peak, and artifacts; reports include model/runtime hashes and a blinded listener rubric; no release is gated on FAD alone.
  Complexity: L

- [ ] P2 — Audit settings and onboarding against actual runtime behavior
  Why: Several controls claim immediate effect without consumers, onboarding completes even when dismissed, and readiness checks can report false-green on exceptions.
  Evidence: `ui/settings_view.py`, `core/settings.py`, `main.py:337-342`, onboarding UI; local-model onboarding patterns.
  Touches: settings schema/consumers, onboarding dialog and readiness probes, first-run state, copy, tests.
  Acceptance: Every visible setting has a tested consumer or is removed; dismissing onboarding does not mark completion; readiness distinguishes installed/downloaded/loadable/loaded/offline/error; disk/VRAM estimates and reopen steps match the selected engine.
  Complexity: M

### P3

- [ ] P3 — Correct constant-power pan in both mixers
  Why: The duplicated cosine formula does not implement the documented constant-power law.
  Evidence: `ui/mixer_view.py:660-663`, `ui/stem_mixer.py:320-324`.
  Touches: shared pan utility, Mixer, Stem Mixer, audio fixtures.
  Acceptance: Center produces equal −3 dB gains, endpoints fully attenuate the opposite channel, mono energy remains constant within tolerance, and both mixers use the same tested implementation.
  Complexity: S
