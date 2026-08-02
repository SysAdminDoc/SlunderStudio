# Changelog

All notable changes to SlunderStudio will be documented in this file.

## Unreleased - 2026-07-29

- Fixed CUDA VRAM diagnostics to use PyTorch's `total_memory` attribute, so GPU status polling and onboarding no longer crash on CUDA systems.
- Added a shared engine capability and activation contract so engines declare required models, run modes, and artifact kinds, and placeholder pipelines require an explicit labeled demo opt-in.
- Corrected the constant-power pan law in both mixers. The duplicated cosine formula left a centred track at 0 dB on both sides instead of -3.01 dB, so panning changed total level; both mixers now share one tested sine/cosine implementation where left^2 + right^2 is 1 at every position.
- Cancelling an SFX batch now keeps the variations that already finished and were verified on disk, removing only the in-flight partial, and shows the kept results with an explicit prompt to retry the rest. CancelledJobError gained preserved/result fields so any task can distinguish completed work from partials.
- Reference track analysis now runs as a cancellable background job with stage progress and a Cancel button, caches results by content hash plus analyzer version, discards results from a superseded selection, and raises on cancellation instead of returning a half-built analysis.
- DiffSinger F0 timing now derives from the loaded model's own sample rate and hop size instead of an unrelated approximation, reads dsconfig in YAML or JSON, and fails explicitly when the model config declares no usable frame timing.
- Cross-module routes now carry a typed artifact instead of switching pages: every route builds a payload with the file, its kind, tempo, key, lyrics and provenance, loads and selects it in the destination, registers it to the open project, and refuses with a visible error when the file is missing. The MIDI-to-Song-Forge and vocals-to-Song-Forge routes previously discarded the file entirely.
- Bounded every recovery artifact and added a Recovery Center in Settings: job records, per-job logs, crash logs, settings backups, and project versions each get age, count, and size limits with safe defaults, a dry-run preview that removes nothing, and protection for active, recoverable, current, newest, and pre-restore records.
- Centralized audio delivery: one export service writes WAV, FLAC, MP3, Ogg Vorbis and Opus, probes codec availability up front and says why a format is unavailable, maps BPM/key/language/rights/revision/ISRC onto the container's metadata standard, produces deterministic filenames, and reopens, hashes and records every successful write in the provenance sidecar. Mixer master export no longer writes PCM16 directly.
- Replaced the RMS loudness approximation with real ITU-R BS.1770-5 measurement: K-weighting derived per sample rate, gated 400 ms blocks, EBU Tech 3342 loudness range, and 4x-oversampled true peak in dBTP. Verified against the EBU Tech 3341/3342 sine cases.
- Mastering results now report measured integrated loudness, loudness range, true peak, and whether the declared target was met, instead of asserting it.
- Vectorized the mastering DSP: the biquad runs through scipy's lfilter and the compressor/limiter gain computation is array-based, so a three-minute 48 kHz stereo render finishes in seconds and is gated by a benchmark budget plus numerical-equivalence tests against the previous per-sample implementations.
- Made Mixer dynamic EQ non-destructive: Analyze, gain-matched Preview, Apply, and Revert are separate actions, and the pre-EQ audio for every affected track stays recoverable.
- Added enforced accessibility and responsive-layout gates: every palette text/surface pair is tested against WCAG 4.5:1 (two failing tokens were corrected), the focus ring is checked against every surface and no rule may hide it without a replacement, the waveform is keyboard-operable with announced position values, toasts are mirrored into a persistent status line and bounded history so timed messages have a non-timed equivalent, and the window minimum now fits a 1024x768 display.
- Consolidated every version string into `core/version.py`: the UI, settings/project schemas, provenance, README badge, artifact names, and the embedded Windows file version now all read one source, and a test fails the build if a module hardcodes its own.
- Removed all code-signing paths from the build. Artifacts ship unsigned, the build resolves the platform icon, names archives `SlunderStudio-vX.Y.Z-<platform>-<arch>.zip`, records SHA-256 hashes, and runs a packaged smoke launch on Windows and POSIX.
- Made the configured autosave interval real: a dirty project is saved and versioned on the timer, versions record whether they are manual, automatic, or pre-restore, any version can be previewed and restored after an automatic pre-restore snapshot, and retention prunes oldest autosaves first while never dropping a pre-restore version.
- Moved the HuggingFace token out of config JSON into the OS credential service (Windows Credential Manager, macOS Keychain, Linux Secret Service, or `keyring`), with a named backend in Settings, an explicit unavailable state instead of a plaintext fallback, and migration that scrubs the token from existing config backups once the store confirms the value.
- Guarded model lifecycle state with an explicit lock and request ticket: loads are serialized, a load superseded by a newer load or unload releases its model instead of overwriting the newer state, activation reports that as cancelled, and a second concurrent download of the same model is refused.
- Serialized all lyrics-history database work behind one operation lock with a bounded busy timeout, explicit transactions, retry on lock contention, atomic batch imports, and a clear closed/reopen state.

- Replaced the legacy ACE-Step integration with the immutable official ACE-Step 1.5 XL Turbo Diffusers adapter and corrected its MIT license, runtime bounds, model size, and capability metadata.
- Added source-preserving Cover, Repaint, and Extend execution with strict source validation, deterministic 48 kHz stereo conditioning, explicit turbo timestep-shift controls, and provenance for the exact adapter revision.
- Locked all optional AI profiles to an ACE-Step-compatible Transformers runtime and added offline clean-room import, engine-contract, migration, UI preflight, and packaged Windows smoke coverage.
- Made AI Producer stop on every required-stage failure, distinguish optional degradation and explicit demo output, and require a new readable owned master before enabling export.
- Added live persisted stage progress, cooperative cancellation with owned-output cleanup, failure-aware durable job records, stale-state revocation on rerun, and explicit Cancel/Retry controls.
- Added validated, duration-preserving polyphase resampling and explicit mono/stereo/surround normalization shared by Mixer, AI Producer, and audio export.
- Mixer now converts every imported track to its visible project sample rate before length calculation, summing, mastering, and export; mixed-rate tone/impulse fixtures verify pitch, alignment, duration, and deterministic output.
- Project imports now use stable asset-ID filenames and exclusive copies for audio and provenance sidecars, preserving same-named originals and rolling back partial imports.
- Project Manager now reconstructs missing, corrupt, incomplete, or externally redirected indexes from canonical project folders and restores readable project metadata backups through a visible Rescan action.
- Unified every waveform preview on one file/array loading contract with validated mono, stereo, and channel-first input, non-stale error states, and repeatable spectrogram transforms.
- Fixed generated MIDI, AI Producer, GPT-SoVITS, and routed Vocal Suite results so their successful audio outputs always reach waveform display.
- Added one typed capability/readiness/result contract across Model Hub, MIDI, SFX, AI Producer, and Vocal Suite, separating installed, verified, active, model, demo, failed, cancelled, and routable states.
- Model Hub can now activate, cancel activation, and deactivate verified local engines; MIDI/SFX/Vocal jobs run asynchronously with explicit demo opt-ins, visible remedies, typed artifacts, cancellation, and stale-route revocation.

## [v0.1.30] - 2026-07-01

### Security & Safety
- Thread-safe double-checked locking on all singletons (Settings, ProjectManager, LyricsDB, VoiceBank, AudioEngine) to prevent split-brain state from concurrent construction.
- VoiceBank now writes atomically (temp file + os.replace) and uses explicit UTF-8 encoding to prevent data corruption on Windows with non-ASCII voice profile names.
- DiffSinger phoneme token IDs now use deterministic MD5 hashing instead of Python's randomized `hash()`, fixing non-reproducible inference across runs.
- Sanitize ffmpeg metadata values in audio export to prevent special character injection.
- ACE-Step offline mode no longer permanently pollutes `HF_HUB_OFFLINE` env var; previous value is restored after pipeline construction.
- SFX demo fallback uses local RNG instances instead of polluting global `np.random` / `random` state.

### Correctness
- AI Producer pipeline now short-circuits after song generation failure instead of continuing to mix/master silence.
- ACE-Step `_find_output` fallback narrows exception handling (ImportError only, not all exceptions) and filters stale WAV files by timestamp to prevent returning audio from previous runs.
- AudioEngine playback callback now respects `loop_end` boundary and raises `CallbackStop` when playback finishes, freeing the audio device.
- Audio clip guard in `save_to_file` MP3 export: data is clamped to [-1.0, 1.0] before int16 conversion to prevent overflow wrapping.
- `trim_audio` now validates `start_sec < end_sec` and clamps both to the audio duration.
- Mastering compression/limiter guards against zero attack/release times (divide-by-zero).
- Project and asset IDs now include UUID suffixes to prevent millisecond-granularity collisions in rapid creation.
- JobStore lock is now per-instance instead of class-level, preventing unrelated job stores from serializing on the same lock.

### Visual Consistency
- Replaced 60+ hardcoded hex color values with Palette tokens across 9 UI files (batch_view, seed_explorer, waveform_widget, song_forge_view, piano_roll, reference_panel, mood_curve_editor, midi_mixer, onboarding).
- Eliminated GitHub Dark color system (#238636, #da3633, #d29922, #58a6ff) in favor of Catppuccin Mocha palette throughout.
- Fixed f-string brace mismatch in batch_view and seed_explorer `_toggle_star` that produced invalid QSS.
- Removed dead gradient code in onboarding wizard.

### UX
- AI Producer now runs on a background thread via InferenceWorker instead of blocking the GUI.
- Mixer export correctly writes FLAC format when selected (was silently writing WAV with .flac extension).

### Testing
- Added 7 regression tests covering thread-safe singletons, unique ID generation, UTF-8 voice bank persistence, trim validation, pipeline short-circuit, and JobStore lock isolation.

## [v0.1.29] - 2026-06-29

- Added Mid/Side mastering gain trims in the DSP chain after stereo width processing.
- Surfaced Mixer Mid and Side dB controls and pass them into a per-run mastering preset without mutating shared presets.
- Added regression tests for Mid/Side side-energy changes and Mixer preset handoff.

## [v0.1.28] - 2026-06-29

- Added a shared delivery LUFS target catalog covering streaming, YouTube, Apple Music, podcast stereo, broadcast, cinema dialog, and loud CD masters.
- Added a Mixer target selector with custom LUFS support and widened target range for cinema/dialog delivery.
- Reused the shared LUFS target catalog in Settings and added regression coverage for Mixer and Settings target availability.

## [v0.1.27] - 2026-06-29

- Added short-term LUFS profile measurement and reference-track loudness matching in the mastering DSP layer.
- Added Mixer reference loading that sets the target LUFS from the reference and reports short-term loudness deltas after mastering.
- Fixed Mixer mastered-waveform preview to use the existing `WaveformWidget.set_audio` API and added regression tests for reference matching.

## [v0.1.26] - 2026-06-29

- Added stem-aware dynamic EQ suggestions in Mixer with deterministic spectral analysis for vocals, bass, drums, instruments, and generic stems.
- Added a Mixer action that applies suggested EQ bands to imported tracks, refreshes track waveforms, and reports the first moves in the status line.
- Added regression tests for stem-specific suggestions, DSP application, Mixer integration, and suggestion reindexing after track removal.

## [v0.1.25] - 2026-06-29

- Added piano roll toolbar actions for quantize, swing timing, and velocity humanization.
- Added MIDI CC automation lanes for mod wheel, volume, pan, expression, and sustain events.
- Added MIDI save/load support for control-change events and regression tests for transforms, CC round-tripping, and piano roll UI actions.

## [v0.1.24] - 2026-06-29

- Added MIDI chord chart export to `.chordpro` and `.crd` formats with optional pasted lyrics.
- Added bar-level chord inference from non-drum MIDI tracks for printable chord sheets.
- Added regression tests for chord inference, ChordPro/CRD formatting, file output, and MIDI Studio export flow.

## [v0.1.23] - 2026-06-29

- Added MIDI Studio drum groove templates for straight rock, hip-hop half-time, trap hats, swing shuffle, four-on-the-floor, and Latin pop.
- Added fallback GM drum-track generation with template swing timing, snare ghost notes, velocity humanization, and deterministic seeded output.
- Added regression tests for groove selection, prompt conditioning, generated drum timing/velocities, fallback inclusion, and MIDI Studio parameter handoff.

## [v0.1.22] - 2026-06-29

- Added MIDI Studio chord-progression priors for text-to-MIDI generation, including `I-V-vi-IV`, `ii-V-I`, blues, and minor-key presets.
- Added Roman-numeral progression parsing to the MIDI-LLM prompt builder and corrected fallback triad spelling so scale degrees such as `vi` stay diatonic.
- Added regression tests for progression parsing, prompt conditioning, generated chord roots, and MIDI Studio parameter handoff.

## [v0.1.21] - 2026-06-29

- Added non-fatal Song Forge vocal-stem recovery after song generation using Demucs when available.
- Added provenance-tracked vocals-only stem exports, job-state output tracking, and a Song Forge "Send Vocal Stem" route to Vocal Suite.
- Added regression tests for recovery provenance, generation metadata, job cleanup path extraction, UI routing, and accessibility coverage.

## [v0.1.20] - 2026-06-29

- Added a Vocal Suite Lyric Melody tab that accepts humming audio, pasted lyrics, and BPM to generate a monophonic MIDI melody.
- Added librosa pYIN-based melody extraction with lyric-to-note alignment, MIDI provenance sidecars, async job-state tracking, and optional DiffSinger rendering when a model is loaded.
- Added regression tests for pitch-frame note extraction, lyric alignment, MIDI/sidecar output, UI handoff, and accessibility coverage.

## [v0.1.19] - 2026-06-29

- Added a Vocal Suite Auto-Tune tab with vocal file input, adjustable correction strength, async processing, waveform preview, and routing to Song Forge/Mixer.
- Added librosa pYIN-based pitch correction that pulls voiced frames toward the nearest semitone and writes provenance-tracked WAV outputs.
- Added regression tests for pitch-correction math, WAV/sidecar generation, and Auto-Tune accessibility coverage.

## [v0.1.18] - 2026-06-29

- Added an English locale catalog and shared i18n helpers for major app chrome, Settings, Lyrics, and Vocal Suite controls.
- Added Settings > Appearance > Default Lyrics Language and wired it into Quick lyrics prompts, Guided lyrics metadata, and new GPT-SoVITS voice profile language defaults.
- Packaged locale assets in PyInstaller builds and added regression tests for catalog coverage and language propagation.

## [v0.1.17] - 2026-06-29

- Added build cleanup for stale `dist/`, one-file, one-folder, ZIP, checksum, and generated spec artifacts.
- Added release ZIP packaging and `dist/SHA256SUMS.txt` generation for local distributables.
- Added packaged-app smoke launch verification that fails on missing, crashed, or recursively spawned processes.
- Added optional Authenticode signing via `SLUNDER_SIGN_CERT_SHA1` or `SLUNDER_SIGN_CERT_FILE` when `signtool` is available.

## [v0.1.16] - 2026-06-29

- Added a redacted Settings health-report export ZIP with JSON and text summaries.
- Included app/dependency versions, GPU and ffmpeg status, model cache state, settings repair status, crash log metadata, and recent failed jobs.
- Kept HuggingFace tokens redacted and omitted job prompts/lyrics unless private job inputs are explicitly included.
- Added diagnostics and Settings export regression tests.

## [v0.1.15] - 2026-06-29

- Added commercial-use, gated-access, license URL, and export-warning metadata to the built-in model registry.
- Surfaced model license, access, and commercial-use status on Model Hub cards.
- Added model license/commercial-use policy to generation provenance and project asset provenance summaries.
- Carried source model license warnings into exported audio sidecars and Song Forge export toast warnings.

## [v0.1.14] - 2026-06-29

- Added voice profile owner, consent source, language, permitted-use, and consent timestamp metadata.
- Blocked RVC conversion and GPT-SoVITS clone profiles when required consent metadata is missing.
- Displayed voice profile consent status in Vocal Suite before clone/convert jobs.
- Embedded voice profile consent provenance in RVC/GPT-SoVITS output sidecars.

## [v0.1.13] - 2026-06-29

- Added a durable JSON job ledger for generation and model-download state.
- Made cancelled generation jobs clean partial audio/provenance files and surface recoverable restart records.
- Changed Model Hub partial downloads to resume in place instead of trashing incomplete cache files first.
- Retained worker thread references until cancellation completes to avoid unsafe QThread teardown.

## [v0.1.12] - 2026-06-29

- Added a shared PySide6 accessibility helper for accessible names, descriptions, focus rings, and tab order.
- Applied the baseline to main navigation, transport controls, Song Forge, Vocal Suite, Model Hub, and Settings.
- Strengthened global focus selectors for buttons, inputs, combo boxes, sliders, tabs, and checkboxes.
- Added headless tests for major view accessibility properties and primary tab order.

## [v0.1.11] - 2026-06-29

- Added numbered settings and project JSON schemas with migration from legacy files.
- Added timestamped backups before settings/project saves and before migration or repair.
- Surfaced settings and project repair status in the GUI when corrupt JSON is encountered.
- Added regression tests for old and corrupt settings/project files.

## [v0.1.10] - 2026-06-29

- Added generation provenance sidecars for song renders, SFX, stems, vocals, MIDI, MIDI renders, AI Producer mixes/masters, and audio exports.
- Project imports now copy adjacent provenance sidecars and store compact provenance summaries on project assets.
- Added an in-app Project Manager action to open asset provenance JSON records.
- Added regression tests for sidecar fields, project import metadata, and SFX demo provenance.

## [v0.1.9] - 2026-06-28

- Added an app trash/quarantine service with manifests, restore, and retention cleanup.
- Moved project, model-cache, and generated SFX deletes to recoverable trash operations.
- Added toast Undo actions for project, model, and generated SFX deletes.
- Added regression tests for trash restore, retention cleanup, failed-delete reporting, and model/project recovery.

## [v0.1.8] - 2026-06-28

- Replaced runtime dependency installation with explicit startup diagnostics and setup commands.
- Converted optional engine dependency helpers to fail with actionable install guidance instead of running pip.
- Changed the PyInstaller build preflight to require PyInstaller explicitly rather than installing it during build.

## [v0.1.7] - 2026-06-28

- Added model download manifests with source, revision, license, trust state, and per-file SHA256 hashes.
- Added download verification that detects missing or tampered model files.
- Added voice-profile provenance, trust metadata, and file hashes.
- Blocked unsafe local PyTorch checkpoint formats unless a voice profile is explicitly trusted.
- Surfaced model revision and trust status in Model Hub cards.

## [v0.1.6] - 2026-06-28

- Blocked unloaded SFX, RVC, and GPT-SoVITS placeholder paths from reporting successful model output.
- Added explicit demo-output metadata and opt-in demo synthesis for SFX fallback generation.
- Added regression tests for non-routable placeholder and demo-output contracts.

## [v0.1.5] - 2026-06-28

- Added GPT-SoVITS reference-sample guardrails for 10-30 second voice cloning onboarding.
- Added Vocal Suite controls for saving validated reference samples as reusable GPT-SoVITS voice profiles.
- Added asynchronous clone dispatch when a GPT-SoVITS base model is loaded.
- Added unit tests for voice-clone reference quality checks and guardrail rejection.

## [v0.1.4] - 2026-06-28

- Added weighted genre fusion presets for Song Forge using existing genre template style tags.
- Added Song Forge controls for primary genre, secondary genre, blend weight, and applying fused tags.
- Added unit tests for genre fusion tag ordering.

## [v0.1.3] - 2026-06-28

- Added audio-CLAP-style reference conditioning that maps reference-track fingerprints to ACE-Step style tags.
- Surfaced CLAP conditioning tags in the Reference Track panel.
- Added unit tests for reference tag export and CLAP-style matching.

## [v0.1.2] - 2026-06-28

- Wired Seed Explorer grid generation to ACE-Step with explicit seed and CFG values per cell.
- Added a Seed Explorer distance slider synchronized with the seed range control.
- Updated Song Forge to load generated seed-cell results back into the waveform and report per-cell failures.

## [v0.1.1] - 2026-06-28

- Added ACE-Step long-form generation with section parsing, per-section rendering, and crossfaded WAV stitching for songs over 2 minutes.
- Added Song Forge advanced-mode control for long-form stitching and batch propagation.
- Added PyInstaller multiprocessing freeze guards for the packaged GUI build.
- Added focused unit tests for long-form section planning and audio stitching.

## [v0.1.0] - %Y->- (HEAD -> main, tag: v0.1.0, origin/main, origin/HEAD)

- up
- up
- v1
