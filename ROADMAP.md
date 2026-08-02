# Slunder Studio Roadmap

Incomplete work only for Slunder Studio, an offline local-first AI music creation suite. Priority mapping: P0 = Now, P1 = Next, P2 = Later, P3 = Under Consideration.

## Planned Features

### Export and integration

- [ ] Wire the existing `.dawproject` exporter/validator into Project Manager and Mixer, make archived media names collision-safe, validate against the official schemas, then validate golden packages in Bitwig, Studio One, and Cubase. The core ZIP/XML export already exists; do not reimplement it.
- [ ] Add stem export naming templates for user-selected DAWs.
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

### P2

- [ ] P2 — Run real MIDI generation as a cancellable background job
  Why: MIDI Studio synchronously calls only the demo generator and does not use the installed MIDI model.
  Evidence: `ui/midi_studio_view.py:421-439`; RVC/ACE-Step resource-aware inference patterns.
  Touches: MIDI Studio, `engines/midi_llm_engine.py`, shared engine/job contract, mute/solo render path, tests.
  Acceptance: A loaded model is used when selected; demo mode is explicit; generation is responsive/cancellable; mute/solo affects preview/export; fixed-seed fixtures are deterministic for the pinned runtime.
  Complexity: M
  Refined by the 2026-08-02 audit — the "mute/solo affects preview/export" clause is not merely
    missing, it is actively mis-reported today, and the render is synchronous:
    `ui/midi_studio_view.py:728-746` (`_on_render`) reads `muted = self._mixer.get_muted_tracks()`
    and `solo = self._mixer.get_solo_track()` and then never uses them — `render_midi_to_audio`
    (`engines/fluidsynth_engine.py:346-349`) has no mute/solo parameters, even though
    `FluidSynthEngine.render_to_numpy` already supports `mute_tracks`/`solo_track`
    (`engines/fluidsynth_engine.py:104-107`). `MidiMixer.mix_changed` (`ui/midi_mixer.py:295-331`)
    is connected nowhere in the repo, so the volume and pan sliders do literally nothing.
    The handler then reports `"Rendered: {output_path}"` as though the mix were honored.
    When wiring, also fix `get_solo_track` (`ui/midi_mixer.py:327-331`), which returns
    `min(self._soloed)` — soloing two tracks would silently solo only one.
    `_on_render` also calls `render_midi_to_audio` directly on the GUI thread, although that
    function's own docstring says "Called by InferenceWorker"; a full FluidSynth render of a
    multi-minute arrangement freezes the UI with no repaint.

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
  Refined by the 2026-08-02 audit — concrete inventory, so the implementer need not re-derive it.
    Settings keys written by a control with ZERO runtime consumers (grep across `ui/`, `core/`, `engines/`, `main.py`):
    `song_forge.timestep_shift`, `song_forge.inference_steps`, `song_forge.batch_count`,
    `song_forge.default_duration` (Song Forge uses its own in-view spinboxes, e.g. `_batch_spin`
    at `ui/song_forge_view.py:728`); `production.mastering_target`, `production.mastering_auto_eq`,
    `production.mastering_auto_compress` (Mixer uses its own combos); `midi_studio.default_bpm`
    (MIDI Studio hard-codes 120); `general.gpu_device`; `general.audio_format`;
    `general.experience_level`; `general.max_cache_gb` — whose help text at
    `ui/settings_view.py:381` ("Auto-cleanup old generations beyond this limit") describes a
    feature that does not exist. `general.output_dir` is read ONLY for path redaction in
    `core/diagnostics.py:231,420`; no export flow defaults to it.
    Controls at `ui/settings_view.py`: Song Forge group :294-328, Production/Mastering group
    :350-366, MIDI `default_bpm` :336-342, `general.gpu_device` :169-178,
    `general.audio_format` :145-150, `general.output_dir` :130-143, `general.max_cache_gb`
    :374-381, `general.experience_level` :208-217.
    Onboarding, exact defects: (a) `main.py:337-341` sets `general.onboarding_complete = True`
    unconditionally after `wizard.exec()`, so closing with X/Esc on page 1 permanently completes
    onboarding and the wizard's own `_finish` gating at `ui/onboarding.py:457-462` is moot —
    only mark complete on an accepted result; (b) `run_checks()` (`ui/onboarding.py:152-193`)
    appends rows to `_checks_layout` without clearing, so Welcome -> Next -> Back -> Next renders
    every system-check row twice, and again per round trip.
    Add a test asserting every key in the settings schema has at least one consumer.

## Audit Findings — 2026-08-02

Baseline recorded before auditing: `py -3.12 -m unittest discover -s tests` = **377 tests, OK**
(no pre-existing failures); `py -3.12 build/build.py` produces
`SlunderStudio-v0.1.31-win-x64.zip` and its packaged smoke check passes. Every finding below is
therefore new work, not a pre-existing red build.

### P0

### P1

### P2

### P3

- [ ] P3 — `RESEARCH.md` is git-tracked despite the all-markdown-except-README ignore rule
  Category: docs
  Where: `.gitignore:14-16`; `git ls-files` shows `README.md`, `RESEARCH.md`, `ROADMAP.md`
  Problem: `.gitignore` declares every `.md` except README local-only, but `RESEARCH.md` (19 KB of
    research notes) and `ROADMAP.md` were committed anyway, so the ignore rule has no effect on
    them — `git check-ignore` exits 1 because tracked files win. `CHANGELOG.md`, `CLAUDE.md`,
    `AGENTS.md` and `LOGO_PROMPTS.md` are correctly ignored. `RESEARCH.md` contradicts the stated
    policy; `ROADMAP.md` may be intentional given it is the task tracker.
  Fix: `git rm --cached RESEARCH.md`; then either `git rm --cached ROADMAP.md` or add
    `!ROADMAP.md` so the ignore file states the actual intent.
  Acceptance: `git ls-files "*.md"` matches what `.gitignore` says should be tracked.
  Confidence: Verified
  Effort: S

- [ ] P3 — Stale duplicated color constants that drift silently from the gated palette
  Category: maintainability
  Where: `ui/accessibility.py:28` (`FOCUS_RING_COLOR = "#f9e2af"` duplicates `Palette.YELLOW`);
    `ui/mood_curve_editor.py:210-211` (fill gradient `QColor(137, 180, 250, ...)` is the old
    Catppuccin blue `#89b4fa` while the curve line above uses `Palette.BLUE` `#a293ff` — two
    different blues in one graphic); `ui/piano_roll.py:93` (selected-note `QColor(255, 180, 60)`
    raw literal, and selection is indicated by this fill alone); `t.get('surface', '#161b22')`-style
    fallbacks throughout `ui/piano_roll.py`, `ui/midi_mixer.py` and `ui/midi_studio_view.py`
    (retired GitHub-dark defaults that are dead code, since `ThemeEngine.get_colors()` always
    supplies every key); `ui/theme.py:235` (`#ff9bad` danger-hover literal inside the theme itself)
  Problem: If a token is retuned — as OVERLAY0 and BLUE already were — these copies diverge and the
    contrast gate cannot see it, since it iterates Palette tokens only.
  Fix: Point all of them at `Palette.*`; replace `t.get(key, stale_default)` with `t[key]`.
  Acceptance: A test asserting `ui/*.py` contains no raw hex literal outside `ui/theme.py` and
    `ui/contrast.py`.
  Confidence: Verified
  Effort: S

- [ ] P3 — Stale comment invites restoring pip-at-runtime, which was deliberately removed
  Category: docs
  Where: `ui/waveform_widget.py:21-24`
  Problem: The comment says "Auto-install and retry", but `core.deps.ensure()` raises
    `MissingDependencyError` (`core/deps.py:110-126`). It misdocuments the deliberate v0.1.8
    hardening ("dependency paths now fail with explicit setup diagnostics instead of running pip")
    and invites a future contributor to "restore" the removed behavior.
  Fix: Rewrite the comment to state that missing dependencies fail closed with diagnostics.
  Acceptance: No comment in the repo describes runtime pip installation as current behavior.
  Confidence: Verified
  Effort: S

- [ ] P3 — Dead animation helpers, one with an invisible-but-clickable trap
  Category: maintainability
  Where: `ui/theme.py:735-785` (`fade_in`, `fade_out`, `slide_in_right`, `slide_out_right`)
  Problem: None of the four has a call site anywhere in the repo (the toast animations are
    self-contained). Beyond being dead, `fade_out` animates opacity to 0 but never hides the widget
    or removes the `QGraphicsOpacityEffect`, and a fully transparent widget still receives mouse
    events — so the first adoption produces an invisible clickable surface. There is also no
    app-wide reduced-motion setting, so adoption would make motion unconditional.
  Fix: Delete them; or fix `fade_out` to `hide()` on `finished` and gate all four behind a
    reduced-motion setting before first use.
  Acceptance: Either the helpers are gone, or they are covered by tests and respect a
    reduced-motion preference.
  Confidence: Verified
  Effort: S

- [ ] P3 — Piano-roll pitch labels and bar numbers are drawn outside the scene rect
  Category: visual
  Where: `ui/piano_roll.py:205, 224-227, 241-245`
  Problem: `setSceneRect(0, 0, total_width + KEY_WIDTH, total_height)` anchors the scene at (0, 0),
    but the C-note labels are placed at x = -42 and the bar numbers at y = -18 — outside the
    scrollable scene, so the pitch gutter and bar numbers the code clearly intends are clipped away.
    The `KEY_WIDTH` allowance was added to the wrong side.
  Fix: Extend the scene rect to include the negative gutter, e.g.
    `setSceneRect(-KEY_WIDTH, -20, total_width + KEY_WIDTH, total_height + 20)`.
  Acceptance: Pitch labels and bar numbers are visible in a rendered piano roll.
  Confidence: Likely — needs a visual check
  Effort: S

- [ ] P3 — Lyrics history offers a Favorites filter that nothing can ever populate
  Category: ux
  Where: `ui/lyrics_view.py:128-133, 163-173`
  Problem: The history pane offers a "★ Favorites" filter and renders stars for
    `entry.is_favorite`, but no UI anywhere sets a favorite — `LyricsDB.get_favorites` has no
    writer-side counterpart in any view. The filter can only ever show an empty list.
  Fix: Add a star toggle on history items (double-click or context menu) that persists
    `is_favorite`.
  Acceptance: A favorite can be set from the UI, survives a restart, and appears under the filter.
  Confidence: Verified
  Effort: S

- [ ] P3 — Remix export hardcodes 44100 Hz instead of the stem mixer's stored rate
  Category: correctness
  Where: `ui/vocal_suite_view.py:2154-2164` (`_on_remix_export`); `ui/stem_mixer.py:263-269`
  Problem: `StemMixer.load_stems` records the separation result's real sample rate, but the export
    writes `wf.setframerate(44100)` unconditionally and `_sample_rate` is never read back. All
    three currently offered Demucs models are 44.1 kHz so it happens to match today, but any future
    model or rate silently produces a pitch-shifted export. The full remix render and WAV write
    also run synchronously on the GUI thread.
  Fix: Expose `StemMixer.sample_rate` and use it in the writer; optionally move the write to a
    worker.
  Acceptance: A test asserting a 48 kHz stem set exports a 48 kHz WAV.
  Confidence: Verified (latent — not reachable with today's model list)
  Effort: S

- [ ] P3 — "demo" is hardcoded into Vocal Suite progress and cancel strings shown during real runs
  Category: ux
  Where: `ui/vocal_suite_view.py:1383` ("Cancelling RVC demo conversion..."), `:1575` ("Cancelling
    GPT-SoVITS demo..."), `:1622` ("Preparing consent-ready GPT-SoVITS demo...", unconditional),
    `:1641` (progress prefix "GPT-SoVITS demo... {pct}%"), `:2353` (tooltip "Produces declared
    outputs: {…}.")
  Problem: The hardening carefully separated demo from real runs — checkboxes, and `run.is_demo`
    aware completion messages at `:1515` and `:1858` — but the in-progress and cancelling strings
    still say "demo" unconditionally. A user running a real activated model is told they are
    running a demo, undermining exactly the truth-in-labelling the demo gates exist to provide.
    "Produces declared outputs:" is engine-contract jargon leaking into a user-facing tooltip.
  Fix: Choose the prefix from the readiness mode or checkbox state, mirroring the completion
    handlers; reword the tooltip in user terms (for example "Output: mixed vocal WAV").
  Acceptance: With a real model active, no progress or cancel string contains the word "demo".
  Confidence: Verified
  Effort: S

- [ ] P3 — Model Hub marks "core" models by title color alone
  Category: a11y
  Where: `ui/model_hub.py:195-198`
  Problem: `color: {Palette.BLUE if self.info.is_core else Palette.TEXT}` is the only signal that a
    model is core, with no badge, label or legend — colour as the sole means of conveying
    information (WCAG 1.4.1). The status badge next to it already demonstrates the right pattern.
  Fix: Add a small "Core" badge or a text qualifier alongside the color.
  Acceptance: Core status is discernible without perceiving color.
  Confidence: Verified
  Effort: S

### Unaudited — needs a pass

- [ ] P2 — Audit `core/` for correctness, security and resource safety
  Category: testing
  Where: `core/audio_engine.py`, `core/audio_buffers.py`, `core/audio_export.py`,
    `core/job_state.py`, `core/trash.py`, `core/provenance.py`, `core/voice_bank.py`,
    `core/midi_utils.py`, `core/chord_chart.py`, `core/dawproject.py`, `core/deps.py`,
    `core/dependency_profiles.py`, `core/i18n.py`, `core/diagnostics.py`, `core/settings.py`,
    `core/project.py`, `core/engine_contract.py`, `core/ace_step_contract.py`
  Problem: The 2026-08-02 audit dispatched a dedicated pass over these modules, and that pass
    terminated early on an API limit before producing findings. Everything above touching `core/`
    was found incidentally while tracing from `ui/` and `engines/` — so these modules have had no
    systematic review of parsing and schema drift, migration and persistence integrity, path
    traversal in archive handling (`core/dawproject.py`, `core/trash.py`), unsafe deserialization,
    resource cleanup on error paths, or thread safety beyond what is already logged.
  Fix: Run a dedicated read-only pass over the list above, tracing cross-module paths
    (settings to consumers, job_state to workers, trash to retention, export to provenance).
  Acceptance: Each module has either a logged finding or an explicit note that it was reviewed and
    found clean.
  Confidence: Verified (this is a known coverage gap, not a suspicion)
  Effort: M

- [ ] P3 — Confirm the visual findings on a real display at 125% scaling
  Category: visual
  Where: the P2 visual items above (wordmark clipping, group-title mnemonic, card description
    clipping, stem border hue)
  Problem: Those four were confirmed by offscreen Qt renders at 1:1 with Windows fonts loaded
    explicitly, because Qt's offscreen platform exposes no font database of its own (measured:
    `QFontDatabase.families()` returns 0 entries under `QT_QPA_PLATFORM=offscreen`). Font fallback
    for symbol glyphs therefore did not match a real desktop, and this machine runs 125% scaling
    that the offscreen renders did not exercise. The four findings are backed by code and metrics
    as well as pixels, so they stand — but layout-sensitive severity should be confirmed on the
    isolated virtual display before large layout changes are made.
  Fix: Launch the app on the isolated display and capture each affected surface at 125%.
  Acceptance: Each visual finding is either confirmed with a screenshot or downgraded with a note.
  Confidence: Verified (a stated limitation of the audit method, not a defect claim)
  Effort: S
