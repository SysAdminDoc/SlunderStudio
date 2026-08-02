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

- [ ] P2 — Piano roll destructive edits have no undo
  Category: ux
  Where: `ui/piano_roll.py:296-305` (delete selected), `:697-735` (quantize / swing / humanize,
    which replace all notes), `:529-540` (Clear CC Lane)
  Problem: All of these mutate or replace `track.notes` / `cc_events` irreversibly; there is no
    undo stack anywhere in `ui/piano_roll.py`, `ui/midi_studio_view.py` or `ui/mixer_view.py`. One
    accidental Ctrl+A followed by Delete, or a mis-set swing amount, destroys the composition with
    no recovery short of re-importing — project versioning helps only if the user happened to
    snapshot. The mixer received a full non-destructive preview/apply/revert workflow in the
    hardening pass; the piano roll got nothing equivalent.
  Fix: Snapshot `track.notes` and `cc_events` (small dataclass lists) into a bounded undo deque
    before each destructive operation, and route delete, quantize, swing, humanize and clear
    through it.
  Acceptance: Extend `tests/test_piano_roll_editing.py` — after each destructive operation, undo
    restores the exact prior note and CC state.
  Confidence: Verified
  Effort: M

- [ ] P2 — Off-palette green action buttons, duplicated in seven files and styled by string replacement
  Category: maintainability
  Where: `#238636`/`#2ea043` blocks at `ui/mixer_view.py:438-443`,
    `ui/midi_studio_view.py:329-333`, `ui/stem_mixer.py:233-238`, `ui/vocal_suite_view.py:145-150`
    and `:1075-1080`, `ui/sfx_view.py:109`, `ui/project_manager.py:201`; checked colors
    `#da3633`/`#d29922` at `ui/mixer_view.py:158, 164`
  Problem: These are retired GitHub-dark colors rather than `Palette.GREEN` (`#72df9d`), so they
    bypass the contrast gate entirely — `failing_text_pairs()` iterates only Palette tokens, which
    is exactly how the failing hover and checked ratios above slipped past a hardened gate. The
    disabled state `color: #555` on `t['border']` measures 2.06:1: WCAG-exempt for disabled
    controls but effectively invisible when themed disabled tokens are available (OVERLAY0 on
    SURFACE0 exceeds 5.5). Worse, two sites style by string surgery —
    `ui/sfx_view.py:109` does `btn_style.replace(t['background'], '#238636').replace(t['text'],
    'white')` and `ui/project_manager.py:201` does `.replace(t['surface'] + ';', '#238636;')` —
    which silently no-op if the template string changes by one character. They already have a live
    side effect: the `:hover` rule is not replaced, so these green primary buttons flip to a dark
    background on hover, losing their color exactly when the user reaches for them.
  Fix: Add a `QPushButton[class="success"]` rule to `build_stylesheet()` using `Palette.GREEN` with
    `Palette.CRUST` text, matching the existing danger and secondary patterns; delete all seven
    inline blocks and both `.replace()` hacks. The contrast gate then covers them permanently.
  Acceptance: A test asserting no `ui/*.py` contains `#238636`, `#2ea043`, `#da3633` or `#d29922`,
    and that the success button style passes 4.5:1 in base, hover and disabled states.
  Confidence: Verified
  Effort: M

- [ ] P2 — The main status bar is permanently hidden but updated every two seconds
  Category: maintainability
  Where: `ui/main_window.py:390-400` (built, then `self._status_bar.hide()`), `:593-619`
    (`_update_gpu_status` writes `_gpu_status_label`, `_vram_label` and `showMessage`)
  Problem: `show()` is never called on the status bar anywhere in the repo, and
    `QStatusBar.showMessage` does not unhide it. The 2-second GPU timer keeps updating three
    widgets and a message the user can never see, including the VRAM percentage colour coding,
    "Active model: X" and "CUDA not available — running on CPU". The command bar duplicates part of
    this, but the active-model name and the VRAM percentage exist only in the dead status bar —
    and accessibility names are installed on invisible widgets (`:403-412`).
  Evidence: `grep -n "_status_bar" ui/main_window.py` shows construction, two `addPermanentWidget`
    calls, `hide()` at `:400`, and three `showMessage` calls — and no `show()`.
  Fix: Either show the status bar or delete it and move "Active model" and VRAM% into the command
    bar; stop updating hidden widgets either way.
  Acceptance: Either the status bar is visible and its content is reachable, or the widgets and
    their update code are gone; no timer writes to a hidden widget.
  Confidence: Verified
  Effort: S

- [ ] P2 — `_clean_pycache()` deletes every bytecode cache on every launch, including when frozen
  Category: perf
  Where: `main.py:106-117`
  Problem: Every startup walks the whole project tree and removes every `__pycache__`, so each
    launch re-parses and re-compiles roughly a hundred modules — precisely the cost `.pyc` caching
    exists to avoid, and Python's own mtime/hash invalidation already handles the stale-bytecode
    concern the comment cites. It also runs in the frozen build, where it is a pure wasted walk of
    the `_internal` tree, and if a virtualenv ever lives in the repo root the walk will delete
    site-packages caches for PySide6, librosa and numba too. Combined with Phase 1 eagerly
    importing all core packages (`main.py:35-42`, where the librosa import alone costs seconds via
    numba), cold start pays the maximum price every single run.
  Fix: Delete the function, or gate it behind an explicit `--clean-bytecode` flag and never run it
    when `_is_frozen()`.
  Acceptance: Startup no longer removes `__pycache__` directories; a second launch is measurably
    faster than the first.
  Confidence: Verified
  Effort: S

- [ ] P2 — The GPU status poll imports torch on the GUI thread and retries a failed import every two seconds
  Category: perf
  Where: `ui/main_window.py:586-591` (`_start_gpu_monitor` calls `_update_gpu_status()`
    synchronously from `__init__`); `core/model_manager.py:487-505` (`get_gpu_info` does
    `import torch` per call)
  Problem: With a torch profile installed, the first status update runs `import torch` — seconds of
    DLL loading — on the GUI thread before the window's first paint, a visible startup freeze.
    Without torch, Python does not cache the failed import, so the `sys.path` scan repeats every
    two seconds for the life of the process.
  Fix: Resolve torch availability once and cache the result (including a negative sentinel), and
    run the first poll from the timer rather than synchronously in the constructor; or move the
    probe to a worker thread.
  Acceptance: `import torch` happens at most once per process from this path, and not before the
    main window is shown.
  Confidence: Verified (code path); the freeze magnitude needs a repro with torch installed
  Effort: S

- [ ] P2 — `--onefile` builds can never pass the smoke gate
  Category: build
  Where: `build/build.py:315-333` (`_smoke_launch_windows`, `len(ids) != 1`), `:336-357`
    (`_smoke_launch_posix`, `count > 1`)
  Problem: A PyInstaller onefile executable runs as a bootloader parent plus a re-spawned child of
    the same executable — two live processes sharing one `ExecutablePath` for the app's whole
    lifetime. The smoke check demands exactly one, so `py build/build.py --onefile` (a supported
    flag, and the stated single-file packaging goal) fails the gate on every successful build,
    indistinguishable from a genuine fork bomb. Onedir passes only because it is single-process.
  Fix: In onefile mode expect exactly two processes (parent plus child), or match on process-tree
    shape rather than a raw count of matching executables.
  Acceptance: `py build/build.py --onefile` completes and its smoke check passes, while a real
    recursive spawn still fails it.
  Confidence: Likely (the code path is certain; the process count is documented PyInstaller onefile
    behavior) — confirm with one real `--onefile` run before changing the gate
  Effort: S

- [ ] P2 — `assets/templates/` ships in every build but nothing reads it
  Category: build
  Where: `build/build.py:59`, `SlunderStudio.spec:8`, `assets/templates/*.json` (34 tracked files)
  Problem: The only reference to `assets/templates` in the entire codebase is the build script's
    `--add-data` line. All genre, mood and structure data actually comes from hardcoded Python in
    `engines/lyrics_templates.py` (`GENRE_TEMPLATES`, `MOODS`, `STANDARD_STRUCTURES`). Editing the
    JSON silently does nothing, so `assets/templates/trap.json` and the `GENRE_TEMPLATES["trap"]`
    entry can diverge with no error, and the dead payload ships in every artifact.
  Evidence: `grep -r "templates"` across `core/`, `engines/`, `ui/` and `tests/` finds only
    `engines.lyrics_templates` imports; no loader ever constructs a path into `assets/templates`.
  Fix: Either delete the directory along with the `--add-data` and spec entries, or make
    `lyrics_templates.py` genuinely load from it and add a consistency test.
  Acceptance: Either the directory is gone from the repo and the bundle, or a test proves editing a
    template JSON changes generation behavior.
  Confidence: Verified
  Effort: S

- [ ] P2 — `SeparationResult` has no `is_success`, so failed separations are recorded as completed jobs
  Category: correctness
  Where: `engines/demucs_engine.py:30-44` (`SeparationResult`); consumed by
    `core/workers.py:113-141`
  Problem: `separate`/`separate_stems` funnel all exceptions into `SeparationResult(error=...)`.
    The worker's semantic-success check reads `result.is_success`, which `SFXResult`,
    `MidiGenResult`, `VoiceResult` and `ProducerResult` all implement — but `SeparationResult` does
    not, so `getattr(..., "is_success", None)` is None and a failed separation is marked
    `mark_completed` in the durable job ledger with zero outputs, surfacing as a successful job in
    history and health reports.
  Evidence: Run stem separation on a corrupt file — `torchaudio.load` raises, an error result is
    returned, and the job state becomes "completed".
  Fix: Add `@property def is_success(self): return self.error is None and bool(self.stems)`.
  Acceptance: A test asserting a separation that fails produces a failed job record.
  Confidence: Verified
  Effort: S

- [ ] P2 — FluidSynth's shared synth carries reverb tail and channel state into the next render
  Category: correctness
  Where: `engines/fluidsynth_engine.py:149-194` (`render_to_numpy`), singleton at `:347-354`
  Problem: Rendering stops sampling immediately after the last chunk. CC 123 stops notes, but the
    synth's reverb and chorus buffers and the note releases still hold energy, and the singleton is
    reused for the next render — so the opening samples of render B contain the decaying tail of
    render A (up to the roughly 1 s release window plus reverb time, audible at the default
    `reverb_room=0.6`). Program selections persist across renders too, mostly masked by the t=0
    program events. Latent secondary defect: the `channels == 1` branch (`:175-176`) reshapes
    FluidSynth's always-interleaved stereo output to `(2*n, 1)`, which would raise a broadcast
    `ValueError` at `:180` — unreachable today because no caller sets `channels=1`
    (`RenderSettings` is never customized in `ui/`), but it will break the first time mono render
    is wired up.
  Fix: Drain and discard a release window of samples after the event loop, or call
    `Synth.system_reset()` (or recreate the synth) per render; delete or fix the mono branch.
  Acceptance: A test rendering a dense passage then a silent one finds the second render's opening
    samples silent.
  Confidence: Likely (standard FluidSynth state behavior; not executed here)
  Effort: S

- [ ] P2 — Fixed pixel widths clip user data today and will clip harder once translated
  Category: ux
  Where: `ui/mixer_view.py:92-93` (track-name `QLabel`, `setFixedWidth(80)`, 11px bold, no elide
    and no tooltip); `ui/ai_producer_view.py:98-99` (stage status label, `setFixedWidth(60)`,
    10px); `ui/onboarding.py:183, 259, 324` (model/step names, fixed 110-140px);
    `ui/model_hub.py:38` (HF-token dialog `setFixedSize(480, 240)`, so validation and error text
    growth at `:91` cannot expand the dialog); `ui/stem_mixer.py:129, 148` and
    `ui/midi_mixer.py:110, 132` (value labels fixed 24-32px, where "L100" is already at the limit)
  Problem: These carry user or dynamic data — track names derived from filenames such as
    "vocals_recovered", stage status plus duration strings — and clip silently with no tooltip
    fallback. The existing i18n roadmap item mentions pseudolocale clipping generically; these are
    the concrete cases that clip in English today, independent of translation.
  Fix: Use `setMinimumWidth` plus `QFontMetrics.elidedText` with a tooltip carrying the full string
    for data labels; replace `setFixedSize` on the dialog with `setMinimumSize` and let the layout
    size it.
  Acceptance: A test asserting a long track name renders elided with a tooltip rather than clipped,
    and that the token dialog grows to fit an error message.
  Confidence: Verified (code); exact pixel repro Likely
  Effort: M

- [ ] P2 — Toasts do not reposition on resize, clip long paths, and use a duration independent of length
  Category: ux
  Where: `ui/toast.py:88, 212-226, 228-243`; `ui/main_window.py` (no `resizeEvent` override)
  Problem: `ToastManager._reposition()` runs only when a toast is added or removed, and the main
    window has no `resizeEvent` hook, so maximizing or restoring while a toast is visible leaves it
    floating mid-window or clipped off-canvas until it expires. `setFixedWidth(380)` with
    `msg_label.setMaximumWidth(320)` and word wrap cannot break the unbreakable file-path tokens
    that error toasts routinely carry, so those clip horizontally. Durations are constant per type
    (info 3 s through error 5 s) regardless of message length, with no hover-to-pause; the gated
    persistent history satisfies SC 2.2.1, but the live toast itself is still unreadable for long
    errors.
  Fix: Reposition from a `resizeEvent` override; scale the duration by message length (for example
    `max(base, 60ms * chars)`); pause the dismiss timer on `enterEvent`.
  Acceptance: A toast stays correctly anchored across a window resize; a long error message stays
    visible long enough to read and is not horizontally clipped.
  Confidence: Verified (resize), Likely (clipping)
  Effort: S

### P3

- [ ] P3 — `core/audio_engine.py` has no test coverage at all
  Category: testing
  Where: `core/audio_engine.py` (entire file); no file under `tests/` imports it
  Problem: Trash restore, dawproject export and job recovery on restart are all genuinely well
    covered (`tests/test_recoverable_trash.py`, `tests/test_dawproject_export.py`,
    `tests/test_job_state.py::test_stale_active_jobs_become_recoverable_on_startup`), but audio
    playback is not — `grep audio_engine tests/` returns nothing. Untested logic includes the
    stream `_callback` (loop wrap at `_loop_end`, end-of-buffer zero fill, `CallbackStop`, volume
    scaling, pause zeroing — `:192-227`), `seek` clamping, `set_loop` region math and
    `_generate_waveform`. The playback-finished race and the silent `load_file` failure logged
    above both live in this untested file.
  Fix: Add `tests/test_audio_engine.py`. Refactor `_callback` into a testable method (or call
    `play()` with `sounddevice` mocked and drive the captured callback with a preallocated
    `outdata`), then assert loop wrap-around, tail zero fill, `CallbackStop` at end, pause
    behavior, `seek` clamping, `format_time`, and `load_file` returning False without mutating
    previously loaded audio.
  Acceptance: The new test file passes and covers each behavior above.
  Confidence: Verified
  Effort: M

- [ ] P3 — Dead `AudioEngine.save_to_file` duplicates the real export path and needs an undeclared dependency
  Category: maintainability
  Where: `core/audio_engine.py:309-353`
  Problem: `save_to_file` has no callers anywhere — real exports go through `core/audio_export.py`,
    which does MP3 via ffmpeg with proper diagnostics. The dead path imports `pydub`, which is
    absent from `requirements.txt`, the lock file and `core/deps._PIP_NAMES`, and swallows every
    failure with a bare `print` that is invisible in the windowed exe — violating the explicit
    diagnostics design in `core/deps.require`. If anyone wires it up, MP3 export fails silently.
  Fix: Delete it and route any future caller to `core.audio_export`.
  Acceptance: The function is gone and the suite still passes.
  Confidence: Verified
  Effort: S

- [ ] P3 — `requirements-lock.txt` is not a lock, and it under-declares scipy
  Category: build
  Where: `requirements-lock.txt`; `core/mastering.py:301, 397, 453, 481`;
    `core/audio_buffers.py:118`
  Problem: The lock pins only the eight top-level packages, with no transitive pins and no hashes.
    `scipy` is imported directly by core DSP code (`lfilter`, `resample_poly`) yet appears in
    neither `requirements.txt` nor the lock — it works only because librosa pulls it in
    transitively, at whatever version pip resolves that day. So `pip install -r
    requirements-lock.txt` does not reproduce the environment: scipy, numba, llvmlite and numexpr
    all float. This is the same drift class the optional-AI profiles already solved with hashes.
  Fix: Declare `scipy>=1.11` explicitly in `requirements.txt`, and regenerate the lock from
    `pip freeze` (full transitive set) or `pip-compile --generate-hashes`.
  Acceptance: A fresh venv installed from the lock file runs the suite green, and a test asserts
    every third-party module imported by `core/` and `engines/` has a matching requirement.
  Confidence: Verified
  Effort: S

- [ ] P3 — Version drift the consistency test cannot see, and two tests that assert on source text
  Category: testing
  Where: `requirements-lock.txt:1` ("# Slunder Studio v0.1.30 — Pinned Dependencies") versus
    `core/version.py:14` (`0.1.31`); `tests/test_version_consistency.py:12-16`;
    `tests/test_version_consistency.py:24-26`; `tests/test_build_artifacts.py:24-26`
  Problem: The consistency suite scans only `*.py`, the README badge and the CHANGELOG, so
    non-Python files carrying version strings drift silently — the lock header is a live instance,
    already one release behind. Separately, `test_numpy_private_exception_module_is_bundled` and
    the build-artifact test assert on source-text substrings of `build.py`, so they pass even if
    the hidden-import list never reaches the PyInstaller command line.
  Fix: Extend the consistency test to assert any `Slunder Studio v\d+\.\d+\.\d+` literal in
    `requirements-lock.txt` equals `APP_VERSION` (or drop the version from that header); change the
    two build tests to assert against the constructed `cmd` list rather than the file's text.
  Acceptance: The extended tests fail on today's `v0.1.30` header and pass once it is corrected.
  Confidence: Verified
  Effort: S

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
