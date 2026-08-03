# Slunder Studio Roadmap

Incomplete work only for Slunder Studio, an offline local-first AI music creation suite. Priority mapping: P0 = Now, P1 = Next, P2 = Later, P3 = Under Consideration.

## Planned Features

### Export and integration

- [ ] Wire the existing `.dawproject` exporter/validator into Project Manager and Mixer, make archived media names collision-safe, validate against the official schemas, then validate golden packages in Bitwig, Studio One, and Cubase. The core ZIP/XML export already exists; do not reimplement it.
  Confirmed 2026-08-02: `core/dawproject.py` (`export_dawproject`, `validate_dawproject`) plus
    `tests/test_dawproject_export.py` are complete and have **zero UI entry points** — no button
    anywhere in `ui/`. `version="1.0"` at `core/dawproject.py:58` is correct; that is what
    implementers emit. Import targets for the golden test: Bitwig 5.0.9+, Studio One 6.5+ (Pro),
    Cubase 14, Cubasis 3.7.1, VST Live 2.2. No competing AI music tool exports DAWproject, so this
    is a genuine differentiator rather than parity work.
- [ ] Add stem export naming templates for user-selected DAWs.
- [ ] Add OSC control with a versioned namespace, loopback default, explicit LAN opt-in, allowlists, and rate/size limits.
- [ ] Add a headless CLI that uses the same engine, job, export, and error contracts as the desktop UI.
  Demand evidence 2026-08-02: Ultimate Vocal Remover has three separate open issues asking for
    exactly this (#359, #274, #654 — 57 combined upvotes) that have gone unshipped for years
    because its GUI-only design made retrofitting one impractical. Slunder's engine/job/export
    contracts already exist, so the CLI is a wrapper rather than a rewrite — but that stays true
    only while every code path keeps going through them.

### Model hub and operations

- [ ] Add model update checks with explicit release notes, immutable target revisions, health validation, and rollback to the last good revision.
  Refined 2026-08-02: two reference implementations worth copying rather than inventing. InvokeAI's
    `model_hash.py` uses BLAKE3 with **prefixed, self-describing digests** (`blake3:ce3f0c…`) and
    hashes a directory model by hashing the hashes of its files, so the algorithm is upgradeable on
    disk without a re-scan — Slunder's SHA-256 manifests should adopt the prefix convention now,
    while it is cheap. Ollama uses a content-addressed OCI-style blob store keyed by digest, which
    gives dedup, resumability and integrity in one structure; with ACE-Step XL at ~9 GB alongside
    Roformer and DiffSinger voicebanks, blob-level dedup across model variants is real disk savings.
    Rollback should snapshot the model set, the way ComfyUI-Manager does, not just the revision pin.
- [ ] Add tested 4-bit/8-bit model variants with measured quality, latency, disk, and VRAM tradeoffs.
- [ ] ~~Add DirectML backend support for AMD GPUs on Windows.~~ **Invalidated 2026-08-02 — do not
  build this.** `torch-directml` is stuck at 0.2.5.dev240914 requiring `torch==2.4.1`, last
  released 2024-09-15 and now roughly two years stale — well below the corrected security floor.
  The replacement is a `windows-rocm` profile: ROCm 7.2.1 components now ship PyTorch 2.9 on
  Windows 11 with Python 3.12 for gfx1100/1101/1200/1201 (RX 7900 XTX, 7700, 9070/XT, 9060 XT,
  AI PRO R9700, W7900), requiring driver 26.2.2. Still a preview, and torch 2.9 does not clear the
  corrected 2.10.0 floor — so either wait for a ROCm-Windows build at >=2.10 or gate the profile
  so it never `torch.load`s an untrusted checkpoint. Also correct the stale rationale string at
  `requirements/profiles.json:131`, which cites "PyTorch 2.3.1" rather than 2.4.1.
- [ ] Add MPS backend support for Apple Silicon.
  Refined 2026-08-02: torch 2.13 is the version that makes this worth doing — FlexAttention landed
    on MPS with hand-written Metal kernels (up to ~12.3x over SDPA on sparse patterns), plus a
    large migration of copy/cast, reductions, cumsum, sort and scatter/gather to native Metal that
    removes per-dispatch MPSGraph compile overhead. The `macos-mps` profile currently leaves all of
    that on the table. Sequence this after the numpy/torch upgrade train.
- [ ] Add central admission control for concurrent model downloads and inference. Resume and SHA-256 verification already exist; retain them instead of duplicating that work.

### Creative workflows

- [ ] Add Ableton Link sync for tempo-matched sessions after transport timing is testable.
- [ ] Add MIDI controller mapping for Mixer and Piano Roll after their keyboard/action contracts are complete.
- [ ] Expand persistent jobs into a batch-queue panel with retry, resume-on-restart, per-job resource estimates, and export selection.
- [ ] Add lyric rhyme/rhythm scoring as advisory feedback, never as an automatic rewrite.
- [ ] Add ACE-Step LoRA dataset validation/training only after the versioned ACE-Step 1.5 inference adapter is stable.
- [ ] Add ACE-Step Vocal-to-BGM, extract/complete, and non-destructive layer workflows through the shared source-conditioned task contract.
- [ ] Add LRC/synchronized-lyrics export from verified alignment data.
  Refined 2026-08-02: target **Enhanced LRC (A2)**, which carries word-level `<mm:ss.xx>` timestamps
    inside each line, not just line-level `.lrc`. Vocal Suite's Lyric Melody tab already produces
    note-aligned lyrics, and Song Forge's long-form mode already plans sections — that is exactly
    the source data, so this is a formatter over existing alignment rather than new analysis. Scope
    note: neither Spotify nor Apple Music accepts externally supplied LRC (Apple uses TTML
    internally), so the value is local playback, karaoke tooling and archival, not DSP delivery.
- [ ] Add a process-isolated engine plug-in SDK only after manifest signing is explicitly excluded, permissions are defined, and a bad extension cannot crash the shell.
  Evidence for keeping this gated, 2026-08-02: the ComfyUI registry shipped the Akira Stealer
    infostealer as `upscaler-4k` / `lonemilk-upscalernew-4k` / `ComfyUI-Upscaler-4K` — **779
    installs**, published Oct 2025, flagged 2026-01-05, re-uploaded under a new account within
    days, still live when reported 2026-01-10. It hid in `/scripts/autoscale.py` rather than the
    node file and used `subprocess` to fetch and run an external script at load. When this item is
    eventually built, adopt ComfyUI-Manager's shape: named security levels, `allow_pip_install`
    and `allow_git_url_install` defaulting to false and **not overridable by the security level**,
    risky operations restricted to loopback, snapshot/restore of the plugin+model set, and a
    remote disable list. Ship all of that before any community installation surface, not after.

## Research-Driven Additions

### P1

### P2

- [ ] P2 — Add a maintained separator adapter with honest model capabilities
  Why: Archived Demucs is the only real backend while current tools expose maintained MDX/MDXC/Roformer/ensemble options and model-specific limitations.
  Evidence: `engines/demucs_engine.py`; UVR, python-audio-separator, Ableton stem separation.
  Touches: separator interface, Vocal Suite, Model Hub, job/export/provenance paths, quality/resource presets.
  Acceptance: Demucs remains one adapter; at least one maintained backend is selectable; each model declares stems, license, device, RAM/VRAM, chunking, quality/speed, and known limitations; originals and per-run settings are preserved.
  Complexity: L
  Refined by the 2026-08-02 research — concrete backend choice and one added acceptance clause.
    `facebookresearch/demucs` is **archived** (last push 2024-04-24); the `adefossez/demucs` fork's
    README states plainly that no new features are coming. Recommended adapter:
    `nomadkaraoke/python-audio-separator` (MIT, releases roughly fortnightly) — it wraps MDX,
    MDXC/Roformer, VR and Demucs behind one `Separator` class with auto-download/cache and a JSON
    model registry queryable as `--list_models --list_filter=drums --format=json`, which is exactly
    the API a GUI should consume. `ZFTurbo/Music-Source-Separation-Training` (MIT) has the widest
    model coverage if more is needed later.
    **New mandatory acceptance clause — per-checkpoint license, not per-framework license.** Several
    best-in-class separation weights (Roformer/SCNet/Mel-Band variants) are published for research
    and non-commercial use while their *framework* is MIT, and python-audio-separator additionally
    carries a credit-to-UVR requirement for UVR-trained models. Without a per-checkpoint license
    field surfaced at run time and stamped into provenance, an MIT product silently ships
    non-commercial weights. Slunder already has model-level license metadata; this needs to be
    per-checkpoint.
    Model tiers worth exposing once the adapter exists: 6-stem, and drum-split (DrumSep Mel-Band
    Roformer separates kick/snare/toms/cymbals) — UVR has open requests for both (#973 drumsep,
    #241 chorus split, #276 piano) that nobody has packaged.

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

- [ ] P2 — Make first run end with a working installation
  Why: The wizard promises model setup and preferences, delivers a static table, and leaves the user
    with zero models and no idea where to get them.
  Evidence: `ui/onboarding.py` is four pages — Welcome, System Check, Model Guide, Quick Start. Its
    own docstring at `:3` promises a "model download prompt" and "preference setup"; `ModelGuidePage`
    (`:210-285`) is a read-only table with no button and no navigation to Model Hub, and no
    preference is collected despite Settings exposing output directory, experience level and default
    language. There is no HuggingFace token step even though gated models need one — the token
    prompt only appears at `ui/model_hub.py:48` *after* a download has already failed. There is no
    skip/close button (`:397` Back and `:409` Next only) and no way to re-open the wizard later.
    System Check has no remediation action: `:166` says "Run setup command from launch diagnostics"
    without showing the command. `:277-279` tells first-run users "generation will use built-in
    fallbacks", contradicting the shipped fail-closed demo behaviour and priming them to expect
    output that will actually be refused or hard-labelled DEMO.
  Touches: `ui/onboarding.py`, `main.py`, `ui/model_hub.py`, `ui/settings_view.py`, tests.
  Acceptance: Onboarding can start a core model download or hand off to Model Hub with the choice
    carried over; it collects output directory and any preference it claims to; a token step exists
    before gated downloads rather than after a failure; System Check offers a working action per
    failed row; an explicit Skip exists and is distinguishable from completion; the wizard can be
    reopened from Settings; the fallback copy matches actual behaviour.
  Complexity: M
  Companion to the existing settings/onboarding audit item above, which covers the
    complete-on-dismiss and duplicate-rows defects — do not fix those twice.

- [ ] P2 — Consolidate audio writing, mixdown and resampling onto one path
  Why: The same six lines are hand-rolled in eight places in a form that cannot express sample rate
    or bit depth, which is what produced the remix-export bug, and three different resamplers
    operate on one signal chain.
  Evidence: Identical `(audio * 32767).clip(...).astype(np.int16)` + `wave.open` +
    `setsampwidth(2)` writers at `engines/ai_producer.py:753`, `engines/demucs_engine.py:374`,
    `engines/diffsinger_engine.py:512`, `engines/fluidsynth_engine.py:236` and `:448`,
    `engines/rvc_engine.py:617` and `:908`, `engines/sfx_engine.py:420`,
    `ui/vocal_suite_view.py:2149` — while `core/audio_export.py` and `engines/ace_step_engine.py`
    correctly use `sf.write` with PCM_16/PCM_24/FLOAT subtypes. `AudioExportSettings.bit_depth`
    already supports 24 and 32 (`core/audio_export.py:51,367-374`), so the int16 writers silently
    discard the app's own capability. Mixdown-plus-peak-normalize is copied three times —
    `ui/stem_mixer.py:300-346`, `ui/mixer_view.py:1276-1325` (identical down to the shared comment)
    and `engines/ai_producer.py:849-859`. Four resampler backends: `scipy.signal.resample_poly`
    (`core/audio_buffers.py:118`), `librosa.resample` (`engines/ace_step_engine.py:182,551`),
    `torchaudio.functional.resample` (`engines/demucs_engine.py:231,323`), plus the limiter's own
    oversampling — three of them with different filter designs on the same generate → separate →
    mix → master chain, and only the `audio_buffers` path covered by tests.
  Touches: `core/audio_export.py`, `core/audio_buffers.py`, all engines, both mixers, tests.
  Acceptance: One writer used everywhere, honouring rate, channel count and bit depth; one mixdown
    helper; one resampler for the signal chain with any exception documented and justified; a test
    asserts no module outside the shared helpers constructs a `wave.open` writer.
  Complexity: M

- [ ] P2 — Route the remaining destructive actions through trash and Undo
  Why: The app already has the right pattern in three places; several equally destructive actions
    bypass it, and one file is internally inconsistent about it.
  Evidence: Trash plus an 8-second Undo toast is used correctly at `ui/project_manager.py:635-651`,
    `ui/model_hub.py:1051-1074` and `ui/sfx_view.py:675-711`. Bypassing it entirely:
    `ui/sfx_view.py:730-736` "Clear All" deletes every card without trash — in the same file whose
    single-card delete does use it; `ui/batch_view.py:365-373` Clear All and `:325` per-card delete;
    `ui/mixer_view.py:910-930` remove track, which also silently discards the track's
    `_dynamic_eq_suggestions`/`_dynamic_eq_originals`; `ui/settings_view.py:662-667` Reset to
    Defaults, which calls `reset_all()` with only a post-hoc warning toast; `ui/seed_explorer.py:309`
    Explore, which overwrites the whole grid of generated variations.
    Note the project's stated principle is immediate action with toast feedback and no confirmation
    dialogs — so the fix is trash plus Undo, matching the existing pattern, not a confirm prompt.
  Touches: `ui/sfx_view.py`, `ui/batch_view.py`, `ui/mixer_view.py`, `ui/settings_view.py`,
    `ui/seed_explorer.py`, `core/trash.py`, tests.
  Acceptance: Every action that destroys generated audio or user configuration is recoverable —
    via trash with an Undo affordance, or a restorable snapshot for settings — and a test enumerates
    destructive handlers and asserts each has a recovery path.
  Complexity: M

- [ ] P2 — Expose asset deletion in Project Manager
  Why: Users can import assets into a project and have no way to remove one.
  Evidence: `core/project.py:999-1024` `delete_asset` is trash-routed and has a matching
    `restore_deleted_asset` at `:1026`; **neither has any caller in `ui/`**. Project Manager's
    button set (`ui/project_manager.py:88-229`) offers Open, Delete project, Save, Save Version,
    Import Asset, Open Provenance and Restore Version — but no asset delete. Related: asset import
    at `:399-403` discards the return value of `mgr.import_asset(...)`, so a failed import produces
    no message and the list simply does not change; the call is also unguarded, so a raise reaches
    `sys.excepthook` and kills the app.
  Touches: `ui/project_manager.py`, tests.
  Acceptance: An asset can be removed from a project, lands in trash with Undo, and can be restored;
    a failed import reports why.
  Complexity: S

- [ ] P2 — Guarantee that any artifact can be re-rendered from its provenance
  Why: Non-determinism and silent post-update regressions are among the most repeated complaints
    about generative music tools, and cloud vendors structurally cannot fix it because they
    deprecate models — this app already stamps everything needed.
  Evidence: `core/provenance.py:171-240` records seed, model id/revision/hash, parameters, prompt,
    lyrics and source paths per artifact. Nothing consumes them to reproduce a render. Community
    reporting documents persona/vocal regressions after model updates and prompts being ignored
    between versions; Suno has publicly committed to deprecating the models users' catalogues were
    made with, which is precisely the failure a local tool can rule out.
  Touches: `core/provenance.py`, engine run paths, `ui/project_manager.py`, tests.
  Acceptance: A "Re-render from provenance" action reproduces an artifact bit-identically when the
    same model revision is present, and refuses with a precise diff (model revision, parameter,
    app version, runtime) when it is not; a test asserts a fixed-seed fixture reproduces exactly on
    the pinned runtime.
  Complexity: M

- [ ] P2 — Recommend models by task using published measurements
  Why: This is the single most-requested thing across the separation ecosystem and nobody ships it;
    users are currently asked to choose between checkpoint filenames.
  Evidence: Ultimate Vocal Remover's most-upvoted issue in its history is #344 (42 upvotes, 29
    comments) — a *user* wrote the "which model is best" documentation the project never shipped —
    paired with #333 asking for processing-method documentation. Slunder's Model Hub currently
    lists models without task guidance. Published SDR figures exist to ground it: a
    BS-Roformer ×2 + Mel-Band Roformer(ft) + SCNet XL ensemble reaches 11.93 SDR vocals / 18.23
    instrumental, with per-stem specialists (DrumSep Mel-Band Roformer kick 22.22, SCNet XL +
    BS-Roformer SW bass 14.87, guitar 9.05, piano 7.83). ACE-Step 1.5's own VRAM ladder is the model
    for generation-side guidance: 2B turbo DiT-only with INT8 and CPU offload at <=6 GB, up to XL
    plus a 4B LM at >=24 GB — which means the README's flat "NVIDIA 8GB+" hides a real tier ceiling.
  Touches: `core/model_manager.py` registry schema, `ui/model_hub.py`, `ui/onboarding.py`, README.
  Acceptance: Each model carries task labels ("best vocal isolation", "fastest", "lowest VRAM"),
    a measured basis with its source and date, and its VRAM tier; Model Hub can filter by task and
    by detected hardware; the recommendation shown for the detected GPU matches the tier that will
    actually run. Depends on the separator adapter item for the separation half.
  Complexity: M

- [ ] P2 — Fix per-word lyric pronunciation without re-rolling the take
  Why: It is the loudest unaddressed complaint across the generative-vocal ecosystem, and this app
    is unusually well placed to solve it because it already has a piano roll.
  Evidence: ACE-Step issues #391 "Dont follow lyrics!" (29 comments), #450 "Need a way to fix
    mispronounced lyrics when everything else is fine", #285 on lyric-audio alignment; GPT-SoVITS
    #1338 on repeated and dropped characters (20 upvotes). No open or closed tool ships a
    phoneme-level override plus regenerate-this-word-only workflow. Slunder already has
    `ui/piano_roll.py` with an undo stack, `engines/melody_extractor.py` for note alignment, and
    DiffSinger phoneme dictionaries (`engines/diffsinger_engine.py`).
  Touches: `ui/piano_roll.py`, `engines/diffsinger_engine.py`, `engines/melody_extractor.py`,
    Vocal Suite, provenance, tests.
  Acceptance: A word or syllable in a rendered vocal can be selected, its phonemes overridden, and
    only that region re-synthesized and crossfaded back, with the rest of the take bit-identical;
    the override is stored in the project and in provenance so the result is reproducible.
  Complexity: L

- [ ] P2 — Design empty and first-use states for every view
  Why: One view out of ten has one; the rest open blank, so a new user cannot tell an empty app from
    a broken one.
  Evidence: `ui/batch_view.py:259-263` is the only designed empty state and is the pattern to copy.
    Missing in Song Forge, Lyrics history (`ui/lyrics_view.py:176-194` renders an empty list plus
    "0 entries"), MIDI Studio (which only reports "Nothing to export" at `:685`/`:702` *after* the
    user acts), all six Vocal Suite tabs, Mixer (`ui/mixer_view.py:457-460` is a bare
    `addStretch()`), SFX results, Project Manager's project list, and Seed Explorer.
    `ui/model_hub.py:810-822` `_filter_cards` hides non-matching cards, so a search matching nothing
    yields a blank scroll area with no "no results" message.
  Touches: all views listed, `ui/widgets.py`.
  Acceptance: Every list, grid and results area has a state that names what belongs there and offers
    the action that fills it; every filterable list distinguishes "empty" from "no matches".
  Complexity: M

- [ ] P2 — Give long operations progress and a way to stop
  Why: Six of roughly ten long-running jobs report only a percentage in a status label, and the
    Mixer cannot be cancelled at all despite its workers supporting it.
  Evidence: Determinate `QProgressBar` exists in only four places — `ui/song_forge_view.py:408`,
    `ui/ai_producer_view.py:415`, `ui/lyrics_view.py:496`, `ui/model_hub.py:294`. Text-only percent
    elsewhere, most starkly `ui/vocal_suite_view.py`, which has 82 `_status.setText` calls and zero
    progress bars across DiffSinger, melody extraction, RVC, cloning, auto-tune and separation.
    `ui/mixer_view.py` workers cancel internally at `:654`, `:914`, `:1248` but expose **no Cancel
    button anywhere** — cancellation only happens as a side effect of removing a track or starting
    another operation. No cancel at all on any export, the MIDI render, the health-report ZIP
    (`ui/settings_view.py:775-794`) or recovery cleanup (`:522-538`).
    `ui/reference_panel.py:283-288` reconstructs its percentage by string-splitting its own label
    text, which will break on any copy change.
  Touches: `ui/vocal_suite_view.py`, `ui/mixer_view.py`, `ui/midi_studio_view.py`,
    `ui/settings_view.py`, `ui/reference_panel.py`, `ui/sfx_view.py`.
  Acceptance: Every operation that can exceed roughly a second shows determinate progress where the
    total is knowable and an explicit indeterminate state where it is not, and exposes Cancel that
    leaves consistent state; no progress value is derived by parsing display text.
  Complexity: M

- [ ] P2 — Harden the UI against display scaling
  Why: The app is developed on a 125% display and styles in device pixels, so text clipping is
    structural rather than incidental — which is why it keeps being fixed one widget at a time.
  Evidence: 261 fixed-size calls across 23 files. Stylesheet font sizes are declared in `px`
    throughout (e.g. `ui/settings_view.py:34,39`), which Qt does not scale with DPI the way `pt`
    does; no `devicePixelRatio` or high-DPI handling appears anywhere in `ui/`. Worst offenders:
    `ui/main_window.py:75` `setFixedWidth(260)` on the entire sidebar with no collapse mode;
    26 `setFixedWidth` calls in `ui/settings_view.py`, tightest at `:135` and `:171` (80 px);
    five `setFixedWidth(320)` control columns in `ui/vocal_suite_view.py`;
    `ui/song_forge_view.py:182` and `:423` together demand 830 px of minimums inside a 1024 px
    minimum window. `ui/main_window.py:336` sets `setMinimumSize(1024, 640)` under a comment
    claiming it must fit 1024x768 at 200% scaling — the value and the comment contradict each other.
    `ui/widgets.py`'s `ElidedLabel` is the only DPI-defensive pattern in the repo and is used in one
    file. Qt's own Windows scaling defects are documented (QTBUG high-DPI reports at 150%).
  Touches: `ui/theme.py`, all views, `ui/widgets.py`, `tests/test_accessibility_gates.py`.
  Acceptance: Font sizes are specified in scalable units; fixed widths are replaced by minimums plus
    layout policies or `ElidedLabel`; the existing no-view-exceeds-1024x768 gate is extended to run
    at a simulated 125% and 150%; the minimum-size comment and value agree.
  Complexity: L

- [ ] P2 — Close the keyboard-reachability gaps the accessibility gates miss
  Why: The gate suite is strong but tests contrast, focus-ring and width rather than whether a
    control can actually be reached, so a fully keyboard-capable widget is unreachable by Tab.
  Evidence: `PianoRollView` implements Ctrl+Z, Delete/Backspace and Ctrl+A at
    `ui/piano_roll.py:403-414` but **never calls `setFocusPolicy`** (`:352-384`), so it inherits
    `QGraphicsView`'s default `WheelFocus` and requires a mouse click before any of it works.
    `ui/stem_mixer.py` and `ui/midi_mixer.py` have no focus or key handling at all — no mute/solo
    from the keyboard. `ui/seed_explorer.py:55` explicitly sets the embedded `MiniWaveform` to
    `NoFocus`, so a seed cell's waveform cannot be seeked by keyboard even though the full
    `WaveformWidget` is the best keyboard citizen in the repo (`ui/waveform_widget.py:430-454`).
    Separately, the four `ui/theme.py` animation helpers were removed rather than gated, so there is
    still no reduced-motion preference to gate future motion behind, and the toast animations are
    unconditional.
  Touches: `ui/piano_roll.py`, `ui/stem_mixer.py`, `ui/midi_mixer.py`, `ui/seed_explorer.py`,
    `ui/toast.py`, `core/settings.py`, `tests/test_accessibility_gates.py`.
  Acceptance: Every widget that handles keys is reachable by Tab from application start; a gate
    asserts that any class defining `keyPressEvent` also sets a focus policy; a reduced-motion
    setting exists and toast animation respects it.
  Complexity: M
  Note on scope: this is keyboard *operability* (WCAG 2.1.1 / 2.4.3), not accelerator shortcuts.
    The project's stated preference is no keyboard shortcuts, and this item does not add any —
    it makes existing controls reachable. If a menu bar or accelerators are ever wanted, that is a
    separate decision that would need the no-shortcuts rule revisited explicitly.

- [ ] P2 — Fix the file-dialog experience across every import and export
  Why: Twelve open dialogs and eight save dialogs each reinvent their own behaviour, and the
    differences are all regressions.
  Evidence: `getOpenFileNames` is used **zero times** anywhere, so importing eight stems means eight
    dialogs. Every `getOpenFileName` passes `""` as the start directory, so the picker resets to the
    working directory each time even though `general.output_dir` exists. Filters disagree for the
    same job: `ui/vocal_suite_view.py:1547` (clone reference) omits `.ogg` that its four siblings
    accept. Save dialogs disagree about formats — `ui/mixer_view.py:1436` is the only one that
    builds its filter from `DELIVERY_FORMATS` and codec availability, while
    `ui/ai_producer_view.py:804`, `ui/vocal_suite_view.py:2144` and `:2180` offer WAV only.
    Drag-and-drop is odd too: `ui/main_window.py:637-652` accepts `.aiff`, which no file dialog
    does, and an audio drop calls `audio.play()` immediately (`:642`) rather than loading it into
    the active view.
  Touches: all views with file dialogs, `core/settings.py`, `core/audio_export.py`.
  Acceptance: One shared helper provides filters from the export/import format tables; multi-select
    works wherever multiple files are meaningful; the last-used directory per operation kind
    persists; accepted extensions match between dialogs and drag-and-drop; dropping audio loads it
    into the current view rather than starting playback.
  Complexity: M

- [ ] P2 — Rewrite user-facing copy that leaks internal vocabulary
  Why: Engine-contract and compliance internals are rendered verbatim in primary status lines across
    several views, not just the one tooltip already tracked.
  Evidence: "Produces declared outputs: {…}" appears in three views —
    `ui/ai_producer_view.py:722`, `ui/midi_studio_view.py:613`, `ui/sfx_view.py:634`.
    `readiness.remedy` is rendered raw as user-facing status or tooltip at
    `ui/ai_producer_view.py:549`, `ui/midi_studio_view.py:481`, `ui/model_hub.py:440` and `:506`.
    `ui/main_window.py:670` shows "Route cancelled: {exc}" and `:696,722,752` embed
    `artifact.context_summary()` — "route" and "artifact" are internal concepts for what the user
    experienced as pressing "Send to Mixer". `ui/ai_producer_view.py:658,669` surface "artifacts",
    "job" and "stage"; `:352` says "a silent placeholder is used". `ui/settings_view.py:699` prints
    `f"Config {state}"` with raw `migrated`/`repaired`/`error` tokens plus a full backup path, and
    `:498,516` render `policy.describe()` internals. `ui/model_hub.py:1058` tells users to
    "uninstall via pip". `ui/main_window.py:294-318` still carries a `PlaceholderPage` whose i18n
    key is literally `placeholder.coming_soon` and which exposes an internal "phase" number —
    the class is dead (all ten pages are real) so it and its key should simply go.
    Typography is inconsistent too: hyphen-as-dash in `ui/settings_view.py` vs em-dash elsewhere.
  Touches: all views, `core/i18n.py`, `assets/locales/en.json`.
  Acceptance: No user-facing string contains "declared outputs", "artifact", "route", "job",
    "stage", "placeholder", "pipeline", "adapter" or a raw state token; `readiness.remedy` is
    mapped to user-facing phrasing rather than rendered raw; dash usage is consistent.
    Coordinate with the localization item so strings are rewritten once, as keys.
  Complexity: M
  Supersedes nothing — the existing P3 item on Vocal Suite "demo" strings is a subset; fix together.

- [ ] P2 — Make the build reproducible and stop shipping UPX-packed binaries
  Why: The build cannot be reproduced or attributed, and it uses a packer that actively costs an
    unsigned application its only defence.
  Evidence: PyInstaller is not pinned anywhere — `build/build.py` merely imports it, so the builder
    version is unrecorded. Neither `SOURCE_DATE_EPOCH` (which PyInstaller honours for the PE header
    timestamp) nor `PYTHONHASHSEED` is set. A stale `SlunderStudio.spec` sits untracked in the repo
    root with `upx=True` at `:28` and `:43` while `build/build.py` generates its own command and
    never passes `--noupx` — PyInstaller uses UPX automatically when it is on PATH. UPX packing is a
    documented antivirus heuristic trigger, and since this project ships unsigned by policy there is
    no signature to offset it. `core/dependency_profiles.py:428` emits CycloneDX `specVersion 1.6`;
    1.7 shipped 2025-10-21 and became ECMA-424 2nd Edition, with the richer ML-BOM profile that
    EU AI Act auditing consumes. The SBOM also describes the wheelhouse, not the frozen bundle.
  Touches: `build/build.py`, `core/dependency_profiles.py`, `SlunderStudio.spec` (delete),
    `tests/test_build_artifacts.py`.
  Acceptance: PyInstaller is pinned in a build-requirements file; `SOURCE_DATE_EPOCH` and
    `PYTHONHASHSEED` are set; `--noupx` is passed; the stale spec file is deleted; the SBOM is
    CycloneDX 1.7 and covers what actually ships; two consecutive builds from the same commit
    produce matching hashes, or the remaining sources of nondeterminism are documented.
    Releases remain unsigned — no signing step is to be added.
  Complexity: M

- [ ] P2 — Remove the dead subsystems, or use them
  Why: Roughly forty unreferenced public functions and a whole unused pipeline framework make it
    impossible to tell scaffolding from intent, and one live subsystem reimplements a dead one.
  Evidence: `core/workers.py:335-470` `WorkflowStep` + `WorkflowQueue` — about 120 lines with seven
    signals — is never constructed anywhere including tests, while `engines/ai_producer.py`
    (literally lyrics → song → mastering) hand-rolls its own sequencing. `core/workers.py:472`
    `DebouncedCallback` is never used while `ui/vocal_suite_view.py:2292-2296` hand-rolls the same
    debounce with a raw QTimer. `engines/audio_analyzer.py:583` `ReferenceLibrary` and
    `ui/main_window.py:293` `PlaceholderPage` are never constructed. The entire settings-preset API
    (`core/settings.py:525-561`: `save_preset`, `load_preset`, `list_presets`, `delete_preset`) is
    unreachable. Seven of ten `ModelManager` signals are emitted with nothing listening
    (`model_loading`, `model_loaded`, `model_unloaded`, `model_error`, `download_started`,
    `download_completed`, `download_error`), and `download_progress` is declared but never emitted.
    Fully dead signals: `ui/batch_view.py:191` `regenerate_similar`, `ui/seed_explorer.py:232`
    `zoom_requested` (with its dead `zoom_into` at `:546`), `ui/waveform_widget.py:39`
    `region_selected`. Plus dead module functions in `core/midi_utils.py` (`export_tracks_separately`,
    `transpose_notes`, `scale_velocity`, `get_time_range`), `core/deps.py:146`,
    `core/mastering.py:898`, `engines/lyrics_templates.py:493,517`.
  Touches: `core/workers.py`, `core/settings.py`, `core/model_manager.py`, `core/midi_utils.py`,
    `engines/audio_analyzer.py`, `ui/main_window.py`, `ui/batch_view.py`, `ui/seed_explorer.py`,
    `ui/waveform_widget.py`.
  Acceptance: Each listed symbol is either deleted or given a caller and a test. Decide
    `WorkflowQueue` deliberately: adopt it in AI Producer or delete it — do not leave both. Note the
    repo has zero TODO/FIXME markers by convention, so unfinished work is invisible to grep; the
    three remaining RVC placeholder paths (`engines/rvc_engine.py:545-553`, `:871`, `:879-884`)
    should be tracked somewhere a tool can find them.
  Complexity: M

- [ ] P2 — Report unload failures instead of asserting success
  Why: A swallowed unload failure leaves VRAM held while the UI says the model is unloaded, so the
    next load fails with an out-of-memory error that has no visible cause.
  Evidence: `core/model_manager.py:951-958` `_release_model_object` swallows failures from
    `unload_model()` / `cleanup()` / `to("cpu")` with `except Exception: pass`; control returns to
    `unload()` which unconditionally sets `ModelStatus.DOWNLOADED` and emits `model_unloaded`
    (`:1164-1169`). Related defence-in-depth defect: `core/model_manager.py:1195-1197` passes
    `execution_consent=bool(info.requires_remote_code or info.allows_unsafe_weights)`, which is
    always `True` for exactly the models the engine-side guard at
    `engines/midi_llm_engine.py:651-656` exists to stop — so that check can never fire. It is not
    currently exploitable because the real gate is upstream at `:973` (`require_verified_model`),
    but a check that always passes is worse than no check.
  Touches: `core/model_manager.py`, `engines/midi_llm_engine.py`, `tests/test_model_trust.py`.
  Acceptance: A failed release leaves the model in an explicit error state, reports it, and does not
    emit `model_unloaded`; `execution_consent` reflects an actual user decision rather than a
    tautology, and a test asserts the guard can fail.
  Complexity: S

- [ ] P2 — Add the EBU R128 delivery preset and correct the loudness citation
  Why: The mastering chain is genuinely standards-conformant, and the one preset broadcast users
    expect is the one that is missing.
  Evidence: `LUFS_TARGETS` (`core/mastering.py:91`) covers streaming -14, YouTube -13, Apple -16,
    podcast -16, ATSC A/85 broadcast -24, cinema -27 and CD -9, but has **no EBU R128 -23 LUFS /
    -1 dBTP entry**. `core/mastering.py:253` cites "BS.1770-4 Annex 2" for the 4x oversampling
    requirement while BS.1770-5 (Nov 2023) is the current revision. Momentary (400 ms), short-term
    (3 s) and integrated readouts already exist but are not labelled with their EBU Mode names.
  Touches: `core/mastering.py`, `ui/mixer_view.py`, `ui/settings_view.py`, README,
    `tests/test_mastering_conformance.py`.
  Acceptance: An R128 target at -23 LUFS / -1 dBTP is selectable; the coefficient derivation and
    true-peak comments are re-verified against BS.1770-5 and cited as such; readouts carry EBU Mode
    labels.
  Complexity: S

- [ ] P2 — Fix project search, which matches the wrong field
  Why: The Project Manager search box does not reliably search project names.
  Evidence: `ui/project_manager.py:684-694` `_on_search` seeds `visible` from
    `query in card._project_id.lower()` — an internal ID, not the name — then iterates
    `findChildren(QLabel)` looking for a stylesheet containing `"bold"` and `break`s after the first
    label regardless of whether it matched, so name matching depends on child-widget ordering.
    Sorting is absent app-wide: no `setSortingEnabled` or `QSortFilterProxyModel` anywhere, so
    lyrics history is fixed to `get_recent()` order, projects appear in scan order and models in
    registry order.
  Touches: `ui/project_manager.py`, `ui/lyrics_view.py`, `ui/model_hub.py`.
  Acceptance: Search matches the project name and notes, not the internal ID, and does not depend on
    widget ordering; the primary lists offer at least name and date sorting.
  Complexity: S

### P3

- [ ] P3 — Reconsider the information architecture around assistive work
  Why: The evidence says the app leads with the capability its target users most reject and buries
    the ones they actively accept.
  Evidence: A Tracklib survey of 1,734 producers (published 2025-11-21) found 38% actively opposed
    to AI and only 17% in favour, **82% opposed to full-song generation from text prompts**, and
    just 6% regularly using fully generative tools — while stem separation, EQ and mastering are
    explicitly the accepted uses. iZotope's framing for the same audience is "intelligent tech that
    guides, not decides". Slunder currently presents Song Forge as the primary surface and the
    assistive tools as downstream. This is a positioning question, not a code defect, and it
    contradicts nothing in the project's philosophy — the local-first, provenance-first design is
    exactly what that audience says it wants.
  Touches: `ui/main_window.py` navigation order, README, onboarding copy.
  Acceptance: A deliberate decision is recorded either way. If adopted: assistive modules lead the
    navigation and the README, generation is presented as one capability rather than the headline,
    and the permanence/ownership guarantee (files on disk, MIT, no revocation, no metered retries)
    is stated explicitly — that guarantee is currently true and unadvertised.
  Complexity: S

- [ ] P3 — Verify model signatures, not just hashes
  Why: Hash manifests prove a file did not change in transit; they cannot prove it was ever
    trustworthy, and the registry this app downloads from was itself breached in 2026.
  Evidence: HuggingFace disclosed a July 2026 incident in which an attacker chained two RCEs in the
    dataset pipeline and leaked cloud and cluster credentials. HF's upload scanning (Picklescan,
    ProtectAI ModelScan, JFrog) has documented bypasses — 7z-instead-of-ZIP, corrupted central
    directories that scanners skip but PyTorch still loads, denylist evasion via subclassing — so
    "HF scanned it" is necessary and not sufficient. OpenSSF Model Signing (OMS) is now a published
    spec using Sigstore bundle format, supporting keyless, self-signed or raw keypairs, and covering
    safetensors, ONNX and GGUF; NVIDIA has signed every NGC-published model with it since March 2025.
    Slunder's existing hash manifests and offline boundary are the right foundation to add this to.
  Touches: `core/model_manager.py`, `tests/test_model_trust.py`, Model Hub trust UI.
  Acceptance: Where a model publishes an OMS signature it is verified before load and the result is
    shown in Model Hub and stamped into provenance; absence of a signature is reported as unsigned
    rather than as verified.
  Complexity: M

- [ ] P3 — Embed C2PA content credentials on export
  Why: Machine-readable provenance is converging on one standard, the app already produces the
    underlying data, and EU AI Act Article 50 transparency obligations begin in August 2026.
  Evidence: C2PA 2.4 (April 2026) adds a dedicated `c2pa.ai-disclosure` assertion for AI
    transparency, and 2.1 was ratified as ISO/IEC 22144; 2.3 added OGG Vorbis embedding.
    `core/provenance.py:171-240` already records the fields such an assertion needs. Watermarking
    (Meta AudioSeal, Google SynthID-Audio) is the complementary layer that survives re-encoding.
    Adoption claims for specific music DSPs came from secondary sources this pass and should be
    re-checked against platform documentation before this is scheduled.
  Touches: `core/provenance.py`, `core/audio_export.py`, a new C2PA dependency, tests.
  Acceptance: Exported audio optionally carries a C2PA manifest whose assertions match the sidecar;
    the feature is off by default and clearly explained; verification round-trips locally. Sequence
    after the DDEX disclosure export, which is the same data in a form users need sooner.
  Complexity: L

- [ ] P3 — Reduce the single-upstream concentration risk on ACE-Step
  Why: The suite's generative capability rests almost entirely on one project, and comparable
    projects have disappeared without warning this year.
  Evidence: ACE-Step 1.5 is the rare permissive option — MIT weights, active, editing-capable — but
    Tencent's SongGeneration/LeVo 2 returned HTTP 404 on GitHub and 401 on HuggingFace when checked
    on 2026-08-02, having been an open release earlier in the year; Riffusion's open repo has been
    dead since 2024 after the company went closed; YuE has not been pushed since 2025-06.
    Permissively-licensed alternatives that exist today: HeartMuLa (Apache-2.0) as a second
    generator, and SoulX-Singer (Apache-2.0, MIDI/melody-conditioned and reported CPU-viable) as a
    vocal engine alongside DiffSinger. Do **not** adopt AudioCraft/MusicGen (CC-BY-NC weights),
    fish-speech or SongPrep (NOASSERTION), or so-vits-svc and seed-vc (AGPL / archived) — an AGPL
    dependency is incompatible with this MIT product.
    Also note DiffSinger v3 is a config-system rewrite (OmegaConf + Pydantic, LoRA, acoustic
    inpainting, new tension/falsetto parameters) — `engines/diffsinger_engine.py` should be checked
    for hardcoded v2 config shape before that lands.
  Touches: `core/engine_contract.py`, `core/model_manager.py` registry, engine adapters.
  Acceptance: A second generator can be added as configuration rather than a rewrite; every model
    entry records a pinned local mirror so an upstream disappearance does not break existing
    projects; the registry refuses non-commercial and unlicensed weights unless explicitly accepted.
  Complexity: L

## Audit Findings — 2026-08-02

### P3

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
  Escalated by the 2026-08-02 audit — this is one symptom of a wider split, and it has a second,
    non-latent defect. `_on_remix_export` is **the only export path in the app that writes no
    provenance sidecar**, so the remix carries no license metadata and
    `get_export_license_warnings` on it always comes back clean regardless of the source model.
    That part is reachable today. See the P2 item on consolidating audio writing — the hardcoded
    44100 exists because this is one of eight hand-rolled int16 `wave.open` writers that cannot
    express a rate or bit depth, sitting alongside a correct `sf.write` path in
    `core/audio_export.py`.

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
  Problem: **Narrowed 2026-08-02 (second pass).** The pass that had terminated early was re-run and
    did complete. It covered exception handling, resource lifetime, dead code and duplication
    across all the modules listed, and its findings are now filed above as the P0/P1/P2 items on
    provenance fail-open, worker/thread teardown, unload reporting, audio-writer duplication and
    dead subsystems. Resource handling in particular was found clean: every `open` /
    `TemporaryDirectory` / `ZipFile` / `sqlite3.connect` site uses a context manager or has a
    matching close, and `core/project.py:979-995` has correct try/rollback semantics.
    What that pass did **not** cover, and what this item now means: **path traversal in archive
    handling** (`core/dawproject.py` ZIP construction and any future import, `core/trash.py`
    manifest-driven restore), **schema drift on unknown-field round-trips**, and **deserialization
    of attacker-influenced JSON** (job records, project files, provenance sidecars, trash
    manifests) — none of which were examined.
  Fix: A focused pass on archive extraction/restore paths and on schema round-tripping, with
    adversarial fixtures (`../` and absolute paths in archive entries, symlink entries, unknown
    fields, truncated and oversized files).
  Acceptance: Archive restore refuses any entry resolving outside its destination root; schema
    round-trips preserve unknown fields; malformed input fails with a diagnostic rather than a
    traceback. Each has a test.
  Confidence: Verified (a scoped remainder, not the original open-ended gap)
  Effort: S

- [ ] P3 — Confirm the visual findings on a real display at 125% scaling
  Category: visual
  Where: the P2 item "Harden the UI against display scaling", and any layout work that follows it.
  Note 2026-08-02: the four original visual findings this item was written against (wordmark
    clipping, group-title mnemonic, card description clipping, stem border hue) have since been
    fixed and removed from this roadmap, so this item now applies to the scaling work instead —
    the underlying method limitation below is unchanged and is why that work needs real-display
    confirmation rather than offscreen renders.
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
