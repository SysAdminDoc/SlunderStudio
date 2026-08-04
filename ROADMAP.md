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
