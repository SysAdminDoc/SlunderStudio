# Slunder Studio

![Version](https://img.shields.io/badge/version-0.1.31-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/Python-3.11--3.12-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![Status](https://img.shields.io/badge/status-active-success)

> Offline AI music generation suite. Generate songs, compose MIDI, synthesize vocals, separate stems, create SFX, and master tracks — all locally on your machine.

![Screenshot](screenshot.png)

## Quick Start

```bash
git clone https://github.com/SysAdminDoc/SlunderStudio.git
cd SlunderStudio
py -3.12 -m pip install -r requirements.txt
py -3.12 main.py
```

Python 3.11 or 3.12 is required. Install core dependencies explicitly before launch; if anything is missing, Slunder Studio opens a diagnostics screen with the exact setup command. AI models are downloaded on-demand from HuggingFace via the built-in Model Hub.

The reproducible engine evaluation harness uses fixed prompts, seeds, durations, and
languages. It records runtime/model provenance, latency, RAM/VRAM, failures, audio
loudness/true peak, artifact hashes, and a separate blinded listener rubric. Run the
report-only fixture pass with:

```bash
py -3.12 tools/evaluate_engines.py --output evaluation-report.json --artifact-dir evaluation-artifacts
```

Pass a project runner using `--runner module:function` for real engine measurements;
the default runner deliberately marks cases as skipped and never gates a release on FAD.

Optional AI runtimes use platform-specific, SHA-256-locked profiles for CPython
3.12. Prepare a wheelhouse while connected, then the installation command
operates with `--no-index` and writes a complete CycloneDX SBOM:

```bash
py -3.12 tools/dependency_profiles.py download windows-cpu --wheelhouse C:\SlunderWheelhouse\windows-cpu
py -3.12 tools/dependency_profiles.py install windows-cpu --wheelhouse C:\SlunderWheelhouse\windows-cpu
py -3.12 tools/dependency_profiles.py smoke windows-cpu
```

Available locks cover Windows/Linux CPU and CUDA 12.6 plus Apple Silicon MPS.
Torch-DirectML is intentionally disabled: Microsoft documents support only
through PyTorch 2.4.1, below the PyTorch 2.10.0 security floor.

## Features

| Module | Description | AI Engine |
|--------|-------------|-----------|
| Song Forge | ACE-Step 1.5 XL Turbo song generation, reference covers, source repaint/extend, stitched long-form songs, and recovered vocal-stem export | ACE-Step, Demucs |
| Lyrics Engine | AI-powered lyrics writing with 33 genre templates | Llama 3.1 8B |
| MIDI Studio | Piano roll editor with quantize/swing/humanize tools, CC lanes, text-to-MIDI composition, groove-template drums, and chord chart export | MIDI-LLM |
| Vocal Suite | Singing synthesis, humming-to-MIDI lyric melody generation, voice conversion, voice cloning, selectable Demucs/MDX/MDXC/Roformer stem separation, and vocal auto-tune pitch correction | DiffSinger, RVC v2, GPT-SoVITS, Audio Separator |
| Stem Separation | Isolate vocals, drums, bass, and other instruments | Demucs (htdemucs) |
| SFX Generator | Text-to-sound-effect generation | Stable Audio Open |
| Mixer | Project-rate mixing with validated resampling, channel normalization, dynamic EQ, Mid/Side trims, reference matching, and mastering | Built-in DSP |
| AI Producer | Cancellable staged production with verified output, explicit demo/degraded states, and retry | Orchestrator |
| Model Hub | Download, verify, activate, deactivate, and switch AI models | HuggingFace Hub |
| Projects | Collision-safe asset imports, recoverable library index, version history, and provenance tracking | — |

## How It Works

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  AI Producer │───>│ Lyrics Engine│───>│  Song Forge  │───>│  MIDI Studio │
│  (One Prompt)│    │  (33 genres) │    │  (ACE-Step)  │    │  (Piano Roll)│
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐
│   Export     │<───│    Mixer     │<───│ SFX Generator│    │ Vocal Suite  │
│  (WAV/FLAC) │    │ (Mastering)  │    │(Stable Audio)│    │(DiffSinger)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

Every module can route audio to any other module. Generate a song in Song Forge, separate stems in Vocal Suite, add SFX, mix everything in the Mixer, and export a mastered track. Engine actions expose their required model/profile and declared output types before they run; Model Hub keeps installed, verified, active, and failed states separate, while algorithmic MIDI/SFX and placeholder voice pipelines require an explicit, visibly labeled demo opt-in. Generated and exported assets write adjacent `.provenance.json` sidecars with app version, prompt/lyrics, seed, model revision/hash metadata, source paths, and render parameters so projects can be audited or reproduced. Project imports store every asset and sidecar under its stable asset ID, preserving same-named originals; Project Manager can rebuild a damaged or incomplete index from valid project folders and timestamped metadata backups. Long-running generation and model-download jobs persist queued/running/completed/failed/cancelled/recoverable state so interrupted sessions can show what needs recovery on restart. Settings and project files use versioned schemas with timestamped backups before migrations, repairs, and saves. Primary creative workflows expose screen-reader names, descriptions, high-contrast focus rings, and predictable tab traversal. Accessibility is gated by tests rather than asserted: every palette text/surface pair must clear WCAG 4.5:1, the focus ring must clear 3:1 on every surface and no stylesheet rule may hide it without a replacement, the waveform is fully keyboard-operable (arrows seek, Page Up/Down scrub, Home/End jump, M switches view) and announces its position as a value, notifications are mirrored into a persistent status line so timed toasts have a non-timed equivalent, and no view may demand more width than a 1024x768 display. For songs over 2 minutes, Song Forge can render structured sections separately and stitch them with crossfades for more stable long-form arrangements; completed Song Forge renders also attempt a Demucs vocal-stem recovery and expose a separate vocals-only route when recovery succeeds. Seed Explorer renders nearby seed/timestep-shift variations from the current lyrics and style prompt so you can compare takes before committing to a full arrangement. Reference Track analysis maps an audio fingerprint to ACE-Step tags for one-click style conditioning, Genre Fusion blends two template tag sets into weighted hybrid prompts, and Voice Cloning validates 10-30s GPT-SoVITS reference samples before saving reusable voice profiles with owner, consent source, language, permitted-use, and sidecar provenance metadata.

The open project autosaves on the interval configured in Settings > General, but only when it actually has unsaved changes. Each autosave writes a versioned snapshot alongside manual ones, and Project Manager can preview any stored version's name, tempo, key, assets, and mixer track count before restoring it. Restoring first snapshots the current state as a pre-restore version, so a restore can be undone. Retention is bounded by Settings > Kept Project Versions: the oldest autosaves are pruned first, manual versions next, and pre-restore versions are never pruned.

Settings > Recovery Center lists every recovery artifact - job records, per-job logs, crash logs, settings backups, and project versions - with its size and retention policy. Preview Cleanup always runs as a dry run and removes nothing; Clean Now applies the age, count, and size limits. Active jobs, recoverable jobs, the current crash log, the newest project version, and pre-restore snapshots are never removed.

Secrets such as the HuggingFace access token are stored in the operating-system credential service — Windows Credential Manager, macOS Keychain, or the Linux Secret Service — and never in the config JSON, its timestamped backups, logs, or diagnostics. Settings names the backend in use, and states plainly when no credential service is available instead of falling back to plaintext. A token found in an older plaintext config is moved into the credential service and removed from the config and its existing backups once the store confirms it can be read back.

Settings can export a redacted health report ZIP with app/dependency versions, GPU and ffmpeg status, model cache state, settings repair status, crash log metadata, and recent failed jobs. HuggingFace tokens are always redacted, and job prompts/lyrics stay out of the report unless the private-input opt-in is enabled.

The app chrome, Settings, Lyrics, and Vocal Suite controls use a catalog-backed locale layer. Settings > Appearance > Interface Language supports English, Arabic with right-to-left layout, and a pseudo-locale for layout QA; the selected locale is restored on restart. Settings > Appearance > Default Lyrics Language feeds Quick lyrics prompts, Guided lyrics metadata, and new GPT-SoVITS voice profile language defaults where supported. MIDI Studio supports explicit chord-progression priors such as `I-V-vi-IV` and `ii-V-I` for text-to-MIDI prompts and fallback generation. MIDI Studio also includes selectable drum groove templates with swing timing, snare ghost notes, and velocity humanization for generated GM drum tracks, `.chordpro` and `.crd` chord chart export with optional pasted lyrics, and piano roll editing tools for quantize, swing, velocity humanize, and MIDI CC automation lanes. Vocal Suite includes a Lyric Melody tab that converts hummed audio into provenance-tracked MIDI, aligns pasted lyrics to detected notes, and can render a routed DiffSinger vocal when a model is loaded. The Auto-Tune tab writes routed, provenance-tracked WAV files with adjustable pitch correction toward the nearest semitone.

Mixer converts every imported track to the configured project sample rate with deterministic polyphase resampling and explicit stereo normalization before calculating duration or summing. It can analyze each stem, infer a stem role from the track name, and apply local dynamic EQ suggestions with per-band gain, frequency, Q, and reasoning before mastering/export. It can also trim Mid and Side gain independently before limiting, load a reference track, match the final master to its integrated LUFS, report short-term LUFS profile deltas for the match, and target streaming, podcast, broadcast, cinema, or loud CD delivery levels.

## Mastering Presets

| Preset | Target LUFS | Character |
|--------|-------------|-----------|
| Balanced | -14.0 | Neutral, general purpose |
| Loud / Radio | -11.0 | Compressed, bright, competitive loudness |
| Warm / Analog | -14.0 | Enhanced lows, rolled-off highs, narrow stereo |
| Bright / Crisp | -14.0 | Enhanced highs, mid presence, wide stereo |
| Hip-Hop / Trap | -12.0 | Heavy sub-bass, punchy compression |
| Cinematic | -16.0 | Dynamic range, wide stereo, gentle compression |
| Lo-Fi | -16.0 | Rolled-off highs, heavy compression, narrow |
| Streaming (Spotify) | -14.0 | Optimized for streaming platform normalization |

Mixer exposes Mid and Side trim controls from -6 dB to +6 dB. The mastering chain applies those trims after stereo-width processing and before limiting/LUFS normalization.

## Delivery LUFS Targets

| Target | LUFS | Use |
|--------|------|-----|
| Streaming | -14.0 | General music streaming normalization |
| YouTube | -13.0 | Online video music delivery |
| Apple Music | -16.0 | Apple Sound Check-style music delivery |
| Podcast stereo | -16.0 | Spoken-word stereo podcast delivery |
| Broadcast | -24.0 | ATSC A/85-style broadcast delivery |
| Cinema dialog | -27.0 | Dialogue-oriented cinema delivery |
| CD / loud master | -9.0 | Loud physical or club master delivery |

## AI Models

Models are downloaded on-demand through the Model Hub. Nothing downloads until you need it.

| Model | Size | Module | Required |
|-------|------|--------|----------|
| ACE-Step 1.5 XL Turbo (DiT) | ~10.4 GB | Song Forge | Recommended |
| Llama 3.1 8B Q4_K_M | ~4.9 GB | Lyrics Engine | Recommended |
| DiffSinger (ONNX) | ~500 MB | Vocal Suite | Optional |
| RVC v2 | ~200 MB/voice | Vocal Suite | Optional |
| Demucs (htdemucs) | ~300 MB | Stem Separation | Optional |
| Stable Audio Open | ~3 GB | SFX Generator | Optional |

All models run entirely on your local machine. No cloud APIs, no subscriptions, no data leaves your computer.
When Offline Mode is enabled in Settings, all HuggingFace API calls and model downloads are blocked — models load from local cache only or fail with an explicit offline error.
Model Hub cards show each model's license, gated/token status, and commercial-use status. Generated and exported provenance sidecars carry source model license policy forward, and Song Forge export warns when a source model is limited, non-commercial, or governed by model-specific terms.
Built-in Hugging Face downloads use reviewed immutable commit revisions and hash every cached file before loading. Remote model code is disabled by default; models that declare custom code or pickle-backed weights require an explicit Model Hub warning and consent scoped to that exact revision.

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 / Linux / macOS | Windows 11 / Ubuntu 22.04+ |
| Python | 3.11 | 3.12 |
| RAM | 8 GB | 16 GB+ |
| GPU | None (CPU mode) | NVIDIA 8GB+ VRAM (CUDA) |
| Disk | 2 GB (app only) | 45 GB+ (full model registry) |

GPU acceleration requires PyTorch with CUDA support. The app runs on CPU without any GPU, but generation will be slower.

## Configuration

Settings are stored in `~/.config/SlunderStudio/` (Linux/macOS) or `%APPDATA%/SlunderStudio/` (Windows).
The configured output directory is used by generated artifacts and renders; the default export
format, sample rate, bit depth, GPU index, MIDI tempo, Song Forge defaults, and mastering stage
switches are consumed by their respective runtime paths. The model-cache limit is an admission
cap: downloads that would exceed it are refused with an actionable message. First-run onboarding
can save the output folder and experience level, securely collect a HuggingFace token before a
gated download, and carry a selected core model into Model Hub for review or download. It can be
reopened from Settings after the first run; choosing Skip leaves setup incomplete.

```
SlunderStudio/
├── settings.json          # App preferences
├── voice_bank.json        # Voice model profiles
├── projects/              # Saved projects with version history
├── models/                # Downloaded AI models
├── voices/                # Voice model files
└── generations/           # All generated outputs
    ├── songs/             # Song Forge output
    ├── lyrics/            # Lyrics Engine output
    ├── midi_studio/       # MIDI generation output
    ├── midi_renders/      # FluidSynth renders
    ├── vocals/            # DiffSinger output
    ├── voice_convert/     # RVC output
    ├── voice_clone/       # GPT-SoVITS output
    ├── stems/             # Demucs separation output
    ├── sfx/               # SFX Generator output
    └── ai_producer/       # AI Producer pipeline output
```

## Building

Create a standalone executable from a temporary, hash-locked build environment:

```powershell
py -3.12 build/build.py --clean-env
py -3.12 build/build.py --clean-env --onefile # Single .exe (Windows)
```

The `--clean-env` build installs `requirements-lock.txt` and the pinned
`build/requirements-build-lock.txt` with `--require-hashes`, rejects extra
installed distributions, excludes test/cloud/unused Qt modules, and audits the
one-folder bundle before it can be packaged. The Windows smoke check always
uses the private non-input virtual-display isolation helper. The build removes
stale `dist/` outputs, resolves the platform icon, verifies one onedir process
or the expected onefile bootloader-parent/child tree, and writes
`dist/SHA256SUMS.txt`. The default build also creates
`dist/SlunderStudio-vX.Y.Z-<platform>-<arch>.zip` beside `dist/SlunderStudio/`.
Every version string — the window title, settings and project schemas,
provenance sidecars, the README badge, artifact names, and the embedded Windows
file version — comes from `core/version.py`.

**Releases are unsigned.** There is no code-signing step and none will be added. Windows SmartScreen will warn on first run of an unsigned executable; choose "More info" then "Run anyway", or verify the download against `SHA256SUMS.txt` first.

## Releases

Windows releases are published on the [GitHub Releases page](https://github.com/SysAdminDoc/SlunderStudio/releases).
Each release includes an unsigned ZIP and `SHA256SUMS.txt`; verify the checksum before running the app.

## Project Structure

```
SlunderStudio/
├── main.py                     # Entry point with dependency diagnostics
├── core/                       # Core infrastructure
│   ├── audio_engine.py         # Playback engine (sounddevice)
│   ├── audio_export.py         # WAV/FLAC/MP3 export
│   ├── chord_chart.py          # MIDI chord inference and ChordPro/CRD export
│   ├── evaluation.py            # Fixed-case engine measurements and listener rubric
│   ├── lyrics_db.py            # Lyrics database with search
│   ├── mastering.py            # DSP mastering chain
│   ├── midi_utils.py           # MIDI I/O (pretty_midi wrapper)
│   ├── model_manager.py        # HuggingFace model downloads
│   ├── disclosure.py           # AI disclosure and human-authorship reports
│   ├── device.py               # Configured CUDA/MPS/CPU device selection
│   ├── provenance.py           # Generation sidecars and project metadata
│   ├── project.py              # Project save/load/versioning
│   ├── settings.py             # Persistent settings
│   ├── voice_bank.py           # Voice profile management
│   └── workers.py              # Background inference workers
├── engines/                    # AI engine wrappers
│   ├── ace_step_engine.py      # ACE-Step song generation
│   ├── ai_producer.py          # One-prompt pipeline orchestrator
│   ├── audio_analyzer.py       # BPM/key/loudness analysis
│   ├── demucs_engine.py        # Stem separation
│   ├── diffsinger_engine.py    # Singing voice synthesis
│   ├── fluidsynth_engine.py    # MIDI-to-audio rendering
│   ├── lyrics_engine.py        # LLM lyrics generation
│   ├── lyrics_templates.py     # 33 genre template definitions
│   ├── midi_llm_engine.py      # Text-to-MIDI generation
│   ├── melody_extractor.py     # Humming-to-MIDI lyric melody extraction
│   ├── rvc_engine.py           # RVC + GPT-SoVITS voice engines
│   ├── vocal_tuning.py         # Vocal auto-tune pitch correction
│   ├── sfx_engine.py           # Stable Audio Open SFX
│   └── style_tags.py           # ACE-Step style tag database
├── ui/                         # PySide6 interface
│   ├── main_window.py          # Main window with sidebar navigation
│   ├── theme.py                # Catppuccin Mocha dark theme
│   ├── onboarding.py           # First-run wizard
│   ├── song_forge_view.py      # Song generation page
│   ├── lyrics_view.py          # Lyrics writing page
│   ├── lyrics_editor.py        # Rich lyrics editor
│   ├── midi_studio_view.py     # MIDI composition page
│   ├── piano_roll.py           # QGraphicsView piano roll
│   ├── midi_mixer.py           # MIDI track mixer
│   ├── vocal_suite_view.py     # Vocal synthesis page
│   ├── stem_mixer.py           # Demucs stem mixer
│   ├── sfx_view.py             # SFX generation page
│   ├── mixer_view.py           # Multi-track mixer + mastering
│   ├── ai_producer_view.py     # AI Producer page
│   ├── project_manager.py      # Project browser
│   ├── model_hub.py            # Model download manager
│   ├── settings_view.py        # Settings page
│   ├── waveform_widget.py      # Audio waveform display
│   ├── mood_curve_editor.py    # Mood/energy curve editor
│   ├── reference_panel.py      # Reference audio panel
│   ├── seed_explorer.py        # Seed variation explorer
│   ├── batch_view.py           # Batch generation
│   └── toast.py                # Toast notifications
├── assets/templates/           # 33 genre JSON templates
├── build/build.py              # locked PyInstaller packaging
├── build/requirements-build.txt
├── build/requirements-build-lock.txt
├── requirements.txt            # Dependencies
└── LICENSE                     # MIT License
```

## FAQ

**Q: Do I need a GPU?**
No. Everything runs on CPU. A CUDA-capable NVIDIA GPU (8GB+ VRAM) dramatically speeds up AI generation but is not required.

**Q: How much disk space do models need?**
About 15.3 GB for the recommended ACE-Step and Llama checkpoints. The full built-in model registry is approximately 38.4 GB. Models download on-demand — nothing installs until you request it.

**Q: Can I use my own voice models?**
Yes. Import RVC `.pth` models or GPT-SoVITS checkpoints through the Voice Bank. The app auto-detects models in standard directories.

**Q: Is any data sent to the cloud?**
No. All processing is local. The only network traffic is model downloads from HuggingFace, which you initiate manually.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built by [SysAdminDoc](https://github.com/SysAdminDoc) with Slunder.
