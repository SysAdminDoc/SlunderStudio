# Slunder Studio

![Version](https://img.shields.io/badge/version-0.1.30-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
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

Python 3.10+ required. Install core dependencies explicitly before launch; if anything is missing, Slunder Studio opens a diagnostics screen with the exact setup command. AI models are downloaded on-demand from HuggingFace via the built-in Model Hub.

## Features

| Module | Description | AI Engine |
|--------|-------------|-----------|
| Song Forge | Full song generation from lyrics + style tags, stitched long-form songs, and recovered vocal-stem export | ACE-Step, Demucs |
| Lyrics Engine | AI-powered lyrics writing with 33 genre templates | Llama 3.2 1B |
| MIDI Studio | Piano roll editor with quantize/swing/humanize tools, CC lanes, text-to-MIDI composition, groove-template drums, and chord chart export | MIDI-LLM |
| Vocal Suite | Singing synthesis, humming-to-MIDI lyric melody generation, voice conversion, voice cloning, and vocal auto-tune pitch correction | DiffSinger, RVC v2, GPT-SoVITS, librosa |
| Stem Separation | Isolate vocals, drums, bass, and other instruments | Demucs (htdemucs) |
| SFX Generator | Text-to-sound-effect generation | Stable Audio Open |
| Mixer | Multi-track mixing with stem-aware dynamic EQ, Mid/Side trims, reference loudness matching, and smart mastering (8 presets) | Built-in DSP |
| AI Producer | One prompt to full song — auto-chains all modules | Orchestrator |
| Model Hub | Download, manage, and switch AI models | HuggingFace Hub |
| Projects | Save/load projects with version history and asset tracking | — |

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

Every module can route audio to any other module. Generate a song in Song Forge, separate stems in Vocal Suite, add SFX, mix everything in the Mixer, and export a mastered track. Generated and exported assets write adjacent `.provenance.json` sidecars with app version, prompt/lyrics, seed, model revision/hash metadata, source paths, and render parameters so projects can be audited or reproduced. Long-running generation and model-download jobs persist queued/running/completed/failed/cancelled/recoverable state so interrupted sessions can show what needs recovery on restart. Settings and project files use versioned schemas with timestamped backups before migrations, repairs, and saves. Primary creative workflows expose screen-reader names, descriptions, high-contrast focus rings, and predictable tab traversal. For songs over 2 minutes, Song Forge can render structured sections separately and stitch them with crossfades for more stable long-form arrangements; completed Song Forge renders also attempt a Demucs vocal-stem recovery and expose a separate vocals-only route when recovery succeeds. Seed Explorer renders nearby seed/CFG variations from the current lyrics and style prompt so you can compare takes before committing to a full arrangement. Reference Track analysis maps an audio fingerprint to ACE-Step tags for one-click style conditioning, Genre Fusion blends two template tag sets into weighted hybrid prompts, and Voice Cloning validates 10-30s GPT-SoVITS reference samples before saving reusable voice profiles with owner, consent source, language, permitted-use, and sidecar provenance metadata.

Settings can export a redacted health report ZIP with app/dependency versions, GPU and ffmpeg status, model cache state, settings repair status, crash log metadata, and recent failed jobs. HuggingFace tokens are always redacted, and job prompts/lyrics stay out of the report unless the private-input opt-in is enabled.

Major app chrome and lyric controls use an English locale catalog, and Settings > Appearance > Default Lyrics Language feeds Quick lyrics prompts, Guided lyrics metadata, and new GPT-SoVITS voice profile language defaults where supported. MIDI Studio supports explicit chord-progression priors such as `I-V-vi-IV` and `ii-V-I` for text-to-MIDI prompts and fallback generation. MIDI Studio also includes selectable drum groove templates with swing timing, snare ghost notes, and velocity humanization for generated GM drum tracks, `.chordpro` and `.crd` chord chart export with optional pasted lyrics, and piano roll editing tools for quantize, swing, velocity humanize, and MIDI CC automation lanes. Vocal Suite includes a Lyric Melody tab that converts hummed audio into provenance-tracked MIDI, aligns pasted lyrics to detected notes, and can render a routed DiffSinger vocal when a model is loaded. The Auto-Tune tab writes routed, provenance-tracked WAV files with adjustable pitch correction toward the nearest semitone.

Mixer can analyze each imported stem, infer a stem role from the track name, and apply local dynamic EQ suggestions with per-band gain, frequency, Q, and reasoning before mastering/export. It can also trim Mid and Side gain independently before limiting, load a reference track, match the final master to its integrated LUFS, report short-term LUFS profile deltas for the match, and target streaming, podcast, broadcast, cinema, or loud CD delivery levels.

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
| ACE-Step | ~3 GB | Song Forge | Recommended |
| Llama 3.2 1B | ~2 GB | Lyrics Engine | Recommended |
| DiffSinger (ONNX) | ~500 MB | Vocal Suite | Optional |
| RVC v2 | ~200 MB/voice | Vocal Suite | Optional |
| Demucs (htdemucs) | ~300 MB | Stem Separation | Optional |
| Stable Audio Open | ~3 GB | SFX Generator | Optional |

All models run entirely on your local machine. No cloud APIs, no subscriptions, no data leaves your computer.
When Offline Mode is enabled in Settings, all HuggingFace API calls and model downloads are blocked — models load from local cache only or fail with an explicit offline error.
Model Hub cards show each model's license, gated/token status, and commercial-use status. Generated and exported provenance sidecars carry source model license policy forward, and Song Forge export warns when a source model is limited, non-commercial, or governed by model-specific terms.

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 / Linux / macOS | Windows 11 / Ubuntu 22.04+ |
| Python | 3.10 | 3.11+ |
| RAM | 8 GB | 16 GB+ |
| GPU | None (CPU mode) | NVIDIA 8GB+ VRAM (CUDA) |
| Disk | 2 GB (app only) | 20 GB+ (with models) |

GPU acceleration requires PyTorch with CUDA support. The app runs on CPU without any GPU, but generation will be slower.

## Configuration

Settings are stored in `~/.config/SlunderStudio/` (Linux/macOS) or `%APPDATA%/SlunderStudio/` (Windows).

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

Create a standalone executable with PyInstaller:

```bash
py -3.12 -m pip install pyinstaller
py -3.12 build/build.py           # One-folder distribution
py -3.12 build/build.py --onefile # Single .exe (Windows)
```

The build script removes stale `dist/` outputs before packaging, smoke-launches the Windows executable to verify only one app process starts, and writes `dist/SHA256SUMS.txt` for the distributable artifacts. The default build also creates a release ZIP beside `dist/SlunderStudio/`. To Authenticode-sign executables, install `signtool` and set either `SLUNDER_SIGN_CERT_SHA1` or `SLUNDER_SIGN_CERT_FILE` before building; `SLUNDER_SIGN_TIMESTAMP_URL` overrides the timestamp server when needed.

## Project Structure

```
SlunderStudio/
├── main.py                     # Entry point with dependency diagnostics
├── core/                       # Core infrastructure
│   ├── audio_engine.py         # Playback engine (sounddevice)
│   ├── audio_export.py         # WAV/FLAC/MP3 export
│   ├── chord_chart.py          # MIDI chord inference and ChordPro/CRD export
│   ├── lyrics_db.py            # Lyrics database with search
│   ├── mastering.py            # DSP mastering chain
│   ├── midi_utils.py           # MIDI I/O (pretty_midi wrapper)
│   ├── model_manager.py        # HuggingFace model downloads
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
├── build/build.py              # PyInstaller packaging
├── requirements.txt            # Dependencies
└── LICENSE                     # MIT License
```

## FAQ

**Q: Do I need a GPU?**
No. Everything runs on CPU. A CUDA-capable NVIDIA GPU (8GB+ VRAM) dramatically speeds up AI generation but is not required.

**Q: How much disk space do models need?**
About 3 GB for the recommended models (ACE-Step + Llama). The full model suite is approximately 10 GB. Models download on-demand — nothing installs until you request it.

**Q: Can I use my own voice models?**
Yes. Import RVC `.pth` models or GPT-SoVITS checkpoints through the Voice Bank. The app auto-detects models in standard directories.

**Q: Is any data sent to the cloud?**
No. All processing is local. The only network traffic is model downloads from HuggingFace, which you initiate manually.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built by [SysAdminDoc](https://github.com/SysAdminDoc) with Slunder.
