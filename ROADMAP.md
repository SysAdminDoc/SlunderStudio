# Slunder Studio Roadmap

Roadmap for Slunder Studio - an offline local-first AI music generation suite (ACE-Step, DiffSinger, RVC, Demucs, Stable Audio Open). Focus: better quality, faster iteration, tighter DAW integration.

## Planned Features

### Generation quality

### Vocals & voice
- Vocal auto-tune (pitch correction to nearest semitone, adjustable strength) - librosa + worldsynth
- Lyric-to-melody generator (humming input -> MIDI line -> DiffSinger render)
- Vocal stem recovery: isolated vocal track exported separately after Song Forge renders

### MIDI & composition
- Text-to-MIDI improvements with chord-progression priors (`I-V-vi-IV`, `ii-V-I`)
- Drum-pattern library with groove templates (swing, ghost notes, velocity humanize)
- MIDI -> chord chart export (`.chordpro`, `.crd`) for lyric + chord sheet printing
- Piano roll quantize, swing, velocity humanize, CC automation lanes

### Mixing & mastering
- Dynamic EQ with AI-suggested curves per stem
- Loudness match to reference track with short-term LUFS tracking
- Reference LUFS targets beyond streaming presets (broadcast, podcast, cinema)
- Mid/Side mastering controls surfaced in Mixer view

### Export & integration
- `.dawproject` export (cross-DAW: Cubase, Studio One, Bitwig)
- Stems export with naming convention matching user-chosen DAW template
- Project-level MP3/FLAC/Opus render with metadata (ID3, Vorbis comments)
- OSC control for external MIDI/DAW remotes
- Headless CLI mode (`slunderstudio --prompt "..." --preset ace-step-long --out song.wav`)

### Model hub & ops
- Model update checker with safe rollback (keep last good version)
- Quantized model variants (4-bit, 8-bit) with quality/VRAM tradeoff picker
- DirectML backend for AMD GPUs on Windows
- Mac MPS backend for Apple Silicon
- Background model download queue with resume + SHA256 verify

## Competitive Research

- **Suno / Udio** - cloud-only, closed. Slunder Studio's edge is fully local generation with full stem + MIDI access; lean into that in marketing and add "Suno-equivalent prompt" translator to import shared Suno prompts.
- **Stable Audio Tools** (Stability) - open local pipeline for SFX, already integrated; upgrade path to Stable Audio 2 when released.
- **AudioCraft/MusicGen** (Meta) - melody-conditioned generation; worth adding MusicGen-Melody as an alternate Song Forge backend so users can hum a melody and have AI complete the arrangement.
- **Riffusion** - real-time spectrogram diffusion; add as a "live jam" mode for latent-space exploration.
- **LM Studio / Jan.ai** - local LLM model-hub UX is mature; mirror their "import GGUF", "change backend", "VRAM estimator" patterns in the Model Hub.

## Nice-to-Haves

- Ableton Link sync for real-time tempo-matched jam sessions
- MIDI controller mappings for Mixer and Piano Roll
- Batch-queue panel ("generate 50 songs overnight with preset X") with resume-on-crash
- In-app lyric scoring vs rhyme/rhythm heuristics during Lyrics Engine edits
- Gradio/webview embed of each engine for remote/headless use
- Plugin SDK so engines can be added without forking (drop a `.py` in `plugins/engines/`)
- Direct export-and-upload pipeline to SunoJump for fingerprint masking before re-upload

## Open-Source Research (Round 2)

### Related OSS Projects
- https://github.com/ace-step/ACE-Step-1.5 — Leading local Suno alternative, <4GB VRAM, LoRA training, Vocal2BGM, multi-track, LRC timestamps
- https://github.com/fspecii/ace-step-ui — Professional Gradio-beyond UI for ACE-Step — UX reference for a desktop wrapper
- https://github.com/multimodal-art-projection/YuE — Full-song foundation model, multilingual (EN/ZH/YUE/JA/KO)
- https://github.com/facebookresearch/demucs — HTDemucs stem separator (industry standard)
- https://github.com/nomadkaraoke/python-audio-separator — MDX/MDXC/VR/Roformer wrapper — single API over many separators
- https://github.com/Anjok07/ultimatevocalremovergui — UVR GUI, huge model catalog
- https://github.com/facebookresearch/audiocraft — MusicGen reference
- https://github.com/magenta/magenta — MIDI-focused generation (melody/harmony RNN/Transformer)
- https://github.com/Zizwar/Awesome-Suno — Awesome list of OSS Suno-adjacent projects

### Features to Borrow
- ACE-Step LoRA training flow — "train on my past songs, generate in my voice/style" (ACE-Step 1.5 v1.5 XL)
- Vocal2BGM mode — user uploads an a cappella, engine generates a backing track (ACE-Step)
- Audio understanding pass — extract BPM / key / time-sig / caption before generation as conditioning (ACE-Step)
- LRC generation for lyric-timestamp export straight into Slunder-Songs vault (ACE-Step)
- python-audio-separator as the stem-sep adapter — one interface, user picks MDX/MDXC/VR/Roformer/HTDemucs at runtime
- UVR's model-downloader UI pattern — users pick models from a gallery, downloads cached to `%LOCALAPPDATA%` (Ultimate Vocal Remover GUI)
- Magenta's MIDI scaffolds (melody-rnn, music-transformer) behind the MIDI Engine tab (magenta)
- Multi-track "Add Layer" flow modeled on Suno Studio (ACE-Step XL)

### Patterns & Architectures Worth Studying
- Gradio server as local sidecar + PyQt6 front-end talking HTTP to it — users can open Gradio UI in browser OR PyQt shell, same backend (fspecii/ace-step-ui model)
- Model registry manifest — `models.json` lists weights + hash + min-VRAM + download URL, settings UI toggles per model (UVR pattern)
- Plugin engine interface (`BaseEngine` with `name, capabilities, generate(params, ctx) -> AudioResult`) — drop-in `plugins/engines/*.py`
- VRAM-aware scheduler — engines declare min-VRAM, scheduler dequeues jobs only when a worker with capacity is free
- Shared audio-tensor cache so stem-sep output can feed mastering/fingerprint-mask without re-decode

## Research-Driven Additions

### P0

### P1

### P2
- [ ] P2 - Build an exportable diagnostics and health report
  Why: Crash logs exist, but users need a single redacted report for model status, dependency versions, GPU/CPU, ffmpeg, config paths, and recent job failures.
  Evidence: `main.py`, `core/model_manager.py`, `core/settings.py`, `core/audio_export.py`.
  Touches: `core/diagnostics.py`, `ui/settings_view.py`, `ui/model_hub.py`, `main.py`, `tests/`.
  Acceptance: Settings includes "Export Health Report" producing a redacted text/JSON bundle with dependency versions, app version, GPU, model statuses, ffmpeg availability, recent errors, and no HF tokens or user lyrics unless opted in.
  Complexity: S

- [ ] P2 - Produce signed, checksummed local distributables
  Why: Build output exists, but releases need clean artifacts, checksums, smoke launch verification, and signing when a certificate is available.
  Evidence: `build/build.py`, `SlunderStudio.spec`, PyInstaller runtime docs.
  Touches: `build/build.py`, `SlunderStudio.spec`, `README.md`, `tests/`.
  Acceptance: Build cleans stale artifacts, creates one-folder and optional onefile outputs, writes SHA256 checksums, smoke-launches the packaged app without spawning extra processes, and signs artifacts when a configured certificate exists.
  Complexity: M

### P3
- [ ] P3 - Add an i18n foundation for UI and lyric workflows
  Why: ACE-Step and GPT-SoVITS support multilingual use, and settings already include a default lyric language, but UI text is hardcoded English.
  Evidence: `core/settings.py`, `ui/*.py`, ACE-Step multilingual lyrics, GPT-SoVITS language support.
  Touches: `core/i18n.py`, `ui/*.py`, `assets/locales/`, `tests/`.
  Acceptance: User-visible strings are routed through a locale catalog, English remains complete, lyric language settings feed prompt/profile metadata, and tests fail on missing keys in major views.
  Complexity: L
