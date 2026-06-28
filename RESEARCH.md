# Research - Slunder Studio

## Executive Summary
Slunder Studio is a local-first PySide6 desktop suite for full-song generation, lyrics, MIDI, voice, stems, SFX, mixing, projects, and model management. Verified: the strongest current shape is not a single model wrapper; it is the integrated offline workflow around ACE-Step, local voice/stem tools, project persistence, and DAW-facing exports. Highest-value direction: make the app trustworthy enough for real creative work before adding more engines. Priority opportunities are: stop placeholder/fallback audio from looking like real generation, harden checkpoint loading and model provenance, remove runtime environment mutation, add generation provenance/reproducibility sidecars, make destructive deletes recoverable, add settings/project migrations, build a screen-reader/accessibility baseline, and surface model license/commercial-use constraints in Model Hub.

## Product Map
- Core workflows: prompt/lyrics/style to ACE-Step song render; reference-track analysis to style tags; seed exploration and batch variation; voice cloning/conversion/singing; stem separation, remix, mastering, and export.
- User personas: local-first musicians avoiding cloud subscriptions; hobby producers iterating against Suno/Udio-style prompts; remixers needing stems and DAW handoff; voice/model tinkerers managing local checkpoints.
- Platforms and distribution: Python 3.10+ desktop app for Windows/Linux/macOS, PyInstaller one-folder/onefile builds, local config under `core/settings.py`, project data under `core/project.py`.
- Key integrations and data flows: Hugging Face model downloads through `core/model_manager.py`; generated audio into module output folders; project assets copied by `ProjectManager.import_asset`; ffmpeg used for MP3/OGG export in `core/audio_export.py`.

## Competitive Landscape
- ACE-Step 1.5: strong local song model with reference audio, cover/repaint, Vocal2BGM, LRC, audio understanding, LoRA, quality scoring, and 10-minute duration support. Learn from its model-zoo/runtime maturity and expose more native capabilities already noted in `ROADMAP.md`; avoid copying its many launch scripts into the desktop app without a single coherent UX.
- Suno Studio: best reference for creator-facing editing, multitrack workflow, DAW stems, MIDI export, personas, and mobile reach. Learn from its "complete creative workspace" positioning; avoid cloud credits/subscriptions because Slunder's differentiator is local ownership.
- Udio and ElevenLabs Music: useful references for polished prompt-to-song onboarding, commercial-rights messaging, and simple public sharing. Learn the quality bar and rights clarity; avoid building social/community hosting before local reliability is solid.
- Demucs, UVR, and python-audio-separator: prove that model choice, model galleries, and quality-vs-speed selection matter more than a single separator backend. Learn UVR-style model browsing and separator adapters; avoid depending only on archived Demucs for future stem quality.
- GPT-SoVITS, RVC, and DiffSinger: show the depth expected from voice workflows: consent-sensitive profiles, language metadata, dataset preparation, segmentation, training/inference separation, and device-specific installs. Learn their profile and dataset guardrails; avoid silent fake outputs from placeholder synthesis.
- AudioCraft, Riffusion, YuE, and Stable Audio Open: useful adjacent engines for melody conditioning, live exploration, full-song multilingual generation, and SFX. Learn alternate-engine interfaces and clear model licensing; avoid adding more engines until current engine lifecycle and provenance are reliable.
- DAWproject, Ableton Link, and CLAP: represent standards-level integration paths. Learn from DAWproject's ZIP/XML project portability and Link tempo sync; avoid a full plugin host before export, sync, and project metadata are dependable.

## Security, Privacy, and Reliability
- Verified risk: `engines/rvc_engine.py` loads GPT-SoVITS profile files with `torch.load(..., weights_only=False)` from user-controlled paths. PyTorch treats untrusted model files as executable-code risk; Slunder needs provenance, hashes, trusted-source UX, and safer formats where available.
- Verified risk: `engines/sfx_engine.py` returns noise-synthesis placeholder results when Stable Audio is not loaded, and `engines/rvc_engine.py` contains placeholder RVC/GPT-SoVITS inference paths. Exports/routing should not present those as real model outputs.
- Verified risk: `main.py`, `core/deps.py`, and `build/build.py` install packages at runtime/build time. This undermines offline reproducibility, complicates support, and can mutate the user's Python environment outside an explicit setup command.
- Verified risk: `core/project.py` deletes project directories with `shutil.rmtree(..., ignore_errors=True)` and no undo/quarantine. Model and asset deletion need the same recoverable-delete policy.
- Verified gap: `core/settings.py` silently falls back on corrupt JSON and silently ignores save failures. Users need backup/repair, migration logs, and visible status when settings cannot persist.
- Verified gap: `core/model_manager.py` writes a completion marker with file count/size/source but no pinned revision, per-file hash manifest, license acceptance record, or rollback metadata.

## Architecture Assessment
- `core/model_manager.py` is the right central boundary for model lifecycle, but it needs a persistent model manifest: repo, revision, files, hashes, license, trusted state, install backend, and compatible devices.
- `core/workers.py` provides the correct QThread boundary, but cancellation is cooperative only; long calls such as ACE-Step pipeline execution need cancel-safe job state, resume/cleanup, and clear "cancel pending" UI.
- `core/project.py` has atomic JSON writes and version snapshots, but lacks schema migrations, project repair, asset collision handling, and trash/undo for destructive actions.
- `main.py` has the PyInstaller freeze guard in the right place, but bootstrap should become diagnostics plus explicit setup instead of implicit pip installs.
- `ui/*` has broad module coverage, but accessibility is not systematic: controls need `accessibleName`/`accessibleDescription`, tested tab order, focus visibility, screen-reader labels, and non-color-only state.
- Test coverage is focused on recent features (`tests/test_*`) but not on persistence corruption, model trust, runtime dependency behavior, export failure paths, accessibility properties, or packaged launch smoke tests.

## Rejected Ideas
- Cloud credits/subscription backend, source Suno/Udio: contradicts Slunder's local-first privacy and offline positioning.
- Mobile-first clone, source Suno mobile app: useful for reach but conflicts with local model/GPU/storage assumptions; revisit only after desktop package reliability is mature.
- Full CLAP/VST host, source CLAP: too much surface before DAWproject export, OSC/Link sync, and project provenance land.
- Direct upload or fingerprint-masking pipeline, source existing `ROADMAP.md` SunoJump note: creates trust and platform-evasion risk; export clean stems/metadata instead.
- Multi-user collaboration/server accounts, source commercial platforms: requires auth, sync, conflict resolution, and privacy policy work that does not support the offline-first promise yet.
- Adding more song engines immediately, source AudioCraft/Riffusion/YuE: current engine lifecycle, provenance, and false-success issues should be fixed before broadening the matrix.

## Sources
OSS music generation:
- https://github.com/ace-step/ACE-Step-1.5
- https://github.com/multimodal-art-projection/YuE
- https://github.com/facebookresearch/audiocraft
- https://github.com/riffusion/riffusion
- https://github.com/Stability-AI/stable-audio-tools
- https://huggingface.co/stabilityai/stable-audio-open-1.0
- https://github.com/Zizwar/Awesome-Suno

Voice and stems:
- https://github.com/facebookresearch/demucs
- https://github.com/nomadkaraoke/python-audio-separator
- https://github.com/Anjok07/ultimatevocalremovergui
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- https://github.com/RVC-Boss/GPT-SoVITS
- https://github.com/openvpi/DiffSinger

Commercial and workflow references:
- https://suno.com/
- https://suno.com/pricing
- https://www.udio.com/
- https://elevenlabs.io/music
- https://github.com/janhq/jan

Standards, security, and platform docs:
- https://github.com/bitwig/dawproject
- https://github.com/Ableton/link
- https://github.com/free-audio/clap
- https://github.com/pytorch/pytorch/security/policy
- https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573
- https://huggingface.co/docs/huggingface_hub/en/guides/download
- https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QWidget.html
- https://github.com/pypa/pip-audit
- https://pyinstaller.org/en/stable/runtime-information.html

## Open Questions
- None blocking prioritization. Implementation should verify target PyTorch, ACE-Step, and PySide6 versions at the start of each roadmap item because this stack changes quickly.
