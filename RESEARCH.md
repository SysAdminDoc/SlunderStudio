# Research — Slunder Studio
Date: 2026-07-29 — replaces all prior research.

## Executive Summary

Slunder Studio is a local-first PySide6 desktop suite spanning song generation, lyrics, MIDI, singing/voice workflows, stem separation, mixing/mastering, project recovery, provenance, and model management. Its strongest shape is the breadth of an offline creative pipeline with unusually good consent, license, provenance, trash, job-state, and schema-migration foundations. Its highest-value direction is not more surface area: it is making every advertised engine and handoff truthful, reversible, secure, and testable. The immediate priorities are:

1. Contain cancellation cleanup so recorded job paths cannot delete arbitrary user files.
2. Pin reviewed model revisions, remove unconditional remote-code trust, and lock optional AI dependency profiles above known vulnerable versions.
3. Correct the mislabeled ACE-Step 1.5 integration and prove cover, extend, and repaint use their source audio.
4. Make AI Producer stage outcomes, progress, cancellation, and final-success state truthful.
5. Make Mixer resample inputs before mixing and route delivery through the shared export/provenance layer.
6. Prevent project-asset overwrite collisions and add real autosave/version restore.
7. Replace downloaded/placeholder/loaded ambiguity with one engine capability and readiness contract.
8. Establish feature-level accessibility, responsive-layout, i18n, and packaged-GUI acceptance gates.
9. Produce reproducible unsigned releases from one version source and tested runtime/backend lock profiles.
10. Validate mastering, metadata, and DAWproject delivery against published standards and real import fixtures.

## Product Map

- Core workflows: prompt/lyrics/style to ACE-Step generation; reference/seed/long-form variation; MIDI composition and rendering; singing, voice conversion, tuning, and separation; multitrack mix/master/export.
- Core workflows: project creation, asset import, manual snapshots, provenance review, recoverable deletion, persistent jobs, settings backup/migration, and redacted diagnostics.
- User personas: privacy-conscious musicians, local-model experimenters, producers handing work to a DAW, remix/restoration users, and creators who need explicit model/voice rights records.
- Platforms and distribution: claimed Windows, Linux, and macOS desktop support; Python 3.12 is the only evidenced release runtime; PyInstaller tooling is currently Windows-shaped and unsigned delivery is required.
- Integrations and data flows: Hugging Face downloads feed engine-specific loaders; generated files and sidecars flow into projects, Mixer, and exports; `.dawproject`, OSC, headless CLI, richer stems, and model variants remain incomplete roadmap work.

## Competitive Landscape

- **ACE-Step 1.5** — Does well: explicit task modes, device/VRAM tiers, source conditioning, editing, LoRA, and multilingual control. Learn: versioned adapters, task-specific validation, and hardware-aware model cards. Avoid: claiming v1.5 while loading the legacy `ACE-Step-v1-3.5B` checkpoint/API.
- **YuE and SongGeneration 2** — Do well: full-song generation, staged previews, continuation, saved sessions, separate vocal/accompaniment output, and multilingual lyrics. Learn: resumable checkpoints and preview-before-refinement. Avoid: importing their heavy training/runtime stack before Slunder's current inference paths work.
- **Suno Studio and Udio Sessions** — Do well: take lanes, region replacement, autosave, snapshots, keyboard editing, and scoped exports. Learn: non-destructive alternatives and restorable history. Avoid: cloud-credit, account, social, or service-lock-in features that contradict local ownership.
- **Moises, Ableton Live, LANDR, and Ozone** — Do well: preserve originals, preview/compare, expose quality tradeoffs, and make export/mastering reversible. Learn: gain-matched A/B, replace-or-save-copy, honest stem limitations, and deterministic delivery. Avoid: describing heuristic DSP or separated stems as production-ready without validation.
- **SoulX-Singer, RVC, GPT-SoVITS, and DiffSinger** — Do well: real preparation/inference pipelines, MIDI/F0 control, multilingual voice work, and explicit model workflows. Learn: wire load, readiness, inference, cancellation, and results end to end. Avoid: presenting status-only buttons or placeholder synthesis as implemented.
- **UVR and python-audio-separator** — Do well: maintained model catalogs, capability-specific stems, ensembles, chunking, hardware profiles, and predictable naming. Learn: a separator adapter with model/settings provenance. Avoid: treating archived Demucs or any single model as the entire separation strategy.
- **ComfyUI Manager and InvokeAI** — Do well: distinguish inventory, installation, activation, update, failure, and security state. Learn: supply-chain checks and constrained credential/filesystem boundaries. Avoid: a marketplace until extension isolation, authorization, compatibility, and rollback exist.
- **DAWproject, Bitwig, Ableton, and Logic** — Do well: portable timelines, scoped export, recovery, keyboard contracts, and failure isolation. Learn: schema/golden-import validation and a shared export contract. Avoid: building a full DAW or plug-in host before reliable handoff works.

## Security, Privacy, and Reliability

- **Verified:** `core/job_state.py:205-216` unlinks every recorded output path without proving it is an app-owned artifact. Restrict cleanup to canonical project/generation/temp roots, reject symlink escapes, and test adversarial paths.
- **Verified:** `engines/lyrics_engine.py:193-201` and `engines/midi_llm_engine.py:622-634` set `trust_remote_code=True`; `core/model_manager.py` defaults revisions to mutable `main`. Built-ins need audited commit SHAs, hash verification before load, safetensors preference, and explicit approval for custom code/pickle.
- **Verified:** optional floors in `requirements.txt` admit PyTorch versions affected by GHSA-53q9-r3pm-6pq6 and Transformers versions affected by GHSA-9356-575x-2w9m. Per-runtime/backend lock profiles must replace broad commented minimums.
- **Verified:** `core/model_manager.py:147-163` calls a legacy 3.5B checkpoint “ACE-Step v1.5,” while `engines/ace_step_engine.py` uses the legacy pipeline. Official v1.5 has different model families, APIs, capability tiers, and Python constraints.
- **Verified:** `engines/ace_step_engine.py:534-552` forwards `src_audio_path` only inside a repaint-range branch; `extend()` and `generate_cover()` set a source but do not set that branch. These modes can execute ordinary generation while reporting source-conditioned work.
- **Verified:** `engines/ai_producer.py:312-335` marks any returned dictionary complete, including dictionaries that encode errors; the outer pipeline then reaches `COMPLETE`. `ui/ai_producer_view.py` also discards detailed progress and can retain a previous successful export after a failed rerun.
- **Verified:** `ui/mixer_view.py:635-672` mixes arrays sample-for-sample using the first track's sample rate, then `ui/mixer_view.py:676-748` writes PCM16 directly. Mixed-rate inputs therefore change timing/pitch, and delivery bypasses shared export validation and provenance.
- **Verified:** `core/project.py:492-528` copies imported assets to a basename destination without collision handling. A second same-named file can overwrite the first and its sidecar.
- **Verified:** `ui/model_hub.py`/`ui/settings_view.py` persist the Hugging Face token through `core/settings.py` plaintext JSON and timestamped backups. Store secrets in the OS credential service and migrate/redact old copies.
- **Verified:** `ui/midi_studio_view.py:581`, `ui/ai_producer_view.py:548`, and `ui/vocal_suite_view.py:1421,1580` call nonexistent `WaveformWidget.load_audio()`; the widget exposes `load_file()` and `set_audio()`.
- **Verified:** `py -3.12 -m pip_audit -r requirements-lock.txt --format json` found no known vulnerabilities in the locked core environment on 2026-07-29. This does not cover the unpinned optional AI stack.
- **Likely:** local-first privacy is a real differentiator, but “offline” should mean each selected engine has a verified local cache and network-denied inference test—not merely that the shell exposes an offline setting.

## Architecture Assessment

- **Verified:** introduce a versioned engine descriptor/result contract covering capability, model revision, license, local path, loaded state, device/dtype, progress, cancellation, output kind, and routable artifacts. This is the prerequisite for truthful Model Hub, SFX, MIDI, Vocal Suite, AI Producer, CLI, and any future plug-in SDK.
- **Verified:** complete or explicitly disable inert paths: MIDI always calls `generate_demo_midi()` (`ui/midi_studio_view.py:421-439`); DiffSinger, RVC, and stem-separation buttons are status-only (`ui/vocal_suite_view.py:1054-1055,1173-1184,1531-1536`); SFX expects a loaded singleton that Model Hub never activates.
- **Verified:** make projects transactional: collision-safe asset IDs, index reconstruction from on-disk projects, bounded backups, schema round trips that preserve unknown fields, real interval autosave, pre-restore snapshots, and restore/compare for `ProjectVersion`.
- **Verified:** centralize decode/resample/mix/master/export. Mixer and Stem Mixer currently duplicate pan/render behavior; Dynamic EQ mutates track arrays in place; mastering loops block the UI; format/provenance behavior differs by surface.
- **Verified:** current loudness is an RMS approximation, not BS.1770/R128 conformance. A production mastering path needs K-weighted gated loudness, oversampled true peak, published vectors, gain-matched preview, apply/revert, and retained originals.
- **Verified:** accessibility is partial. The only theme uses small muted text at approximately 3.36:1 on the base color, stylesheet focus outlines are removed, custom waveform/piano-roll controls lack keyboard semantics, and the 1200×800 minimum/fixed sidebar force overflow on smaller displays.
- **Verified:** i18n has an English catalog and helper, but only a small subset of views use it; there is no locale selector, pseudolocale, RTL pass, or extraction/completeness gate.
- **Verified:** release metadata spans `README.md`, source headers, Settings, the spec, and build scripts at both 0.1.29 and 0.1.30. `build/build.py` is Windows-specific, references the wrong icon path, and contains signing behavior that must be removed; releases must remain unsigned.
- **Verified:** the remaining direct dependency review found `sounddevice==0.5.5`, `pyqtgraph==0.14.0`, and `psutil==7.2.2` at their current stable releases; test SoundFile 0.14's concurrent-open fix, NumPy 2, and Hugging Face Hub/Transformers combinations in compatibility lanes instead of changing the release pins blindly.
- **Verified:** add concurrency guards to `core/lyrics_db.py` and `core/model_manager.py`, move SFX/MIDI/reference analysis off the GUI thread, bound job/log/backup retention, and surface resume/discard/retry in one recovery view.
- **Likely:** Stack Overflow PySide reports, KVR separation discussions, and Audacity recovery failures repeat the same need for signal-driven workers, honest stem limitations, and pre-restore snapshots.
- **Needs live validation:** no model weights or supported DAWs were available for real inference/import testing. Implementation acceptance must include fixed-model generation fixtures, network-denied cached runs, packaged GUI smoke tests on each claimed OS, and golden `.dawproject` imports in Bitwig, Studio One, and Cubase.
- **Verified:** the core unit suite passed 147 tests on Python 3.12. Test gaps remain for real inference, complete AI Producer runs, model supply-chain policy, mixed sample rates, project collision/restore, accessibility roles/actions, NumPy 2/Python 3.13+, and packaged releases.

## Rejected Ideas

- Direct SunoJump upload/fingerprint masking — source: prior `ROADMAP.md`; platform-evasion and trust risk with no product-fit benefit.
- Cloud accounts, social hosting, multi-user sync, or mobile-first expansion — sources: Suno, Udio, Moises; conflict with the local-first desktop promise and add identity/sync/policy risk before correctness.
- Full VST/CLAP host or node-graph marketplace — sources: Bitwig, ComfyUI, InvokeAI; requires process isolation, registry security, authorization, compatibility, and crash recovery far beyond current needs.
- In-app ACE-Step/RVC/GPT-SoVITS training now — sources: ACE-Step 1.5, RVC, GPT-SoVITS; inference activation and recovery are incomplete, so training would multiply support burden.
- Riffusion live-jam or another built-in MusicGen backend now — sources: Riffusion, AudioCraft; novelty is lower-value than fixing existing engines, and MusicGen weights are noncommercial.
- A “Suno-equivalent prompt” translator — source: prior `ROADMAP.md`; Suno exposes no stable portable prompt schema, so this would be brittle branding-specific heuristics.
- Direct Magenta RNN scaffolds — source: prior `ROADMAP.md`; adding a TensorFlow-era stack before the existing MIDI-LLM path works increases packaging cost without fixing the verified gap.
- A shared decoded-audio tensor cache now — source: prior `ROADMAP.md`; first centralize decode/resample ownership and measure repeated-decode cost, otherwise cache lifetime and RAM pressure become new failure modes.
- Gradio sidecar plus PySide shell — source: ACE-Step community UIs; duplicates state, security, accessibility, and packaging surfaces without solving the current engine contract.
- Automatic installation of gated, custom-code, or noncommercial models — sources: Hugging Face, Stable Audio Open, AudioCraft; require explicit license/security approval and immutable provenance.
- MIDI 2.0/UMP implementation in this cycle — source: MIDI Association; worthwhile only after MIDI 1 generation, mute/solo, rendering, routing, and controller mapping work.
- FAD-only release gating — sources: VERSA, MAD/MusicPrefs, MusicEval; use fixed technical metrics plus a small blinded listener rubric because no single metric captures musical quality.
- A generic “upgrade Qt for CVEs” item — sources: PySide6/Qt advisories; the locked 6.11.1 includes the relevant SVG fix and the QDom advisory does not match current imports.
- Live biometric phrase verification for local voice profiles — source: Suno Voices; current owner/source/scope/consent metadata is proportionate for an offline tool, while biometric storage would create a new privacy burden.

## Sources

### OSS, releases, and issues

- https://github.com/ace-step/ACE-Step-1.5
- https://github.com/ace-step/ACE-Step-1.5/releases/tag/v0.1.8
- https://github.com/ace-step/ACE-Step-1.5/pull/1035
- https://github.com/ace-step/ACE-Step-1.5/issues/1271
- https://github.com/multimodal-art-projection/YuE
- https://github.com/joeljuvel/YuE-UI
- https://github.com/tencent-ailab/SongGeneration
- https://github.com/ASLP-lab/DiffRhythm
- https://github.com/Soul-AILab/SoulX-Singer
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- https://github.com/RVC-Boss/GPT-SoVITS
- https://github.com/openvpi/DiffSinger
- https://github.com/nomadkaraoke/python-audio-separator
- https://github.com/Anjok07/ultimatevocalremovergui
- https://github.com/facebookresearch/audiocraft
- https://github.com/riffusion/riffusion
- https://github.com/fspecii/ace-step-ui
- https://github.com/Comfy-Org/ComfyUI-Manager/pull/2732
- https://github.com/invoke-ai/InvokeAI/releases/tag/v6.13.7
- https://github.com/bitwig/dawproject
- https://github.com/Ableton/link
- https://huggingface.co/stabilityai/stable-audio-open-1.0
- https://github.com/backblaze-labs/awesome-audio-generation
- https://github.com/ad-si/awesome-music-production

### Commercial and production workflows

- https://suno.com/release-notes
- https://help.suno.com/en/articles/7940161
- https://help.suno.com/en/articles/8128193
- https://help.suno.com/en/articles/11362369
- https://help.suno.com/en/articles/12702337
- https://help.udio.com/en/articles/11649525-sessions-udio-s-timeline-editing-view
- https://help.udio.com/en/articles/10754328-create-music-with-your-own-audio
- https://help.udio.com/en/articles/12683565-changes-associated-with-the-universal-music-group-umg-partnership
- https://help.moises.ai/hc/en-us/articles/21745204066076-Moises-AI-Studio-Your-All-in-One-AI-Music-Creation-Platform
- https://www.ableton.com/en/live-manual/12/stem-separation/
- https://www.ableton.com/en/live-manual/12/accessibility-and-keyboard-navigation/
- https://help.ableton.com/hc/en-us/articles/115001878844-Recovering-a-Set-manually-after-a-crash
- https://www.bitwig.com/userguide/latest/vst_plug-in_handling_and_options/
- https://support.apple.com/guide/logicpro/export-tracks-as-audio-files-lgcpb27f70f9/10.7/mac/11.6.1
- https://www.landr.com/online-audio-mastering/
- https://www.izotope.com/products/ozone-advanced
- https://www.lalal.ai/desktop-app/

### Standards, security, and dependencies

- https://github.com/huggingface/transformers/security
- https://huggingface.co/docs/transformers/models
- https://huggingface.co/docs/hub/security-pickle
- https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6
- https://github.com/advisories/GHSA-9356-575x-2w9m
- https://pip.pypa.io/en/stable/topics/secure-installs/
- https://www.itu.int/rec/R-REC-BS.1770-5-202311-I
- https://tech.ebu.ch/files/live/sites/tech/files/shared/r/r128.pdf
- https://www.w3.org/TR/WCAG22/
- https://doc.qt.io/qt-6/accessible-qwidget.html
- https://opensoundcontrol.stanford.edu/spec-1_0.html
- https://midi.org/midi-2-0-core-specification-collection
- https://id3.org/id3v2.4.0-frames
- https://xiph.org/vorbis/doc/v-comment.html
- https://www.rfc-editor.org/info/rfc7845/
- https://tech.ebu.ch/docs/tech/tech3285.pdf
- https://doc.qt.io/qtforpython-6/release_notes/pyside6_release_notes.html
- https://github.com/spatialaudio/python-sounddevice/releases
- https://pyqtgraph.readthedocs.io/en/pyqtgraph-0.14.0/
- https://psutil.readthedocs.io/stable/
- https://huggingface.co/docs/huggingface_hub/concepts/migration
- https://www.qt.io/blog/security-advisory-type-confusion-and-heap-buffer-overflow-vulnerability-in-qt-svg-marker-handling
- https://www.qt.io/blog/security-advisory-cve-2026-15037-xml-injection
- https://numpy.org/doc/stable/numpy_2_0_migration_guide.html
- https://librosa.org/doc/0.11.0/changelog.html
- https://docs.python.org/3.14/library/removed.html
- https://github.com/bastibe/python-soundfile/releases

### Research and community signal

- https://arxiv.org/abs/2506.00045
- https://arxiv.org/abs/2503.01183
- https://arxiv.org/abs/2412.17667
- https://arxiv.org/abs/2503.16669
- https://arxiv.org/abs/2501.10811
- https://www.reddit.com/r/udiomusic/comments/1fn90km/stuck_in_generation_queue/
- https://www.reddit.com/r/udiomusic/comments/1rm7nb3/unusable_mess/
- https://www.reddit.com/r/udiomusic/comments/1ok8rp8/10_hoursday_for_15_months_300_songs_now_locked_we/
- https://www.reddit.com/r/SunoAI/comments/1stwgii/paying_for_broken_generations/
- https://www.reddit.com/r/audioengineering/comments/1gbcyfn/follow_up_to_ai_stem_splitter_post/
- https://news.ycombinator.com/item?id=40559507
- https://stackoverflow.com/questions/72754426/gui-freezing-in-python-pyside6
- https://www.kvraudio.com/forum/viewtopic.php?t=576467
- https://forum.audacityteam.org/t/crash-recovery-has-erased-my-project-any-way-back/64261

## Open Questions

None block prioritization. Model inference, packaged cross-platform behavior, and DAW imports require implementation-time hardware/application validation rather than more desk research.
