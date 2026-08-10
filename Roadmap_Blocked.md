# Blocked Roadmap Items

## Model hub — Add DirectML backend support for AMD GPUs on Windows

Blocked on 2026-08-03 because the roadmap item is explicitly invalidated: the available
`torch-directml` package requires the stale PyTorch 2.4.1 runtime, while the repository's
security floor is newer. Resume only when a supported Windows ROCm profile reaches the required
runtime floor and can be hash-locked without loading untrusted checkpoints.

## Model Hub — Add MPS backend support for Apple Silicon

Blocked on 2026-08-03 because the roadmap explicitly sequences this profile after the NumPy →
SciPy → torch → Transformers upgrade train, which remains blocked until the ABI-matched
`torchaudio==2.13.0` artifacts are published. Do not build an MPS profile against an unverified
runtime floor; resume it after the upgrade train is available and hash-locked.

## P1 — Run the NumPy → SciPy → torch → Transformers upgrade train

Blocked on 2026-08-02 because the required `torchaudio==2.13.0` artifacts are not published.
The independent core-runtime portion is complete: `requirements-lock.txt` now pins NumPy 2.4.6
and SciPy 1.18.0, with NumPy constrained below 2.5 for Numba 0.66.0.

Evidence checked against upstream package sources:

- PyPI lists `torchaudio` through 2.11.0 and has no 2.12.0 or 2.13.0 release:
  https://pypi.org/project/torchaudio/
- The official `pytorch/audio` repository has a v2.11.0 tag and no v2.12.0 or v2.13.0 tag:
  https://github.com/pytorch/audio/tags
- The official PyTorch CPU and CUDA wheel indexes contain no torchaudio 2.13 wheel for any
  of the configured profile indexes: https://download.pytorch.org/whl/

Do not substitute torchaudio 2.11 for torch 2.13 or fabricate a lock entry; the versions must
remain ABI-matched and hash-verifiable. Resume the profile portion when an upstream 2.13 release
is available, then update all five profile locks and rerun the full upgrade compatibility suite.

## P3 — Reconsider the information architecture around assistive work

Blocked on 2026-08-03 because acceptance requires a deliberate product-positioning decision about
whether assistive modules should lead navigation and README messaging. This cannot be inferred
from the codebase without human judgment about product direction and audience positioning.

The proposed adoption would reorder `ui/main_window.py`, update README/onboarding copy, present
generation as one capability rather than the headline, and explicitly state the local ownership
guarantee. Preserve the evidence and alternatives in the active roadmap when this decision is
revisited.

## Export — Validate DAWproject against official schemas and golden DAWs

Blocked on 2026-08-03 because the remaining acceptance gate requires installed, licensed target
applications and an operator-controlled import check in Bitwig, Studio One, and Cubase. Slunder's
Project Manager and Mixer export paths now use the existing structural validator and collision-safe
media packaging, but this environment cannot prove official-schema acceptance or target-DAW import
behavior without those external applications and their approved test fixtures.
## Creative Workflows — Add Ableton Link sync

Blocked on 2026-08-03 because the roadmap requires transport timing to be testable first. The
current transport and OSC surfaces expose command dispatch only; they do not provide a measured
tempo clock, start/stop phase contract, jitter budget, or headless timing fixture. Resume after
that timing contract exists and can be verified without driving the user's desktop or an external
DAW session.

## Creative Workflows — Add ACE-Step LoRA dataset validation/training

Blocked on 2026-08-03 because the versioned inference adapter is stable, but this repository has no
verified ACE-Step training runtime, tensor-preprocessing contract, or safe LoRA adapter-loading
contract. The upstream workflow requires a separate training stack and substantial local GPU
resources; adding an unpinned trainer or a local HTTP dependency would weaken the project's
offline, hash-locked release boundary. Resume after a versioned training runtime/profile and
adapter format are explicitly selected and can be tested with local fixtures.

## Creative Workflows — Add ACE-Step Vocal-to-BGM, extract/complete, and non-destructive layers

Blocked on 2026-08-03 because the pinned `ACE-Step/acestep-v15-xl-turbo-diffusers` contract in
this repository exposes only text-to-music, cover, repaint, and application-level extend. The
upstream advanced extract/lego/complete modes are not capabilities of that XL Turbo model, so
shipping UI calls now would claim support the active model cannot provide. Resume after a
compatible model revision is selected, pinned, cached, and covered by the shared source-task
contract and local fixtures.

## Creative Workflows — Add a process-isolated engine plug-in SDK

Blocked on 2026-08-03 because the item is explicitly gated on a permission model, crash
containment contract, and the security decision to exclude manifest signing. Those boundaries
require deliberate security/product judgment before an extension surface can be implemented
safely; do not add a community plug-in loader until they are selected and testable.

## P3 — Confirm the visual findings on a real display at 125% scaling

Blocked on 2026-08-03 because the safe isolation harness can prove that the app window is on the
private non-input desktop and exact fourth virtual display, but its screenshot command captures
only the active display desktop and cannot expose pixels from that private desktop. Switching
desktops would violate the repository's invisible-testing contract. Resume when a private-desktop
capture API is available or an operator supplies an equivalent safe visual fixture.
