# Blocked Roadmap Items

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
