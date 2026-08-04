"""OMS signature discovery and verification for downloaded model caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.model_manager import ModelInfo


SIGNATURE_UNSIGNED = "unsigned"
SIGNATURE_VERIFIED = "verified"
SIGNATURE_MISSING = "missing"
SIGNATURE_INVALID = "invalid"
SIGNATURE_UNTRUSTED = "untrusted"
SIGNATURE_UNAVAILABLE = "unavailable"

_DISCOVERY_NAMES = (
    "model.sig",
    "model.oms.sig",
    "model.sigstore.json",
    "model.oms.json",
    "oms.sig",
)


@dataclass(frozen=True)
class SignatureVerification:
    """The recorded result of checking an optional OMS signature."""

    status: str
    reason: str
    signature_path: str = ""
    signer_identity: str = ""
    oidc_issuer: str = ""
    verifier: str = ""

    @property
    def is_acceptable(self) -> bool:
        """Return whether the model may proceed past the signature gate."""
        return self.status in {SIGNATURE_UNSIGNED, SIGNATURE_VERIFIED}

    def as_manifest_fields(self) -> dict[str, str]:
        """Return the stable fields persisted in the download marker."""
        return {
            "signature_status": self.status,
            "signature_reason": self.reason,
            "signature_path": self.signature_path,
            "signature_identity": self.signer_identity,
            "signature_oidc_issuer": self.oidc_issuer,
            "signature_verifier": self.verifier,
        }


def _configured_signature_path(cache_path: Path, info: "ModelInfo") -> tuple[Path | None, bool]:
    """Resolve a configured signature path without allowing cache escape."""
    configured = str(getattr(info, "signature_path", "") or "").strip()
    if not configured:
        return None, False

    candidate = Path(configured)
    if not candidate.is_absolute():
        candidate = cache_path / candidate
    try:
        resolved = candidate.resolve(strict=False)
        root = cache_path.resolve(strict=False)
        if not resolved.is_relative_to(root):
            return None, True
    except OSError:
        return None, True
    return candidate, True


def find_oms_signature(cache_path: str | Path, info: "ModelInfo") -> tuple[Path | None, bool]:
    """Find an OMS signature and report whether the registry explicitly expects one.

    The registry may name a signature when a publisher has a non-default filename. When
    it does not, the small set of standard detached names is searched at the cache root.
    No arbitrary recursive search is performed because a nested signature could otherwise
    be mistaken for the signature of the model being activated.
    """
    root = Path(cache_path)
    configured, expected = _configured_signature_path(root, info)
    if expected:
        if configured is None or not configured.is_file():
            return None, True
        return configured, True

    for name in _DISCOVERY_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate, False
    return None, False


def _signer_metadata(info: "ModelInfo") -> tuple[str, str, str]:
    """Return the configured signer identity, issuer, and verifier label."""
    identity = str(getattr(info, "signature_identity", "") or "")
    issuer = str(getattr(info, "signature_oidc_issuer", "") or "")
    public_key = str(getattr(info, "signature_public_key", "") or "")
    certificate_chain = getattr(info, "signature_certificate_chain", ()) or ()
    if public_key:
        return identity, issuer, "OMS public key"
    if certificate_chain:
        return identity, issuer, "OMS certificate chain"
    if identity and issuer:
        return identity, issuer, "Sigstore identity"
    return identity, issuer, ""


def verify_oms_signature(
    cache_path: str | Path,
    info: "ModelInfo",
) -> SignatureVerification:
    """Verify a publisher-provided OMS signature against the complete model directory.

    Unsigned caches remain loadable but are explicitly marked unsigned. A discovered
    signature is fail-closed: the model-signing verifier, a signer policy, and the
    signed file manifest must all succeed before activation can continue.
    """
    root = Path(cache_path)
    signature_path, expected = find_oms_signature(root, info)
    identity, issuer, verifier = _signer_metadata(info)
    if signature_path is None:
        if expected:
            return SignatureVerification(
                status=SIGNATURE_MISSING,
                reason="The registry expects an OMS signature, but the detached signature is missing.",
                signer_identity=identity,
                oidc_issuer=issuer,
                verifier=verifier,
            )
        return SignatureVerification(
            status=SIGNATURE_UNSIGNED,
            reason="No OMS signature was published with this model revision.",
        )

    if not verifier:
        return SignatureVerification(
            status=SIGNATURE_UNTRUSTED,
            reason=(
                "An OMS signature is present, but no trusted signer policy is configured "
                "for this model revision."
            ),
            signature_path=str(signature_path),
            signer_identity=identity,
            oidc_issuer=issuer,
        )

    try:
        from model_signing import verifying

        config = verifying.Config()
        public_key = str(getattr(info, "signature_public_key", "") or "")
        certificate_chain = getattr(info, "signature_certificate_chain", ()) or ()
        if public_key:
            config.use_elliptic_key_verifier(public_key=public_key)
        elif certificate_chain:
            config.use_certificate_verifier(certificate_chain=certificate_chain)
        else:
            config.use_sigstore_verifier(
                identity=identity,
                oidc_issuer=issuer,
            )
        # The detached signature is kept beside the cache and the app's completion
        # marker is added after signing. Compare only the signed resource set.
        config.set_ignore_unsigned_files(True)
        config.verify(root, signature_path)
    except ImportError as exc:
        return SignatureVerification(
            status=SIGNATURE_UNAVAILABLE,
            reason=(
                "The OMS verifier is not installed; install the locked model-signing "
                "runtime before activating this signed model."
            ),
            signature_path=str(signature_path),
            signer_identity=identity,
            oidc_issuer=issuer,
            verifier=verifier,
        )
    except Exception as exc:
        return SignatureVerification(
            status=SIGNATURE_INVALID,
            reason=f"OMS signature verification failed: {type(exc).__name__}: {exc}",
            signature_path=str(signature_path),
            signer_identity=identity,
            oidc_issuer=issuer,
            verifier=verifier,
        )

    return SignatureVerification(
        status=SIGNATURE_VERIFIED,
        reason="OMS signature verified against the local model file manifest.",
        signature_path=str(signature_path),
        signer_identity=identity,
        oidc_issuer=issuer,
        verifier=verifier,
    )


def signature_metadata_label(metadata: dict[str, Any]) -> str:
    """Format a concise, user-facing OMS state for Model Hub and diagnostics."""
    status = str(metadata.get("signature_status", SIGNATURE_UNSIGNED) or SIGNATURE_UNSIGNED)
    reason = str(metadata.get("signature_reason", "") or "")
    identity = str(metadata.get("signature_identity", "") or "")
    if status == SIGNATURE_VERIFIED:
        return f"OMS signature: verified{f' ({identity})' if identity else ''}"
    if status == SIGNATURE_UNSIGNED:
        return "OMS signature: unsigned"
    if status == SIGNATURE_MISSING:
        return "OMS signature: missing"
    if status == SIGNATURE_UNTRUSTED:
        return "OMS signature: signer policy missing"
    if status == SIGNATURE_UNAVAILABLE:
        return "OMS signature: verifier unavailable"
    if status == SIGNATURE_INVALID:
        return "OMS signature: invalid"
    return f"OMS signature: {status}{f' - {reason}' if reason else ''}"
