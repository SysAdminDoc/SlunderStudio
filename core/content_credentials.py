"""Optional C2PA Content Credentials for exported audio.

The export path remains unsigned unless a user explicitly enables Content
Credentials and supplies a C2PA claim-signing certificate and private key.
Credentials are read from the configured paths for one export and are never
copied into provenance sidecars or application logs.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.provenance import read_provenance_sidecar
from core.version import APP_NAME, APP_VERSION


C2PA_SPEC_VERSION = "2.4"
C2PA_SUPPORTED_FORMATS = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
}
_DIGITAL_CREATION = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCreation"
)
_TRAINED_ALGORITHMIC_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
)
_SIGNATURE_FAILURE_CODES = {
    "claimSignature.mismatch",
    "claimSignature.missing",
    "claimSignature.invalid",
}


class ContentCredentialsError(RuntimeError):
    """Raised when an explicitly requested C2PA export cannot complete."""


class ContentCredentialsUnsupportedFormat(ContentCredentialsError):
    """Raised when the installed C2PA binding cannot embed this format."""


@dataclass(frozen=True)
class C2PAConfig:
    """User-supplied C2PA signer locations.

    The paths point to a PEM certificate chain and an unencrypted PEM private
    key.  ``timestamp_url`` is optional; an empty value keeps the local export
    offline and omits an RFC 3161 timestamp.
    """

    certificate_path: str = ""
    private_key_path: str = ""
    timestamp_url: str = ""


@dataclass(frozen=True)
class C2PAResult:
    """Safe, sidecar-ready result of embedding and reading a manifest back."""

    status: str
    specification: str
    format: str
    provenance_digest: str
    validation_state: str
    validation_codes: tuple[str, ...]
    manifest_labels: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "specification": self.specification,
            "format": self.format,
            "provenance_digest": self.provenance_digest,
            "validation_state": self.validation_state,
            "validation_codes": list(self.validation_codes),
            "manifest_labels": list(self.manifest_labels),
        }


def configured_c2pa_config() -> C2PAConfig:
    """Read C2PA credential paths without importing Settings at module load."""
    from core.settings import Settings

    settings = Settings()
    return C2PAConfig(
        certificate_path=str(
            settings.get("general.c2pa_certificate_path", "") or ""
        ).strip(),
        private_key_path=str(
            settings.get("general.c2pa_private_key_path", "") or ""
        ).strip(),
        timestamp_url=str(
            settings.get("general.c2pa_timestamp_url", "") or ""
        ).strip(),
    )


def c2pa_format_for_export(export_format: str) -> str:
    """Return the C2PA MIME type or fail before an unsigned export is made."""
    normalized = str(export_format or "").strip().lower().lstrip(".")
    mime = C2PA_SUPPORTED_FORMATS.get(normalized)
    if mime:
        return mime
    supported = ", ".join(fmt.upper() for fmt in C2PA_SUPPORTED_FORMATS)
    raise ContentCredentialsUnsupportedFormat(
        f"C2PA Content Credentials currently support {supported}; "
        f"{normalized.upper() or 'this'} export is not supported by the installed binding."
    )


def validate_c2pa_config(config: C2PAConfig) -> tuple[Path, Path]:
    """Validate signer locations without exposing their contents."""
    certificate = _required_file(config.certificate_path, "certificate")
    private_key = _required_file(config.private_key_path, "private key")
    if config.timestamp_url and not config.timestamp_url.startswith(
        ("http://", "https://")
    ):
        raise ContentCredentialsError(
            "C2PA timestamp URL must start with http:// or https://."
        )
    return certificate, private_key


def _required_file(value: str, label: str) -> Path:
    if not value:
        raise ContentCredentialsError(
            f"C2PA export is enabled but the signer {label} path is not configured."
        )
    path = Path(value).expanduser()
    if not path.is_file():
        raise ContentCredentialsError(
            f"C2PA signer {label} file was not found: {path}"
        )
    return path


def _stable_provenance_projection(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Select sidecar fields that stay stable when the signed hash changes."""
    fields = (
        "schema_version",
        "app_version",
        "module",
        "operation",
        "output_kind",
        "model",
        "seed",
        "prompt",
        "lyrics",
        "parameters",
        "source_asset_ids",
        "source_paths",
        "source_hashes",
        "export_format",
        "rerender_key",
    )
    projection = {
        field: sidecar.get(field)
        for field in fields
        if field in sidecar
    }

    extra = dict(sidecar.get("extra") or {})
    # The embedded assertion must not include its own result or the changing
    # artifact hash.  Delivery writer/tags remain useful stable assertions.
    extra.pop("c2pa", None)
    delivery = dict(extra.get("delivery") or {})
    delivery.pop("verification", None)
    if delivery:
        extra["delivery"] = delivery
    else:
        extra.pop("delivery", None)
    if extra:
        projection["extra"] = extra
    return projection


def provenance_digest(sidecar: dict[str, Any]) -> str:
    """Hash the stable sidecar projection used by the embedded assertion."""
    encoded = json.dumps(
        _stable_provenance_projection(sidecar),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_c2pa_manifest(
    sidecar: dict[str, Any],
    *,
    export_format: str,
) -> tuple[dict[str, Any], str]:
    """Build a C2PA 2.4 manifest and the sidecar identity it covers."""
    mime = c2pa_format_for_export(export_format)
    digest = provenance_digest(sidecar)
    model = dict(sidecar.get("model") or {})
    model_id = str(model.get("id") or "").strip()
    model_name = str(model.get("name") or model_id).strip()
    source_type = _TRAINED_ALGORITHMIC_MEDIA if model_id else _DIGITAL_CREATION

    assertions: list[dict[str, Any]] = [
        {
            "label": "c2pa.actions",
            "data": {
                "actions": [
                    {
                        "action": "c2pa.created",
                        "digitalSourceType": source_type,
                    }
                ]
            },
        },
    ]
    if model_id:
        model_metadata = {
            key: model[key]
            for key in (
                "revision",
                "resolved_revision",
                "hash",
                "source",
                "license",
            )
            if model.get(key)
        }
        assertions.append(
            {
                "label": "c2pa.ai-disclosure",
                "data": {
                    "modelType": "c2pa.types.model.huggingface.transformers",
                    "modelName": model_name,
                    "modelIdentifier": model_id,
                    "contentProfile": {
                        "humanOversightLevel": "prompt_guided",
                    },
                    **({"metadata": model_metadata} if model_metadata else {}),
                },
            }
        )

    assertions.append(
        {
            "label": "org.slunderstudio.provenance",
            "data": {
                "schemaVersion": 1,
                "provenanceDigest": digest,
                "sidecar": _stable_provenance_projection(sidecar),
            },
        }
    )
    manifest = {
        "claim_generator_info": [
            {"name": APP_NAME, "version": APP_VERSION},
        ],
        "format": mime,
        "title": str((sidecar.get("artifact") or {}).get("name") or "Audio export"),
        "assertions": assertions,
    }
    return manifest, digest


def embed_c2pa_manifest(
    artifact_path: str | Path,
    *,
    config: C2PAConfig | None = None,
) -> C2PAResult:
    """Sign an existing export, replace it atomically, and read it back."""
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise ContentCredentialsError(
            f"Cannot add C2PA Content Credentials to missing export: {artifact}"
        )
    export_format = artifact.suffix.lstrip(".").lower()
    c2pa_format_for_export(export_format)
    signer_config = config or configured_c2pa_config()
    certificate_path, private_key_path = validate_c2pa_config(signer_config)
    sidecar = read_provenance_sidecar(artifact)
    if not sidecar:
        raise ContentCredentialsError(
            "C2PA export requires the adjacent provenance sidecar."
        )
    manifest, digest = build_c2pa_manifest(
        sidecar,
        export_format=export_format,
    )

    temporary_path = _temporary_sibling(artifact)
    try:
        _sign_file(
            artifact,
            temporary_path,
            manifest,
            certificate_path=certificate_path,
            private_key_path=private_key_path,
            timestamp_url=signer_config.timestamp_url,
        )
        os.replace(temporary_path, artifact)
    except ContentCredentialsError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize native errors
        raise ContentCredentialsError(
            f"C2PA signing failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            Path(temporary_path).unlink(missing_ok=True)
        except OSError:
            pass

    return _verify_c2pa_round_trip(artifact, digest, export_format)


def _temporary_sibling(artifact: Path) -> str:
    fd, path = tempfile.mkstemp(
        prefix=f".{artifact.stem}.c2pa-",
        suffix=artifact.suffix,
        dir=str(artifact.parent),
    )
    os.close(fd)
    return path


def _sign_file(
    source: Path,
    destination: str,
    manifest: dict[str, Any],
    *,
    certificate_path: Path,
    private_key_path: Path,
    timestamp_url: str,
) -> None:
    try:
        import c2pa
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError as exc:
        raise ContentCredentialsError(
            "C2PA export requires the locked c2pa-python dependency."
        ) from exc

    try:
        certificate_data = certificate_path.read_bytes()
        private_key_data = private_key_path.read_bytes()
        private_key = serialization.load_pem_private_key(
            private_key_data,
            password=None,
        )
        if not isinstance(private_key, ec.EllipticCurvePrivateKey):
            raise ContentCredentialsError(
                "C2PA currently requires an EC private key for ES256 signing."
            )
        if private_key.curve.name != "secp256r1":
            raise ContentCredentialsError(
                "C2PA ES256 signing requires a P-256 (secp256r1) private key."
            )
        certificate = x509.load_pem_x509_certificate(certificate_data)
        if certificate.public_key().public_numbers() != private_key.public_key().public_numbers():
            raise ContentCredentialsError(
                "C2PA certificate and private key do not describe the same signer."
            )
        certificate_text = certificate_data.decode("utf-8")
    except ContentCredentialsError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise ContentCredentialsError(
            f"C2PA signer credentials could not be loaded: {type(exc).__name__}."
        ) from exc

    def callback(data: bytes) -> bytes:
        return private_key.sign(data, ec.ECDSA(hashes.SHA256()))

    context_settings = {
        "verify": {
            "remote_manifest_fetch": False,
            "ocsp_fetch": False,
        }
    }
    timestamp = timestamp_url or None
    with c2pa.Context.from_dict(context_settings) as context:
        with c2pa.Signer.from_callback(
            callback,
            c2pa.C2paSigningAlg.ES256,
            certificate_text,
            timestamp,
        ) as signer:
            with c2pa.Builder(manifest, context) as builder:
                builder.sign_file(source, destination, signer)


def _verify_c2pa_round_trip(
    artifact: Path,
    digest: str,
    export_format: str,
) -> C2PAResult:
    try:
        import c2pa
    except ImportError as exc:
        raise ContentCredentialsError(
            "C2PA export requires the locked c2pa-python dependency."
        ) from exc

    try:
        context_settings = {
            "verify": {
                "remote_manifest_fetch": False,
                "ocsp_fetch": False,
            }
        }
        with c2pa.Context.from_dict(context_settings) as context:
            with c2pa.Reader(artifact, context=context) as reader:
                payload = json.loads(reader.json())
                validation_state = str(reader.get_validation_state() or "")
    except Exception as exc:  # noqa: BLE001 - normalize native errors
        raise ContentCredentialsError(
            f"C2PA round-trip verification failed: {type(exc).__name__}: {exc}"
        ) from exc

    statuses = payload.get("validation_status") or []
    codes = tuple(
        str(item.get("code"))
        for item in statuses
        if isinstance(item, dict) and item.get("code")
    )
    failures = sorted(set(codes).intersection(_SIGNATURE_FAILURE_CODES))
    if failures:
        raise ContentCredentialsError(
            "C2PA round-trip reported invalid claim signature: "
            + ", ".join(failures)
        )

    active_id = payload.get("active_manifest")
    active = (payload.get("manifests") or {}).get(active_id, {})
    assertions = active.get("assertions") or []
    labels = tuple(
        str(assertion.get("label"))
        for assertion in assertions
        if isinstance(assertion, dict) and assertion.get("label")
    )
    provenance_assertion = next(
        (
            assertion
            for assertion in assertions
            if isinstance(assertion, dict)
            and assertion.get("label") == "org.slunderstudio.provenance"
        ),
        None,
    )
    embedded_digest = (
        ((provenance_assertion or {}).get("data") or {}).get("provenanceDigest")
    )
    if embedded_digest != digest:
        raise ContentCredentialsError(
            "C2PA round-trip provenance does not match the export sidecar."
        )
    return C2PAResult(
        status="embedded",
        specification=C2PA_SPEC_VERSION,
        format=export_format,
        provenance_digest=digest,
        validation_state=validation_state,
        validation_codes=codes,
        manifest_labels=labels,
    )
