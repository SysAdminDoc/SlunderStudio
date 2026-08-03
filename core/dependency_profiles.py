"""
Reproducible, hash-locked optional AI dependency profiles.

Runtime code never installs packages. Installation is an explicit operator
action through tools/dependency_profiles.py and is forced offline once a
wheelhouse has been prepared and verified.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import unquote


PROFILE_SCHEMA_VERSION = 1
HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})(?:\s|$)", re.IGNORECASE)
REQUIREMENT_RE = re.compile(
    r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)(?:\s+(.+))?$"
)
DENYLISTED_PROFILE_PACKAGES = frozenset({"kernels"})


class DependencyProfileError(RuntimeError):
    """Raised when a profile, lock, wheelhouse, or runtime is unsafe."""


@dataclass(frozen=True)
class LockEntry:
    name: str
    version: str
    hashes: tuple[str, ...]

    @property
    def normalized_name(self) -> str:
        return normalize_name(self.name)


@dataclass(frozen=True)
class DependencyProfile:
    name: str
    enabled: bool
    backend: str
    system: str
    machines: tuple[str, ...]
    python_tag: str
    lock_path: Optional[Path]
    index_url: str
    extra_index_url: str
    roots: Mapping[str, str]
    smoke_imports: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class WheelArtifact:
    entry: LockEntry
    path: Path
    sha256: str


def assert_profile_dependency_policy(
    entries: Mapping[str, LockEntry],
    profile_name: str = "profile",
) -> None:
    """Reject packages that would re-enable known unsafe Transformers paths."""
    denied = sorted(
        DENYLISTED_PROFILE_PACKAGES.intersection(
            normalize_name(name) for name in entries
        )
    )
    if denied:
        raise DependencyProfileError(
            f"Dependency profile {profile_name!r} contains deny-listed package(s): "
            f"{', '.join(denied)}"
        )


def assert_no_denylisted_packages_installed() -> None:
    """Fail closed if a deny-listed distribution is present in the runtime."""
    for package in sorted(DENYLISTED_PROFILE_PACKAGES):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise DependencyProfileError(
            f"Deny-listed package {package!r} is installed ({version}); "
            "remove it before starting Slunder Studio."
        )


def resource_root() -> Path:
    """Return the source or PyInstaller data root."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parents[1]


def profiles_file() -> Path:
    return resource_root() / "requirements" / "profiles.json"


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name).strip()).lower()


def _version_numbers(version: str) -> tuple[int, ...]:
    public = str(version).split("+", 1)[0]
    match = re.match(r"^(\d+(?:\.\d+)*)", public)
    if not match:
        raise DependencyProfileError(f"Unsupported version format: {version!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def version_at_least(version: str, minimum: str) -> bool:
    actual = _version_numbers(version)
    floor = _version_numbers(minimum)
    width = max(len(actual), len(floor))
    return actual + (0,) * (width - len(actual)) >= floor + (0,) * (width - len(floor))


def version_less_than(version: str, maximum: str) -> bool:
    actual = _version_numbers(version)
    ceiling = _version_numbers(maximum)
    width = max(len(actual), len(ceiling))
    return actual + (0,) * (width - len(actual)) < ceiling + (0,) * (width - len(ceiling))


def load_registry(path: Optional[str | Path] = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else profiles_file()
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyProfileError(
            f"Cannot read dependency profile registry: {registry_path}"
        ) from exc
    if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise DependencyProfileError(
            f"Unsupported dependency profile schema: {payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("profiles"), dict):
        raise DependencyProfileError("Dependency profile registry has no profiles map")
    if not isinstance(payload.get("advisory_floors"), dict):
        raise DependencyProfileError("Dependency profile registry has no advisory floors")
    payload["_path"] = registry_path.resolve()
    return payload


def get_profile(
    name: str,
    *,
    registry: Optional[Mapping[str, Any]] = None,
    require_enabled: bool = True,
) -> DependencyProfile:
    data = dict(registry or load_registry())
    raw_profiles = data["profiles"]
    if name not in raw_profiles:
        options = ", ".join(sorted(raw_profiles))
        raise DependencyProfileError(f"Unknown dependency profile {name!r}; choose: {options}")
    raw = raw_profiles[name]
    enabled = bool(raw.get("enabled"))
    reason = str(raw.get("reason", "")).strip()
    if require_enabled and not enabled:
        raise DependencyProfileError(
            f"Dependency profile {name!r} is disabled: {reason or 'no safe runtime is available'}"
        )

    registry_path = Path(data.get("_path") or profiles_file())
    lock_value = raw.get("lock")
    lock_path = (registry_path.parent / str(lock_value)).resolve() if lock_value else None
    roots = raw.get("roots") or {}
    if enabled and (not roots or not lock_path):
        raise DependencyProfileError(f"Enabled profile {name!r} is incomplete")
    for package, version in roots.items():
        if not package or not version or any(token in str(version) for token in "<>=,*"):
            raise DependencyProfileError(
                f"Profile {name!r} root {package!r} is not exactly pinned"
            )

    return DependencyProfile(
        name=name,
        enabled=enabled,
        backend=str(raw.get("backend", "")),
        system=str(raw.get("system", "")),
        machines=tuple(str(item) for item in raw.get("machines", [])),
        python_tag=str(raw.get("python_tag", "")),
        lock_path=lock_path,
        index_url=str(raw.get("index_url", "")),
        extra_index_url=str(raw.get("extra_index_url", "")),
        roots={normalize_name(key): str(value) for key, value in roots.items()},
        smoke_imports=tuple(str(item) for item in raw.get("smoke_imports", [])),
        reason=reason,
    )


def parse_lock(path: str | Path) -> dict[str, LockEntry]:
    lock_path = Path(path)
    try:
        raw_lines = lock_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DependencyProfileError(f"Cannot read dependency lock: {lock_path}") from exc

    logical_lines: list[str] = []
    pending = ""
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--index-url") or line.startswith("--extra-index-url"):
            continue
        if line.startswith(("-e ", "--editable", "http://", "https://", "git+")):
            raise DependencyProfileError(f"Executable or direct dependency is forbidden: {line}")
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        logical_lines.append(pending)
        pending = ""
    if pending:
        raise DependencyProfileError("Dependency lock ends with an incomplete continuation")

    entries: dict[str, LockEntry] = {}
    for line in logical_lines:
        match = REQUIREMENT_RE.fullmatch(line)
        if not match:
            raise DependencyProfileError(f"Dependency lock entry is not exactly pinned: {line}")
        name, version, options = match.groups()
        hashes = tuple(sorted(set(value.lower() for value in HASH_RE.findall(options or ""))))
        if not hashes:
            raise DependencyProfileError(f"Dependency lock entry has no SHA-256 hash: {line}")
        remaining = HASH_RE.sub("", options or "").strip()
        if remaining:
            raise DependencyProfileError(f"Unsupported dependency lock option: {remaining}")
        normalized = normalize_name(name)
        if normalized in entries:
            raise DependencyProfileError(f"Duplicate dependency lock entry: {name}")
        entries[normalized] = LockEntry(name=name, version=version, hashes=hashes)
    if not entries:
        raise DependencyProfileError(f"Dependency lock is empty: {lock_path}")
    return entries


def validate_profile(
    name: str,
    *,
    registry_path: Optional[str | Path] = None,
    check_host: bool = False,
) -> tuple[DependencyProfile, dict[str, LockEntry]]:
    registry = load_registry(registry_path)
    profile = get_profile(name, registry=registry)
    if check_host:
        assert_host_compatible(profile)
    if profile.lock_path is None:
        raise DependencyProfileError(f"Profile {name!r} has no lock")
    entries = parse_lock(profile.lock_path)
    assert_profile_dependency_policy(entries, profile.name)

    for package, expected_version in profile.roots.items():
        entry = entries.get(package)
        if entry is None:
            raise DependencyProfileError(
                f"Profile {name!r} lock is missing root package {package}"
            )
        if entry.version != expected_version:
            raise DependencyProfileError(
                f"Profile {name!r} lock drift for {package}: "
                f"expected {expected_version}, got {entry.version}"
            )

    floors = registry["advisory_floors"]
    for package, advisory in floors.items():
        normalized = normalize_name(package)
        entry = entries.get(normalized)
        if entry is None:
            continue
        minimum = str(advisory.get("minimum", ""))
        if not minimum or not version_at_least(entry.version, minimum):
            raise DependencyProfileError(
                f"{entry.name} {entry.version} is below security floor {minimum} "
                f"({advisory.get('advisory', 'advisory')})"
            )
    for package, constraint in registry.get("compatibility_ceilings", {}).items():
        normalized = normalize_name(package)
        entry = entries.get(normalized)
        if entry is None:
            continue
        maximum = str(constraint.get("maximum_exclusive", ""))
        if not maximum or not version_less_than(entry.version, maximum):
            raise DependencyProfileError(
                f"{entry.name} {entry.version} is incompatible with "
                f"{constraint.get('source', 'the selected runtime')}; "
                f"required <{maximum}"
            )
    return profile, entries


def validate_profile_registry_security(
    registry_path: Optional[str | Path] = None,
) -> None:
    """Validate every profile lock and the active interpreter deny-list policy."""
    registry = load_registry(registry_path)
    for name in sorted(registry["profiles"]):
        profile = get_profile(name, registry=registry, require_enabled=False)
        if profile.lock_path is None:
            continue
        entries = parse_lock(profile.lock_path)
        assert_profile_dependency_policy(entries, profile.name)
    assert_no_denylisted_packages_installed()


def assert_host_compatible(profile: DependencyProfile) -> None:
    system = platform.system()
    machine = platform.machine()
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    allowed_machines = {item.casefold() for item in profile.machines}
    if system != profile.system:
        raise DependencyProfileError(
            f"Profile {profile.name!r} targets {profile.system}, not {system}"
        )
    if allowed_machines and machine.casefold() not in allowed_machines:
        raise DependencyProfileError(
            f"Profile {profile.name!r} does not support machine {machine}"
        )
    if python_tag != profile.python_tag:
        raise DependencyProfileError(
            f"Profile {profile.name!r} requires {profile.python_tag}, not {python_tag}"
        )


def lock_sha256(path: str | Path) -> str:
    return sha256_file(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_identity(path: Path) -> tuple[str, str]:
    if path.suffix.lower() != ".whl":
        raise DependencyProfileError(f"Only wheel archives are accepted: {path.name}")
    parts = unquote(path.name[:-4]).split("-")
    if len(parts) < 5:
        raise DependencyProfileError(f"Invalid wheel filename: {path.name}")
    return normalize_name(parts[0]), parts[1]


def verify_wheelhouse(
    entries: Mapping[str, LockEntry],
    wheelhouse: str | Path,
) -> dict[str, WheelArtifact]:
    root = Path(wheelhouse).expanduser().resolve()
    if not root.is_dir():
        raise DependencyProfileError(f"Wheelhouse does not exist: {root}")

    candidates: dict[tuple[str, str], list[tuple[Path, str]]] = {}
    for path in sorted(root.glob("*.whl")):
        if path.is_symlink() or not path.is_file():
            raise DependencyProfileError(f"Wheelhouse contains a non-regular wheel: {path.name}")
        package, version = _wheel_identity(path)
        digest = sha256_file(path)
        candidates.setdefault((package, version), []).append((path, digest))

    artifacts: dict[str, WheelArtifact] = {}
    for normalized, entry in entries.items():
        matches = candidates.get((normalized, entry.version), [])
        if not matches:
            raise DependencyProfileError(
                f"Wheelhouse is missing {entry.name}=={entry.version}"
            )
        invalid = [(path, digest) for path, digest in matches if digest not in entry.hashes]
        if invalid:
            names = ", ".join(path.name for path, _ in invalid)
            raise DependencyProfileError(
                f"Wheelhouse hash mismatch for {entry.name}=={entry.version}: {names}"
            )
        if len(matches) != 1:
            raise DependencyProfileError(
                f"Wheelhouse contains multiple candidates for {entry.name}=={entry.version}"
            )
        path, digest = matches[0]
        artifacts[normalized] = WheelArtifact(entry=entry, path=path, sha256=digest)
    return artifacts


def offline_install_command(
    profile: DependencyProfile,
    wheelhouse: str | Path,
    *,
    python_executable: Optional[str] = None,
) -> list[str]:
    if profile.lock_path is None:
        raise DependencyProfileError(f"Profile {profile.name!r} has no lock")
    assert_profile_dependency_policy(parse_lock(profile.lock_path), profile.name)
    return [
        python_executable or sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(Path(wheelhouse).expanduser().resolve()),
        "--only-binary=:all:",
        "--require-hashes",
        "-r",
        str(profile.lock_path),
    ]


def download_command(
    profile: DependencyProfile,
    wheelhouse: str | Path,
    *,
    python_executable: Optional[str] = None,
) -> list[str]:
    if profile.lock_path is None:
        raise DependencyProfileError(f"Profile {profile.name!r} has no lock")
    assert_profile_dependency_policy(parse_lock(profile.lock_path), profile.name)
    return [
        python_executable or sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--require-hashes",
        "-r",
        str(profile.lock_path),
        "--dest",
        str(Path(wheelhouse).expanduser().resolve()),
    ]


def create_sbom(
    profile: DependencyProfile,
    artifacts: Mapping[str, WheelArtifact],
) -> dict[str, Any]:
    if profile.lock_path is None:
        raise DependencyProfileError(f"Profile {profile.name!r} has no lock")
    locked_entries = parse_lock(profile.lock_path)
    if set(artifacts) != set(locked_entries):
        missing = sorted(set(locked_entries) - set(artifacts))
        extra = sorted(set(artifacts) - set(locked_entries))
        raise DependencyProfileError(
            f"SBOM inventory does not match the lock; missing={missing}, extra={extra}"
        )
    lock_digest = lock_sha256(profile.lock_path)
    components = []
    for normalized, artifact in sorted(artifacts.items()):
        components.append({
            "type": "library",
            "bom-ref": f"pkg:pypi/{normalized}@{artifact.entry.version}",
            "name": artifact.entry.name,
            "version": artifact.entry.version,
            "purl": f"pkg:pypi/{normalized}@{artifact.entry.version}",
            "hashes": [{"alg": "SHA-256", "content": artifact.sha256}],
            "properties": [
                {"name": "slunderstudio:wheel", "value": artifact.path.name},
            ],
        })
    serial = uuid.uuid5(uuid.NAMESPACE_URL, f"slunderstudio:{profile.name}:{lock_digest}")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "component": {
                "type": "application",
                "name": "Slunder Studio optional AI runtime",
                "version": profile.name,
            },
            "properties": [
                {"name": "slunderstudio:profile", "value": profile.name},
                {"name": "slunderstudio:backend", "value": profile.backend},
                {"name": "slunderstudio:lock-sha256", "value": lock_digest},
            ],
        },
        "components": components,
        "dependencies": [{"ref": component["bom-ref"]} for component in components],
    }


def write_sbom(
    profile: DependencyProfile,
    artifacts: Mapping[str, WheelArtifact],
    output: str | Path,
) -> Path:
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = create_sbom(profile, artifacts)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def install_profile(
    name: str,
    wheelhouse: str | Path,
    *,
    registry_path: Optional[str | Path] = None,
    python_executable: Optional[str] = None,
    sbom_output: Optional[str | Path] = None,
) -> Path:
    profile, entries = validate_profile(name, registry_path=registry_path, check_host=True)
    assert_no_denylisted_packages_installed()
    artifacts = verify_wheelhouse(entries, wheelhouse)
    output = Path(sbom_output) if sbom_output else (
        Path(wheelhouse) / f"slunderstudio-{name}.cdx.json"
    )
    sbom_path = write_sbom(profile, artifacts, output)
    env = os.environ.copy()
    env["PIP_NO_INDEX"] = "1"
    subprocess.run(
        offline_install_command(
            profile,
            wheelhouse,
            python_executable=python_executable,
        ),
        check=True,
        env=env,
    )
    return sbom_path


def download_profile(
    name: str,
    wheelhouse: str | Path,
    *,
    registry_path: Optional[str | Path] = None,
    python_executable: Optional[str] = None,
    sbom_output: Optional[str | Path] = None,
) -> Path:
    profile, entries = validate_profile(name, registry_path=registry_path, check_host=True)
    root = Path(wheelhouse).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        download_command(profile, root, python_executable=python_executable),
        check=True,
    )
    artifacts = verify_wheelhouse(entries, root)
    output = Path(sbom_output) if sbom_output else root / f"slunderstudio-{name}.cdx.json"
    return write_sbom(profile, artifacts, output)


def smoke_profile(
    name: str,
    *,
    registry_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    profile, entries = validate_profile(name, registry_path=registry_path, check_host=True)
    imported = {}
    for import_name in profile.smoke_imports:
        module = importlib.import_module(import_name)
        imported[import_name] = str(getattr(module, "__version__", "available"))

    torch = importlib.import_module("torch")
    if profile.backend == "cuda":
        if not torch.cuda.is_available():
            raise DependencyProfileError("CUDA profile smoke test found no CUDA device")
        device = "cuda"
    elif profile.backend == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not mps.is_available():
            raise DependencyProfileError("MPS profile smoke test found no MPS device")
        device = "mps"
    elif profile.backend == "cpu":
        device = "cpu"
    else:
        raise DependencyProfileError(f"Unsupported smoke-test backend: {profile.backend}")

    result = (torch.tensor([1], device=device) + torch.tensor([2], device=device)).item()
    if result != 3:
        raise DependencyProfileError(f"Tensor smoke test returned {result!r}, expected 3")

    for package, expected in profile.roots.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise DependencyProfileError(f"Required package is not installed: {package}") from exc
        if installed != expected:
            raise DependencyProfileError(
                f"Installed {package} version {installed} does not match lock {expected}"
            )
    return {
        "profile": profile.name,
        "backend": profile.backend,
        "device": device,
        "packages": len(entries),
        "imports": imported,
        "tensor_result": result,
    }


def registry_diagnostics(
    *,
    registry_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    profiles: dict[str, Any] = {}
    for name in sorted(registry["profiles"]):
        try:
            profile = get_profile(name, registry=registry, require_enabled=False)
            compatible = False
            compatibility_error = ""
            if profile.enabled:
                try:
                    assert_host_compatible(profile)
                    compatible = True
                except DependencyProfileError as exc:
                    compatibility_error = str(exc)
                _, entries = validate_profile(name, registry_path=registry["_path"])
                lock_digest = lock_sha256(profile.lock_path) if profile.lock_path else ""
            else:
                entries = {}
                lock_digest = ""
            profiles[name] = {
                "enabled": profile.enabled,
                "backend": profile.backend,
                "target": f"{profile.system}/{','.join(profile.machines)}",
                "python_tag": profile.python_tag,
                "host_compatible": compatible,
                "compatibility_error": compatibility_error,
                "package_count": len(entries),
                "lock_sha256": lock_digest,
                "reason": profile.reason,
            }
        except DependencyProfileError as exc:
            profiles[name] = {
                "enabled": False,
                "valid": False,
                "error": str(exc),
            }

    installed: dict[str, Any] = {}
    for package, advisory in registry["advisory_floors"].items():
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = ""
        minimum = str(advisory.get("minimum", ""))
        installed[package] = {
            "version": version,
            "minimum": minimum,
            "advisory": str(advisory.get("advisory", "")),
            "compliant": bool(version and minimum and version_at_least(version, minimum)),
        }
    return {
        "schema_version": registry["schema_version"],
        "generated_on": registry.get("generated_on", ""),
        "profiles": profiles,
        "advisory_status": installed,
    }


def lock_from_pip_report(
    profile_name: str,
    report_path: str | Path,
    *,
    registry_path: Optional[str | Path] = None,
) -> str:
    registry = load_registry(registry_path)
    profile = get_profile(profile_name, registry=registry)
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyProfileError(f"Cannot read pip resolver report: {report_path}") from exc

    entries: dict[str, LockEntry] = {}
    for item in report.get("install", []):
        metadata = item.get("metadata") or {}
        download = item.get("download_info") or {}
        hashes = (download.get("archive_info") or {}).get("hashes") or {}
        sha256 = str(hashes.get("sha256", "")).lower()
        name = str(metadata.get("name", ""))
        version = str(metadata.get("version", ""))
        if not name or not version or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise DependencyProfileError(
                f"pip resolver report has an unhashed or incomplete archive: {name or '<unknown>'}"
            )
        normalized = normalize_name(name)
        if normalized in entries:
            raise DependencyProfileError(f"pip resolver report repeats package {name}")
        entries[normalized] = LockEntry(name=name, version=version, hashes=(sha256,))

    for package, version in profile.roots.items():
        entry = entries.get(package)
        if entry is None or entry.version != version:
            actual = entry.version if entry else "missing"
            raise DependencyProfileError(
                f"pip resolver report drift for {package}: expected {version}, got {actual}"
            )

    header = [
        f"# Slunder Studio optional AI profile: {profile.name}",
        f"# Resolved with pip {report.get('pip_version', 'unknown')} on "
        f"{registry.get('generated_on', 'unknown')}.",
        "# Every selected archive is exactly pinned and SHA-256 verified.",
        f"--index-url {profile.index_url}",
    ]
    if profile.extra_index_url:
        header.append(f"--extra-index-url {profile.extra_index_url}")
    lines = header + [""]
    for normalized in sorted(entries):
        entry = entries[normalized]
        hashes = " ".join(f"--hash=sha256:{value}" for value in entry.hashes)
        lines.append(f"{entry.name}=={entry.version} {hashes}")
    return "\n".join(lines) + "\n"
