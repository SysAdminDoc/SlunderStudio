#!/usr/bin/env python3
"""Prepare, verify, install, inventory, and smoke-test optional AI profiles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dependency_profiles import (  # noqa: E402
    DependencyProfileError,
    download_profile,
    get_profile,
    install_profile,
    load_registry,
    lock_from_pip_report,
    registry_diagnostics,
    smoke_profile,
    validate_profile,
    verify_wheelhouse,
    write_sbom,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage Slunder Studio hash-locked optional AI runtimes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="Show profiles, target platforms, and lock status.")

    validate = subparsers.add_parser("validate", help="Validate one profile and its lock.")
    validate.add_argument("profile")

    for command, help_text in (
        ("download", "Download a verified wheelhouse while network access is available."),
        ("verify", "Verify every locked archive in an existing wheelhouse."),
        ("install", "Install exclusively from a verified offline wheelhouse."),
        ("sbom", "Write a complete CycloneDX inventory for a verified wheelhouse."),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("profile")
        child.add_argument("--wheelhouse", required=True, type=Path)
        if command in {"download", "install", "sbom"}:
            child.add_argument("--sbom-output", type=Path)
        if command in {"download", "install"}:
            child.add_argument("--python", dest="python_executable")

    smoke = subparsers.add_parser("smoke", help="Run imports and a backend tensor operation.")
    smoke.add_argument("profile")

    lock_report = subparsers.add_parser(
        "lock-report",
        help="Convert an audited pip --report JSON file into a hashed lock.",
    )
    lock_report.add_argument("profile")
    lock_report.add_argument("report", type=Path)
    lock_report.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            print(json.dumps(registry_diagnostics(), indent=2, sort_keys=True))
            return 0
        if args.command == "validate":
            profile, entries = validate_profile(args.profile)
            print(f"{profile.name}: valid ({len(entries)} packages)")
            return 0
        if args.command == "download":
            path = download_profile(
                args.profile,
                args.wheelhouse,
                python_executable=args.python_executable,
                sbom_output=args.sbom_output,
            )
            print(f"Verified wheelhouse and wrote SBOM: {path}")
            return 0
        if args.command == "verify":
            profile, entries = validate_profile(args.profile, check_host=True)
            artifacts = verify_wheelhouse(entries, args.wheelhouse)
            print(f"{profile.name}: verified {len(artifacts)} wheel archives")
            return 0
        if args.command == "install":
            path = install_profile(
                args.profile,
                args.wheelhouse,
                python_executable=args.python_executable,
                sbom_output=args.sbom_output,
            )
            print(f"Offline installation complete; SBOM: {path}")
            return 0
        if args.command == "sbom":
            profile, entries = validate_profile(args.profile, check_host=True)
            artifacts = verify_wheelhouse(entries, args.wheelhouse)
            output = args.sbom_output or (
                args.wheelhouse / f"slunderstudio-{profile.name}.cdx.json"
            )
            path = write_sbom(profile, artifacts, output)
            print(path)
            return 0
        if args.command == "smoke":
            print(json.dumps(smoke_profile(args.profile), indent=2, sort_keys=True))
            return 0
        if args.command == "lock-report":
            profile = get_profile(args.profile, registry=load_registry())
            output = args.output or profile.lock_path
            if output is None:
                raise DependencyProfileError(f"Profile {profile.name!r} has no lock path")
            content = lock_from_pip_report(profile.name, args.report)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with Path(output).open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            print(output)
            return 0
    except (DependencyProfileError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
