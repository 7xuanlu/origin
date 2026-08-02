#!/usr/bin/env python3
"""Inspect and execute the freshly packaged archives in an unprivileged build job."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from release_archive import ArchiveError, safe_extract_archive
from release_targets import TargetError, release_assets, require_target


class SmokeError(RuntimeError):
    """A canonical archive does not pass its native smoke."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as payload:
        while chunk := payload.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def smoke_target(target: str, dist_dir: Path, windows_smoke_script: Path) -> None:
    target_entry = require_target(target)
    assets = release_assets(target=target)
    expected_names = {asset["name"] for asset in assets}
    actual_names = {path.name for path in dist_dir.iterdir()}
    if actual_names != expected_names:
        raise SmokeError(
            f"dist inventory mismatch: missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )

    with tempfile.TemporaryDirectory(prefix="wenlan-archive-smoke-") as raw:
        root = Path(raw)
        extracted_by_name: dict[str, Path] = {}
        for index, asset in enumerate(assets):
            archive = dist_dir / asset["name"]
            destination = root / str(index)
            safe_extract_archive(
                archive, destination, asset["archive"], asset["members"]
            )
            extracted_by_name[asset["name"]] = destination

        main_name = f"{target_entry['artifact_name']}.{target_entry['archive']}"
        main_dir = extracted_by_name[main_name]
        if target == "aarch64-apple-darwin":
            parity = [
                (
                    main_dir / "wenlan",
                    extracted_by_name["wenlan-cli-darwin-arm64.tar.gz"] / "wenlan",
                ),
                (
                    main_dir / "wenlan-mcp",
                    extracted_by_name["wenlan-mcp-darwin-arm64.tar.gz"] / "wenlan-mcp",
                ),
            ]
            if any(_sha256(main) != _sha256(standalone) for main, standalone in parity):
                raise SmokeError("Darwin standalone archive bytes differ from the multi archive")
        suffix = ".exe" if target == "x86_64-pc-windows-msvc" else ""
        for binary in ("wenlan", "wenlan-server", "wenlan-mcp"):
            executable = main_dir / f"{binary}{suffix}"
            if os.name != "nt":
                executable.chmod(0o755)
            result = subprocess.run(
                [str(executable), "--help"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                raise SmokeError(
                    f"{executable.name} --help failed with {result.returncode}"
                )

        if target == "x86_64-pc-windows-msvc":
            if not windows_smoke_script.is_file():
                raise SmokeError(f"Windows semantic smoke is missing: {windows_smoke_script}")
            result = subprocess.run(
                [
                    "pwsh",
                    "-NoProfile",
                    "-File",
                    str(windows_smoke_script),
                    "-ExePath",
                    str(main_dir / "wenlan-server.exe"),
                ],
                stdin=subprocess.DEVNULL,
                timeout=300,
                check=False,
            )
            if result.returncode != 0:
                raise SmokeError(
                    f"packaged Windows semantic smoke failed with {result.returncode}"
                )


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dist-dir", required=True, type=Path)
    parser.add_argument(
        "--windows-smoke-script",
        type=Path,
        default=Path(__file__).with_name("smoke-windows.ps1"),
    )
    args = parser.parse_args(argv)
    smoke_target(args.target, args.dist_dir, args.windows_smoke_script)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except (ArchiveError, SmokeError, TargetError, OSError, subprocess.SubprocessError) as error:
        print(f"smoke-release-archive: {error}", file=sys.stderr)
        raise SystemExit(1)
