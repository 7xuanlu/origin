#!/usr/bin/env python3
"""Expose Wenlan's canonical shipped release-target matrix."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


class TargetError(ValueError):
    """A requested target is not part of the shipped release surface."""


_TARGETS = (
    {
        "target": "aarch64-apple-darwin",
        "os": "macos-14",
        "archive": "tar.gz",
        "artifact_name": "wenlan-darwin-arm64",
    },
    # Intel macOS is intentionally absent: ort 2.x has no matching prebuilt
    # runtime, so restoring it requires a separate source-build decision.
    {
        "target": "aarch64-unknown-linux-gnu",
        "os": "ubuntu-24.04-arm",
        "archive": "tar.gz",
        "artifact_name": "wenlan-linux-arm64",
    },
    {
        "target": "x86_64-unknown-linux-gnu",
        "os": "ubuntu-24.04",
        "archive": "tar.gz",
        "artifact_name": "wenlan-linux-x64",
    },
    {
        "target": "x86_64-pc-windows-msvc",
        "os": "windows-2022",
        "archive": "zip",
        "artifact_name": "wenlan-windows-x64",
    },
)


def release_matrix() -> dict:
    """Return a caller-owned copy of the canonical GitHub Actions matrix."""

    return {"include": copy.deepcopy(list(_TARGETS))}


def require_target(target: str) -> dict:
    """Return a caller-owned target entry or fail closed."""

    for entry in _TARGETS:
        if entry["target"] == target:
            return copy.deepcopy(entry)
    raise TargetError(f"{target!r} is not a shipped release target")


def _write_github_output(path: str, matrix_json: str) -> None:
    if "\n" in matrix_json or "\r" in matrix_json:
        raise TargetError("compact release matrix unexpectedly contains a newline")
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(f"release-targets={matrix_json}\n")


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--github-output")

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--target", required=True)

    arguments = parser.parse_args(argv)
    if arguments.command == "check":
        entry = require_target(arguments.target)
        print(json.dumps(entry, separators=(",", ":"), sort_keys=True))
        return 0

    matrix_json = json.dumps(
        release_matrix(),
        separators=(",", ":"),
        sort_keys=True,
    )
    print(matrix_json)
    if arguments.github_output:
        _write_github_output(arguments.github_output, matrix_json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except TargetError as error:
        print(f"release_targets: {error}", file=sys.stderr)
        raise SystemExit(1)
