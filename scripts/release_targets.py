#!/usr/bin/env python3
"""Expose Wenlan's canonical shipped release-target matrix."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections.abc import Iterable
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

_ASSETS = (
    {
        "name": "wenlan-darwin-arm64.tar.gz",
        "target": "aarch64-apple-darwin",
        "archive": "tar.gz",
        "members": ["wenlan", "wenlan-server", "wenlan-mcp"],
    },
    {
        "name": "wenlan-cli-darwin-arm64.tar.gz",
        "target": "aarch64-apple-darwin",
        "archive": "tar.gz",
        "members": ["wenlan"],
    },
    {
        "name": "wenlan-mcp-darwin-arm64.tar.gz",
        "target": "aarch64-apple-darwin",
        "archive": "tar.gz",
        "members": ["wenlan-mcp"],
    },
    {
        "name": "wenlan-linux-arm64.tar.gz",
        "target": "aarch64-unknown-linux-gnu",
        "archive": "tar.gz",
        "members": ["wenlan", "wenlan-server", "wenlan-mcp"],
    },
    {
        "name": "wenlan-linux-x64.tar.gz",
        "target": "x86_64-unknown-linux-gnu",
        "archive": "tar.gz",
        "members": ["wenlan", "wenlan-server", "wenlan-mcp"],
    },
    {
        "name": "wenlan-windows-x64.zip",
        "target": "x86_64-pc-windows-msvc",
        "archive": "zip",
        "members": [
            "wenlan.exe",
            "wenlan-server.exe",
            "wenlan-mcp.exe",
            "onnxruntime.dll",
            "vulkan-1.dll",
            "VulkanRT-License.txt",
        ],
    },
)


def release_matrix(*, exclude_targets: Iterable[str] = ()) -> dict:
    """Return a caller-owned matrix, optionally excluding validated targets."""

    excluded = set(exclude_targets)
    known = {entry["target"] for entry in _TARGETS}
    unknown = excluded - known
    if unknown:
        raise TargetError(f"cannot exclude unknown shipped targets: {sorted(unknown)}")
    return {
        "include": copy.deepcopy(
            [entry for entry in _TARGETS if entry["target"] not in excluded]
        )
    }


def release_preflight_matrix(*, event_name: str, ref: str, head_ref: str) -> dict:
    """Return the event-scoped preflight matrix without weakening release proof."""

    if event_name == "pull_request":
        if head_ref.startswith("release-please--branches--"):
            return release_matrix()
        return release_matrix(exclude_targets=["x86_64-pc-windows-msvc"])
    if event_name == "push":
        if ref != "refs/heads/main":
            raise TargetError(f"release preflight does not accept push ref {ref!r}")
        return release_matrix(
            exclude_targets=[
                "aarch64-apple-darwin",
                "aarch64-unknown-linux-gnu",
                "x86_64-unknown-linux-gnu",
            ]
        )
    if event_name == "workflow_dispatch":
        return release_matrix()
    raise TargetError(f"release preflight does not accept event {event_name!r}")


def require_target(target: str) -> dict:
    """Return a caller-owned target entry or fail closed."""

    for entry in _TARGETS:
        if entry["target"] == target:
            return copy.deepcopy(entry)
    raise TargetError(f"{target!r} is not a shipped release target")


def release_assets(*, target: str | None = None) -> list[dict]:
    """Return the exact caller-owned release payload inventory."""

    if target is not None:
        require_target(target)
    assets = [entry for entry in _ASSETS if target is None or entry["target"] == target]
    return copy.deepcopy(assets)


def _write_github_output(path: str, output_name: str, matrix_json: str) -> None:
    if "\n" in matrix_json or "\r" in matrix_json:
        raise TargetError("compact release matrix unexpectedly contains a newline")
    if re.fullmatch(r"[A-Za-z0-9_-]+", output_name) is None:
        raise TargetError("GitHub output name contains unsupported characters")
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(f"{output_name}={matrix_json}\n")


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--github-output")
    matrix_parser.add_argument("--output-name", default="release-targets")
    matrix_parser.add_argument("--exclude-target", action="append", default=[])

    preflight_parser = subparsers.add_parser("preflight-matrix")
    preflight_parser.add_argument("--event-name", required=True)
    preflight_parser.add_argument("--ref", required=True)
    preflight_parser.add_argument("--head-ref", default="")
    preflight_parser.add_argument("--github-output")
    preflight_parser.add_argument("--output-name", default="release-preflight-targets")

    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--target", required=True)

    assets_parser = subparsers.add_parser("assets")
    assets_parser.add_argument("--target")

    arguments = parser.parse_args(argv)
    if arguments.command == "check":
        entry = require_target(arguments.target)
        print(json.dumps(entry, separators=(",", ":"), sort_keys=True))
        return 0
    if arguments.command == "assets":
        print(
            json.dumps(
                release_assets(target=arguments.target),
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0

    if arguments.command == "preflight-matrix":
        matrix = release_preflight_matrix(
            event_name=arguments.event_name,
            ref=arguments.ref,
            head_ref=arguments.head_ref,
        )
    else:
        matrix = release_matrix(exclude_targets=arguments.exclude_target)
    matrix_json = json.dumps(matrix, separators=(",", ":"), sort_keys=True)
    print(matrix_json)
    if arguments.github_output:
        _write_github_output(arguments.github_output, arguments.output_name, matrix_json)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main(sys.argv[1:]))
    except TargetError as error:
        print(f"release_targets: {error}", file=sys.stderr)
        raise SystemExit(1)
