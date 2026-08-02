#!/usr/bin/env python3
"""Publish one crate with exact, bounded sparse-index reconciliation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Protocol


INDEX_BASE_URL = "https://index.crates.io"
FAILURE_POLL_ATTEMPTS = 12
SUCCESS_POLL_ATTEMPTS = 6
POLL_INTERVAL_SECONDS = 5.0
REQUEST_TIMEOUT_SECONDS = 5.0
MAX_INDEX_BYTES = 8 * 1024 * 1024
CRATE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


class PublishError(RuntimeError):
    """The publish operation cannot safely continue."""


class CompletedCommand(Protocol):
    returncode: int


def _validate_inputs(crate_name: str, version: str) -> None:
    if CRATE_NAME_RE.fullmatch(crate_name) is None:
        raise PublishError(f"invalid crate name: {crate_name!r}")
    if (
        not version
        or len(version) > 128
        or any(character.isspace() or ord(character) < 0x20 for character in version)
    ):
        raise PublishError(f"invalid crate version: {version!r}")


def sparse_index_path(crate_name: str) -> str:
    """Return Cargo's lowercase sparse-index path for a crate name."""

    _validate_inputs(crate_name, "0")
    name = crate_name.lower()
    if len(name) == 1:
        prefix = "1"
    elif len(name) == 2:
        prefix = "2"
    elif len(name) == 3:
        prefix = f"3/{name[0]}"
    else:
        prefix = f"{name[:2]}/{name[2:4]}"
    return f"{prefix}/{name}"


def sparse_index_contains(body: bytes, crate_name: str, version: str) -> bool:
    """Parse Cargo index NDJSON and match the exact crate and version fields."""

    _validate_inputs(crate_name, version)
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PublishError("sparse index response is not valid UTF-8") from error

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PublishError(
                f"sparse index line {line_number} is not valid JSON"
            ) from error
        if not isinstance(record, dict):
            raise PublishError(f"sparse index line {line_number} is not an object")
        if not isinstance(record.get("name"), str) or not isinstance(
            record.get("vers"), str
        ):
            raise PublishError(
                f"sparse index line {line_number} lacks string name/vers fields"
            )
        if record["name"] == crate_name and record["vers"] == version:
            return True
    return False


def sparse_version_exists(
    crate_name: str,
    version: str,
    *,
    index_base_url: str = INDEX_BASE_URL,
    request_timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> bool:
    """Fetch one sparse package file and look for an exact version record."""

    url = f"{index_base_url.rstrip('/')}/{sparse_index_path(crate_name)}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/plain",
            "Cache-Control": "no-cache",
            "User-Agent": "wenlan-release-publisher/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            body = response.read(MAX_INDEX_BYTES + 1)
    except urllib.error.HTTPError as error:
        if error.code in {404, 410, 451}:
            return False
        raise PublishError(
            f"sparse index request failed with HTTP {error.code}: {url}"
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise PublishError(f"sparse index request failed: {url}: {error}") from error
    if len(body) > MAX_INDEX_BYTES:
        raise PublishError(f"sparse index response exceeds {MAX_INDEX_BYTES} bytes")
    return sparse_index_contains(body, crate_name, version)


def _poll_for_exact_version(
    crate_name: str,
    version: str,
    *,
    attempts: int,
    probe: Callable[[str, str], bool],
    sleeper: Callable[[float], None],
    poll_interval: float,
) -> bool:
    if attempts < 1:
        raise PublishError("poll attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            if probe(crate_name, version):
                print(f"{crate_name} {version} is visible on the sparse index")
                return True
        except PublishError as error:
            print(
                f"WARNING: sparse index attempt {attempt}/{attempts} failed: {error}",
                file=sys.stderr,
            )
        if attempt < attempts:
            print(
                f"Waiting for exact sparse index version {crate_name} {version} "
                f"({attempt}/{attempts})"
            )
            sleeper(poll_interval)
    return False


def publish_crate(
    crate_name: str,
    version: str,
    *,
    registry_token: str | None,
    cargo_bin: str = "cargo",
    probe: Callable[[str, str], bool] = sparse_version_exists,
    run_command: Callable[[Sequence[str]], CompletedCommand] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    poll_interval: float = POLL_INTERVAL_SECONDS,
    failure_poll_attempts: int = FAILURE_POLL_ATTEMPTS,
    success_poll_attempts: int = SUCCESS_POLL_ATTEMPTS,
) -> int:
    """Publish or reconcile one exact crate version; return a process status."""

    _validate_inputs(crate_name, version)
    try:
        already_published = probe(crate_name, version)
    except PublishError as error:
        raise PublishError(
            "pre-publish exact-version sparse check failed; refusing to upload: "
            f"{error}"
        ) from error
    if already_published:
        print(f"{crate_name} {version} already exists; skipping cargo publish")
        return 0
    if not registry_token:
        raise PublishError(
            "CARGO_REGISTRY_TOKEN is required because "
            f"{crate_name} {version} is not published"
        )

    command = [cargo_bin, "publish", "-p", crate_name]
    runner = run_command or (lambda argv: subprocess.run(argv, check=False))
    publish_result = runner(command)
    if not isinstance(publish_result.returncode, int):
        raise PublishError("cargo publish returned an invalid process status")

    if publish_result.returncode != 0:
        print(
            f"cargo publish failed with {publish_result.returncode}; "
            "checking whether the upload nevertheless reached the exact sparse index",
            file=sys.stderr,
        )
        if _poll_for_exact_version(
            crate_name,
            version,
            attempts=failure_poll_attempts,
            probe=probe,
            sleeper=sleeper,
            poll_interval=poll_interval,
        ):
            return 0
        print(
            f"ERROR: {crate_name} {version} is still absent; preserving cargo publish "
            f"status {publish_result.returncode}",
            file=sys.stderr,
        )
        return publish_result.returncode

    if _poll_for_exact_version(
        crate_name,
        version,
        attempts=success_poll_attempts,
        probe=probe,
        sleeper=sleeper,
        poll_interval=poll_interval,
    ):
        return 0
    raise PublishError(
        f"cargo publish succeeded but {crate_name} {version} did not become visible "
        f"after {success_poll_attempts} short sparse-index checks"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, dest="crate_name")
    parser.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    try:
        return publish_crate(
            args.crate_name,
            args.version,
            registry_token=os.environ.get("CARGO_REGISTRY_TOKEN"),
        )
    except PublishError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
