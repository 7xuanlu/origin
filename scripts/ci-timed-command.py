#!/usr/bin/env python3
"""Run one benchmark command and always write a machine-readable timing receipt."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash", dest="hash_paths", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tool_version(command):
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def main():
    args = parse_args()
    configured_target = os.environ.get("CARGO_TARGET_DIR")
    disk_path = Path(configured_target) if configured_target else Path.cwd()
    if not disk_path.exists():
        disk_path = Path.cwd()
    disk = shutil.disk_usage(disk_path)
    tool_versions = {
        "cargo": tool_version(["cargo", "-V"]),
        "rustc": tool_version(["rustc", "-vV"]),
        "nextest": tool_version(["cargo", "nextest", "--version"]),
    }
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    try:
        exit_code = subprocess.run(args.command, check=False).returncode
    except OSError as error:
        exit_code = 127
        command_error = str(error)
    else:
        command_error = None
    completed_at = datetime.now(timezone.utc)
    duration = time.monotonic() - started

    outputs = []
    for path in args.hash_paths:
        outputs.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    receipt = {
        "schema_version": 1,
        "label": args.label,
        "command": args.command,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": round(duration, 3),
        "exit_code": exit_code,
        "command_error": command_error,
        "runner": {
            "os": os.environ.get("RUNNER_OS"),
            "arch": os.environ.get("RUNNER_ARCH"),
            "name": os.environ.get("RUNNER_NAME"),
            "image_os": os.environ.get("ImageOS"),
            "image_version": os.environ.get("ImageVersion"),
            "temp": os.environ.get("RUNNER_TEMP"),
            "workspace": os.environ.get("GITHUB_WORKSPACE"),
            "sha": os.environ.get("GITHUB_SHA"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "requested_label": os.environ.get("BENCH_REQUESTED_RUNNER"),
        },
        "tool_versions": tool_versions,
        "build_environment": {
            "rustflags": os.environ.get("RUSTFLAGS"),
            "cargo_home": os.environ.get("CARGO_HOME"),
            "cargo_target_dir": os.environ.get("CARGO_TARGET_DIR"),
            "release_codegen_units": os.environ.get(
                "CARGO_PROFILE_RELEASE_CODEGEN_UNITS"
            ),
            "release_lto": os.environ.get("CARGO_PROFILE_RELEASE_LTO"),
            "release_strip": os.environ.get("CARGO_PROFILE_RELEASE_STRIP"),
            "release_incremental": os.environ.get(
                "CARGO_PROFILE_RELEASE_INCREMENTAL"
            ),
        },
        "disk": {
            "path": str(disk_path),
            "cwd": str(Path.cwd()),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
