#!/usr/bin/env python3
"""Verify the actual crates.io ort-sys source behind Wenlan's Windows DLL pin."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


EXPECTED_CRATE_VERSION = "2.0.0-rc.11"
EXPECTED_COMMIT = "2de34065983a5c034f5afcc072b23b99479f465b"
EXPECTED_ORT_VERSION = "1.23.2"
EXPECTED_API_VERSION = 23
WINDOWS_TARGET = "x86_64-pc-windows-msvc"


def locate_ort_sys_source(repo_root: Path) -> Path:
    result = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)
    matches = [
        package
        for package in metadata["packages"]
        if package["name"] == "ort-sys"
        and package["version"] == EXPECTED_CRATE_VERSION
        and package["source"].startswith("registry+")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "expected one crates.io ort-sys "
            f"{EXPECTED_CRATE_VERSION}, found {len(matches)}"
        )
    return Path(matches[0]["manifest_path"]).parent


def verify_ort_sys_source(source_root: Path) -> list[str]:
    violations = []

    vcs = json.loads(
        (source_root / ".cargo_vcs_info.json").read_text(encoding="utf-8")
    )
    actual_commit = vcs.get("git", {}).get("sha1")
    if actual_commit != EXPECTED_COMMIT:
        violations.append(
            f"ort-sys package commit mismatch: expected {EXPECTED_COMMIT}, got {actual_commit}"
        )
    if vcs.get("path_in_vcs") != "ort-sys":
        violations.append("ort-sys package path_in_vcs is not 'ort-sys'")

    dist_lines = (
        source_root / "build/download/dist.txt"
    ).read_text(encoding="utf-8").splitlines()
    windows_cpu_rows = [
        line.split("\t")
        for line in dist_lines
        if line.startswith(f"none\t{WINDOWS_TARGET}\t")
    ]
    expected_dist_fragment = f"/ms@{EXPECTED_ORT_VERSION}/"
    if len(windows_cpu_rows) != 1 or expected_dist_fragment not in windows_cpu_rows[0][2]:
        violations.append(
            "ort-sys Windows x64 CPU distribution does not pin "
            f"ONNX Runtime {EXPECTED_ORT_VERSION}"
        )

    version_source = (source_root / "src/version.rs").read_text(encoding="utf-8")
    api_match = re.search(r"ORT_API_VERSION:\s*u32\s*=\s*(\d+)", version_source)
    actual_api_version = int(api_match.group(1)) if api_match else None
    if actual_api_version != EXPECTED_API_VERSION:
        violations.append(
            f"ort-sys must expose ORT API version {EXPECTED_API_VERSION}, "
            f"got {actual_api_version}"
        )

    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ort-sys-dir", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    source_root = args.ort_sys_dir or locate_ort_sys_source(repo_root)
    violations = verify_ort_sys_source(source_root)
    if violations:
        print("ort-sys source pin verification failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1

    print(
        "verified crates.io ort-sys "
        f"{EXPECTED_CRATE_VERSION} ({EXPECTED_COMMIT}) -> "
        f"ONNX Runtime {EXPECTED_ORT_VERSION}, API {EXPECTED_API_VERSION}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
