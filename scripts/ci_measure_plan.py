#!/usr/bin/env python3
"""Plan post-merge measurements from an exact two-tree Git diff.

The planner is deliberately fail-closed.  Helpers raise ``PlanError`` for
invalid inputs or Git data; the CLI converts those errors into a plan that
runs both expensive measurements so a routing failure cannot hide them.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
MAX_GIT_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024

RELEASE_MANAGED_PATHS = frozenset(
    {
        ".release-please-manifest.json",
        "CHANGELOG.md",
        "Cargo.lock",
        "Cargo.toml",
        "app/Cargo.toml",
        "app/tauri.conf.json",
        "crates/wenlan-cli/npm/package.json",
        "crates/wenlan-mcp/npm/package.json",
        "package.json",
        "plugin-codex/.codex-plugin/plugin.json",
        "plugin-codex/skills/setup/SKILL.md",
        "plugin/.claude-plugin/plugin.json",
        "plugin/skills/setup/SKILL.md",
        "version.txt",
    }
)

BOTH_OWNER_FILES = frozenset(
    {
        "Cargo.lock",
        "Cargo.toml",
        "rust-toolchain.toml",
        "scripts/ci_measure_plan.py",
        "scripts/ci_measure_plan.test.py",
    }
)
BOTH_OWNER_PREFIXES = (
    ".cargo/",
    "app/eval/",
    "crates/wenlan-core/",
    "crates/wenlan-types/",
)
CANARY_OWNER_FILES = frozenset(
    {".config/nextest.toml", ".github/workflows/main-canary.yml"}
)
COVERAGE_OWNER_FILES = frozenset({".github/workflows/coverage.yml"})
COVERAGE_OWNER_PREFIXES = ("crates/wenlan-server/",)

# These are closed, understood surfaces which do not alter either measured
# Rust target.  A path outside the owner and non-owner sets runs both jobs.
KNOWN_NONOWNER_FILES = frozenset(
    {
        ".gitattributes",
        ".gitignore",
        "clippy.toml",
        "Cross.toml",
        "install.sh",
        "LICENSE",
        "plugin-contract.json",
        "release-please-config.json",
        "crates/wenlan-core/src/drift_guard.rs",
    }
)
KNOWN_NONOWNER_PREFIXES = (
    ".agents/",
    ".claude/",
    ".claude-plugin/",
    ".githooks/",
    ".github/ISSUE_TEMPLATE/",
    ".github/workflows/",
    "crates/wenlan-cli/",
    "crates/wenlan-mcp/",
    "docker/",
    "docs/",
    "homebrew/",
    "plugin/",
    "plugin-codex/",
    "scripts/",
)


class PlanError(RuntimeError):
    """The measurement plan cannot be proven from the supplied inputs."""


def _full_sha(value: str, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise PlanError(f"{label} is not a full lowercase commit SHA")
    return value


def _run_git(repository: Path, arguments: list[str]) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise PlanError(f"could not execute git: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise PlanError(f"git {' '.join(arguments[:2])} failed: {detail}")
    if len(result.stdout) > MAX_GIT_OUTPUT_BYTES:
        raise PlanError("git output exceeds the size bound")
    return result.stdout


def _require_commit(repository: Path, sha: str, label: str) -> None:
    output = _run_git(repository, ["rev-parse", "--verify", f"{sha}^{{commit}}"])
    try:
        resolved = output.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise PlanError(f"git returned a non-ASCII {label}") from error
    if resolved != sha:
        raise PlanError(f"{label} did not resolve to the exact supplied commit")


def parse_name_status_z(payload: bytes) -> list[tuple[str, str]]:
    """Parse ``git diff --name-status -z --no-renames`` output."""

    if not isinstance(payload, bytes):
        raise PlanError("Git diff payload is not bytes")
    fields = payload.split(b"\0")
    if fields[-1:] != [b""]:
        raise PlanError("Git diff payload is not NUL terminated")
    fields.pop()
    if len(fields) % 2:
        raise PlanError("Git diff payload has an incomplete record")
    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for offset in range(0, len(fields), 2):
        try:
            status = fields[offset].decode("ascii")
            path = fields[offset + 1].decode("utf-8")
        except UnicodeDecodeError as error:
            raise PlanError("Git diff contains invalid text encoding") from error
        if status not in {"A", "M", "D"}:
            raise PlanError(f"Git diff contains unsupported status {status!r}")
        if (
            not path
            or path.startswith("/")
            or "\x00" in path
            or "\n" in path
            or "\r" in path
            or path in seen
        ):
            raise PlanError("Git diff contains an invalid or duplicate path")
        seen.add(path)
        records.append((status, path))
    return records


def exact_two_dot_diff(
    repository: Path | str, base_sha: str, head_sha: str
) -> list[tuple[str, str]]:
    """Compare exactly ``base_sha..head_sha``; never substitute a merge base."""

    root = Path(repository)
    base = _full_sha(base_sha, "base SHA")
    head = _full_sha(head_sha, "head SHA")
    if not root.is_dir():
        raise PlanError("repository is not a directory")
    _require_commit(root, base, "base SHA")
    _require_commit(root, head, "head SHA")
    payload = _run_git(
        root,
        ["diff", "--name-status", "--no-renames", "-z", f"{base}..{head}", "--"],
    )
    return parse_name_status_z(payload)


def _git_text(repository: Path, sha: str, path: str) -> str:
    payload = _run_git(repository, ["show", f"{sha}:{path}"])
    if len(payload) > MAX_SOURCE_BYTES:
        raise PlanError(f"release-managed file {path!r} exceeds the size bound")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PlanError(f"release-managed file {path!r} is not UTF-8") from error
    if "\x00" in text:
        raise PlanError(f"release-managed file {path!r} contains NUL")
    return text


def _stable_version(value: str, label: str) -> tuple[int, int, int]:
    if VERSION_RE.fullmatch(value) is None:
        raise PlanError(f"{label} is not a stable semantic version")
    return tuple(int(part) for part in value.split("."))


def _is_exact_release_transform(
    repository: Path,
    base_sha: str,
    head_sha: str,
    records: list[tuple[str, str]],
) -> bool:
    if set(records) != {("M", path) for path in RELEASE_MANAGED_PATHS}:
        return False

    old_version = _git_text(repository, base_sha, "version.txt").strip()
    new_version = _git_text(repository, head_sha, "version.txt").strip()
    old_parts = _stable_version(old_version, "base version")
    new_parts = _stable_version(new_version, "head version")
    if new_parts != (old_parts[0], old_parts[1], old_parts[2] + 1):
        return False

    for path in sorted(RELEASE_MANAGED_PATHS):
        old = _git_text(repository, base_sha, path)
        new = _git_text(repository, head_sha, path)
        if path == "CHANGELOG.md":
            removed: list[str] = []
            added: list[str] = []
            for line in difflib.ndiff(old.splitlines(), new.splitlines()):
                if line.startswith("- "):
                    removed.append(line[2:])
                elif line.startswith("+ "):
                    added.append(line[2:])
            if len(added) > 1000 or removed or not any(
                line.startswith(f"## [{new_version}]") for line in added
            ):
                return False
        elif old_version not in old or new != old.replace(old_version, new_version):
            return False
    return True


def _is_markdown(path: str) -> bool:
    return path.casefold().endswith(".md")


def _is_known_nonowner(path: str) -> bool:
    if path in BOTH_OWNER_FILES:
        return False
    if path in KNOWN_NONOWNER_FILES or _is_markdown(path):
        return True
    if path == ".github/pull_request_template.md":
        return True
    return path.startswith(KNOWN_NONOWNER_PREFIXES)


def build_plan(
    *,
    event_name: str,
    repository: Path | str = ".",
    base_sha: str = "",
    head_sha: str = "",
) -> dict[str, object]:
    """Return a deterministic measurement plan or raise ``PlanError``."""

    if event_name == "workflow_dispatch":
        return {
            "main_canary_required": True,
            "coverage_required": True,
            "reason": "manual dispatch keeps both measurement backstops",
        }
    if event_name != "push":
        raise PlanError(f"unsupported event {event_name!r}")

    root = Path(repository)
    base = _full_sha(base_sha, "base SHA")
    head = _full_sha(head_sha, "head SHA")
    records = exact_two_dot_diff(root, base, head)
    if not records:
        return {
            "main_canary_required": False,
            "coverage_required": False,
            "reason": "exact two-tree diff is empty",
        }
    if _is_exact_release_transform(root, base, head, records):
        return {
            "main_canary_required": False,
            "coverage_required": False,
            "reason": "exact release-managed next-patch transform",
        }

    canary = False
    coverage = False
    unknown: list[str] = []
    owners: list[str] = []
    for _status, path in records:
        if path in CANARY_OWNER_FILES:
            canary = True
            owners.append(path)
        elif path in COVERAGE_OWNER_FILES:
            coverage = True
            owners.append(path)
        elif path in BOTH_OWNER_FILES:
            canary = True
            coverage = True
            owners.append(path)
        elif _is_known_nonowner(path):
            continue
        elif path.startswith(BOTH_OWNER_PREFIXES):
            canary = True
            coverage = True
            owners.append(path)
        elif path.startswith(COVERAGE_OWNER_PREFIXES):
            coverage = True
            owners.append(path)
        else:
            unknown.append(path)

    if unknown:
        canary = True
        coverage = True
        reason = f"unknown path fails closed: {sorted(unknown)[0]}"
    elif owners:
        reason = f"measurement owner changed: {sorted(owners)[0]}"
    else:
        reason = "only explicit measurement nonowners changed"
    return {
        "main_canary_required": canary,
        "coverage_required": coverage,
        "reason": reason,
    }


def fail_closed_plan(error: BaseException) -> dict[str, object]:
    detail = " ".join(str(error).splitlines()).strip() or type(error).__name__
    return {
        "main_canary_required": True,
        "coverage_required": True,
        "reason": f"planner failure fails closed: {detail}",
    }


def write_github_output(path: Path | str, plan: dict[str, object]) -> None:
    reason = plan.get("reason")
    canary = plan.get("main_canary_required")
    coverage = plan.get("coverage_required")
    if not isinstance(reason, str) or "\n" in reason or "\r" in reason:
        raise PlanError("plan reason is not a single line")
    if not isinstance(canary, bool) or not isinstance(coverage, bool):
        raise PlanError("plan required values are not booleans")
    with Path(path).open("a", encoding="utf-8", newline="\n") as output:
        output.write(f"main-canary-required={str(canary).lower()}\n")
        output.write(f"coverage-required={str(coverage).lower()}\n")
        output.write(f"reason={reason}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument(
        "--base-sha", "--before-sha", "--before", dest="base_sha", default=""
    )
    parser.add_argument(
        "--head-sha", "--after-sha", "--head", dest="head_sha", default=""
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--github-output")
    arguments = parser.parse_args(argv)

    try:
        plan = build_plan(
            event_name=arguments.event_name,
            repository=arguments.repository,
            base_sha=arguments.base_sha,
            head_sha=arguments.head_sha,
        )
    except (PlanError, OSError, ValueError) as error:
        plan = fail_closed_plan(error)

    print(json.dumps(plan, separators=(",", ":"), sort_keys=True))
    if arguments.github_output:
        write_github_output(arguments.github_output, plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
